"""
title: Prompt Injection Detection Filter Offline
author: open-webui
date: 2024-11-20
version: 3.3
license: MIT
description: Offline pipeline for detecting prompt injections with chunking and sentence-level detection
requirements: transformers>=4.35.0, torch>=2.0.0
"""

from typing import List, Optional
from pydantic import BaseModel
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        threshold: float = 0.3  # Еще снизил порог
        enable_filtering: bool = True
        chunk_size: int = 256
        chunk_overlap: int = 50
        max_chunks: int = 10
        block_on_detection: bool = True
        check_sentences: bool = True  # Проверять отдельные предложения

    def __init__(self):
        self.type = "filter"
        self.id = "prompt_injection_detector"
        self.name = "Prompt Injection Detector"
        
        self.valves = self.Valves()
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.model_path = "/app/model/prompt-injection"

    async def on_startup(self):
        try:
            print(f"🚀 Загрузка модели с {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            print(f"✅ Prompt Injection Detector loaded on {self.device}")
            print(f"⚙️ Порог срабатывания: {self.valves.threshold:.0%}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model_loaded = False

    def split_into_sentences(self, text: str) -> List[str]:
        """Разбивает текст на предложения"""
        # Регулярка для разбиения на предложения
        sentence_endings = r'(?<=[.!?])\s+(?=[А-ЯA-Z])'
        sentences = re.split(sentence_endings, text)
        
        # Дополнительно разбиваем по переносам строк
        result = []
        for sent in sentences:
            parts = sent.split('\n')
            result.extend([p.strip() for p in parts if p.strip()])
        
        return result

    def split_into_chunks(self, text: str) -> List[str]:
        """
        Разбивает текст на перекрывающиеся чанки
        """
        if not self.tokenizer:
            return [text]
            
        # Сначала токенизируем весь текст
        tokens = self.tokenizer.encode(text, truncation=False)
        
        if len(tokens) <= self.valves.chunk_size:
            return [text]
        
        chunks = []
        chunk_size = self.valves.chunk_size
        overlap = self.valves.chunk_overlap
        step = chunk_size - overlap
        
        # Разбиваем на чанки с перекрытием
        for i in range(0, len(tokens), step):
            if i >= self.valves.max_chunks * step:
                break
                
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Пытаемся найти границу предложения
            sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
            if len(sentences) > 1:
                # Берем текст до последней точки
                chunk_text = ' '.join(sentences[:-1])
            
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks

    def predict_single(self, text: str):
        """
        Предсказание для одного текста
        """
        if not self.model_loaded or not self.model or not self.tokenizer:
            return True, 0.0
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                prob = F.softmax(outputs.logits, dim=-1)[0, 1].item()
            
            return prob < self.valves.threshold, prob
        except Exception as e:
            print(f"⚠️ Ошибка при предсказании: {e}")
            return True, 0.0

    def predict_chunked(self, text: str):
        """
        Предсказание с разбиением на чанки и предложения
        Возвращает: (is_safe, max_risk, all_suspicious_items)
        """
        if not self.model_loaded:
            print("⚠️ Модель не загружена, пропускаем проверку")
            return True, 0.0, []
        
        print(f"\n🔍 Анализ текста длиной {len(text)} символов")
        
        all_risks = []
        suspicious_items = []
        
        # 1. Проверяем весь текст целиком
        _, risk_whole = self.predict_single(text)
        all_risks.append(("весь текст", risk_whole))
        print(f"  Весь текст: риск {risk_whole:.2%}")
        
        if risk_whole > self.valves.threshold:
            suspicious_items.append({
                "type": "full_text",
                "risk": risk_whole,
                "text": text[:200] + "..." if len(text) > 200 else text
            })
        
        # 2. Проверяем по предложениям (самое важное!)
        if self.valves.check_sentences:
            sentences = self.split_into_sentences(text)
            print(f"  Разбито на {len(sentences)} предложений")
            
            for i, sentence in enumerate(sentences):
                if len(sentence) < 10:  # Пропускаем слишком короткие
                    continue
                    
                _, risk_sent = self.predict_single(sentence)
                all_risks.append((f"предложение {i+1}", risk_sent))
                
                if risk_sent > self.valves.threshold * 0.8:  # Чуть ниже порога для логирования
                    print(f"    Предложение {i+1}: риск {risk_sent:.2%} - {sentence[:100]}...")
                
                if risk_sent > self.valves.threshold:
                    suspicious_items.append({
                        "type": "sentence",
                        "index": i,
                        "risk": risk_sent,
                        "text": sentence
                    })
        
        # 3. Проверяем по чанкам
        chunks = self.split_into_chunks(text)
        if len(chunks) > 1:  # Если текст действительно длинный
            print(f"  Разбито на {len(chunks)} чанков")
            
            for i, chunk in enumerate(chunks):
                _, risk_chunk = self.predict_single(chunk)
                all_risks.append((f"чанк {i+1}", risk_chunk))
                
                if risk_chunk > self.valves.threshold:
                    suspicious_items.append({
                        "type": "chunk",
                        "index": i,
                        "risk": risk_chunk,
                        "text": chunk[:200] + "..." if len(chunk) > 200 else chunk
                    })
        
        # Находим максимальный риск
        max_risk_item = max(all_risks, key=lambda x: x[1])
        max_risk = max_risk_item[1]
        
        print(f"\n  Максимальный риск: {max_risk:.2%} (в {max_risk_item[0]})")
        
        # Определяем безопасность: если max_risk >= threshold - НЕ безопасно
        is_safe = max_risk < self.valves.threshold
        
        print(f"  Результат: is_safe={is_safe}, max_risk={max_risk:.2%}, threshold={self.valves.threshold:.0%}")
        print(f"  Подозрительных элементов: {len(suspicious_items)}")
        
        return is_safe, max_risk, suspicious_items

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Входной фильтр для сообщений
        """
        print("\n" + "="*60)
        print("🔍 Prompt Injection Detector: проверка сообщения")
        print("="*60)
        
        # Проверяем настройки
        if not self.valves.enable_filtering:
            print("⚠️ Фильтрация отключена в настройках")
            return body
        
        if not self.model_loaded:
            print("⚠️ Модель не загружена, пропускаем проверку")
            return body
        
        # Получаем сообщение
        messages = body.get("messages", [])
        if not messages:
            print("⚠️ Нет сообщений в запросе")
            return body
        
        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            print("⚠️ Последнее сообщение не от пользователя")
            return body
        
        user_message = last_msg.get("content", "").strip()
        if not user_message:
            print("⚠️ Пустое сообщение")
            return body
        
        print(f"📝 Сообщение от пользователя: {user_message[:200]}..." if len(user_message) > 200 else f"📝 Сообщение: {user_message}")
        
        # Проверяем сообщение
        is_safe, max_risk, suspicious_items = self.predict_chunked(user_message)
        
        # Логируем результат
        print(f"\n📊 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
        print(f"  Безопасно: {is_safe}")
        print(f"  Макс. риск: {max_risk:.2%}")
        print(f"  Порог: {self.valves.threshold:.0%}")
        
        if suspicious_items:
            print(f"  Найдено подозрительных элементов: {len(suspicious_items)}")
            for item in suspicious_items:
                print(f"    • {item['type']} (риск {item['risk']:.1%}): {item['text'][:150]}...")
        
        # БЛОКИРУЕМ если не безопасно
        if not is_safe and self.valves.block_on_detection:
            error_msg = f"🚫 ЗАПРОС ЗАБЛОКИРОВАН: Обнаружена prompt injection\n"
            error_msg += f"Максимальный риск: {max_risk:.1%} (порог: {self.valves.threshold:.0%})\n"
            
            if suspicious_items:
                error_msg += f"\nОбнаруженные подозрительные элементы:\n"
                for item in suspicious_items[:5]:  # Показываем первые 5
                    error_msg += f"  • {item['type']} (риск {item['risk']:.1%}): {item['text'][:200]}\n"
            
            print(f"\n❌ {error_msg}")
            print("="*60)
            
            # ВЫБРАСЫВАЕМ ИСКЛЮЧЕНИЕ ДЛЯ БЛОКИРОВКИ
            raise Exception(error_msg)
        
        # Если безопасно или блокировка отключена, добавляем метаданные
        if is_safe:
            print("\n✅ Сообщение безопасно, пропускаем")
        else:
            print(f"\n⚠️ Обнаружена инжекция, но блокировка отключена (block_on_detection=False)")
        
        # Добавляем метаданные о проверке
        if "metadata" not in body:
            body["metadata"] = {}
        
        body["metadata"]["prompt_injection_check"] = {
            "max_risk": max_risk,
            "threshold": self.valves.threshold,
            "is_safe": is_safe,
            "suspicious_items": len(suspicious_items)
        }
        
        print("="*60 + "\n")
        return body

    async def on_shutdown(self):
        """Очистка при выключении"""
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🧹 Модель выгружена")
