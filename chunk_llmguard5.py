"""
title: Prompt Injection Detection Filter Offline
author: open-webui
date: 2024-11-20
version: 3.2
license: MIT
description: Offline pipeline for detecting prompt injections with chunking and proper blocking
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
        threshold: float = 0.5  # Снижен порог для лучшей чувствительности
        enable_filtering: bool = True
        chunk_size: int = 256
        chunk_overlap: int = 50
        max_chunks: int = 10
        block_on_detection: bool = True  # Явный флаг для блокировки

    def __init__(self):
        self.type = "filter"
        self.id = "prompt_injection_detector chunking 5"
        self.name = "Prompt Injection Detector chunking 5"
        
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

    def find_suspicious_sentences(self, text: str, threshold: float) -> List[tuple]:
      """
      Находит подозрительные предложения в тексте
      """
      sentences = re.split(r'(?<=[.!?])\s+', text)
      suspicious = []
      
      for sentence in sentences:
          if not sentence.strip():
              continue
              
          _, risk_score = self.predict_single(sentence)
          if risk_score > threshold:  # Используем точный порог, без умножения
              suspicious.append((sentence, risk_score))
      
      return suspicious

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
      Предсказание с разбиением на чанки
      Возвращает: (is_safe, max_risk, suspicious_chunks, suspicious_sentences)
      """
      if not self.model_loaded:
          print("⚠️ Модель не загружена, пропускаем проверку")
          return True, 0.0, [], []
      
      print(f"\n🔍 Анализ текста длиной {len(text)} символов")
      
      # Проверяем весь текст целиком
      _, risk_score_whole = self.predict_single(text)
      max_risk = risk_score_whole
      
      print(f"  Весь текст: риск {max_risk:.2%}")
      
      # Разбиваем на чанки и проверяем каждый
      chunks = self.split_into_chunks(text)
      print(f"  Разбито на {len(chunks)} чанков")
      
      suspicious_chunks = []
      
      for i, chunk in enumerate(chunks):
          _, chunk_risk = self.predict_single(chunk)
          print(f"    Чанк {i+1}: риск {chunk_risk:.2%}")
          
          max_risk = max(max_risk, chunk_risk)
          
          if chunk_risk > self.valves.threshold:
              suspicious_chunks.append({
                  "chunk_index": i,
                  "risk": chunk_risk,
                  "text": chunk[:150] + "..." if len(chunk) > 150 else chunk
              })
              print(f"    ⚠️ Подозрительный чанк {i+1} (риск {chunk_risk:.1%})")
          
          # Ранний выход если риск слишком высокий
          if max_risk > 0.95:
              print(f"  ⚠️ Очень высокий риск {max_risk:.1%}, прерываем проверку")
              break
    
      # Ищем подозрительные предложения
      suspicious_sentences = self.find_suspicious_sentences(text, self.valves.threshold)
      
      # ВАЖНО: проверяем и чанки, и предложения
      # Если есть подозрительные предложения с риском выше порога - это не безопасно
      has_high_risk_sentences = any(risk > self.valves.threshold for _, risk in suspicious_sentences)
      
      # Определяем безопасность: 
      # - max_risk должен быть меньше порога
      # - И не должно быть подозрительных предложений выше порога
      is_safe = (max_risk < self.valves.threshold) and not has_high_risk_sentences
      
      print(f"\n  Результат: is_safe={is_safe}, max_risk={max_risk:.2%}, threshold={self.valves.threshold:.0%}")
      print(f"  Подозрительных чанков: {len(suspicious_chunks)}, предложений: {len(suspicious_sentences)}")
      if suspicious_sentences:
          print(f"  Есть предложения выше порога: {has_high_risk_sentences}")
      
      return is_safe, max_risk, suspicious_chunks, suspicious_sentences

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
        is_safe, max_risk, suspicious_chunks, suspicious_sentences = self.predict_chunked(user_message)
        
        # Логируем результат
        print(f"\n📊 ИТОГОВЫЙ РЕЗУЛЬТАТ:")
        print(f"  Безопасно: {is_safe}")
        print(f"  Макс. риск: {max_risk:.2%}")
        print(f"  Порог: {self.valves.threshold:.0%}")
        
        if suspicious_chunks:
            print(f"  Подозрительные чанки: {len(suspicious_chunks)}")
            for chunk in suspicious_chunks:
                print(f"    • Чанк {chunk['chunk_index']+1}: риск {chunk['risk']:.1%}")
                print(f"      Текст: {chunk['text']}")
        
        if suspicious_sentences:
            print(f"  Подозрительные предложения: {len(suspicious_sentences)}")
            for sent, risk in suspicious_sentences[:3]:
                sent_short = sent[:100] + "..." if len(sent) > 100 else sent
                print(f"    • Риск {risk:.1%}: {sent_short}")
        
        # БЛОКИРУЕМ если не безопасно
        if not is_safe and self.valves.block_on_detection:
            error_msg = f"🚫 ЗАПРОС ЗАБЛОКИРОВАН: Обнаружена prompt injection\n"
            error_msg += f"Максимальный риск: {max_risk:.1%} (порог: {self.valves.threshold:.0%})\n"
            
            if suspicious_chunks:
                error_msg += f"\nПодозрительные фрагменты ({len(suspicious_chunks)}):\n"
                for chunk in suspicious_chunks[:3]:
                    error_msg += f"  • Риск {chunk['risk']:.1%}: {chunk['text']}\n"
            
            if suspicious_sentences:
                error_msg += f"\nПодозрительные предложения ({len(suspicious_sentences)}):\n"
                for sent, risk in suspicious_sentences[:3]:
                    sent_short = sent[:150] + "..." if len(sent) > 150 else sent
                    error_msg += f"  • Риск {risk:.1%}: {sent_short}\n"
            
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
            "chunks_checked": len(self.split_into_chunks(user_message)),
            "suspicious_chunks": len(suspicious_chunks),
            "suspicious_sentences": len(suspicious_sentences)
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
