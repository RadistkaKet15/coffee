"""
title: Prompt Injection Detection Filter Offline
author: open-webui
date: 2024-11-20
version: 3.1
license: MIT
description: Offline pipeline for detecting prompt injections with chunking
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
        threshold: float = 0.75
        enable_filtering: bool = True
        chunk_size: int = 256  # Размер чанка в токенах
        chunk_overlap: int = 50  # Перекрытие чанков для надежности
        max_chunks: int = 10  # Максимальное количество чанков для проверки

    def __init__(self):
        self.type = "filter"
        self.id = "prompt_injection_detector_ chanking"
        self.name = "Prompt Injection Detector Chanking"
        
        self.valves = self.Valves()
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.model_path = "/app/model/prompt-injection"

    async def on_startup(self):
        try:
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
            print("✅ Prompt Injection Detector loaded with chunking support")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model_loaded = False

    def split_into_chunks(self, text: str) -> List[str]:
        """
        Разбивает текст на перекрывающиеся чанки по границам предложений
        """
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
            if risk_score > threshold:
                suspicious.append((sentence, risk_score))
        
        return suspicious

    def predict_single(self, text: str):
        """
        Предсказание для одного текста
        """
        if not self.model_loaded:
            return True, 0.0
        
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

    def predict_chunked(self, text: str):
        """
        Предсказание с разбиением на чанки
        """
        if not self.model_loaded:
            return True, 0.0, []
        
        # Проверяем весь текст целиком
        is_safe_whole, risk_score_whole = self.predict_single(text)
        
        # Разбиваем на чанки и проверяем каждый
        chunks = self.split_into_chunks(text)
        max_risk = risk_score_whole
        suspicious_chunks = []
        
        for i, chunk in enumerate(chunks):
            _, chunk_risk = self.predict_single(chunk)
            max_risk = max(max_risk, chunk_risk)
            
            if chunk_risk > self.valves.threshold:
                suspicious_chunks.append({
                    "chunk_index": i,
                    "risk": chunk_risk,
                    "text": chunk[:100] + "..." if len(chunk) > 100 else chunk
                })
            
            # Ранний выход если риск слишком высокий
            if max_risk > 0.95:
                break
        
        # Ищем подозрительные предложения в самом рискованном чанке
        if suspicious_chunks:
            highest_risk_chunk = max(suspicious_chunks, key=lambda x: x["risk"])
            suspicious_sentences = self.find_suspicious_sentences(
                highest_risk_chunk["text"], 
                self.valves.threshold * 0.9
            )
        else:
            suspicious_sentences = self.find_suspicious_sentences(text, self.valves.threshold)
        
        is_safe = max_risk < self.valves.threshold
        
        return is_safe, max_risk, suspicious_chunks, suspicious_sentences

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if not self.valves.enable_filtering or not self.model_loaded:
            return body
        
        messages = body.get("messages", [])
        if not messages:
            return body
        
        last_msg = messages[-1]
        if last_msg.get("role") != "user":
            return body
        
        user_message = last_msg.get("content", "").strip()
        if not user_message:
            return body
        
        # Используем чанкованную проверку
        is_safe, max_risk, suspicious_chunks, suspicious_sentences = self.predict_chunked(user_message)
        
        # Детальное логирование для отладки
        if suspicious_chunks:
            print(f"⚠️ Обнаружены подозрительные чанки: {len(suspicious_chunks)}")
            for chunk in suspicious_chunks:
                print(f"  Чанк {chunk['chunk_index']}: риск {chunk['risk']:.1%}")
                print(f"  Текст: {chunk['text']}")
        
        if suspicious_sentences:
            print(f"⚠️ Обнаружены подозрительные предложения: {len(suspicious_sentences)}")
            for sent, risk in suspicious_sentences[:3]:  # Показываем первые 3
                print(f"  Предложение: {sent[:100]}... риск: {risk:.1%}")
        
        if not is_safe:
            # Формируем детальное сообщение об ошибке
            error_msg = f"🚫 Обнаружена prompt injection\n"
            error_msg += f"Максимальный риск: {max_risk:.1%}\n"
            
            if suspicious_chunks:
                error_msg += f"Подозрительные чанки: {len(suspicious_chunks)}\n"
                for chunk in suspicious_chunks[:2]:  # Показываем первые 2
                    error_msg += f"  - Риск {chunk['risk']:.1%}: {chunk['text']}\n"
            
            if suspicious_sentences:
                error_msg += f"Подозрительные предложения: {len(suspicious_sentences)}\n"
                for sent, risk in suspicious_sentences[:2]:
                    error_msg += f"  - Риск {risk:.1%}: {sent[:100]}...\n"
            
            raise Exception(error_msg)
        
        # Добавляем метаданные о проверке в тело запроса
        if "metadata" not in body:
            body["metadata"] = {}
        
        body["metadata"]["prompt_injection_check"] = {
            "max_risk": max_risk,
            "chunks_checked": len(self.split_into_chunks(user_message)),
            "suspicious_chunks": len(suspicious_chunks),
            "suspicious_sentences": len(suspicious_sentences)
        }
        
        return body

    async def on_shutdown(self):
        if self.model:
            del self.model
