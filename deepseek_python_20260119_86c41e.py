"""
title: Multi-Language Toxicity Filter
author: open-webui
date: 2024-05-30
version: 2.1
license: MIT
description: Toxicity filter for both English and Russian
requirements: detoxify transformers torch
"""

from typing import List, Optional
from pydantic import BaseModel
import os
import re

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        toxicity_threshold: float = 0.5
        language_detection: bool = True

    def __init__(self):
        self.type = "filter"
        self.name = "Multi-Language Toxicity Filter"
        self.valves = self.Valves(pipelines=["*"])
        
        self.detoxify_model = None
        self.russian_model = None
        
        # Простые ключевые слова для русского (как fallback)
        self.russian_toxic_keywords = [
            "мудак", "долбоёб", "урод", "идиот", "дебил", "кретин",
            "соси", "отъебись", "пошёл нахуй", "пиздец", "блядь",
            "хуй", "пизда", "ебать", "гондон", "пидор", "педик"
        ]

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        
        # Пробуем загрузить Detoxify для английского
        try:
            from detoxify import Detoxify
            self.detoxify_model = Detoxify("original")
            print("✅ Detoxify загружен (для английского)")
        except ImportError:
            print("⚠️ Detoxify не установлен: pip install detoxify")
        
        # Пробуем загрузить русскую модель
        try:
            from transformers import pipeline
            self.russian_model = pipeline(
                "text-classification",
                model="cointegrated/rubert-tiny-toxicity"
            )
            print("✅ Русская модель токсичности загружена")
        except ImportError:
            print("⚠️ transformers не установлен: pip install transformers torch")

    def detect_language(self, text: str) -> str:
        """Простое определение языка по символам"""
        # Считаем кириллические символы
        cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        
        if cyrillic_count > latin_count:
            return "ru"
        else:
            return "en"

    def check_russian_keywords(self, text: str) -> float:
        """Проверка по ключевым словам (fallback)"""
        text_lower = text.lower()
        matches = sum(1 for word in self.russian_toxic_keywords 
                     if word in text_lower)
        
        if matches > 0:
            return min(0.5 + (matches * 0.1), 1.0)
        return 0.0

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if "messages" not in body or len(body["messages"]) == 0:
            return body
        
        last_message = body["messages"][-1]
        if last_message["role"] != "user":
            return body
        
        user_message = last_message["content"]
        print(f"🔍 Проверка: {user_message[:50]}...")
        
        # Определяем язык
        language = self.detect_language(user_message)
        print(f"🌐 Определен язык: {language}")
        
        toxicity_score = 0.0
        
        if language == "en" and self.detoxify_model:
            # Английский - используем Detoxify
            result = self.detoxify_model.predict(user_message)
            toxicity_score = result["toxicity"]
            print(f"🇬🇧 Detoxify score: {toxicity_score:.3f}")
            
        elif language == "ru":
            if self.russian_model:
                # Русский - используем трансформер модель
                result = self.russian_model(user_message)[0]
                label = result["label"]
                toxicity_score = result["score"] if label == "toxic" else 1 - result["score"]
                print(f"🇷🇺 Russian model: {label} ({toxicity_score:.3f})")
            else:
                # Fallback - проверка по ключевым словам
                toxicity_score = self.check_russian_keywords(user_message)
                print(f"🇷🇺 Keywords check: {toxicity_score:.3f}")
        
        # Применяем порог
        if toxicity_score > self.valves.toxicity_threshold:
            print(f"🚫 Токсичность превышена: {toxicity_score:.3f} > {self.valves.toxicity_threshold}")
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Токсичное сообщение обнаружено",
                    "score": toxicity_score,
                    "language": language,
                    "threshold": self.valves.toxicity_threshold
                }
            )
        
        print(f"✅ Сообщение безопасно: {toxicity_score:.3f}")
        return body