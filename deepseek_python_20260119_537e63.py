"""
title: Simple Russian Toxicity Filter
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: Simple toxicity filter for Russian.
requirements: transformers
"""

from typing import List, Optional
from pydantic import BaseModel
import os


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0

    def __init__(self):
        self.type = "filter"
        self.name = "Simple Russian Toxicity Filter"
        self.valves = self.Valves(**{"pipelines": ["*"]})
        self.model = None

    async def on_startup(self):
        print(f"🚀 Simple Russian Toxicity Filter запущен")
        try:
            from transformers import pipeline
            self.model = pipeline(
                "text-classification", 
                model="cointegrated/rubert-tiny-toxicity"
            )
            print("✅ Модель загружена")
        except Exception as e:
            print(f"⚠️ Модель не загрузилась: {e}")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if self.model and "messages" in body and body["messages"]:
            msg = body["messages"][-1]["content"]
            
            # Простая проверка на русские оскорбления
            toxic_words = ["хуй", "пизд", "ебан", "сука", "бля", "дебил", "идиот"]
            
            if any(word in msg.lower() for word in toxic_words):
                # Дополнительная проверка моделью
                result = self.model(msg)[0]
                if result["label"] == "toxic" and result["score"] > 0.8:
                    raise Exception(f"Токсичное сообщение: {result['score']:.2f}")
        
        return body