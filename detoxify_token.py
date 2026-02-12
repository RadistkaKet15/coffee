"""
title: Detoxify Multilingual Debiased Filter Offline
author: open-webui
date: 2024-11-20
version: 9.0
license: MIT
description: Fully offline Detoxify multilingual debiased model with local tokenizer
requirements: transformers>=4.35.0, torch>=2.0.0
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import torch

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        toxicity_threshold: float = 0.5
        enable_filtering: bool = True

    def __init__(self):
        self.type = "filter"
        self.id = "detoxify_multilingual_offline_v9"
        self.name = "Detoxify Multilingual Offline v9"
        
        self.valves = self.Valves(
            pipelines=["*"],
            toxicity_threshold=0.5,
            enable_filtering=True
        )
        
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.model_path = "/app/model/multilingual_debiased-0b549669.ckpt"
        self.tokenizer_path = "/app/model/tokenizer"
        
        self.class_names = [
            "toxicity", "severe_toxicity", "obscene", "identity_attack",
            "insult", "threat", "sexual_explicit", "male", "female",
            "homosexual_gay_or_lesbian", "christian", "jewish", "muslim",
            "black", "white", "psychiatric_or_mental_illness"
        ]

    async def on_startup(self):
        print("🚀 Загружаем Detoxify Multilingual Debiased...")
        
        try:
            # 1. Загружаем ТОКЕНИЗАТОР из локальной папки
            from transformers import XLMRobertaTokenizer
            print(f"📁 Загрузка токенизатора из: {self.tokenizer_path}")
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                self.tokenizer_path,
                local_files_only=True
            )
            print(f"✅ Токенизатор загружен (словарь: {self.tokenizer.vocab_size})")
            
            # 2. Загружаем МОДЕЛЬ из чекпоинта
            print(f"📁 Загрузка модели из: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig
            
            config = XLMRobertaConfig.from_pretrained(
                self.tokenizer_path,
                num_labels=16,
                local_files_only=True
            )
            
            self.model = XLMRobertaForSequenceClassification(config)
            
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.eval()
            
            self.model_loaded = True
            print("✅ Модель Detoxify Multilingual Debiased загружена успешно!")
            
            # 3. Тест
            test_texts = ["hello", "fuck you", "ты идиот", "nigga"]
            for text in test_texts:
                score = self.predict_toxicity(text).get("toxicity", 0)
                color = "🔴" if score > 0.5 else "🟡" if score > 0.3 else "🟢"
                print(f"  {color} {text}: {score:.3f}")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False

    def predict_toxicity(self, text: str) -> Dict[str, float]:
        if not self.model_loaded:
            return {"toxicity": 0.1}
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.sigmoid(outputs.logits).squeeze()
        
        results = {}
        if scores.dim() == 0:
            results["toxicity"] = float(scores)
        else:
            for i, score in enumerate(scores.tolist()):
                if i < len(self.class_names):
                    results[self.class_names[i]] = float(score)
        return results

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if not self.valves.enable_filtering or not self.model_loaded:
            return body
        
        try:
            messages = body.get("messages", [])
            if not messages:
                return body
            
            last_msg = messages[-1]
            if last_msg.get("role") != "user":
                return body
            
            user_message = last_msg.get("content", "").strip()
            if not user_message:
                return body
            
            print(f"📨 Анализ: {user_message[:50]}...")
            
            scores = self.predict_toxicity(user_message)
            toxicity = scores.get("toxicity", 0)
            
            if toxicity > self.valves.toxicity_threshold:
                raise Exception(f"🚫 Сообщение заблокировано (токсичность: {toxicity:.1%})")
            
            print(f"✅ OK (токсичность: {toxicity:.3f})")
            return body
            
        except Exception as e:
            if "заблокировано" in str(e):
                raise e
            return body

    async def on_shutdown(self):
        print("🔧 Остановка")
        if self.model:
            del self.model
