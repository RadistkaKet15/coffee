"""
title: Prompt Injection Detection Filter Offline
author: open-webui
date: 2024-11-20
version: 3.0
license: MIT
description: Offline pipeline for detecting prompt injections
requirements: transformers>=4.35.0, torch>=2.0.0
"""

from typing import List, Optional
from pydantic import BaseModel
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        threshold: float = 0.75
        enable_filtering: bool = True

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
            print("✅ Prompt Injection Detector loaded")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model_loaded = False

    def predict(self, text: str):
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
        
        is_safe, risk_score = self.predict(user_message)
        
        if not is_safe:
            raise Exception(
                f"🚫 Prompt injection detected (risk: {risk_score:.1%})"
            )
        
        return body

    async def on_shutdown(self):
        if self.model:
            del self.model
