"""
title: LLM Guard Filter Pipeline
author: jannikstdl
date: 2024-05-30
version: 1.1
license: MIT
description: A pipeline for filtering out potential prompt injections using the LLM Guard library.
requirements: llm-guard
"""

from typing import List, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel

# Обход проблемы с импортом
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils")

try:
    from llm_guard.input_scanners import PromptInjection
    from llm_guard.input_scanners.prompt_injection import MatchType
except ImportError as e:
    print(f"Error importing llm_guard: {e}")
    print("Trying alternative import...")
    # Альтернативный импорт
    import importlib
    llm_guard = importlib.import_module('llm_guard')
    PromptInjection = llm_guard.input_scanners.PromptInjection
    MatchType = llm_guard.input_scanners.prompt_injection.MatchType

import os

class Pipeline:
    def __init__(self):
        self.type = "filter"
        self.id = "llmguard_prompt_injection_filter_pipeline"
        self.name = "LLMGuard Prompt Injection Filter"

        class Valves(BaseModel):
            pipelines: List[str] = []
            priority: int = 0
            threshold: float = 0.8

        self.valves = Valves(
            **{
                "pipelines": ["*"],
                "threshold": 0.8
            }
        )

        self.model = None

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        try:
            self.model = PromptInjection(
                threshold=self.valves.threshold, 
                match_type=MatchType.FULL
            )
            print("LLM Guard model loaded successfully")
        except Exception as e:
            print(f"Error loading LLM Guard model: {e}")
            self.model = None

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        if self.model:
            try:
                # Обновляем threshold если нужно
                self.model.threshold = self.valves.threshold
            except:
                pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")
        
        if not self.model:
            print("LLM Guard model not loaded, skipping check")
            return body
        
        try:
            user_message = body["messages"][-1]["content"]
            sanitized_prompt, is_valid, risk_score = self.model.scan(user_message)
            
            if risk_score > self.valves.threshold:
                print(f"Prompt injection detected! Risk score: {risk_score}")
                raise Exception(f"Prompt injection detected (risk score: {risk_score})")
                
        except Exception as e:
            if "Prompt injection detected" in str(e):
                raise e
            else:
                print(f"Error in prompt injection check: {e}")
                
        return body
