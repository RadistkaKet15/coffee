"""
title: Bilingual Toxicity Filter Pipeline
author: open-webui
date: 2024-05-30
version: 2.0
license: MIT
description: A pipeline for filtering out toxic messages in English and Russian.
requirements: detoxify, transformers, torch
"""

from typing import List, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel
from detoxify import Detoxify
import os


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        # e.g. ["llama3:latest", "gpt-3.5-turbo"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0
        
        # Дополнительные настройки для русской модели
        russian_model_enabled: bool = True
        toxicity_threshold: float = 0.5
        language_detection: bool = True

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is not to specify the id so that it can be automatically inferred from the filename,
        # so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens.
        # It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "detoxify_filter_pipeline"
        self.name = "Bilingual Toxicity Filter"

        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "russian_model_enabled": os.getenv("RUSSIAN_MODEL_ENABLED", "true").lower() == "true",
                "toxicity_threshold": float(os.getenv("TOXICITY_THRESHOLD", "0.5")),
                "language_detection": os.getenv("LANGUAGE_DETECTION", "true").lower() == "true",
            }
        )

        self.english_model = None
        self.russian_model = None
        self.language_detector = None

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")

        # Загружаем английскую модель Detoxify
        print("Loading English toxicity model (Detoxify)...")
        self.english_model = Detoxify("original")
        print("✓ English model loaded")

        # Загружаем русскую модель, если включена
        if self.valves.russian_model_enabled:
            try:
                print("Loading Russian toxicity model...")
                from transformers import pipeline
                self.russian_model = pipeline(
                    "text-classification",
                    model="cointegrated/rubert-tiny-toxicity",
                    tokenizer="cointegrated/rubert-tiny-toxicity",
                    device=-1  # CPU
                )
                print("✓ Russian model loaded")
            except ImportError as e:
                print(f"⚠️ Russian model not available: {e}")
                print("Install: pip install transformers torch")
                self.valves.russian_model_enabled = False
            except Exception as e:
                print(f"⚠️ Error loading Russian model: {e}")
                self.valves.russian_model_enabled = False

        # Простой детектор языка (по наличию кириллицы)
        if self.valves.language_detection:
            print("Language detection enabled")
        else:
            print("Language detection disabled")

        print(f"Toxicity threshold: {self.valves.toxicity_threshold}")
        print("✓ Filter initialized successfully")

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    def _detect_language(self, text: str) -> str:
        """Простой детектор языка"""
        if not self.valves.language_detection:
            return "auto"
        
        # Считаем кириллические символы
        cyrillic_count = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        cyrillic_ratio = cyrillic_count / max(len(text), 1)
        
        # Если больше 30% кириллицы - считаем русским
        if cyrillic_ratio > 0.3:
            return "ru"
        
        # Проверяем английские слова
        english_words = ["the", "and", "you", "that", "this", "for", "with", "have"]
        text_lower = text.lower()
        english_word_count = sum(1 for word in english_words if word in text_lower)
        
        if english_word_count >= 2 or len(text.split()) < 5:
            return "en"
        
        return "unknown"

    def _check_toxicity_english(self, text: str) -> float:
        """Проверка токсичности на английском"""
        if self.english_model is None:
            return 0.0
        
        try:
            result = self.english_model.predict(text)
            toxicity_score = result.get("toxicity", 0.0)
            print(f"English toxicity score: {toxicity_score:.4f}")
            return toxicity_score
        except Exception as e:
            print(f"Error in English model: {e}")
            return 0.0

    def _check_toxicity_russian(self, text: str) -> float:
        """Проверка токсичности на русском"""
        if not self.valves.russian_model_enabled or self.russian_model is None:
            return 0.0
        
        try:
            result = self.russian_model(text)[0]
            # Модель возвращает {"label": "toxic"/"neutral", "score": 0.95}
            if result["label"] == "toxic":
                toxicity_score = result["score"]
            else:
                toxicity_score = 1.0 - result["score"]  # Преобразуем neutral в toxicity
            
            print(f"Russian toxicity score: {toxicity_score:.4f} (label: {result['label']})")
            return toxicity_score
        except Exception as e:
            print(f"Error in Russian model: {e}")
            return 0.0

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This filter is applied to the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")
        print(f"Body keys: {list(body.keys())}")

        if "messages" not in body or len(body["messages"]) == 0:
            print("No messages in body, skipping filter")
            return body

        user_message = body["messages"][-1]["content"]
        print(f"User message: {user_message[:100]}...")

        # Детектируем язык
        language = self._detect_language(user_message)
        print(f"Detected language: {language}")

        # Выбираем модель для проверки
        toxicity_score = 0.0
        
        if language == "ru":
            # Для русского текста используем русскую модель
            toxicity_score = self._check_toxicity_russian(user_message)
            
            # Если русская модель не загружена или дала 0, пробуем английскую
            if toxicity_score == 0.0 and self.english_model:
                print("Falling back to English model for Russian text")
                toxicity_score = self._check_toxicity_english(user_message)
        
        elif language == "en":
            # Для английского текста используем английскую модель
            toxicity_score = self._check_toxicity_english(user_message)
        
        else:
            # Для неизвестного языка пробуем обе модели и берем максимум
            print("Unknown language, trying both models...")
            score_en = self._check_toxicity_english(user_message)
            score_ru = self._check_toxicity_russian(user_message)
            toxicity_score = max(score_en, score_ru)
            print(f"Max toxicity score: {toxicity_score:.4f}")

        # Применяем порог
        print(f"Final toxicity score: {toxicity_score:.4f}, Threshold: {self.valves.toxicity_threshold}")
        
        if toxicity_score > self.valves.toxicity_threshold:
            error_msg = (
                f"Toxic message detected (score: {toxicity_score:.2f}, "
                f"threshold: {self.valves.toxicity_threshold})"
            )
            print(f"🚫 {error_msg}")
            raise Exception(error_msg)

        print("✅ Message passed toxicity check")
        return body