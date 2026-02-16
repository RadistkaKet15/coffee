"""
title: Presidio PII Redaction Pipeline (Fully Offline)
author: open-webui
date: 2024-11-20
version: 1.0.0
license: MIT
description: Fully offline pipeline for redacting PII using local Presidio with pre-downloaded spaCy model
requirements: presidio-analyzer, presidio-anonymizer, spacy
"""

import os
import subprocess
import sys
from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage

# Полностью отключаем интернет
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        enabled_for_admins: bool = False
        entities_to_redact: List[str] = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", 
            "CREDIT_CARD", "IP_ADDRESS", "US_PASSPORT", "LOCATION",
            "DATE_TIME", "NRP", "MEDICAL_LICENSE", "URL"
        ]
        language: str = "en"
        spacy_model_name: str = "en_core_web_lg"
        redaction_text: str = "[PII REDACTED]"
        auto_install: bool = True

    def __init__(self):
        self.type = "filter"
        self.name = "Presidio PII Redaction (Offline)"
        
        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("PII_REDACT_PIPELINES", "*").split(","),
                "enabled_for_admins": os.getenv("PII_REDACT_ENABLED_FOR_ADMINS", "false").lower() == "true",
                "entities_to_redact": os.getenv("PII_REDACT_ENTITIES", ",".join(self.Valves().entities_to_redact)).split(","),
                "language": os.getenv("PII_REDACT_LANGUAGE", "en"),
                "spacy_model_name": os.getenv("SPACY_MODEL_NAME", "en_core_web_lg"),
                "redaction_text": os.getenv("PII_REDACTION_TEXT", "[PII REDACTED]"),
                "auto_install": os.getenv("PII_AUTO_INSTALL", "true").lower() == "true",
            }
        )
        
        self.analyzer = None
        self.anonymizer = None
        self.initialized = False
        
        # Пути к локальным файлам в контейнере
        self.model_path = "/app/model"
        self.spacy_wheel = f"{self.model_path}/en_core_web_lg-3.7.1-py3-none-any.whl"
        self.wheels_dir = f"{self.model_path}/pip_wheels"

    def install_dependencies(self):
        """Устанавливает все зависимости из локальных wheel-файлов"""
        try:
            print("🔄 Установка зависимостей из локальных wheel-файлов...")
            
            # Проверяем наличие файлов
            if not os.path.exists(self.spacy_wheel):
                print(f"❌ Файл не найден: {self.spacy_wheel}")
                print(f"📁 Содержимое {self.model_path}:")
                for f in os.listdir(self.model_path):
                    print(f"  - {f}")
                return False
            
            if not os.path.exists(self.wheels_dir):
                print(f"❌ Папка не найдена: {self.wheels_dir}")
                return False
            
            # Устанавливаем spaCy модель
            print(f"📦 Установка spaCy модели из: {self.spacy_wheel}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", self.spacy_wheel])
            
            # Устанавливаем Presidio и все зависимости
            print(f"📦 Установка Presidio из: {self.wheels_dir}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--no-index", "--find-links", self.wheels_dir,
                "presidio-analyzer", "presidio-anonymizer"
            ])
            
            print("✅ Все зависимости успешно установлены!")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка установки зависимостей: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def on_startup(self):
        print(f"🚀 Запуск Presidio PII Redaction Pipeline (Offline)")
        
        try:
            # Устанавливаем зависимости если нужно
            if self.valves.auto_install:
                success = self.install_dependencies()
                if not success:
                    print("⚠️ Продолжаем с существующими зависимостями...")
            
            # Импортируем Presidio после установки
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
            
            # Сохраняем классы для использования
            self.AnalyzerEngine = AnalyzerEngine
            self.NlpEngineProvider = NlpEngineProvider
            self.AnonymizerEngine = AnonymizerEngine
            self.OperatorConfig = OperatorConfig
            
            # Проверяем доступность spaCy модели
            try:
                import spacy
                nlp = spacy.load(self.valves.spacy_model_name)
                print(f"✅ spaCy модель {self.valves.spacy_model_name} загружена")
            except Exception as e:
                print(f"❌ Ошибка загрузки spaCy модели: {e}")
                print(f"📁 Доступные модели spaCy:")
                subprocess.call([sys.executable, "-m", "spacy", "info"])
                return
            
            # Настраиваем NLP engine с локальной моделью
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {
                        "lang_code": self.valves.language,
                        "model_name": self.valves.spacy_model_name
                    }
                ]
            }
            
            print(f"📁 Используем локальную spaCy модель: {self.valves.spacy_model_name}")
            
            # Создаем провайдера NLP engine
            nlp_engine_provider = self.NlpEngineProvider(nlp_configuration=nlp_configuration)
            nlp_engine = nlp_engine_provider.create_engine()
            
            # Создаем AnalyzerEngine с локальным NLP engine
            self.analyzer = self.AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=[self.valves.language]
            )
            
            # Создаем AnonymizerEngine
            self.anonymizer = self.AnonymizerEngine()
            
            # Тестируем на простом примере
            test_text = "John Doe from New York called 555-123-4567"
            test_results = self.analyzer.analyze(
                text=test_text,
                language=self.valves.language,
                entities=["PERSON", "LOCATION", "PHONE_NUMBER"]
            )
            
            print(f"✅ Presidio инициализирован успешно!")
            print(f"📊 Найдено сущностей в тесте: {len(test_results)}")
            for result in test_results:
                print(f"   - {result.entity_type}: '{test_text[result.start:result.end]}'")
            
            self.initialized = True
            
        except Exception as e:
            print(f"❌ Ошибка инициализации Presidio: {e}")
            import traceback
            traceback.print_exc()
            self.initialized = False

    def redact_pii(self, text: str) -> str:
        """Обнаруживает и заменяет PII в тексте"""
        if not self.initialized or not self.analyzer or not self.anonymizer:
            return text
        
        try:
            # Анализируем текст
            results = self.analyzer.analyze(
                text=text,
                language=self.valves.language,
                entities=self.valves.entities_to_redact if self.valves.entities_to_redact else None
            )
            
            if not results:
                return text
            
            # Анонимизируем
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={
                    "DEFAULT": self.OperatorConfig("replace", {"new_value": self.valves.redaction_text})
                }
            )
            
            return anonymized.text
            
        except Exception as e:
            print(f"⚠️ Ошибка обработки PII: {e}")
            return text

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Фильтр входящих сообщений"""
        
        if not self.initialized:
            print("⚠️ Presidio не инициализирован, пропускаем обработку")
            return body
        
        # Проверяем, нужно ли обрабатывать
        if user is None or user.get("role") != "admin" or self.valves.enabled_for_admins:
            messages = body.get("messages", [])
            redacted_count = 0
            redacted_entities = set()
            
            for message in messages:
                if message.get("role") == "user" and message.get("content"):
                    original = message["content"]
                    redacted = self.redact_pii(original)
                    
                    if original != redacted:
                        redacted_count += 1
                        message["content"] = redacted
                        
                        # Определяем какие сущности были найдены
                        results = self.analyzer.analyze(
                            text=original,
                            language=self.valves.language,
                            entities=self.valves.entities_to_redact
                        )
                        for r in results:
                            redacted_entities.add(r.entity_type)
            
            if redacted_count > 0:
                print(f"🔒 Заменено PII в {redacted_count} сообщении(ях)")
                print(f"📋 Найденные сущности: {', '.join(redacted_entities)}")
        
        return body

    async def on_shutdown(self):
        print(f"🔧 Остановка Presidio PII Redaction Pipeline")

    async def on_valves_updated(self):
        """Обновление настроек"""
        print(f"⚙️ Обновлены настройки:")
        print(f"  • entities_to_redact: {len(self.valves.entities_to_redact)} сущностей")
        print(f"  • language: {self.valves.language}")
        print(f"  • spacy_model: {self.valves.spacy_model_name}")
        print(f"  • redaction_text: {self.valves.redaction_text}")
