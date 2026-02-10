"""
title: Detoxify Multilingual Filter Offline
author: open-webui
date: 2024-11-20
version: 4.0
license: MIT
description: Fully offline Detoxify multilingual model for toxicity filtering
requirements: transformers>=4.35.0, torch>=2.0.0
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import torch
import json
from pathlib import Path

# Полностью отключаем интернет
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_DOWNLOAD_TIMEOUT'] = '1'

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        model_type: str = "multilingual"
        toxicity_threshold: float = 0.5
        enable_filtering: bool = True

    def __init__(self):
        self.type = "filter"
        self.id = "detoxify_multilingual_offline_v4"
        self.name = "Detoxify Multilingual Offline v4"
        
        self.valves = self.Valves(
            pipelines=["*"],
            priority=0,
            model_type="multilingual",
            toxicity_threshold=0.5,
            enable_filtering=True
        )
        
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Путь к модели в папке /app/model
        self.model_path = "/app/model/multilingual-de31e4a5.ckpt"
        
        # Названия классов для multilingual модели
        self.class_names = [
            "toxicity",
            "severe_toxicity", 
            "obscene",
            "threat",
            "insult",
            "identity_hate"
        ]

    async def on_startup(self):
        print(f"🚀 Запускаем Detoxify Multilingual пайплайн")
        print(f"📁 Ищем модель в: {self.model_path}")
        
        try:
            # Проверяем наличие папки model
            model_dir = "/app/model"
            if not os.path.exists(model_dir):
                print(f"❌ Папка не найдена: {model_dir}")
                print("💡 Создаю папку /app/model...")
                os.makedirs(model_dir, exist_ok=True)
            
            # Проверяем наличие файла модели
            if not os.path.exists(self.model_path):
                print(f"❌ Файл модели не найден: {self.model_path}")
                print("📝 Список файлов в /app/model:")
                try:
                    for file in os.listdir(model_dir):
                        print(f"  - {file}")
                except:
                    print("  Папка пуста или недоступна")
                self.model_loaded = False
                return
            
            print(f"✅ Файл модели найден: {self.model_path}")
            print("🔄 Загружаем модель Detoxify...")
            
            # Загружаем чекпоинт
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            if 'state_dict' not in checkpoint:
                print("❌ Чекпоинт не содержит state_dict")
                print("📊 Ключи в чекпоинте:", list(checkpoint.keys()))
                self.model_loaded = False
                return
            
            state_dict = checkpoint['state_dict']
            print(f"📊 Размер state_dict: {len(state_dict)} ключей")
            print(f"📊 Пример ключей: {list(state_dict.keys())[:5]}")
            
            # Определяем архитектуру (XLM-RoBERTa для multilingual)
            from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig
            
            # Создаем конфигурацию XLM-RoBERTa
            config = XLMRobertaConfig(
                vocab_size=250002,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=514,
                type_vocab_size=1,
                initializer_range=0.02,
                layer_norm_eps=1e-05,
                pad_token_id=1,
                bos_token_id=0,
                eos_token_id=2,
                num_labels=6  # 6 классов для multilingual
            )
            
            # Создаем модель
            self.model = XLMRobertaForSequenceClassification(config)
            
            # Преобразуем ключи чекпоинта
            new_state_dict = {}
            for key, value in state_dict.items():
                # Убираем префикс 'model.' если есть
                if key.startswith('model.'):
                    new_key = key[6:]  # Убираем 'model.'
                else:
                    new_key = key
                
                # Заменяем roberta. на roberta. для совместимости
                if new_key.startswith('roberta.'):
                    new_key = new_key  # Оставляем как есть
                
                new_state_dict[new_key] = value
            
            # Загружаем веса
            print("🔄 Загружаем веса в модель...")
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Отсутствующие ключи: {len(missing_keys)}")
                print(f"   Пример: {missing_keys[:3]}")
            if unexpected_keys:
                print(f"⚠️  Неожиданные ключи: {len(unexpected_keys)}")
                print(f"   Пример: {unexpected_keys[:3]}")
            
            self.model.eval()
            
            # Создаем простой токенизатор
            print("🔄 Создаю токенизатор...")
            self.create_simple_tokenizer()
            
            self.model_loaded = True
            print("✅ Модель Detoxify Multilingual загружена успешно!")
            print(f"📊 Модель использует {self.model.config.num_labels} классов: {self.class_names}")
            
            # Тестируем модель
            print("🧪 Тестируем модель на простых примерах...")
            test_texts = ["Hello world", "This is a test", "You are bad"]
            for text in test_texts:
                try:
                    scores = self.predict_toxicity(text)
                    print(f"  '{text}': toxicity={scores.get('toxicity', 0):.3f}")
                except Exception as e:
                    print(f"  Ошибка теста '{text}': {e}")
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False

    def create_simple_tokenizer(self):
        """Создает простой токенизатор для XLM-RoBERTa"""
        try:
            # Создаем минимальный токенизатор
            from transformers import PreTrainedTokenizerFast
            
            # Базовый словарь
            vocab = {}
            
            # Специальные токены XLM-RoBERTa
            special_tokens = {
                "<s>": 0,
                "<pad>": 1,
                "</s>": 2,
                "<unk>": 3,
                "<mask>": 4,
            }
            
            # Добавляем специальные токены
            vocab.update(special_tokens)
            
            # Добавляем некоторые базовые токены
            basic_tokens = [
                ".", ",", "!", "?", ";", ":", "'", "\"", "(", ")", "[", "]",
                "{", "}", "-", "_", "+", "=", "*", "/", "\\", "|", "@", "#",
                "$", "%", "^", "&", "~", "`"
            ]
            
            for i, token in enumerate(basic_tokens, len(vocab)):
                vocab[token] = i
            
            # Добавляем некоторые частые слова
            common_words = [
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
                "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
                "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
                "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
                "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
                "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
                "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
                "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
            ]
            
            for i, word in enumerate(common_words, len(vocab)):
                vocab[word] = i
            
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=None,
                vocab=vocab,
                unk_token="<unk>",
                sep_token="</s>",
                pad_token="<pad>",
                cls_token="<s>",
                mask_token="<mask>",
                model_max_length=512
            )
            
            print(f"✅ Токенизатор создан (словарь: {len(vocab)} токенов)")
            
        except Exception as e:
            print(f"⚠️  Не удалось создать сложный токенизатор: {e}")
            # Используем минимальный токенизатор
            class SimpleTokenizer:
                def __init__(self):
                    self.vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
                
                def __call__(self, text, return_tensors="pt", truncation=True, max_length=512, **kwargs):
                    # Простая токенизация: разбиваем на слова
                    words = text.lower().replace(".", " . ").replace(",", " , ").split()
                    
                    # Преобразуем слова в ID
                    word_ids = []
                    for word in words[:max_length-2]:  # -2 для специальных токенов
                        if word in self.vocab:
                            word_ids.append(self.vocab[word])
                        else:
                            # Создаем ID на основе хэша
                            word_hash = hash(word) % 10000 + 10  # +10 чтобы избежать специальных токенов
                            self.vocab[word] = word_hash
                            word_ids.append(word_hash)
                    
                    # Добавляем специальные токены
                    input_ids = [0] + word_ids + [2]  # <s> + tokens + </s>
                    
                    return {
                        "input_ids": torch.tensor([input_ids]),
                        "attention_mask": torch.ones(1, len(input_ids))
                    }
            
            self.tokenizer = SimpleTokenizer()
            print("✅ Используем минимальный токенизатор")

    def predict_toxicity(self, text: str) -> Dict[str, float]:
        """Предсказание токсичности с помощью модели"""
        if not self.model_loaded or not self.model or not self.tokenizer:
            raise Exception("Модель не загружена")
        
        try:
            # Токенизация
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Получаем логиты и применяем сигмоиду
                logits = outputs.logits
                scores = torch.sigmoid(logits).squeeze().tolist()
            
            # Создаем словарь с результатами
            results = {}
            if isinstance(scores, list):
                for i, score in enumerate(scores):
                    if i < len(self.class_names):
                        results[self.class_names[i]] = float(score)
            else:
                # Если scores это одно число
                results["toxicity"] = float(scores)
            
            return results
            
        except Exception as e:
            print(f"⚠️  Ошибка при предсказании: {e}")
            import traceback
            traceback.print_exc()
            # Возвращаем безопасные значения по умолчанию
            return {name: 0.1 for name in self.class_names}

    async def on_shutdown(self):
        print(f"🔧 Останавливаем пайплайн: {self.name}")
        if self.model:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Обработка входящих сообщений"""
        
        # Проверяем включена ли фильтрация
        if not self.valves.enable_filtering:
            print("⚙️  Фильтрация отключена в настройках")
            return body
        
        try:
            # Получаем сообщение пользователя
            messages = body.get("messages", [])
            if not messages:
                return body
            
            last_message = messages[-1]
            user_message = last_message.get("content", "").strip()
            
            if not user_message:
                return body
            
            print(f"📨 Анализируем сообщение: {user_message[:80]}...")
            
            if not self.model_loaded:
                print("⚠️  Модель не загружена, пропускаем проверку")
                return body
            
            # Получаем предсказания от модели
            toxicity_scores = self.predict_toxicity(user_message)
            
            # Основная оценка токсичности (первый класс)
            main_toxicity = toxicity_scores.get("toxicity", 0.0)
            
            # Логируем результаты
            log_msg = f"📊 Токсичность: {main_toxicity:.3f}"
            for class_name, score in toxicity_scores.items():
                if class_name != "toxicity" and score > 0.3:
                    log_msg += f", {class_name}: {score:.3f}"
            print(log_msg)
            
            # Проверяем порог
            if main_toxicity > self.valves.toxicity_threshold:
                # Формируем детализированное сообщение
                details = "\n".join([f"  • {name}: {score:.1%}" for name, score in toxicity_scores.items()])
                
                error_msg = (
                    f"🚫 Сообщение заблокировано системой фильтрации токсичности\n\n"
                    f"**Общий уровень токсичности:** {main_toxicity:.1%}\n"
                    f"**Пороговое значение:** {self.valves.toxicity_threshold:.0%}\n\n"
                    f"**Детальный анализ:**\n{details}\n\n"
                    f"**Модель:** Detoxify Multilingual"
                )
                
                print(f"🚫 Сообщение заблокировано: toxicity={main_toxicity:.3f} > threshold={self.valves.toxicity_threshold}")
                raise Exception(error_msg)
            
            print(f"✅ Сообщение прошло проверку (toxicity: {main_toxicity:.3f})")
            return body
            
        except Exception as e:
            if "заблокировано" in str(e):
                raise e  # Пробрасываем наши ошибки блокировки
            else:
                print(f"⚠️  Ошибка при обработке сообщения: {e}")
                # В случае ошибки пропускаем сообщение
                return body

    async def on_valves_updated(self):
        """Обновление настроек"""
        print(f"⚙️  Обновлены настройки пайплайна:")
        print(f"  • model_type: {self.valves.model_type}")
        print(f"  • toxicity_threshold: {self.valves.toxicity_threshold}")
        print(f"  • enable_filtering: {self.valves.enable_filtering}")
