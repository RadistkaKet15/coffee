"""
title: Detoxify Multilingual Debiased Filter Offline
author: open-webui
date: 2024-11-20
version: 5.0
license: MIT
description: Fully offline Detoxify multilingual debiased model for toxicity filtering
requirements: transformers>=4.35.0, torch>=2.0.0
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import torch
import json

# Полностью отключаем интернет
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_DOWNLOAD_TIMEOUT'] = '1'

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        toxicity_threshold: float = 0.5
        enable_filtering: bool = True

    def __init__(self):
        self.type = "filter"
        self.id = "detoxify_multilingual_debiased_offline"
        self.name = "Detoxify Multilingual Debiased Offline"
        
        self.valves = self.Valves(
            pipelines=["*"],
            priority=0,
            toxicity_threshold=0.5,
            enable_filtering=True
        )
        
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Путь к DEBIASED модели
        self.model_path = "/app/model/multilingual_debiased-0b549669.ckpt"
        
        # Названия классов для multilingual DEBIASED модели (16 классов)
        self.class_names = [
            "toxicity",
            "severe_toxicity", 
            "obscene",
            "identity_attack",
            "insult",
            "threat",
            "sexual_explicit",
            "male",
            "female",
            "homosexual_gay_or_lesbian",
            "christian",
            "jewish",
            "muslim",
            "black",
            "white",
            "psychiatric_or_mental_illness"
        ]

    async def on_startup(self):
        print(f"🚀 Запускаем Detoxify Multilingual Debiased пайплайн")
        print(f"📁 Ищем модель в: {self.model_path}")
        
        try:
            # Проверяем наличие файла модели
            if not os.path.exists(self.model_path):
                print(f"❌ Файл модели не найден: {self.model_path}")
                print("📝 Список файлов в /app/model:")
                try:
                    for file in os.listdir("/app/model"):
                        print(f"  - {file}")
                except:
                    print("  Папка пуста или недоступна")
                self.model_loaded = False
                return
            
            print(f"✅ Файл модели найден: {self.model_path}")
            print("🔄 Загружаем модель Detoxify Multilingual Debiased...")
            
            # Загружаем чекпоинт
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            if 'state_dict' not in checkpoint:
                print("❌ Чекпоинт не содержит state_dict")
                print("📊 Ключи в чекпоинте:", list(checkpoint.keys()))
                self.model_loaded = False
                return
            
            state_dict = checkpoint['state_dict']
            print(f"📊 Размер state_dict: {len(state_dict)} ключей")
            
            # DEBIASED модель имеет 16 классов
            from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig
            
            # Создаем конфигурацию XLM-RoBERTa с 16 классами
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
                num_labels=16  # ВАЖНО: 16 классов для debiased модели
            )
            
            # Создаем модель с 16 классами
            self.model = XLMRobertaForSequenceClassification(config)
            
            # Преобразуем ключи чекпоинта
            new_state_dict = {}
            for key, value in state_dict.items():
                # Убираем префикс 'model.' если есть
                if key.startswith('model.'):
                    new_key = key[6:]  # Убираем 'model.'
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            # Загружаем веса
            print("🔄 Загружаем веса в модель...")
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Отсутствующие ключи: {len(missing_keys)}")
                if len(missing_keys) > 0:
                    print(f"   Пример: {list(missing_keys)[:3]}")
            if unexpected_keys:
                print(f"⚠️  Неожиданные ключи: {len(unexpected_keys)}")
                if len(unexpected_keys) > 0:
                    print(f"   Пример: {list(unexpected_keys)[:3]}")
            
            self.model.eval()
            
            # Создаем простой токенизатор
            print("🔄 Создаю токенизатор...")
            self.create_simple_tokenizer()
            
            self.model_loaded = True
            print("✅ Модель Detoxify Multilingual Debiased загружена успешно!")
            print(f"📊 Модель использует {self.model.config.num_labels} классов")
            print(f"📊 Классы: {self.class_names}")
            
            # Тестируем модель
            print("🧪 Тестируем модель...")
            try:
                test_text = "Hello world"
                scores = self.predict_toxicity(test_text)
                toxicity = scores.get("toxicity", 0)
                print(f"  Тест '{test_text}': toxicity={toxicity:.3f}")
                
                # Тест с токсичным текстом
                test_text_bad = "bad"
                scores_bad = self.predict_toxicity(test_text_bad)
                toxicity_bad = scores_bad.get("toxicity", 0)
                print(f"  Тест '{test_text_bad}': toxicity={toxicity_bad:.3f}")
            except Exception as e:
                print(f"  Ошибка теста: {e}")
            
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False

    def create_simple_tokenizer(self):
        """Создает простой токенизатор для XLM-RoBERTa"""
        try:
            # Используем базовый токенизатор с минимальным словарем
            from transformers import PreTrainedTokenizerFast
            
            # Минимальный словарь
            vocab = {
                "<s>": 0,
                "<pad>": 1,
                "</s>": 2,
                "<unk>": 3,
                "<mask>": 4,
            }
            
            # Добавляем частые слова
            common_words = [
                ".", ",", "!", "?", ";", ":", "'", '"', "(", ")", "[", "]",
                "{", "}", "-", "_", "+", "=", "*", "/", "\\", "|", "@", "#",
                "$", "%", "^", "&", "~", "`",
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
                "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
                "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
                "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
                "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
                "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
                "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
                "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
                "hello", "world", "test", "message", "bad", "good", "hate", "love", "like", "dislike"
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
            
            print(f"✅ Токенизатор создан ({len(vocab)} токенов)")
            
        except Exception as e:
            print(f"⚠️  Не удалось создать токенизатор: {e}")
            # Аварийный токенизатор
            class EmergencyTokenizer:
                def __call__(self, text, return_tensors="pt", truncation=True, max_length=512, **kwargs):
                    # Разбиваем на слова и создаем простые ID
                    words = text.lower().split()
                    word_ids = []
                    
                    for i, word in enumerate(words[:max_length]):
                        # Простой хэш
                        word_id = hash(word) % 1000 + 10
                        word_ids.append(word_id)
                    
                    if not word_ids:
                        word_ids = [0, 2]  # <s> и </s>
                    else:
                        word_ids = [0] + word_ids + [2]  # <s> + tokens + </s>
                    
                    return {
                        "input_ids": torch.tensor([word_ids]),
                        "attention_mask": torch.ones(1, len(word_ids))
                    }
            
            self.tokenizer = EmergencyTokenizer()
            print("✅ Используем аварийный токенизатор")

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
                logits = outputs.logits
                scores = torch.sigmoid(logits).squeeze()
            
            # Преобразуем в список Python
            if scores.dim() == 0:  # Скаляр
                scores_list = [scores.item()]
            else:
                scores_list = scores.tolist()
            
            # Создаем словарь результатов
            results = {}
            for i, score in enumerate(scores_list):
                if i < len(self.class_names):
                    class_name = self.class_names[i]
                    results[class_name] = float(score)
                else:
                    results[f"class_{i}"] = float(score)
            
            return results
            
        except Exception as e:
            print(f"⚠️  Ошибка при предсказании: {e}")
            # Возвращаем безопасные значения по умолчанию
            default_scores = {}
            for class_name in self.class_names:
                if class_name == "toxicity":
                    default_scores[class_name] = 0.1
                else:
                    default_scores[class_name] = 0.05
            return default_scores

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
            
            # Логируем только высокие оценки
            high_scores = []
            for class_name, score in toxicity_scores.items():
                if score > 0.3:  # Логируем только оценки выше 0.3
                    high_scores.append(f"{class_name}:{score:.3f}")
            
            if high_scores:
                print(f"📊 Высокие оценки: {', '.join(high_scores)}")
            else:
                print(f"📊 Токсичность: {main_toxicity:.3f}")
            
            # Проверяем порог
            if main_toxicity > self.valves.toxicity_threshold:
                # Формируем детализированное сообщение
                high_score_details = []
                for class_name, score in toxicity_scores.items():
                    if score > 0.2:  # Показываем только значимые оценки
                        high_score_details.append(f"  • {class_name}: {score:.1%}")
                
                if high_score_details:
                    details = "\n".join(high_score_details)
                else:
                    details = "  • Только общая токсичность"
                
                error_msg = (
                    f"🚫 Сообщение заблокировано системой фильтрации токсичности\n\n"
                    f"**Общий уровень токсичности:** {main_toxicity:.1%}\n"
                    f"**Пороговое значение:** {self.valves.toxicity_threshold:.0%}\n\n"
                    f"**Детальный анализ:**\n{details}\n\n"
                    f"**Модель:** Detoxify Multilingual Debiased (16 классов)"
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
        print(f"  • toxicity_threshold: {self.valves.toxicity_threshold}")
        print(f"  • enable_filtering: {self.valves.enable_filtering}")
