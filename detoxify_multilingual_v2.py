"""
title: Detoxify Multilingual Filter
author: open-webui
date: 2024-05-30
version: 2.0
license: MIT
description: A pipeline for filtering out toxic messages using Detoxify multilingual model.
requirements: detoxify
"""

from typing import List, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel
from detoxify import Detoxify
import os


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0

    def __init__(self):
        self.type = "filter"

        # ВАЖНО: Уникальный ID 👇
        self.id = "detoxify_multilingual_v2"

        # ВАЖНО: Уникальное имя 👇
        self.name = "Detoxify Multilingual v2"

        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
            }
        )

        self.model = None
        pass

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        # Используем multilingual модель 👇
        self.model = Detoxify("multilingual")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")
        print(body)

        user_message = body["messages"][-1]["content"]
        toxicity = self.model.predict(user_message)
        print(f"Multilingual toxicity scores: {toxicity}")

        if toxicity["toxicity"] > 0.5:
            raise Exception("Toxic message detected (multilingual v2)")

        return body