# services/memory_service.py

from typing import List
from mem0 import Memory
from config.settings import settings
from utils.logger import logger


class Mem0Service:
    def __init__(self):
        try:
            mem0_config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": settings.OPENAI_MODEL_NAME,  # Explicitly uses gpt-4o-mini
                        "temperature": 0.1,
                    }
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "location": ":memory:",
                    }
                }
            }
            if settings.MEM0_API_KEY:
                mem0_config["api_key"] = settings.MEM0_API_KEY

            self.memory = Memory.from_config(mem0_config)
            logger.info("[MEM0] Memory Service initialized with gpt-4o-mini.")
        except Exception as e:
            logger.warning(f"[MEM0] Initialization warning: {str(e)}")
            self.memory = None

    def get_user_memories(self, user_id: str) -> List[str]:
        if not self.memory:
            return []
        try:
            results = self.memory.get_all(user_id=user_id)
            memories = []
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict) and "memory" in item:
                        memories.append(item["memory"])
            elif isinstance(results, dict) and "results" in results:
                memories = [m.get("memory", "") for m in results.get("results", [])]
            return memories
        except Exception as e:
            logger.error(f"[MEM0] Failed to fetch memories: {str(e)}")
            return []

    def add_user_memory(self, user_id: str, interaction: str):
        if not self.memory:
            return
        try:
            self.memory.add(interaction, user_id=user_id)
        except Exception as e:
            logger.error(f"[MEM0] Failed to save memory: {str(e)}")

    def close(self):
        try:
            if hasattr(self, "memory") and self.memory and hasattr(self.memory, "vector_store"):
                if hasattr(self.memory.vector_store, "client") and hasattr(self.memory.vector_store.client, "close"):
                    self.memory.vector_store.client.close()
        except Exception:
            pass


mem0_service = Mem0Service()