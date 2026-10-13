import os
import uuid
from typing import Any
from time import time
from pal.shared.constants.memory import MEMORY_TYPES
from pal.shared.schemas.memory import MemoryType, Base
from pal.clients.embedding_client import EmbeddingClient

class AgentMemory():
    def __init__(self, run_id, embedding_model):
        self.store = self._initialise_memory(run_id)
        self.embedding_client = EmbeddingClient(embedding_model)


    def _initialise_memory(self, run_id: str, selected_memory_types: list = MEMORY_TYPES):

        memory = {}

        for mem_type in selected_memory_types:
            memory[mem_type] = {}

        return memory

    def remember(self, agent_id: str, run_id: str, memory_type: MemoryType, original_data: Any):

        base = Base(
            agent_id = agent_id,
            run_id = run_id,
            mem_id = uuid.uuid4(),
            mem_type = memory_type,
            data_type = original_data.dtype(),
            original_data = original_data,
            embedding = self.embedding_client.vectorise(original_data),
        )


        if memory_type == 'semantic':
            self.store[memory_type].extend(
                {
                    
                }
            )

        if memory_type == 'episodic':
            self.store[memory_type].extend(
                {

                }
            ) 

        if memory_type == 'goal_oriented':
            self.store[memory_type].extend(
                {

                }
            )

        if memory_type == 'user_oriented':
            self.store[memory_type].extend(
                {

                }
            )



        SEMANTIC = auto()
    EPISODIC = auto()
    GOAL_ORIENTED = auto()
    USER_ORIENTED = auto()

    
