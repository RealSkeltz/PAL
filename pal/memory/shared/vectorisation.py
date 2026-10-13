"""
Here we define the functions needed for vectorising a knowledge base of variable length. It may be one or multiple documents of different kinds.
Each document is chunked and vectorised, after which the vectors are stored. Then, the query is vectorised and compared to the chunks to find the highest match.
"""

from pal.clients import get_client

client = get_client('embedding:embeddinggemma-2:740m')

