import semantic_search
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection
import keyword_search

# Setup iniziale (eseguire UNA VOLTA)
conn = get_connection()
keyword_search.setup_fts_table(conn)

"""docs = semantic_search.semantic_search("come implementare un piano paghe?")
for d in docs:
    print(d["similarity"], d["titolo"])"""