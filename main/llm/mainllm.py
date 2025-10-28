
import search

def llm_call(user_prompt):
    top_docs, output_lines = search.semantic_search(user_prompt)
    return output_lines[0]

