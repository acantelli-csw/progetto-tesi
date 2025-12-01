import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_embedding.db_connection import get_connection
import search
import llm
from ragas import GenerationPipeline, TestsetGenerator

def load_all_docs():
    cursor = get_connection().cursor()
    cursor.execute("""  SELECT ID, NumRI, Progressivo, Cliente, Titolo, Autore, Documento, Url_doc, Content
                        FROM DocumentChunks """)
    rows = cursor.fetchall()
    docs = []
    for row in rows:
        doc_id, numero, progressivo, cliente, titolo, autore, documento, url_doc, content = row
        docs.append({
            "id": doc_id,
            "numero": numero,
            "progressivo": progressivo,
            "cliente": cliente,
            "titolo": titolo,
            "autore": autore,
            "documento": documento,
            "autore": autore,
            "url_doc": url_doc,
            "content": content,
        })
    return docs

def retrieve_fn(user_prompt: str):
    # 1️ - Decisione strumenti
    tools = llm.decide_tools(user_prompt)

    # 1.5 - Recupero documenti
    all_documents = []
    if tools["use_semantic"]:
        print("\nUso la ricerca SEMANTICA\n")
        all_documents += search.semantic_search(user_prompt, 25)

    if tools["use_keyword"]:
        print("\nUso la ricerca per KEYWORDS\n")
        all_documents += search.keyword_search(user_prompt, 25)

    # 2 - Selezione documenti in base alla coerenza
    document_selection = []
    selected_docs = []
    if tools["use_semantic"] or tools["use_keyword"]:
        if all_documents:
            document_selection = llm.select_documents(user_prompt, all_documents)

            # Selezione documenti rilevanti
            selected_docs = [all_documents[i] for i in document_selection['relevant_docs']]
    return selected_docs

def answer_fn(user_prompt: str, selected_docs: list[str]):
    return llm.generate_final_answer(user_prompt, selected_docs, [])

# Creazione pipeline Ragas
documents = load_all_docs()
pipeline = GenerationPipeline.from_docs(
    documents=documents,
    llm=None,
    retrieve_fn=retrieve_fn,
    # answer_fn=answer_fn  # solo se vuoi valutare end-to-end
)

# Generazione testset
generator = TestsetGenerator(pipeline)
testset = generator.generate(n_questions=30)

print("Testset generato con successo! Numero domande:", len(testset))

with open("testset_ragas.json", "w", encoding="utf-8") as f:
    json.dump(testset, f, ensure_ascii=False, indent=4)

with open("testset_ragas.json", "r", encoding="utf-8") as f:
    testset_final = json.load(f)

print(testset_final)



