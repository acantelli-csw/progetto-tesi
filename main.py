from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SQLServerVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = SQLServerVectorStore(connection_string=CONN_STR, table_name="Documents", embedding_function=embeddings)

llm = ChatOpenAI(model="gpt-4.1")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

response = qa.run("Quali sono i punti principali del documento X?")