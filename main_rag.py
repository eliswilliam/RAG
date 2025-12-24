from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

modelo = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    groq_api_key=groq_api_key
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

arquivos = [
    "java (1).pdf"
]

documentos = sum(
    [
        PyPDFLoader(arquivo).load() for arquivo in arquivos
    ], []
)

pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
).split_documents(documentos)

dados_recuperados = FAISS.from_documents(
    pedacos, embeddings
).as_retriever(search_kwargs={"k":2})

prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido"),
        ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta:")
    ]
)

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

def responder(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(um_trecho.page_content for um_trecho in trechos)
    return cadeia.invoke({
        "query": pergunta, "contexto":contexto
    })

print(responder("O que e uma classe abstrata em Java?"))