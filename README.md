RAG (Retrieval-Augmented Generation) - Sistema de Q&A em Português

Sistema de perguntas e respostas baseado em RAG (Retrieval-Augmented Generation) que utiliza documentos PDF como base de conhecimento para responder perguntas em português sobre Java. 

📋 Descrição

Este projeto implementa um pipeline RAG que:
- Carrega documentos PDF
- Divide o conteúdo em chunks menores
- Cria embeddings vetoriais dos documentos
- Armazena os embeddings em um banco vetorial FAISS
- Recupera informações relevantes para responder perguntas
- Utiliza um LLM (Llama 3.3) via Groq para gerar respostas contextualizadas

🛠️ Tecnologias Utilizadas

- **LangChain**: Framework para desenvolvimento de aplicações com LLMs
- **Groq**: API para acesso ao modelo Llama 3.3 70B
- **HuggingFace**: Embeddings usando `sentence-transformers/all-MiniLM-L6-v2`
- **FAISS**: Banco de dados vetorial para busca semântica
- **Python-dotenv**: Gerenciamento de variáveis de ambiente

📦 Instalação

1. Clone o repositório: 
```bash
git clone https://github.com/eliswilliam/RAG.git
cd RAG
```

2. Instale as dependências:
```bash
pip install langchain-groq langchain-huggingface langchain-community
pip install faiss-cpu python-dotenv pypdf
```

3. Configure as variáveis de ambiente: 
Crie um arquivo `.env` na raiz do projeto com sua chave da API Groq:
```
GROQ_API_KEY=sua_chave_api_aqui
```

## 🚀 Uso

1.  Coloque seus arquivos PDF na raiz do projeto (atualmente configurado para `java (1).pdf`)

2. Execute o script: 
```bash
python main_rag.py
```

3. Para fazer suas próprias perguntas, modifique a última linha do código: 
```python
print(responder("Sua pergunta aqui? "))
```

## 🔧 Configurações

### Ajustar o Modelo LLM
```python
modelo = ChatGroq(
    model="llama-3.3-70b-versatile",  # Modelo utilizado
    temperature=0.5,                   # Ajusta a criatividade (0-1)
    groq_api_key=groq_api_key
)
```

### Ajustar Chunking
```python
pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Tamanho de cada pedaço
    chunk_overlap=100   # Sobreposição entre pedaços
).split_documents(documentos)
```

### Ajustar Recuperação
```python
dados_recuperados = FAISS.from_documents(
    pedacos, embeddings
).as_retriever(search_kwargs={"k":2})  # Número de chunks retornados
```

## 📂 Estrutura do Projeto

```
RAG/
├── main_rag.py          # Script principal
├── java (1).pdf         # Documento fonte (exemplo)
├── .env                 # Variáveis de ambiente (não versionado)
└── README.md            # Este arquivo
```

## 🔍 Como Funciona

1. **Carregamento**:  PDFs são carregados usando `PyPDFLoader`
2. **Divisão**:  Documentos são divididos em chunks de 1000 caracteres com overlap de 100
3. **Embedding**: Cada chunk é convertido em vetor usando modelo HuggingFace
4. **Armazenamento**:  Vetores são armazenados no FAISS para busca eficiente
5. **Recuperação**: Ao fazer uma pergunta, o sistema busca os 2 chunks mais relevantes
6. **Geração**: O LLM gera uma resposta baseada exclusivamente no contexto recuperado

## 🎯 Exemplo de Uso

```python
from main_rag import responder

# Fazer uma pergunta
resposta = responder("O que é uma classe abstrata em Java?")
print(resposta)
```

## 📝 Notas

- O sistema responde **exclusivamente** com base no conteúdo dos documentos fornecidos
- A temperatura está configurada em 0.5 para balancear criatividade e precisão
- O modelo de embeddings é otimizado para textos em português

## 🔑 Obtendo a API Key do Groq

1. Acesse [console.groq.com](https://console.groq.com)
2. Faça login ou crie uma conta
3. Navegue até a seção de API Keys
4. Crie uma nova chave e copie para seu arquivo `.env`

## 🤝 Contribuindo

Contribuições são bem-vindas!  Sinta-se à vontade para abrir issues ou pull requests.



Desenvolvido por [@eliswilliam](https://github.com/eliswilliam)
