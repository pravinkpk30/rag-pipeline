# RAG Pipeline with MongoDB Vector Search

A comprehensive **Retrieval-Augmented Generation (RAG)** pipeline built with MongoDB Atlas Vector Search, LangChain, HuggingFace Embeddings, and Google Gemini LLM. This project demonstrates how to build an AI-powered question-answering system that retrieves relevant context from documents and generates accurate responses.

## 🚀 Features

- **Vector Search**: MongoDB Atlas vector search for semantic similarity matching
- **PDF Document Processing**: Automatic PDF loading and text chunking
- **Smart Embeddings**: HuggingFace embeddings with automatic fallback from Google AI
- **LLM Integration**: Google Gemini 2.0 Flash for intelligent response generation
- **Production-Ready**: Error handling, retry logic, and quota management

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## 🏗️ Architecture

The RAG pipeline consists of the following stages:

```
PDF Document → Text Splitting → Embeddings → MongoDB Vector Store → Query → LLM → Answer
```

### Workflow:

1. **Document Ingestion**: Load PDF documents and split into chunks
2. **Embedding Generation**: Convert text chunks to vector embeddings (384 dimensions)
3. **Vector Storage**: Store embeddings in MongoDB Atlas with vector search index
4. **Query Processing**: Convert user queries to embeddings
5. **Similarity Search**: Find relevant document chunks using cosine similarity
6. **Context Generation**: Retrieve top-k most relevant documents
7. **LLM Response**: Generate answers using retrieved context with Google Gemini

## 📦 Prerequisites

- **Python**: 3.11 or higher
- **MongoDB Atlas**: Account with M10+ cluster (for vector search)
- **Google AI API Key**: For Gemini LLM access
- **UV Package Manager**: For dependency management (recommended)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-pipeline
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

Using UV (recommended):

```bash
uv add -r requirements.txt
```

Or using pip:

```bash
pip install -r requirements.txt
```

### Dependencies Include:

- `pymongo` - MongoDB driver
- `langchain` - LLM framework
- `langchain_community` - Community integrations
- `langchain_google_genai` - Google Gemini integration
- `langchain-huggingface` - HuggingFace embeddings
- `sentence-transformers` - Pre-trained embedding models
- `pypdf` - PDF processing
- `python-dotenv` - Environment variable management

## ⚙️ Configuration

### 1. Create `.env` File

```bash
cp .env.example .env
```

### 2. Add Your Credentials

```env
# Google AI API Key
GOOGLE_API_KEY=your_google_api_key_here

# MongoDB Atlas Connection String
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?appName=Rag
```

### 3. MongoDB Atlas Setup

> **📺 Video Tutorial**: New to MongoDB Atlas? Watch this step-by-step guide: [How to Register MongoDB Cluster](https://www.youtube.com/watch?v=FepDo-0DrSo)

1. Create a MongoDB Atlas account at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Create an M10+ cluster (required for vector search)
3. Create a database named `sample_mflix`
4. Create a collection named `ragpdf`
5. Get your connection string from Atlas

## 🎯 Usage

### Running the Jupyter Notebook

```bash
jupyter notebook rag.ipynb
```

### Step-by-Step Execution:

#### 1. **Initialize LLM**

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
```

#### 2. **Setup Embeddings**

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

#### 3. **Load and Process Documents**

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("https://investors.mongodb.com/node/12236/pdf")
data = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
documents = text_splitter.split_documents(data)
```

#### 4. **Generate Embeddings**

```python
docs_to_insert = [{
    "text": doc.page_content,
    "embedding": get_embedding(doc.page_content)
} for doc in documents]
```

#### 5. **Insert into MongoDB**

```python
from pymongo import MongoClient

client = MongoClient(os.getenv("MONGODB_URI"))
collection = client["sample_mflix"]["ragpdf"]
result = collection.insert_many(docs_to_insert)
```

#### 6. **Create Vector Search Index**

```python
from pymongo.operations import SearchIndexModel

search_index_model = SearchIndexModel(
    definition={
        "fields": [{
            "type": "vector",
            "numDimensions": 384,
            "path": "embedding",
            "similarity": "cosine"
        }]
    },
    name="vector_index_rag",
    type="vectorSearch"
)
collection.create_search_index(model=search_index_model)
```

#### 7. **Query the System**

```python
query = "What are MongoDB's latest AI announcements?"
context_docs = get_query_results(query)
context_string = " ".join([doc["text"] for doc in context_docs])

prompt = f"""Use the following pieces of context to answer the question at the end.
    {context_string}
    Question: {query}
"""

completion = llm.invoke(prompt)
print(completion.content)
```

## 🔍 Pipeline Components

### 1. **Embedding Function**

The `get_embedding()` function handles text-to-vector conversion:

```python
def get_embedding(text, input_type="document"):
    """
    Get embeddings for the given text using HuggingFace.

    Args:
        text: The text to embed (can be a string or list of strings)
        input_type: Type of input - "document" or "query"

    Returns:
        Embedding vector(s) of 384 dimensions
    """
    if isinstance(text, str):
        embedding = embeddings.embed_query(text)
        print(f"Embedding dimension: {len(embedding)}")
        return embedding
    else:
        embedding_list = embeddings.embed_documents(text)
        print(f"Generated {len(embedding_list)} embeddings")
        return embedding_list
```

**Features:**

- 384-dimensional embeddings using `all-MiniLM-L6-v2` model
- Normalized embeddings for better similarity matching
- Supports both single text and batch processing
- No API quota limits (runs locally)

### 2. **Vector Search Function**

The `get_query_results()` function performs semantic search:

```python
def get_query_results(query):
    """Gets results from a vector search query."""

    query_embedding = get_embedding(query, input_type="query")

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index_rag",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 384,
                "limit": 5
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1
            }
        }
    ]

    results = collection.aggregate(pipeline)
    return list(results)
```

**Parameters:**

- `index`: Name of the vector search index
- `queryVector`: The query embedding (384 dimensions)
- `numCandidates`: Number of candidates to consider (384)
- `limit`: Top-k results to return (5)
- `similarity`: Cosine similarity metric

### 3. **Document Processing**

**Text Splitting Strategy:**

- `chunk_size`: 400 characters per chunk
- `chunk_overlap`: 20 characters overlap between chunks
- Ensures context preservation across chunk boundaries

**Why These Values?**

- 400 chars ≈ 100 tokens (optimal for embedding models)
- 20 char overlap prevents context loss at boundaries
- Balances granularity vs. context completeness

## 📚 API Reference

### Environment Variables

| Variable         | Description                     | Required |
| ---------------- | ------------------------------- | -------- |
| `GOOGLE_API_KEY` | Google AI API key for Gemini    | Yes      |
| `MONGODB_URI`    | MongoDB Atlas connection string | Yes      |

### MongoDB Collections

| Database       | Collection | Purpose                    |
| -------------- | ---------- | -------------------------- |
| `sample_mflix` | `ragpdf`   | Stores document embeddings |

### Vector Search Index Schema

```json
{
  "fields": [
    {
      "type": "vector",
      "numDimensions": 384,
      "path": "embedding",
      "similarity": "cosine"
    }
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

#### 1. **Google API Quota Exceeded**

**Error:** `ResourceExhausted: 429 Resource has been exhausted`

**Solution:** The system automatically falls back to HuggingFace embeddings (local, no quota limits)

#### 2. **MongoDB Connection Failed**

**Error:** `ServerSelectionTimeoutError: localhost:27017`

**Solutions:**

- Ensure MongoDB Atlas cluster is running
- Check connection string in `.env`
- Verify network access (whitelist your IP in Atlas)
- Use M10+ cluster (required for vector search)

#### 3. **Import Errors**

**Error:** `ModuleNotFoundError: No module named 'langchain_huggingface'`

**Solution:**

```bash
uv pip install langchain-huggingface sentence-transformers
```

#### 4. **Embedding Dimension Mismatch**

**Error:** Vector search returns no results

**Solution:**

- Ensure index `numDimensions` matches embedding size (384)
- Recreate the search index if needed
- Verify embeddings are normalized

#### 5. **LLM Response Format Error**

**Error:** `AttributeError: 'AIMessage' object has no attribute 'choices'`

**Solution:** Use `completion.content` instead of `completion.choices[0].message.content`

### Performance Optimization

1. **Batch Processing**: Process multiple documents at once
2. **Index Tuning**: Adjust `numCandidates` based on dataset size
3. **Chunk Size**: Experiment with different chunk sizes for your use case
4. **Caching**: Cache frequently accessed embeddings

## 🔐 Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use environment variables** for all sensitive data
3. **Rotate API keys** regularly
4. **Restrict MongoDB network access** to specific IPs
5. **Use MongoDB Atlas encryption** at rest and in transit

## 📊 Performance Metrics

- **Embedding Generation**: ~0.1s per document (local HuggingFace)
- **Vector Search**: ~50ms per query (MongoDB Atlas)
- **LLM Response**: ~1-3s (Google Gemini 2.0 Flash)
- **Total Pipeline**: ~2-4s per query

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **MongoDB** for Atlas Vector Search
- **LangChain** for the RAG framework
- **HuggingFace** for embedding models
- **Google** for Gemini LLM
- **Sentence Transformers** for pre-trained models

## 📞 Support

For issues and questions:

- Open an issue on GitHub
- Check the [troubleshooting section](#troubleshooting)
- Review MongoDB Atlas [documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/)

---

**Built with ❤️ using MongoDB, LangChain, and Google Gemini**
