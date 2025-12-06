import os
import time
from typing import Union, List

# Try importing different embedding providers
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Google Generative AI not available")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("HuggingFace embeddings not available")

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class EmbeddingManager:
    """Manages multiple embedding providers with automatic fallback"""
    
    def __init__(self, provider="google"):
        """
        Initialize embedding manager
        
        Args:
            provider: "google", "huggingface", or "openai"
        """
        self.provider = provider
        self.embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model based on provider"""
        if self.provider == "google" and GOOGLE_AVAILABLE:
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    task_type="retrieval_document"
                )
                print("✓ Using Google Generative AI Embeddings")
            except Exception as e:
                print(f"Failed to initialize Google embeddings: {e}")
                self._fallback_to_huggingface()
        
        elif self.provider == "huggingface" and HUGGINGFACE_AVAILABLE:
            self._initialize_huggingface()
        
        elif self.provider == "openai" and OPENAI_AVAILABLE:
            self.embeddings = OpenAIEmbeddings()
            print("✓ Using OpenAI Embeddings")
        
        else:
            self._fallback_to_huggingface()
    
    def _initialize_huggingface(self):
        """Initialize HuggingFace embeddings (free, local, no API limits)"""
        print("Initializing HuggingFace embeddings (this may take a moment on first run)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Using HuggingFace Embeddings (all-MiniLM-L6-v2)")
    
    def _fallback_to_huggingface(self):
        """Fallback to HuggingFace if other providers fail"""
        if HUGGINGFACE_AVAILABLE:
            print("⚠ Falling back to HuggingFace embeddings...")
            self._initialize_huggingface()
        else:
            raise RuntimeError("No embedding provider available. Install langchain-huggingface or langchain-google-genai")
    
    def get_embedding(self, text: Union[str, List[str]], input_type: str = "document", max_retries: int = 3):
        """
        Get embeddings for the given text with retry logic and error handling.
        
        Args:
            text: The text to embed (can be a string or list of strings)
            input_type: Type of input - "document" or "query"
            max_retries: Maximum number of retry attempts
        
        Returns:
            Embedding vector(s)
        """
        # Update task_type for Google embeddings
        if self.provider == "google" and hasattr(self.embeddings, 'task_type'):
            if input_type == "query":
                self.embeddings.task_type = "retrieval_query"
            else:
                self.embeddings.task_type = "retrieval_document"
        
        # Retry logic for handling rate limits
        for attempt in range(max_retries):
            try:
                # Get embeddings
                if isinstance(text, str):
                    # Single text
                    embedding = self.embeddings.embed_query(text)
                    print(f"✓ Embedding dimension: {len(embedding)}")
                    return embedding
                else:
                    # Multiple texts
                    embedding_list = self.embeddings.embed_documents(text)
                    print(f"✓ Generated {len(embedding_list)} embeddings, dimension: {len(embedding_list[0])}")
                    return embedding_list
            
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a quota/rate limit error
                if "ResourceExhausted" in error_msg or "429" in error_msg or "quota" in error_msg.lower():
                    print(f"\n⚠ API Quota Exhausted! Switching to HuggingFace embeddings...")
                    self.provider = "huggingface"
                    self._initialize_huggingface()
                    # Retry with new provider
                    continue
                
                # Check if it's a temporary error
                elif attempt < max_retries - 1 and ("timeout" in error_msg.lower() or "503" in error_msg):
                    wait_time = (attempt + 1) * 2
                    print(f"⚠ Temporary error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                else:
                    print(f"✗ Error getting embeddings: {e}")
                    raise
        
        raise RuntimeError(f"Failed to get embeddings after {max_retries} attempts")


# Initialize with Google by default (will auto-fallback to HuggingFace if quota exceeded)
embedding_manager = EmbeddingManager(provider="google")

def get_embedding(text: Union[str, List[str]], input_type: str = "document"):
    """
    Convenience function to get embeddings using the global embedding manager.
    
    Args:
        text: The text to embed (can be a string or list of strings)
        input_type: Type of input - "document" or "query"
    
    Returns:
        Embedding vector(s)
    """
    return embedding_manager.get_embedding(text, input_type)


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n" + "="*60)
    print("Testing Embedding Function")
    print("="*60 + "\n")
    
    # Test with a single document
    print("1. Testing single document embedding:")
    single_embedding = get_embedding("This is a sample document", input_type="document")
    print(f"   Result: {len(single_embedding)} dimensions\n")
    
    # Test with a query
    print("2. Testing query embedding:")
    query_embedding = get_embedding("What is this about?", input_type="query")
    print(f"   Result: {len(query_embedding)} dimensions\n")
    
    # Test with multiple documents
    print("3. Testing multiple documents:")
    docs = ["First document", "Second document", "Third document"]
    multi_embeddings = get_embedding(docs, input_type="document")
    print(f"   Result: {len(multi_embeddings)} embeddings generated\n")
