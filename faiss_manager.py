import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Any, Union
from langchain_ollama import OllamaEmbeddings

class SimpleFaissManager:
    """
    A simple FaissDB manager with mxbai-embed-large via Ollama and auto-incrementing IDs.
    
    Features:
    - Auto-incrementing IDs
    - Add text or vectors with metadata
    - Remove vectors by ID
    - Search for similar content
    - Save/load database
    - Uses mxbai-embed-large through Ollama
    """
    
    def __init__(self, index_type: str = "flat", model_name: str = "mxbai-embed-large", base_url: str = "http://localhost:11434"):
        """
        Initialize the Faiss manager with mxbai-embed-large via Ollama.
        
        Args:
            index_type: Type of index ("flat", "ivf", "hnsw")
            model_name: Ollama model name (default: mxbai-embed-large)
            base_url: Ollama server URL
        """
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )
        
        # Get embedding dimension by testing with a sample text
        sample_embedding = self.embeddings.embed_query("test")
        self.dimension = len(sample_embedding)
        
        self.index_type = index_type
        self.next_id = 0
        
        # Create the appropriate index
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError("Unsupported index type. Use 'flat', 'ivf', or 'hnsw'")
        
        # Add ID mapping to track vectors
        self.index = faiss.IndexIDMap(self.index)
        
        # Store metadata and original text for each vector
        self.metadata = {}
        self.texts = {}
    
    def add(self, content: Union[str, np.ndarray], metadata: Any = None) -> int:
        """
        Add text or vector to the database.
        
        Args:
            content: Text string or numpy array vector
            metadata: Optional metadata to store with the content
            
        Returns:
            The auto-generated ID for the content
        """
        if isinstance(content, str):
            # Generate embedding for text
            vector = np.array(self.embeddings.embed_query(content), dtype=np.float32)
            text = content
        else:
            # Use provided vector
            vector = content.astype(np.float32)
            text = None
        
        # Ensure vector is the right shape
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        if vector.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vector.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Get the next ID
        current_id = self.next_id
        self.next_id += 1
        
        # Add to index with ID
        ids = np.array([current_id], dtype=np.int64)
        self.index.add_with_ids(vector, ids)
        
        # Store metadata and text
        if metadata is not None:
            self.metadata[current_id] = metadata
        if text is not None:
            self.texts[current_id] = text
        
        return current_id
    
    def add_batch(self, contents: List[Union[str, np.ndarray]], metadata_list: List[Any] = None) -> List[int]:
        """
        Add multiple texts or vectors at once.
        
        Args:
            contents: List of text strings or numpy arrays
            metadata_list: Optional list of metadata for each content
            
        Returns:
            List of auto-generated IDs
        """
        vectors = []
        texts = []
        
        for content in contents:
            if isinstance(content, str):
                vector = np.array(self.embeddings.embed_query(content), dtype=np.float32)
                vectors.append(vector)
                texts.append(content)
            else:
                vectors.append(content.astype(np.float32))
                texts.append(None)
        
        vectors = np.vstack(vectors)
        n_vectors = vectors.shape[0]
        
        # Generate IDs
        ids = list(range(self.next_id, self.next_id + n_vectors))
        self.next_id += n_vectors
        
        # Add to index
        ids_array = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(vectors, ids_array)
        
        # Store metadata and texts
        for i, current_id in enumerate(ids):
            if metadata_list and i < len(metadata_list):
                self.metadata[current_id] = metadata_list[i]
            if texts[i] is not None:
                self.texts[current_id] = texts[i]
        
        return ids
    
    def remove(self, vector_id: int) -> bool:
        """
        Remove a vector by ID.
        
        Args:
            vector_id: ID of the vector to remove
            
        Returns:
            True if removed successfully, False if ID not found
        """
        try:
            # Remove from index
            ids_to_remove = np.array([vector_id], dtype=np.int64)
            self.index.remove_ids(ids_to_remove)
            
            # Remove metadata and text
            self.metadata.pop(vector_id, None)
            self.texts.pop(vector_id, None)
            
            return True
        except Exception as e:
            print(f"Error removing vector {vector_id}: {e}")
            return False
    
    def search(self, query: Union[str, np.ndarray], k: int = 5, return_metadata: bool = True) -> List[dict]:
        """
        Search for similar vectors.
        
        Args:
            query: Text string or numpy array to search for
            k: Number of results to return
            return_metadata: Whether to include metadata in results
            
        Returns:
            List of dictionaries with search results
        """
        if isinstance(query, str):
            # Generate embedding for text query
            query_vector = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        else:
            query_vector = query.astype(np.float32)
        
        # Ensure query vector is the right shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search
        distances, ids = self.index.search(query_vector, k)
        
        results = []
        for i in range(len(ids[0])):
            if ids[0][i] != -1:  # Valid result
                result = {
                    'id': int(ids[0][i]),
                    'distance': float(distances[0][i]),
                    'similarity': 1 / (1 + distances[0][i])  # Convert distance to similarity
                }
                
                # Add text if available
                if ids[0][i] in self.texts:
                    result['text'] = self.texts[ids[0][i]]
                
                # Add metadata if requested and available
                if return_metadata and ids[0][i] in self.metadata:
                    result['metadata'] = self.metadata[ids[0][i]]
                
                results.append(result)
        
        return results
    
    def get_by_id(self, vector_id: int) -> Optional[dict]:
        """
        Get vector information by ID.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Dictionary with vector information or None if not found
        """
        result = {'id': vector_id}
        
        if vector_id in self.texts:
            result['text'] = self.texts[vector_id]
        
        if vector_id in self.metadata:
            result['metadata'] = self.metadata[vector_id]
        
        return result if len(result) > 1 else None
    
    def get_stats(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with stats
        """
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'next_id': self.next_id,
            'has_metadata': len(self.metadata),
            'has_text': len(self.texts)
        }
    
    def save(self, filepath: str):
        """
        Save the database to disk.
        
        Args:
            filepath: Path to save the database (without extension)
        """
        # Save Faiss index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save metadata, texts, and other info
        data = {
            'metadata': self.metadata,
            'texts': self.texts,
            'next_id': self.next_id,
            'dimension': self.dimension,
            'index_type': self.index_type
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Database saved to {filepath}.faiss and {filepath}.pkl")
    
    def load(self, filepath: str):
        """
        Load the database from disk.
        
        Args:
            filepath: Path to load the database from (without extension)
        """
        # Load Faiss index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata and other info
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.metadata = data['metadata']
        self.texts = data['texts']
        self.next_id = data['next_id']
        self.dimension = data['dimension']
        self.index_type = data['index_type']
        
        print(f"Database loaded from {filepath}.faiss and {filepath}.pkl")
    
    def clear(self):
        """Clear all vectors from the database."""
        # Recreate the index
        if self.index_type == "flat":
            base_index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            base_index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "hnsw":
            base_index = faiss.IndexHNSWFlat(self.dimension, 32)
        
        self.index = faiss.IndexIDMap(base_index)
        self.metadata.clear()
        self.texts.clear()
        self.next_id = 0
        
        print("Database cleared")


# Example usage
if __name__ == "__main__":
    # Initialize the manager
    db = SimpleFaissManager()
    
    # Add some documents
    doc1_id = db.add("Python is a great programming language for data science")
    doc2_id = db.add("Machine learning models require large datasets", metadata={"category": "ML"})
    doc3_id = db.add("Faiss is efficient for similarity search", metadata={"category": "Search"})
    
    print(f"Added documents with IDs: {doc1_id}, {doc2_id}, {doc3_id}")
    
    # Search for similar documents
    results = db.search("programming languages", k=2)
    print("\nSearch results for 'programming languages':")
    for result in results:
        print(f"ID: {result['id']}, Similarity: {result['similarity']:.3f}")
        print(f"Text: {result['text']}")
        if 'metadata' in result:
            print(f"Metadata: {result['metadata']}")
        print()
    
    # Get database stats
    stats = db.get_stats()
    print("Database stats:", stats)
    
    # Save the database
    # db.save("my_vector_db")
    
    # Load the database
    # db.load("my_vector_db")