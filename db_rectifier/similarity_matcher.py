import numpy as np
import faiss
import pickle
import os
from langchain_ollama import OllamaEmbeddings

class NewsSimilarityDB:
    def __init__(self, db_path="news_similarity_db"):
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text:v1.5")
        self.news_list = []
        self.faiss_index = None
        self.dim = None
        self.db_path = db_path
        self.db_file = f"{db_path}.pkl"
        self.index_file = f"{db_path}.index"
    
    def add_news(self, news_items):
        """Add news items to the database"""
        if isinstance(news_items, str):
            news_items = [news_items]
        
        # Add to news list
        self.news_list.extend(news_items)
        
        # Generate embeddings for all news
        doc_embeddings = self.embedding_model.embed_documents(self.news_list)
        doc_embeddings = np.array(doc_embeddings).astype("float32")
        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        
        # Rebuild FAISS index
        self.dim = doc_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index.add(doc_embeddings)
        
        print(f"✅ Added {len(news_items)} news item(s). Total news in DB: {len(self.news_list)}")
        
        # Auto-save after adding news
        self.save_database()
    
    def check_similarity(self, query, k=3, similarity_threshold=0.9):
        """Check similarity of query against news database"""
        if not self.news_list:
            print("❌ No news in database. Please add some news first.")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding = np.array(query_embedding).astype("float32")
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search in FAISS index
        k = min(k, len(self.news_list))  # Don't search for more items than we have
        D, I = self.faiss_index.search(np.array([query_embedding]), k)
        
        # print(f"\n🔎 Query: {query}\n")
        # print("📊 Similarity Results:")
        # print("-" * 50)
        
        results = []
        for rank, idx in enumerate(I[0]):
            score = D[0][rank]
            news_item = self.news_list[idx]
            
            # Determine similarity level
            if score > similarity_threshold:
                status = "✅ Very Similar"
            elif score > 0.6:
                status = "⚠️ Somewhat Similar"
            else:
                status = "❌ Not Similar"
            
            # print(f"{rank+1}. '{news_item}'")
            # print(f"   Score: {score:.4f} - {status}\n")
            
            results.append({
                'news': news_item,
                'score': score,
                'rank': rank + 1,
                'is_similar': score > similarity_threshold
            })
        
        return results
    
    def get_all_news(self):
        """Get all news items in the database"""
        return self.news_list.copy()
    
    def save_database(self):
        """Save the database to local files"""
        try:
            # Save news list and metadata
            db_data = {
                'news_list': self.news_list,
                'dim': self.dim
            }
            with open(self.db_file, 'wb') as f:
                pickle.dump(db_data, f)
            
            # Save FAISS index
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, self.index_file)
            
            print(f"💾 Database saved to {self.db_file} and {self.index_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving database: {e}")
            return False
    
    def load_database(self):
        """Load the database from local files"""
        try:
            # Check if files exist
            if not os.path.exists(self.db_file):
                print(f"📂 No existing database found at {self.db_file}")
                return False
            
            # Load news list and metadata
            with open(self.db_file, 'rb') as f:
                db_data = pickle.load(f)
            
            self.news_list = db_data['news_list']
            self.dim = db_data['dim']
            
            # Load FAISS index
            if os.path.exists(self.index_file):
                self.faiss_index = faiss.read_index(self.index_file)
            
            print(f"📂 Database loaded successfully. Total news: {len(self.news_list)}")
            return True
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return False
    
    def database_exists(self):
        """Check if database files exist"""
        return os.path.exists(self.db_file) and os.path.exists(self.index_file)

# # Example usage
# if __name__ == "__main__":
#     # Initialize the database
#     news_db = NewsSimilarityDB()
    
#     # Add initial news items
#     initial_news = [
#         "Maruti bought the blueprint for the new Suzuki 500 car model",
#         "Tata launches a new electric SUV in India",
#         "Hyundai partners with Google for AI-based navigation",
#     ]
#     news_db.add_news(initial_news)
    
#     # Add more news items
#     news_db.add_news("Tesla announces new Model Y variant for Indian market")
#     news_db.add_news("Mahindra unveils electric truck prototype")
    
#     # Check similarity
#     query1 = "Maruti invested on car blueprint for a new car model"
#     results1 = news_db.check_similarity(query1, k=3, similarity_threshold=0.75)
    
#     # Check another query
#     query2 = "New electric vehicle launched in India"
#     results2 = news_db.check_similarity(query2, k=3, similarity_threshold=0.75)
    
#     # Show all news in database
#     print("\n📰 All news in database:")
#     for i, news in enumerate(news_db.get_all_news(), 1):
#         print(f"{i}. {news}")