import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

class RetentionKnowledgeBase:
    def __init__(self, persist_directory="./data/chroma_db"):
        self.persist_directory = persist_directory
        
        # 🟢 100% FREE & LOCAL: Using Hugging Face MiniLM for fast, lightweight embeddings
        print("⏳ Loading local Hugging Face embedding model (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vector_store = self._initialize_db()

    def _initialize_db(self):
        """Creates the database and loads it with Telecom Retention SOPs."""
        os.makedirs(self.persist_directory, exist_ok=True)

        vector_store = Chroma(
            collection_name="retention_sops",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        # Only load documents if the database is empty
        if vector_store._collection.count() == 0:
            print("🧠 Initializing Vector DB with Telecom SOPs...")
            
            # --- NEW: Read from the Markdown file ---
            from langchain_community.document_loaders import TextLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            # Load the markdown file
            loader = TextLoader("./data/knowledge_base/telecom_sop_2026.md")
            docs = loader.load()
            
            # Split the document into logical chunks for the Vector DB
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=50,
                separators=["## ", "### ", "\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(docs)
            
            vector_store.add_documents(splits)
            print(f"✅ Vector DB populated with {len(splits)} knowledge chunks.")
        else:
            print(f"✅ Vector DB loaded. Found {vector_store._collection.count()} chunks.")
            
        return vector_store

    def retrieve_strategy(self, query: str, k: int = 1) -> str:
        """Searches the database for the best SOP based on the churn driver."""
        results = self.vector_store.similarity_search(query, k=k)
        if results:
            return results[0].page_content
        return "No specific strategy found. Default to standard retention discount."

# For testing the module directly
if __name__ == "__main__":
    kb = RetentionKnowledgeBase()
    test_query = "Customer is churning because of too many customer service calls."
    print("\n🔍 Test Retrieval for:", test_query)
    print("💡 Result:", kb.retrieve_strategy(test_query))