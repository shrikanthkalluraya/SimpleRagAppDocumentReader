from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings


class LocalRAGDemo:
    def __init__(self):
        # Initialize LOCAL embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use CPU
        )
        self.vectorstore = None

    def process_documents(self, texts):
        """Process documents using local embeddings"""
        # Split texts
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = []
        for text in texts:
            chunks.extend(text_splitter.split_text(text))

        # Create vector store with LOCAL embeddings
        self.vectorstore = Chroma.from_texts(
            chunks,
            self.embeddings,
            persist_directory="./local_chroma_db"
        )

        return f"✅ Processed {len(chunks)} chunks locally!"

    def query(self, question, k=3):
        """Query using local vector search"""
        if not self.vectorstore:
            return "❌ No documents processed yet!"

        # LOCAL similarity search
        docs = self.vectorstore.similarity_search(question, k=k)

        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"{i}. {doc.page_content}")

        return "\n\n".join(results)


# Demo usage
if __name__ == "__main__":
    rag = LocalRAGDemo()

    # Sample documents
    docs = [
        "Python is a programming language. It's great for beginners and experts alike.",
        "Machine learning uses algorithms to find patterns in data.",
        "Dogs are loyal pets that love to play and go for walks.",
        "Cats are independent animals that like to climb and hunt mice."
    ]

    # Process documents locally
    print(rag.process_documents(docs))

    # Query locally
    print("\n" + "=" * 50)
    print("Query: Tell me about pets")
    print("=" * 50)
    print(rag.query("Tell me about pets"))

    print("\n" + "=" * 50)
    print("Query: What is programming?")
    print("=" * 50)
    print(rag.query("What is programming?"))