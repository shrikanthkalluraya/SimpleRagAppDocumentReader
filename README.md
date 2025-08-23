Simple Document Q&A RAG App
Project Overview: Build a RAG system that can answer questions about uploaded documents using LangChain, LangGraph for workflow orchestration, and LangSmith for monitoring.Simple RAG App with LangChain EcosystemDocument 

# Simple RAG App with LangChain Ecosystem

## Project Overview
Build a document Q&A system that demonstrates RAG, LangGraph, LangChain, and LangSmith integration.

## Features
- Upload and process PDF/text documents
- Vector-based document retrieval
This project gives you a complete RAG application that demonstrates all the key components:

RAG Concepts: Document chunking, vector embeddings, similarity search, and context-aware generation

LangGraph: Orchestrates the workflow with clear states (retrieve → generate) and shows how to build complex AI workflows

LangChain: Handles all the heavy lifting - document loading, text splitting, embeddings, vector stores, and LLM interactions

LangSmith: Automatically traces every step, giving you insights into performance, costs, and errors

The project is simple enough to understand quickly but comprehensive enough to showcase real RAG capabilities. You can start with basic text files and gradually add more complex documents to see how the system performs.


## Installation

1. **Clone the repository**

   ```bash
   git clone git@github.com:shrikanthkalluraya/SimpleRagAppDocumentReader.git
   cd SimpleRagAppDocumentReader
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Flow Chart for the applications**

   ```bash
   You give text to robot
       ↓
   Robot breaks it into pieces  
       ↓
   Embeddings turn pieces into vectors
       ↓  
   CHROMA stores the vectors
       ↓
   You ask a question

4. **Key Learning Points**
   
   &emsp;***RAG Concepts Demonstrated:***

      &emsp;&emsp;****Document Loading****: PDF/text file processing
   
      &emsp;&emsp;****Text Chunking****: Breaking documents into manageable pieces
   
      &emsp;&emsp;****Embeddings****: Converting text to vectors
   
      &emsp;&emsp;****Vector Search****: Finding relevant document chunks
   
      &emsp;&emsp;****Context Injection****: Adding retrieved context to prompts&emsp;
   
   &emsp;***LangGraph Features:***
   
      &emsp;&emsp;****State Management:**** Tracking workflow state
   
      &emsp;&emsp;****Node Definition:**** Individual processing steps
   
      &emsp;&emsp;****Edge Configuration:**** Workflow transitions
   
      &emsp;&emsp;****Workflow Orchestration:**** Managing the complete pipeline
   
   &emsp;***LangChain Integration:***
   
      &emsp;&emsp;****Document Loaders:**** File processing utilities
      
      &emsp;&emsp;****Text Splitters:**** Chunking strategies
      
      &emsp;&emsp;****Vector Stores:**** ChromaDB integration
      
      &emsp;&emsp;****Retrievers:**** Search functionality
      
      &emsp;&emsp;****Prompt Templates:**** Structured prompts
   
   &emsp;***LangSmith Monitoring:***
   
      &emsp;&emsp;****Automatic Tracing:**** Every LLM call traced
      
      &emsp;&emsp;****Performance Metrics:**** Response times, token usage
   
     &emsp;&emsp;**** Error Tracking:**** Failed operations
   
      &emsp;&emsp;****Workflow Visualization:**** Step-by-step execution
             ↓
         Chroma finds similar vectors
             ↓
         Robot uses found pieces to answer
   
