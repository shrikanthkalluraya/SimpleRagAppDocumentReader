flowchart TD
    A[Start RAG Application] --> B{Knowledge Base Ready?}
    B -->|No| C[Load Documents]
    B -->|Yes| I[Wait for User Query]
    
    C --> D[Split into Chunks]
    D --> E[Generate Embeddings]
    E --> F[Store in Vector Database]
    F --> I
    
    I --> J[Receive User Query]
    J --> K[Generate Query Embedding]
    K --> L[Search Vector Database]
    L --> M[Retrieve Top-K Similar Chunks]
    M --> N[Combine Chunks into Context]
    N --> O[Create Prompt with Context + Query]
    O --> P[Send to Language Model]
    P --> Q[Generate Response]
    Q --> R[Return Response + Sources]
    R --> S[Display to User]
    S --> T{Continue?}
    T -->|Yes| I
    T -->|No| U[End]

    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style K fill:#e8f5e8
    style L fill:#e8f5e8
    style M fill:#e8f5e8
    style N fill:#e8f5e8
    style O fill:#fff3e0
    style P fill:#fff3e0
    style Q fill:#fff3e0
    style U fill:#ffebee
