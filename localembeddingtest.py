from langchain_huggingface import HuggingFaceEmbeddings

# This will download the model first time (be patient!)
print("📥 Downloading model for first time...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Model ready!")

# Test it out!
text1 = "I love dogs"
text2 = "Puppies are great"
text3 = "Cars need gas"

vector1 = embeddings.embed_query(text1)
vector2 = embeddings.embed_query(text2)
vector3 = embeddings.embed_query(text3)

print(f"Dog vector (first 5 numbers): {vector1[:5]} length: {len(vector1)}")
print(f"Puppy vector (first 5 numbers): {vector2[:5]} length: {len(vector2)}")
print(f"Car vector (first 5 numbers): {vector3[:5]} length: {len(vector3)}")
