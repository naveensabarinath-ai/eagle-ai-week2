from langchain_huggingface import HuggingFaceEmbeddings

# A small popular model for embeddings is "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("STARTING EMBEDDING")
embedded = embeddings.embed_documents(["Hello world", "This is a test"])
print(embedded)



