from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores  import FAISS





texts = ["Cricket is one of the famous sports in the world", 
         "Still Football is the most popular sport in the world",
           "Basketball is a popular sport in the United States",
             "Tennis is a popular sport in Europe",
               "Golf is a popular sport in Asia",
               "Sachin Tendulkar is a famous cricketer from India",
               "Lionel Messi is a famous footballer from Argentina",
               "LeBron James is a famous basketball player from the United States",
               "Roger Federer is a famous tennis player from Switzerland",
               "Tiger Woods is a famous golfer from the United States"]

# free local and cheap embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store the embeddings in a FAISS index
vector_store   = FAISS.from_texts(texts, embeddings)


vector_store.save_local("my_vector_store")  

retriever = vector_store.as_retriever(search_kwargs={"k": 1})



query = "Who is the best cricketer in the world?"
results = retriever.invoke (query)

for result in results:
    print(result.page_content)
