# A Vector Store is a database that stores vectors, which are numerical representations of data. In the context of machine learning and natural language processing, vectors are often used to represent text, images, or other types of data in a way that allows for efficient similarity searches and retrieval.

# In this example, we will use the FAISS library to create a simple vector store that can store and retrieve vectors. We will also use the OpenAI API to generate embeddings for text data.

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

bestCricketPlayer = vector_store.similarity_search("Who is the best cricketer in the world?", k=1)
footballPlayer = vector_store.similarity_search("Who is the best footballer in the world?", k=1)
tennisPlayer = vector_store.similarity_search("Who is the best tennis player in the world?", k=1)

print("Best Cricket Player: ", bestCricketPlayer)
print("Best Football Player: ", footballPlayer)
print("Best Tennis Player: ", tennisPlayer)


vector_store.save_local("my_vector_store")  # Save the vector store to disk







