import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="adhd_knowledge")
print("Neçə hissə var:", collection.count())