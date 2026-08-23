import chromadb
import ollama
import os

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="adhd_knowledge")

knowledge_folder = "knowledge"

for filename in os.listdir(knowledge_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(knowledge_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Sadə chunk-lama: paraqraflara böl
        chunks = [text.strip()]

        for i, chunk in enumerate(chunks):
            response = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
            embedding = response["embedding"]

            collection.add(
                ids=[f"{filename}_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": filename}]
            )

        print(f"Əlavə edildi: {filename} ({len(chunks)} hissə)")

print("Bitdi! Bilik bazası hazırdır.")