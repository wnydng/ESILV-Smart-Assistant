import ollama

# Test embedding
resp = ollama.embeddings(
    model="mxbai-embed-large",
    prompt="hello world"
)

print(len(resp["embedding"]))
print(resp["embedding"][:10])  # aperçu
