from sentence_transformers import SentenceTransformer

def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embed_texts(model, texts):
    vectors = model.encode(texts, convert_to_numpy=True)
    # Convert each vector element to Python float
    vectors = [[float(x) for x in vec] for vec in vectors]
    vec = vectors[0]
    print(type(vec[0]), vec[:10])
    return vectors
