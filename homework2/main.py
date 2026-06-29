import numpy as np

from embedder import Embedder
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]

def main():
    query = "How does approximate nearest neighbor search work?"
    embedder = Embedder()
    vector = embedder.encode(query)

    # The embedder returns a vector of 384 numbers. What's the first value (v[0])?
    print(vector[0]) # -0.02058200593003704

def cosineSimilarity(documents):
    query = "How does approximate nearest neighbor search work?"
    embedder = Embedder()
    query_vector = embedder.encode(query)

    document = next(
        d
        for d in documents
        if d["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md"
    )
    document_vector = embedder.encode(document["content"])

    similarity = np.dot(query_vector, document_vector)
    print(similarity) # 0.36107029062443674

def chunkAndSearch():
    query = "How does approximate nearest neighbor search work?"
    embedder = Embedder()

    chunks = chunk_documents(documents, size=2000, step=1000)
    v = embedder.encode(query)
    X = embedder.encode_batch([chunk["content"] for chunk in chunks])
    scores = X.dot(v)

    best_idx = int(np.argmax(scores))
    print(chunks[best_idx]["filename"])  # 02-vector-search/lessons/07-sqlitesearch-vector.md

def vectorSearchMinSearch():
    query = "What metric do we use to evaluate a search engine?"
    embedder = Embedder()

    chunks = chunk_documents(documents, size=2000, step=1000)
    X = embedder.encode_batch([chunk["content"] for chunk in chunks])
    v = embedder.encode(query)

    vindex = VectorSearch(keyword_fields=["filename"])
    vindex.fit(X, chunks)
    results = vindex.search(v, num_results=5)

    print(results[0]["filename"])  # 04-evaluation/lessons/05-search-metrics.md

def textSearchVectorSearch():
    query = "How do I store vectors in PostgreSQL?"
    embedder = Embedder()

    chunks = chunk_documents(documents, size=2000, step=1000)

    text_index = Index(text_fields=["content"], keyword_fields=["filename"])
    text_index.fit(chunks)
    text_results = text_index.search(query, num_results=5)
    text_filenames = {result["filename"] for result in text_results}

    X = embedder.encode_batch([chunk["content"] for chunk in chunks])
    v = embedder.encode(query)
    vindex = VectorSearch(keyword_fields=["filename"])
    vindex.fit(X, chunks)
    vector_results = vindex.search(v, num_results=5)

    vector_only = [
        result["filename"]
        for result in vector_results
        if result["filename"] not in text_filenames
    ]
    print(vector_only[0])  # 02-vector-search/lessons/08-pgvector.md

def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def hybridSearch():
    query = "How do I give the model access to tools?"
    embedder = Embedder()

    chunks = chunk_documents(documents, size=2000, step=1000)

    text_index = Index(text_fields=["content"], keyword_fields=["filename"])
    text_index.fit(chunks)
    text_results = text_index.search(query, num_results=5)

    X = embedder.encode_batch([chunk["content"] for chunk in chunks])
    v = embedder.encode(query)
    vindex = VectorSearch(keyword_fields=["filename"])
    vindex.fit(X, chunks)
    vector_results = vindex.search(v, num_results=5)

    results = rrf([vector_results, text_results])
    print(results[0]["filename"])  # 01-agentic-rag/lessons/13-function-calling.md

if __name__ == "__main__":
    main()
    cosineSimilarity(documents)
    chunkAndSearch()
    vectorSearchMinSearch()
    textSearchVectorSearch()
    hybridSearch()
