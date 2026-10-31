import json
import time
import sys
import os
import faiss
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

def run_stress_test(multiplier=100):
    print("==================================================")
    print("🚀 INITIALIZING VECTOR DATABASE ARCHITECTURE TEST")
    print("==================================================")

    # 1. Load your original curated domain data
    files = ["naruto_s1_backup.json", "naruto_s2.json"]
    real_docs, real_ids, real_meta = [], [], []

    for file_name in files:
        if not os.path.exists(file_name):
            print(f"❌ Error: {file_name} missing from path. Run your Git extraction step.")
            sys.exit(1)
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data:
            real_docs.append(doc["raw_text"])
            real_ids.append(str(doc["id"]))
            real_meta.append({"url": doc["url"]})

    print(f"📦 Successfully parsed baseline files ({len(real_docs)} source records).")

    # 2. Synthetic scaling loop to test multi-node limits
    scaled_docs, scaled_ids, scaled_metadata = [], [], []
    for i in range(multiplier):
        for idx, (doc, doc_id, meta) in enumerate(zip(real_docs, real_ids, real_meta)):
            scaled_docs.append(f"[Simulated Data Node-{i}] {doc}")
            scaled_ids.append(f"{doc_id}_{i}")
            scaled_metadata.append({"url": meta["url"], "archive_id": i})

    total_records = len(scaled_docs)
    print(f"🔥 Dataset synthetically scaled up to {total_records:,} total documents.")

    # 3. Generate normalized embeddings vector block
    print("🧠 Generating text embeddings using all-MiniLM-L6-v2 (please wait)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(scaled_docs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    v_dim = embeddings.shape[1]
    print(f"✅ Generated embedding matrix layout shape: {embeddings.shape}")

    # ==================================================
    # INDEX BUILDING STEPS
    # ==================================================
    print("\n🏗️ Building Index Structures...")

    # Index Type A: FAISS Exact Flat
    index_faiss_flat = faiss.IndexFlatIP(v_dim)
    index_faiss_flat.add(embeddings)

    # Index Type B: FAISS Approximate Quantized IVF (Inverted File Index)
    nlist = int(np.sqrt(total_records)) # Standard heuristic for partitioning cluster space
    quantizer = faiss.IndexFlatIP(v_dim)
    index_faiss_ivf = faiss.IndexIVFFlat(quantizer, v_dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index_faiss_ivf.train(embeddings) # Map cluster boundaries
    index_faiss_ivf.add(embeddings)
    index_faiss_ivf.nprobe = int(nlist * 0.1) # Scan closest 10% of clusters

    # Index Type C: ChromaDB HNSW Proximity Graph
    chroma_client = chromadb.PersistentClient(path="./chroma_stress_db")
    if "naruto_scaled" in {c.name for c in chroma_client.list_collections()}:
        chroma_client.delete_collection("naruto_scaled")
    chroma_collection = chroma_client.create_collection(
        name="naruto_scaled",
        configuration={"hnsw": {"space": "cosine"}}
    )
    
    # Safely batch store items into disk directory
    batch_size = 5000
    for i in range(0, total_records, batch_size):
        chroma_collection.add(
            ids=scaled_ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size].tolist(),
            documents=scaled_docs[i:i+batch_size],
            metadatas=scaled_metadata[i:i+batch_size]
        )

    # ==================================================
    # MULTI-QUERY EVALUATION BENCHMARK
    # ==================================================
    print("\n⏱️ Running Stress Test Search Queries...")
    test_queries = [
        "Nine tail fox attack",
        "Sasuke fights Gaara",
        "Naruto learns a new technique",
        "Chunin Exams tournament battle"
    ]
    
    # Warmup round
    q_emb_warm = model.encode([test_queries[0]], normalize_embeddings=True).astype("float32")
    index_faiss_flat.search(q_emb_warm, 5)
    index_faiss_ivf.search(q_emb_warm, 5)
    chroma_collection.query(query_embeddings=q_emb_warm.tolist(), n_results=5)

    # Execution tracking timers
    t_flat, t_ivf, t_chroma = [], [], []

    for query in test_queries:
        q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
        
        # Track FAISS Flat
        start = time.perf_counter()
        index_faiss_flat.search(q_emb, 5)
        t_flat.append(time.perf_counter() - start)
        
        # Track FAISS IVF
        start = time.perf_counter()
        index_faiss_ivf.search(q_emb, 5)
        t_ivf.append(time.perf_counter() - start)
        
        # Track ChromaDB
        start = time.perf_counter()
        chroma_collection.query(query_embeddings=q_emb.tolist(), n_results=5)
        t_chroma.append(time.perf_counter() - start)

    print("\n==================================================")
    print("📊 ARCHITECTURAL SYSTEM BENCHMARK METRICS SUMMARY")
    print("==================================================")
    print(f"Data Payload Size: {total_records:,} vectors")
    print(f"FAISS Flat (Exhaustive Brute Force):  {np.mean(t_flat):.6f} seconds (Average)")
    print(f"FAISS IVF (Approximate Clustering):    {np.mean(t_ivf):.6f} seconds (Average)")
    print(f"ChromaDB (Hierarchical HNSW Graph):   {np.mean(t_chroma):.6f} seconds (Average)")
    print("==================================================")

if __name__ == "__main__":
    # Multiplier=100 will expand your 442 items to ~44,200 vectors
    run_stress_test(multiplier=100)
