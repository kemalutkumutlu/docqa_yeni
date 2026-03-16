with open("src/core/indexing.py", "r", encoding="utf-8") as f:
    text = f.read()

# Remove global locks
text = text.replace("import threading\n", "")
text = text.replace("_CHROMA_WRITE_LOCKS: dict[str, threading.Lock] = {}\n_CHROMA_WRITE_LOCKS_GUARD = threading.Lock()\n\n\ndef _chroma_write_lock(chroma_dir: Path | str) -> threading.Lock:\n    key = str(Path(chroma_dir).resolve())\n    with _CHROMA_WRITE_LOCKS_GUARD:\n        lock = _CHROMA_WRITE_LOCKS.get(key)\n        if lock is None:\n            lock = threading.Lock()\n            _CHROMA_WRITE_LOCKS[key] = lock\n        return lock\n", "")

# Remove `with _chroma_write_lock(chroma_dir):` from build
build_old = """        store = ChromaStore(persist_dir=str(chroma_dir), collection_name=collection_name_use)
        with _chroma_write_lock(chroma_dir):
            # Prevent stale chunks accumulating for the same doc_id(s) in the persistent store.
            doc_ids = sorted({c.doc_id for c in chunks})
            if doc_ids:
                if len(doc_ids) == 1:
                    store.delete_where(where={"doc_id": doc_ids[0]})
                else:
                    store.delete_where(where={"$or": [{"doc_id": did} for did in doc_ids]})

            # Upsert all chunks (parents + children)
            store.upsert_chunks(chunks, embeddings=embeddings)"""
            
build_new = """        store = ChromaStore(persist_dir=str(chroma_dir), collection_name=collection_name_use)
        # Prevent stale chunks accumulating for the same doc_id(s) in the persistent store.
        doc_ids = sorted({c.doc_id for c in chunks})
        if doc_ids:
            if len(doc_ids) == 1:
                store.delete_where(where={"doc_id": doc_ids[0]})
            else:
                store.delete_where(where={"$or": [{"doc_id": did} for did in doc_ids]})

        # Upsert all chunks (parents + children)
        store.upsert_chunks(chunks, embeddings=embeddings)"""
        
text = text.replace(build_old, build_new)

# Remove `with _chroma_write_lock(self.chroma_dir or ""):` from add_chunks
add_old = """        with _chroma_write_lock(self.chroma_dir or ""):
            # Clean stale chunks for the new doc_id(s) in Chroma.
            new_doc_ids = sorted({c.doc_id for c in new_chunks})
            if new_doc_ids:
                if len(new_doc_ids) == 1:
                    self.store.delete_where(where={"doc_id": new_doc_ids[0]})
                else:
                    self.store.delete_where(where={"$or": [{"doc_id": did} for did in new_doc_ids]})

            # If any of these doc_ids were already indexed, we are REPLACING them.
            # Chroma is cleaned above; we must also prevent sparse duplicates.
            already_indexed = set(new_doc_ids) & set(self.allowed_doc_ids)
            if already_indexed:
                self.bm25.remove_doc_ids(already_indexed)

            # Embed only the NEW chunks.
            embeddings = self.embedder.embed_chunks(new_chunks)
            self.store.upsert_chunks(new_chunks, embeddings=embeddings)

            # Extend BM25 incrementally.
            self.bm25.extend(new_chunks)

            # Track new doc_ids.
            self.allowed_doc_ids.update(new_doc_ids)

            # Persist updated BM25
            if self.bm25_path:
                self.bm25.save(self.bm25_path)"""
                
add_new = """        # Clean stale chunks for the new doc_id(s) in Chroma.
        new_doc_ids = sorted({c.doc_id for c in new_chunks})
        if new_doc_ids:
            if len(new_doc_ids) == 1:
                self.store.delete_where(where={"doc_id": new_doc_ids[0]})
            else:
                self.store.delete_where(where={"$or": [{"doc_id": did} for did in new_doc_ids]})

        # If any of these doc_ids were already indexed, we are REPLACING them.
        # Chroma is cleaned above; we must also prevent sparse duplicates.
        already_indexed = set(new_doc_ids) & set(self.allowed_doc_ids)
        if already_indexed:
            self.bm25.remove_doc_ids(already_indexed)

        # Embed only the NEW chunks.
        embeddings = self.embedder.embed_chunks(new_chunks)
        self.store.upsert_chunks(new_chunks, embeddings=embeddings)

        # Extend BM25 incrementally.
        self.bm25.extend(new_chunks)

        # Track new doc_ids.
        self.allowed_doc_ids.update(new_doc_ids)

        # Persist updated BM25
        if self.bm25_path:
            self.bm25.save(self.bm25_path)"""

text = text.replace(add_old, add_new)

with open("src/core/indexing.py", "w", encoding="utf-8") as f:
    f.write(text)
    
print("done")
