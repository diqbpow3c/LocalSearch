import os

from PySide6.QtCore import (
    Signal,
    QObject,
)
import logging
import faiss
import bm25s
from bm25s.tokenization import Tokenizer
import Stemmer
from tokenizers import Tokenizer as EmbTokenizer
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import rjieba
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer as EmbTokenizer
import onnxruntime as ort

import pickle
import configs
from configs import *

logger = logging.getLogger(__name__)


class Searcher(QObject):
    """
    A class for performing hybrid search (BM25 for chunks, BM25 for documents,
    embedding similarity, and a combination of all) on files and folders.
    It uses FAISS for embedding similarity and rank_bm25 for BM25 search.
    """

    index_rebuild_progress_signal = Signal(str)

    def __init__(self, file_paths=None, folder_paths=None):
        """
        Initializes the HybridSearch object.

        Args:
            file_paths (list): A list of initial file paths to index.
            folder_paths (list): A list of initial folder paths to index.
        """
        super().__init__()
        self.device = configs.DEVICE
        self.file_paths = set(file_paths) if file_paths else set()
        self.folder_paths = set(folder_paths) if folder_paths else set()

        # Data structures to store indexed content and metadata
        self.documents = (
            []
        )  # List of dictionaries: {'path': ..., 'content': ..., 'mtime': ..., 'size': ...}
        self.chunks = []  # List of dictionaries: {'doc_idx': ..., 'text': ...}
        self.old_chunks = None
        self.faiss_index = None
        self.bm25_chunk_model = None
        self.bm25_doc_model = None
        self.embedding_model = None

        self.chunk_size = CHUNK_SIZE  # Characters per chunk
        self.chunk_overlap = CHUNK_OVERLAP  # Overlap between chunks

        self.stemmer = Stemmer.Stemmer("english")
        self.tokenizer_doc = Tokenizer(
            stemmer=self.stemmer, stopwords=STOPWORDS, splitter=self._tokenize
        )
        self.tokenizer_chunk = Tokenizer(
            stemmer=self.stemmer, stopwords=STOPWORDS, splitter=self._tokenize
        )
        self._load_embedding_model()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def _load_embedding_model(self):
        self.embedding_tokenizer = EmbTokenizer.from_file(EMBEDDING_TOKENIZER_FILE)
        self.embedding_tokenizer.enable_padding(length=EMBEDDING_MODEL_TOKEN_LENGTH)

        session_options = ort.SessionOptions()
        # session_options.log_severity_level=1
        model_file = EMBEDDING_MODEL_ONNX_FILE
        if self.device == 'cpu':
            session = ort.InferenceSession(model_file, sess_options=session_options,providers=["CPUExecutionProvider"])
        else:
            ort.preload_dlls()
            onnx_gpu_file = EMBEDDING_MODEL_ONNX_FILE.replace("model.onnx", "model_gpu.onnx")
            if os.path.exists(onnx_gpu_file):
                model_file = onnx_gpu_file
            session = ort.InferenceSession(model_file, sess_options=session_options,providers=["CUDAExecutionProvider","CPUExecutionProvider"])
        self.embedding_model = session
        logger.info(f"Using onnx file {model_file}")

    def _encode(self, texts, batch_size=EMBEDDING_ENCODING_BATCH_SIZE):
        all_outputs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            res = self.embedding_tokenizer.encode_batch_fast(batch_texts)
            input_ids = [e.ids for e in res]
            attention_mask = [e.attention_mask for e in res]
            token_type_ids = [e.type_ids for e in res]
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            outputs = self.embedding_model.run(None, inputs)

            attention_mask = np.array(inputs["attention_mask"])
            mask = attention_mask[..., None]
            embedding = (outputs[0] * mask).sum(axis=1) / mask.sum(axis=1)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            all_outputs.append(embedding)  # Assuming the first output is needed
        if len(all_outputs) >= 1:
            embeddings = np.concatenate(all_outputs, axis=0)
        else:
            embeddings = []
        return embeddings

    @staticmethod
    def _read_file_content(file_path):
        # avoid bloat in unstructured[pdf]. # elements = partition(file_path, strategy='fast')
        logger.info(f"Parsing{file_path}")
        try:
            if str(Path(file_path).suffix) == ".pdf":
                reader = PdfReader(file_path)
                full_text = "\n".join([page.extract_text() for page in reader.pages])
            elif str(Path(file_path).suffix) in NO_CONTENT_PARSING_EXTENSIONS:
                return ""
            else:
                elements = partition(str(Path(file_path)))
                full_text = "\n".join(
                    element.text for element in elements if element.text
                )
            logger.info(f"Successfully parsed file {file_path}")
            return full_text
        except Exception as e:
            logger.info(f"Can't parse file content for {file_path}, {e}")
            return None

    @staticmethod
    def _tokenize(text):
        return rjieba.cut_for_search(text.lower())

    def rebuild_index(self):
        """
        Rebuilds the entire search index from scratch.
        This involves reading files, chunking, generating embeddings,
        and building BM25 and FAISS indexes.
        """
        if not self.embedding_model:
            logger.info("Embedding model not loaded. Cannot rebuild index.")
            return False

        logger.info("Rebuilding index...")
        self.documents = []
        self.old_chunks = self.chunks
        self.chunks = []
        all_chunk_texts = []
        self.faiss_index = None
        self.bm25_chunk_model = None
        self.bm25_doc_model = None
        self.tokenizer_doc = Tokenizer(
            stemmer=self.stemmer, stopwords=STOPWORDS, splitter=self._tokenize
        )
        self.tokenizer_chunk = Tokenizer(
            stemmer=self.stemmer, stopwords=STOPWORDS, splitter=self._tokenize
        )

        files_to_index = set()
        for folder_path in self.folder_paths:
            if os.path.isdir(folder_path):
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if str(Path(file).suffix) in SUPPORTED_EXTENSIONS:
                            files_to_index.add(os.path.join(root, file))
        files_to_index.update(self.file_paths)
        total_files = len(files_to_index)
        for idx, file_path in enumerate(files_to_index):
            self.index_rebuild_progress_signal.emit(
                f"Rebuild index,{idx + 1}/{total_files}, {str(Path(file_path))}"
            )
            if not os.path.exists(file_path):
                logger.info(f"File not found, skipping: {file_path}")
                continue

            content = self._read_file_content(file_path)
            if content is None:  # reading file contents for images would return ""
                continue

            try:
                mtime = os.path.getmtime(file_path)
                size = os.path.getsize(file_path)
            except OSError as e:
                logger.info(f"Could not get metadata for {file_path}: {e}")
                continue

            doc_idx = idx
            self.documents.append(
                {
                    "path": str(Path(file_path)),
                    "content": content,
                    "mtime": mtime,
                    "size": size,
                    # 'emb_needs_update': emb_needs_update
                }
            )

            file_chunks = self.splitter.split_text(content)
            for chunk_text in file_chunks:
                self.chunks.append(
                    {
                        "doc_idx": doc_idx,
                        "text": chunk_text,
                        # 'emb_needs_update': emb_needs_update
                    }
                )
                all_chunk_texts.append(chunk_text)

        if not self.documents:
            logger.info("No documents to index.")
            return True

        # Build BM25 Document Model
        logger.info("Rebuilding index, setting up tokenizer")
        self.index_rebuild_progress_signal.emit(
            f"Rebuilding index...Processing keyword index..."
        )
        corpus_doc = [doc["path"] + doc["content"] for doc in self.documents]
        corpus_doc_tokens = self.tokenizer_doc.tokenize(
            corpus_doc, update_vocab=True, allow_empty=True
        )
        self.bm25_doc_model = bm25s.BM25(method="bm25+")
        self.bm25_doc_model.index(corpus_doc_tokens)
        logger.info(f"BM25 Document Model built with {len(self.documents)} documents.")

        # Build BM25 Chunk Model
        if all_chunk_texts:
            logger.info("Rebuilding index, preparing to build chunk BM25 model")
            corpus_chunk_tokens = self.tokenizer_chunk.tokenize(
                all_chunk_texts, update_vocab=True, allow_empty=True
            )
            self.bm25_chunk_model = bm25s.BM25(method="bm25+")
            self.bm25_chunk_model.index(corpus_chunk_tokens)
            logger.info(f"BM25 Chunk Model built with {len(self.chunks)} chunks.")
            logger.info(f"Generating embeddings for {len(all_chunk_texts)} chunks...")

            # reuse embeddings in self.old_chunks
            chunks_subset_to_update = []
            for chunk in self.chunks:
                match = next(
                    (
                        c
                        for c in self.old_chunks
                        if c["doc_idx"] == chunk["doc_idx"]
                           and c["text"] == chunk["text"]
                    ),
                    None,
                )
                if match and "embedding" in match.keys():
                    chunk["embedding"] = match["embedding"]
                else:
                    chunks_subset_to_update.append(chunk)
            texts_to_embed = [CHUNK_PREFIX + chunk["text"] for chunk in chunks_subset_to_update]
            self.index_rebuild_progress_signal.emit(
                f"Rebuilding index...Calculating embeddings..."
            )
            chunks_subset_embeddings = self._encode(texts_to_embed)
            for idx, emb in enumerate(chunks_subset_embeddings):
                chunks_subset_to_update[idx]["embedding"] = emb

            chunk_embeddings = np.stack(
                [chunk["embedding"] for chunk in self.chunks], axis=0
            )
            d = chunk_embeddings.shape[1]  # Dimension of embeddings
            param = "HNSW64"
            self.faiss_index = faiss.index_factory(d, param, faiss.METRIC_INNER_PRODUCT)
            self.faiss_index.add(chunk_embeddings)
            logger.info(
                f"FAISS index built with {self.faiss_index.ntotal} embeddings."
            )

        else:
            logger.info("No chunks to index for embeddings and BM25 chunk model.")
        logger.info("Index rebuilding complete.")
        self.index_rebuild_progress_signal.emit(f"Index successfully rebuilt.")
        return True

    def search(self, query, mode="hybrid", top_k=SEARCH_ENTRIES_TOPK, deduplicate=False):
        """
        Performs a search based on the specified mode.

        Args:
            deduplicate: Whether to return only a single entry for each document, instead of potentially returning multiple chunks for a single doc
            query (str): The search query.
            mode (str): The search mode ('bm25_chunk', 'bm25_document', 'embedding', 'hybrid').
            top_k (int): The number of top results to retrieve for each sub-search.

        Returns:
            list: A list of dictionaries, each representing a search result.
                  Each dictionary contains file metadata
        """
        assert mode in ["hybrid", "bm25_document", "embedding"]
        results = []
        query_tokens = self.tokenizer_doc.tokenize([query], allow_empty=False)
        highlight_tokens = self.tokenizer_doc.tokenize([query], allow_empty=False, return_as="string")[
            0]  # type: ignore
        if mode == "hybrid":  # bm25_chunk retrieval
            if self.bm25_chunk_model and self.chunks:
                logger.info(f"Performing BM25 chunk search for '{query}'...")
                k = min(top_k, len(self.chunks))
                tmp, scores = self.bm25_chunk_model.retrieve(query_tokens, k=k)
                chunk_scores = scores[0]
                ranked_chunk_indices = np.argsort(chunk_scores)[::-1]

                bm25_chunk_results = []
                for idx in ranked_chunk_indices:
                    if idx < len(self.chunks):  # Ensure index is valid
                        chunk = self.chunks[idx]
                        doc = self.documents[chunk["doc_idx"]]
                        # sometimes the result does not contain any query token.
                        # can't use score to discern
                        if not any(word in chunk["text"] for word in highlight_tokens):
                            continue
                        bm25_chunk_results.append(
                            {
                                "path": doc["path"],
                                "content": chunk["text"],
                                "mtime": doc["mtime"],
                                "size": doc["size"],
                                "score": chunk_scores[idx],
                                "source_mode": "BM25 Chunk",
                            }
                        )
                results.extend(bm25_chunk_results)

        if mode == "bm25_document" or mode == "hybrid":
            if self.bm25_doc_model and self.documents:
                logger.info(f"Performing BM25 document search for '{query}'...")
                # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
                # To return docs instead of IDs, set the `corpus=corpus` parameter.
                k = min(top_k, len(self.documents))
                tmp, scores = self.bm25_doc_model.retrieve(query_tokens, k=k)
                doc_scores = scores[0]
                ranked_doc_indices = np.argsort(doc_scores)[::-1]

                bm25_doc_results = []
                for idx in ranked_doc_indices:
                    if idx < len(self.documents):
                        doc = self.documents[idx]
                        if not any(word in doc["path"]+doc["content"] for word in highlight_tokens):
                            continue
                        bm25_doc_results.append(
                            {
                                "path": doc["path"],
                                "content": doc["content"],
                                "mtime": doc["mtime"],
                                "size": doc["size"],
                                "score": doc_scores[idx],
                                "source_mode": "BM25 Document",
                            }
                        )
                bm25_doc_results = sorted(
                    bm25_doc_results, key=lambda d: d["score"], reverse=True
                )
                results.extend(bm25_doc_results)

        if mode == "hybrid" or mode == "embedding":
            if self.faiss_index and self.embedding_model:
                logger.info(f"Performing embedding similarity search for '{query}'...")
                _tmp_query = QUERY_PREFIX + query  # make multilingual-e5 model happy
                query_embedding = self._encode([_tmp_query])
                D, I = self.faiss_index.search(
                    query_embedding, top_k
                )  # D: distances, I: indices

                embedding_results = []
                for i, score in zip(I[0], D[0]):
                    if i != -1 and i < len(self.chunks):  # Check for valid index
                        chunk = self.chunks[i]
                        doc = self.documents[chunk["doc_idx"]]
                        embedding_results.append(
                            {
                                "path": doc["path"],
                                "content": chunk["text"],
                                "mtime": doc["mtime"],
                                "size": doc["size"],
                                "score": score,  # D is not always the similarity, depending on the metric used
                                "source_mode": "Embedding",
                            }
                        )
                embedding_results = sorted(
                    embedding_results, key=lambda d: d["score"], reverse=True
                )
                results.extend(embedding_results)

        # For hybrid mode, combine, and use reciprocal rank fusion
        if mode == "hybrid":
            for i, item in enumerate(bm25_doc_results):
                item["rank"] = i + 1
            for i, item in enumerate(embedding_results):
                item["rank"] = i + 1
            for i, item in enumerate(bm25_chunk_results):
                item["rank"] = i + 1
            k = 60  # parameter for rrf
            for res in results:
                r_b_chunk = (
                    res["rank"]
                    if res["source_mode"] == "BM25 Chunk"
                    else len(bm25_chunk_results) + 1
                )
                r_b_doc = (
                    res["rank"]
                    if res["source_mode"] == "BM25 Document"
                    else len(bm25_doc_results) + 1
                )
                r_e = (
                    res["rank"]
                    if res["source_mode"] == "Embedding"
                    else len(embedding_results) + 1
                )
                rrf_score = 1 / (k + r_b_chunk) + 1 / (k + r_b_doc) + 1 / (k + r_e)
                res["rrf_score"] = rrf_score
                # count how many highlight_tokens appeared in the content
                res["highlight_tokens_appearance"] = sum(
                    1 for ht in highlight_tokens if ht in res["path"]+res["content"]
                )
            results = sorted(results, key=lambda x: x["rrf_score"], reverse=True)

            # stage 2 ranking. prioritize results containing more hits for query
            best_hits = []
            good_hits = []
            n_highlight_tokens = len(highlight_tokens)
            for i, res in enumerate(results[:]):  # iterate over a copy
                if query in res["path"]+res["content"]:
                    best_hits.append(res)
                    results.remove(res)
                elif (
                        n_highlight_tokens > 2
                        and res["highlight_tokens_appearance"] > 0.5 * n_highlight_tokens
                ):
                    good_hits.append(res)
                    results.remove(res)
            tmp_results = best_hits + good_hits + results
            # internal deduplication based on both content and file path (composite key)
            # (not the deduplication specified by user)
            final_results = []
            seen_pairs = set()
            for res in tmp_results:
                if res["source_mode"] in ["BM25 Chunk", "Embedding"]:
                    # Create a composite key from content and path
                    pair = (res["content"], res["path"])
                    if pair not in seen_pairs:
                        final_results.append(res)
                        seen_pairs.add(pair)
                else:
                    final_results.append(res) # BM25 Doc
            results = final_results
        results = results[:top_k]
        # deduplication
        if deduplicate:
            docs_set = set()
            deduped_results = []
            for res in results:
                if res["path"] not in docs_set:
                    deduped_results.append(res)
                    docs_set.add(res["path"])
            results = deduped_results
        return results, highlight_tokens

    def get_current_mtimes(self):
        """
        Gets the current modification times for all indexed files.
        """
        current_mtimes = {}
        files_to_check = set()

        for folder_path in self.folder_paths:
            if os.path.isdir(folder_path):
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if str(Path(file).suffix) in SUPPORTED_EXTENSIONS:
                            files_to_check.add(os.path.join(root, file))
        files_to_check.update(self.file_paths)

        for file_path in files_to_check:
            try:
                if os.path.exists(file_path):
                    current_mtimes[file_path] = os.path.getmtime(file_path)
            except OSError as e:
                logger.info(f"Error getting mtime for {file_path}: {e}")
        return current_mtimes

    def get_saved_mtimes(self):
        if os.path.exists(os.path.join(INDEX_DIR, "files_mtimes.pkl")):
            with open(os.path.join(INDEX_DIR, "files_mtimes.pkl"), "rb") as f:
                files_mtimes = pickle.load(f)
        else:
            files_mtimes = None
        return files_mtimes

    def save_index_components(self, index_dir=INDEX_DIR):
        os.makedirs(index_dir, exist_ok=True)
        docs_chunks_data = {
            "documents": self.documents,
            "chunks": self.chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        # skip saving if the data have not changed
        old_docs_chunks_data = {
            "documents": self.documents,
            "chunks": self.old_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        if self.faiss_index is not None:
            faiss.write_index(
                self.faiss_index, os.path.join(index_dir, "faiss_index.bin")
            )

        self.tokenizer_doc.save_vocab(save_dir=str(Path(index_dir) / "tokenizer_doc"))
        self.tokenizer_chunk.save_vocab(
            save_dir=str(Path(index_dir) / "tokenizer_chunk")
        )
        if docs_chunks_data != old_docs_chunks_data:
            with open(os.path.join(index_dir, "docs_chunks_data.pkl"), "wb") as f:
                pickle.dump(docs_chunks_data, f)
        if self.bm25_doc_model and self.bm25_chunk_model:
            self.bm25_doc_model.save(str(Path(index_dir) / "bm25_doc_model"))
            self.bm25_chunk_model.save(str(Path(index_dir) / "bm25_chunk_model"))
        logger.info(f"Index components saved to {index_dir}")

        # save modification times
        files_mtimes = self.get_current_mtimes()
        with open(os.path.join(index_dir, "files_mtimes.pkl"), "wb") as f:
            pickle.dump(files_mtimes, f)
        return True

    def load_index_components(self, index_dir=INDEX_DIR):
        """Loads the index components from disk."""
        if not os.path.exists(index_dir):
            logger.info(f"Index directory not found: {index_dir}")
            return False

        try:
            with open(os.path.join(index_dir, "docs_chunks_data.pkl"), "rb") as f:
                docs_chunks_data = pickle.load(f)
            self.tokenizer_doc.load_vocab(str(Path(index_dir) / "tokenizer_doc"))
            self.tokenizer_chunk.load_vocab(str(Path(index_dir) / "tokenizer_chunk"))

            self.bm25_doc_model = bm25s.BM25.load(
                str(Path(index_dir) / "bm25_doc_model")
            )
            self.bm25_chunk_model = bm25s.BM25.load(
                str(Path(index_dir) / "bm25_chunk_model")
            )

            self.documents = docs_chunks_data.get("documents", [])
            self.chunks = docs_chunks_data.get("chunks", [])
            self.chunk_size = docs_chunks_data.get("chunk_size", 200)
            self.chunk_overlap = docs_chunks_data.get("chunk_overlap", 30)

            faiss_index_path = os.path.join(index_dir, "faiss_index.bin")
            if os.path.exists(faiss_index_path):
                self.faiss_index = faiss.read_index(faiss_index_path)
            else:
                self.faiss_index = None

            logger.info(f"Index components loaded from {index_dir}")
            return True
        except Exception as e:
            logger.info(f"Error loading index components from {index_dir}: {e}")
            # Reset all index related attributes on failure
            self.documents = []
            self.chunks = []
            self.faiss_index = None
            self.bm25_chunk_model = None
            self.bm25_doc_model = None
            return False
