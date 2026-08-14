import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = 'vectorstore/db_faiss'

_VECTORSTORE_CACHE = None
_BM25_RETRIEVER_CACHE = None

# Medical Synonym Mapping for Query Understanding & Expansion
MEDICAL_SYNONYM_EXPANSION = {
    "heart attack": "myocardial infarction",
    "stroke": "cerebrovascular accident stroke",
    "shortness of breath": "dyspnea shortness of breath",
    "breathlessness": "dyspnea",
    "dizziness": "vertigo dizziness",
    "stomach pain": "abdominal pain gastritis peptic ulcer",
    "high blood pressure": "hypertension blood pressure",
    "fever": "pyrexia fever",
    "headache": "cephalalgia headache migraine",
    "kidney failure": "renal failure kidney disease",
    "blood clot": "thrombosis pulmonary embolism"
}


import re

def rewrite_query_with_context(query: str, context: Optional[str] = None) -> str:
    """
    Rewrites ambiguous multi-turn follow-up queries into self-contained standalone search queries.
    Example:
      Query: "What about treatment?" + Context: "diabetes mellitus"
      Standalone Output: "What about treatment? for diabetes mellitus"
    """
    if not query or not str(query).strip():
        return ""
        
    q_str = str(query).strip()
    q_lower = q_str.lower()
    
    if not context or not str(context).strip():
        return q_str
        
    ctx_str = str(context).strip()
    
    # Check if query contains pronouns or is a short follow-up
    has_pronoun_or_short = False
    if len(q_lower.split()) <= 4:
        has_pronoun_or_short = True
    else:
        for p in ["it", "its", "this", "that", "these", "those", "they", "them", "what about", "how about"]:
            if re.search(r'\b' + re.escape(p) + r'\b', q_lower):
                has_pronoun_or_short = True
                break
                
    if not has_pronoun_or_short:
        return q_str
        
    # Extract clinical subject entity from context
    ctx_words = [w for w in re.findall(r'\b[A-Za-z]{3,}\b', ctx_str) if w.lower() not in {"what", "are", "the", "is", "for", "and", "in", "of", "to", "how", "tell", "me", "about", "document", "topic", "section", "source", "page"}]
    if ctx_words:
        subject = " ".join(ctx_words[:3])
        if subject.lower() not in q_lower:
            return f"{q_str} for {subject}"
            
    return q_str


def expand_medical_query(query: str) -> str:
    """
    Expands query terms with medical synonyms to bridge layperson and clinical terminology.
    """
    q_lower = str(query).lower().strip()
    expansions = []
    for lay_term, med_term in MEDICAL_SYNONYM_EXPANSION.items():
        if lay_term in q_lower:
            expansions.append(med_term)
    if expansions:
        return f"{query} ({' '.join(expansions)})"
    return query


def load_hybrid_retrievers():
    """
    Loads and caches FAISS VectorStore and BM25Retriever in memory with error boundaries.
    """
    global _VECTORSTORE_CACHE, _BM25_RETRIEVER_CACHE
    if _VECTORSTORE_CACHE is not None and _BM25_RETRIEVER_CACHE is not None:
        return _VECTORSTORE_CACHE, _BM25_RETRIEVER_CACHE
        
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(
            f"FAISS index folder not found at '{DB_FAISS_PATH}'. Please run `python create_memory_for_llm.py` to index the medical PDFs."
        )
        
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        _VECTORSTORE_CACHE = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        
        # Build BM25Retriever over all document chunks stored in FAISS docstore
        all_docs = list(_VECTORSTORE_CACHE.docstore._dict.values())
        if all_docs:
            _BM25_RETRIEVER_CACHE = BM25Retriever.from_documents(all_docs)
            _BM25_RETRIEVER_CACHE.k = 10
        else:
            _BM25_RETRIEVER_CACHE = None
            
        return _VECTORSTORE_CACHE, _BM25_RETRIEVER_CACHE
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS vector database or BM25 index from '{DB_FAISS_PATH}': {str(e)}")


def load_vectorstore():
    db, _ = load_hybrid_retrievers()
    return db


def reciprocal_rank_fusion(vector_docs: List[Any], bm25_docs: List[Any], c: int = 60, top_k: int = 4) -> List[Any]:
    """
    Combines dense vector search and sparse BM25 search using Reciprocal Rank Fusion (RRF).
    RRF Score = 1 / (60 + rank_vector) + 1 / (60 + rank_bm25)
    """
    doc_scores: Dict[str, float] = {}
    doc_map: Dict[str, Any] = {}
    
    for rank, doc in enumerate(vector_docs, 1):
        doc_id = getattr(doc, 'page_content', str(doc)).strip()
        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (c + rank))
        doc_map[doc_id] = doc
        
    for rank, doc in enumerate(bm25_docs, 1):
        doc_id = getattr(doc, 'page_content', str(doc)).strip()
        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + (1.0 / (c + rank))
        doc_map[doc_id] = doc
        
    sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids[:top_k]]


def search_medical_documents(query: str, topic_filter: Optional[str] = None, k: int = 4, conversation_context: Optional[str] = None) -> List[Any]:
    """
    Performs Hybrid RAG Retrieval (Query Rewriting + Dense FAISS MMR + Sparse BM25 + Medical Synonym Expansion + RRF Reranking).
    """
    if not query or not str(query).strip():
        return []
        
    try:
        db, bm25_retriever = load_hybrid_retrievers()
        
        # Stage 0: Standalone Query Rewriting for Multi-Turn Follow-ups
        standalone_query = rewrite_query_with_context(query, conversation_context)
        
        # Stage 1: Query Understanding & Medical Synonym Expansion
        expanded_query = expand_medical_query(standalone_query)
        
        # Stage 2: Parallel Hybrid Retrieval (Vector + BM25)
        vector_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
        )
        vector_docs = vector_retriever.invoke(expanded_query)
        
        bm25_docs = []
        if bm25_retriever is not None:
            try:
                bm25_retriever.k = 10
                bm25_docs = bm25_retriever.invoke(expanded_query)
            except Exception:
                bm25_docs = []
                
        # Stage 3: Reciprocal Rank Fusion (RRF) Reranker
        if bm25_docs:
            reranked_docs = reciprocal_rank_fusion(vector_docs, bm25_docs, c=60, top_k=max(k, 8))
        else:
            reranked_docs = vector_docs
            
        # Stage 4: Topic / Source Filtering
        if topic_filter:
            tf_lower = str(topic_filter).lower().strip()
            filtered = [
                d for d in reranked_docs 
                if tf_lower in d.metadata.get("topic", "").lower() or tf_lower in d.metadata.get("source", "").lower()
            ]
            if filtered:
                return filtered[:k]
                
        return reranked_docs[:k]
    except Exception as e:
        print(f"Warning: Document retrieval encountered an error: {str(e)}")
        return []


@tool
def medical_knowledge_search(query: str, topic_filter: Optional[str] = None) -> str:
    """
    Searches indexed medical reference books and clinical handbooks for verified medical information using Hybrid Search (Vector + BM25 + RRF).
    Handles missing index, empty retrieval, and invalid query inputs cleanly.
    
    Args:
        query: Medical topic, condition, differential diagnosis, or symptom to search for.
        topic_filter: Optional keyword to filter search results by document name, topic, or section.
    """
    if not query or not isinstance(query, str) or not query.strip():
        return "Error: Medical search query cannot be empty or invalid."
        
    try:
        docs = search_medical_documents(query=query, topic_filter=topic_filter, k=4)
        if not docs:
            return f"No relevant medical reference information was found for query: '{query}'."
            
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Medical Reference')
            page = doc.metadata.get('page', doc.metadata.get('page_label', 'N/A'))
            topic = doc.metadata.get('topic', 'Clinical Reference')
            section = doc.metadata.get('section', 'General Section')
            
            content = getattr(doc, 'page_content', str(doc)).strip()
            results.append(
                f"[Document {i} | Topic: {topic} | Section: {section} | Source: {source} | Page: {page}]\n{content}"
            )
            
        return "\n\n".join(results)
    except Exception as e:
        return f"Error executing medical search: {str(e)}"
