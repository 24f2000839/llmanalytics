import time
import hashlib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta

app = FastAPI()

# ======================================
# CONFIG
# ======================================
MODEL_COST_PER_1M = 1.00
AVG_TOKENS_PER_REQUEST = 3000
TTL_HOURS = 24
MAX_CACHE_SIZE = 1500
SIMILARITY_THRESHOLD = 0.95

# ======================================
# ANALYTICS METRICS
# ======================================
total_requests = 0
cache_hits = 0
cache_misses = 0

total_tokens = 0
cached_tokens = 0

# ======================================
# MODELS
# ======================================
class QueryRequest(BaseModel):
    query: str
    application: str


class QueryResponse(BaseModel):
    answer: str
    cached: bool
    latency: int
    cacheKey: str


class AnalyticsResponse(BaseModel):
    hitRate: float
    totalRequests: int
    cacheHits: int
    cacheMisses: int
    cacheSize: int
    costSavings: float
    savingsPercent: float
    strategies: list


# ======================================
# CACHE ENTRY
# ======================================
class CacheEntry:
    def __init__(self, query, answer, embedding):
        self.query = query
        self.answer = answer
        self.embedding = embedding
        self.timestamp = datetime.utcnow()


cache = OrderedDict()

# ======================================
# UTILITIES
# ======================================
def get_md5(text):
    return hashlib.md5(text.encode()).hexdigest()


def generate_embedding(text):
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(384)


def is_expired(entry):
    return datetime.utcnow() - entry.timestamp > timedelta(hours=TTL_HOURS)


def evict_if_needed():
    while len(cache) > MAX_CACHE_SIZE:
        cache.popitem(last=False)  # LRU eviction


def semantic_search(query_embedding):
    if not cache:
        return None

    keys = list(cache.keys())
    embeddings = np.array([cache[k].embedding for k in keys])
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    best_index = np.argmax(similarities)
    best_score = similarities[best_index]

    if best_score > SIMILARITY_THRESHOLD:
        return keys[best_index]

    return None


def call_llm(query):
    time.sleep(1.5)  # simulate API latency
    return f"Summary of: {query}"


# ======================================
# HEALTH
# ======================================
@app.get("/")
def health():
    return {"status": "AI caching system running"}


# ======================================
# MAIN QUERY ENDPOINT
# ======================================
@app.post("/", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    global total_requests, cache_hits, cache_misses
    global total_tokens, cached_tokens

    start_time = time.time()
    total_requests += 1
    total_tokens += AVG_TOKENS_PER_REQUEST

    query = request.query
    md5_key = get_md5(query)

    # -------------------------
    # 1️⃣ Exact Match
    # -------------------------
    if md5_key in cache:
        entry = cache[md5_key]
        if not is_expired(entry):
            cache_hits += 1
            cached_tokens += AVG_TOKENS_PER_REQUEST
            cache.move_to_end(md5_key)

            latency = int((time.time() - start_time) * 1000)
            return QueryResponse(
                answer=entry.answer,
                cached=True,
                latency=latency,
                cacheKey=md5_key
            )
        else:
            del cache[md5_key]

    # -------------------------
    # 2️⃣ Semantic Caching
    # -------------------------
    query_embedding = generate_embedding(query)
    semantic_key = semantic_search(query_embedding)

    if semantic_key:
        entry = cache[semantic_key]
        if not is_expired(entry):
            cache_hits += 1
            cached_tokens += AVG_TOKENS_PER_REQUEST
            cache.move_to_end(semantic_key)

            latency = int((time.time() - start_time) * 1000)
            return QueryResponse(
                answer=entry.answer,
                cached=True,
                latency=latency,
                cacheKey=semantic_key
            )

    # -------------------------
    # 3️⃣ Cache Miss
    # -------------------------
    cache_misses += 1

    answer = call_llm(query)

    new_entry = CacheEntry(query, answer, query_embedding)
    cache[md5_key] = new_entry
    cache.move_to_end(md5_key)
    evict_if_needed()

    latency = int((time.time() - start_time) * 1000)

    return QueryResponse(
        answer=answer,
        cached=False,
        latency=latency,
        cacheKey=md5_key
    )


# ======================================
# ANALYTICS ENDPOINT
# ======================================
@app.get("/analytics", response_model=AnalyticsResponse)
def analytics():

    hit_rate = cache_hits / total_requests if total_requests else 0

    # Correct cost calculation formula:
    # savings = (total_tokens - cached_tokens) * model_cost / 1M

    actual_tokens_used = total_tokens - cached_tokens
    baseline_cost = (total_tokens * MODEL_COST_PER_1M) / 1_000_000
    actual_cost = (actual_tokens_used * MODEL_COST_PER_1M) / 1_000_000
    savings = baseline_cost - actual_cost

    savings_percent = (savings / baseline_cost * 100) if baseline_cost else 0

    return AnalyticsResponse(
        hitRate=round(hit_rate, 2),
        totalRequests=total_requests,
        cacheHits=cache_hits,
        cacheMisses=cache_misses,
        cacheSize=len(cache),
        costSavings=round(savings, 2),
        savingsPercent=round(savings_percent, 2),
        strategies=[
            "exact match caching",
            "semantic caching (embedding similarity)",
            "LRU eviction policy",
            "TTL expiration"
        ]
    )
