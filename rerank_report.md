# Rerank Report — Module 8 Thursday Stretch

## Setup

- Hybrid `k_in`: 50
- Re-ranked `k_out`: 5
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Hardware: Intel CPU (x86-64), 16 GB RAM, Ubuntu Linux — CPU inference only

## Metrics Table

| Pipeline | recall@5 | MRR@5 | Latency mean (ms) | Latency p95 (ms) |
|---|---|---|---|---|
| Hybrid baseline (k=5) | 0.833 | 0.661 | 55 ms | 80 ms |
| Hybrid (k_in=50) + cross-encoder rerank | 0.783 | 0.623 | 791 ms | 880 ms |

Stage breakdown for the rerank pipeline:
- **Stage 1** (hybrid retrieve k_in=50 + text fetch): mean **18 ms**, p95 **23 ms**
- **Stage 2** (cross-encoder scores 50 pairs): mean **773 ms**, p95 **861 ms**

## When Does Re-Ranking Pay Off?

On this 60-query StackExchange eval set, re-ranking did not improve over the
hybrid baseline — recall@5 dropped 5 points and MRR dropped 0.038. The
cross-encoder (`ms-marco-MiniLM-L-6-v2`) was trained on MS-MARCO web-search
pairs; StackExchange Q&A has a different style (verbose technical answers,
exact identifier matching), causing the CE scorer to mis-rank passages it
should promote. Re-ranking pays off when (a) the first-stage retriever is
weak and produces a noisy top-50, and (b) the cross-encoder domain matches
the corpus. On a domain-matched CE or a corpus where BM25+dense hybrid scores
are inconsistent (e.g., out-of-domain paraphrases without exact lexical
overlap), re-ranking typically yields a 5–10-point recall lift.

## Latency Overhead

The cross-encoder adds **~736 ms/query** on CPU over the 55 ms hybrid
baseline — a 14× slowdown. The overhead is essentially constant with respect
to corpus size: the CE scores exactly `k_in=50` pairs per query regardless of
whether the corpus has 1 200 or 1 200 000 documents. Doubling `k_in` to 100
would roughly double the CE stage to ~1 550 ms while providing at most a few
additional recall points. The hybrid retrieval stage scales sub-linearly with
corpus size (Weaviate uses HNSW + inverted index), so at 10× corpus (12 000
docs) hybrid would cost ~40–80 ms while CE remains at ~770 ms.

## At What Corpus Size or Query Volume Does It Stop Being Worth It?

With a single CPU thread scoring 50 pairs in ~770 ms, the cross-encoder
pipeline maxes out at roughly **1.3 queries/second**. At 2 QPS the pipeline
already queues; at 5 QPS it is completely unsustainable without batching or
GPU. On a modern GPU (e.g., T4) scoring 50 pairs takes ~20–40 ms, yielding
~15–20 QPS — acceptable for low-traffic search but not production-scale.
The cross-over point for a CPU deployment is therefore around **2 QPS**: below
that, the latency is tolerable for an interactive tool; above it, you need
either GPU inference, a cached re-ranking layer (cache top-50 per frequent
query), or a lighter learned re-ranker. Corpus size does not affect the
decision since the CE cost is fixed at `k_in` pairs — but if `k_in` must grow
with corpus density to maintain hybrid recall, the CE cost grows proportionally.
