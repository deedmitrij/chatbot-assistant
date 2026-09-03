import math

from tests.vector_db.conftest import get_all_test_cases_from_file

# K=3 matches VectorDBService.search()'s default n_results, which is also
# what KnowledgeManager.get_relevant_context() hands to the LLM in
# production (backend/managers/knowledge_manager.py).
K = 3


def hit_rate_at_k(results, k):
    """
    What it measures: did we find AT LEAST ONE relevant document somewhere
    in the top-K results? A simple yes/no per query, averaged over all
    queries.

    How: for each query, check if any of the top-K retrieved ids is in that
    query's relevant set (1 if yes, 0 if no), then average that over all
    queries.

    Example: query A hits (relevant doc at rank 2, within top-3), query B
    misses (its relevant doc is rank 5) -> HitRate@3 = (1 + 0) / 2 = 0.5

    Useful for: a quick, easy-to-explain "does search basically work?"
    check. It doesn't care about exact rank or about multiple relevant
    documents, which is exactly why Recall@K and MRR exist alongside it.
    """
    if not results:
        return 0.0
    hits = sum(
        1
        for entry in results
        if set(entry["retrieved_ids"][:k]) & set(entry["relevant_ids"])
    )
    return hits / len(results)


def recall_at_k(results, k):
    """
    What it measures: of ALL the documents that would correctly answer a
    query, what fraction did we actually retrieve in the top-K? Only
    differs from HitRate@K when a query has more than one relevant document.

    How: for each query, divide (relevant docs found in top-K) by (total
    relevant docs for that query), then average over all queries.

    Example: a query has 2 relevant docs and only 1 shows up in the top-K
    -> that query's recall is 1/2 = 0.5, averaged with the other queries.

    Useful for: catching "partial" retrieval failures that HitRate@K
    hides — e.g. finding 1 of 2 correct answers still counts as a HitRate@K
    success, but Recall@K correctly shows it as only half right.
    """
    if not results:
        return 0.0
    per_query = [
        len(set(entry["retrieved_ids"][:k]) & set(entry["relevant_ids"])) / len(entry["relevant_ids"])
        for entry in results
    ]
    return sum(per_query) / len(per_query)


def precision_at_k(results, k):
    """
    What it measures: of the K documents we actually retrieved, what
    fraction are actually relevant? The mirror image of Recall@K — Recall
    asks "did we find everything relevant?", Precision asks "is what we
    found mostly relevant, or mostly noise?".

    How: for each query, divide (relevant docs found in top-K) by K itself,
    then average over all queries.

    Example: top-3 results contain 1 relevant document and 2 irrelevant
    ones -> that query's precision is 1/3, averaged with the other queries.

    Useful for: judging how much irrelevant "noise" gets handed to the LLM
    alongside the right answer. Note that on a small corpus with only 1-2
    truly relevant documents, Precision@3 is structurally capped well below
    1.0 even for perfect retrieval — that's expected, not a bug.
    """
    if not results:
        return 0.0
    per_query = [
        len(set(entry["retrieved_ids"][:k]) & set(entry["relevant_ids"])) / k
        for entry in results
    ]
    return sum(per_query) / len(per_query)


def mrr(results):
    """
    What it measures: how early does the FIRST relevant document show up in
    the ranking? Unlike the @K metrics above, it looks at the whole ranked
    list, not just a fixed cutoff.

    How: for each query, take 1 / (rank of the first relevant document
    found), or 0 if none is found at all, then average over all queries.

    Example: query A's first relevant doc is rank 1 (1/1 = 1.0), query B's
    is rank 4 (1/4 = 0.25) -> MRR = (1.0 + 0.25) / 2 = 0.625

    Useful for: measuring ranking quality when only the single best answer
    matters (e.g. what the user sees first), independent of any fixed K.
    """
    if not results:
        return 0.0
    reciprocal_ranks = []
    for entry in results:
        relevant = set(entry["relevant_ids"])
        rr = 0.0
        for rank, doc_id in enumerate(entry["retrieved_ids"], start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_at_k(results, k):
    """
    What it measures: not just WHETHER relevant documents were retrieved,
    but whether the MOST relevant ones were ranked highest. Needs graded
    relevance (e.g. 3 = highly relevant, 1 = partially relevant, 0 = not
    relevant) rather than a plain relevant/not-relevant split — this is
    what separates it from HitRate/Recall/Precision above.

    How: each retrieved document's relevance grade is discounted by its
    rank (a hit at rank 1 counts more than the same hit at rank 3), summed
    into a "DCG" score; that's then divided by the best possible DCG for
    that query (putting every relevant document in its ideal order) to get
    a 0-1 score. Averaged over all queries.

    Example: a query has one grade-3 and one grade-1 relevant document.
    If retrieval returns the grade-3 doc first and the grade-1 doc second,
    that's the ideal order -> NDCG = 1.0. If it returns the grade-1 doc
    first and buries the grade-3 doc, DCG drops even though both documents
    were technically "found" -> NDCG < 1.0.

    Useful for: catching a ranking problem that HitRate/Recall/Precision
    all miss entirely — they only ask "was it retrieved?", not "was the
    best answer put first?".
    """
    if not results:
        return 0.0

    per_query_ndcg = []
    for entry in results:
        grades = entry["relevance_grades"]

        dcg = sum(
            grades.get(doc_id, 0) / math.log2(rank + 1)
            for rank, doc_id in enumerate(entry["retrieved_ids"][:k], start=1)
        )

        ideal_order = sorted(grades.values(), reverse=True)[:k]
        idcg = sum(
            grade / math.log2(rank + 1)
            for rank, grade in enumerate(ideal_order, start=1)
        )

        per_query_ndcg.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(per_query_ndcg) / len(per_query_ndcg)


def test_vector_db_retrieval_metrics(vector_db_service):
    """
    Aggregate retrieval quality over a controlled multi-query dataset
    (test_retrieval_metrics.json), distinct from the other vector_db tests
    which each check one query's behavior in isolation.

    Computes HitRate@1, HitRate@K, Recall@K, Precision@K, MRR, and NDCG@K
    across every case in one pass. Two cases have more than one relevant
    document (making Recall@K/Precision@K meaningfully different from
    HitRate@K), and one case has graded relevance — a partially-relevant
    document alongside two highly-relevant ones — making NDCG@K
    meaningfully different from the binary metrics.

    Regression floors were set below a measured baseline (HitRate@1=1.000,
    HitRate@3=1.000, Recall@3=0.929, Precision@3=0.381, MRR=1.000,
    NDCG@3=0.937), each with margin for one query's result getting modestly
    worse without failing the test on normal noise, while still catching a
    real regression.
    """
    suites = get_all_test_cases_from_file("test_retrieval_metrics.json")
    dataset, cases = suites[0][0], [case for _, case in suites]

    existing_ids = vector_db_service.collection.get()["ids"]
    vector_db_service.delete_by_ids(existing_ids)
    vector_db_service.upsert_batch(
        documents=dataset["documents"],
        ids=dataset["ids"],
        metadatas=dataset["metadatas"]
    )

    results = []
    for case in cases:
        search_result = vector_db_service.search(
            query_text=case["query"],
            n_results=len(dataset["ids"])
        )
        results.append({
            "retrieved_ids": search_result["ids"][0],
            "relevant_ids": case["expected_ids"],
            "relevance_grades": case["relevance_grades"]
        })

    hit_rate_1 = hit_rate_at_k(results, k=1)
    hit_rate_k = hit_rate_at_k(results, k=K)
    recall_k = recall_at_k(results, k=K)
    precision_k = precision_at_k(results, k=K)
    mrr_score = mrr(results)
    ndcg_k = ndcg_at_k(results, k=K)

    print(f"\nRetrieval metrics over {len(results)} queries:")
    print(f"  HitRate@1   = {hit_rate_1:.3f}")
    print(f"  HitRate@{K}   = {hit_rate_k:.3f}")
    print(f"  Recall@{K}    = {recall_k:.3f}")
    print(f"  Precision@{K} = {precision_k:.3f}")
    print(f"  MRR         = {mrr_score:.3f}")
    print(f"  NDCG@{K}      = {ndcg_k:.3f}")

    assert hit_rate_1 >= 0.85, f"HitRate@1 regressed: {hit_rate_1:.3f}"
    assert hit_rate_k >= 0.85, f"HitRate@{K} regressed: {hit_rate_k:.3f}"
    assert recall_k >= 0.75, f"Recall@{K} regressed: {recall_k:.3f}"
    assert precision_k >= 0.30, f"Precision@{K} regressed: {precision_k:.3f}"
    assert mrr_score >= 0.85, f"MRR regressed: {mrr_score:.3f}"
    assert ndcg_k >= 0.80, f"NDCG@{K} regressed: {ndcg_k:.3f}"
