"""Unstructured VLM description baseline with the same strict provenance boundary."""

from dataclasses import asdict
import hashlib
from time import perf_counter

from visionevidence.retrieval.bm25 import TOKENIZER
from visionevidence.vlm.prompts import get_prompt

PROMPT_VERSION = "concise_visual_description_v1"
PROMPT = get_prompt() + "\nUse one concise English prose paragraph of at most 60 words. No headings, Markdown, or JSON."


def describe_and_retrieve(request, vlm, index, *, k=10):
    if request.request_id is None or request.prompt != PROMPT:
        raise ValueError("baseline request requires an ID and the fixed description prompt")
    request.validate_image()
    image_hash = hashlib.sha256(request.image_path.read_bytes()).hexdigest()
    started = perf_counter()
    response = vlm.generate(request)
    elapsed = (perf_counter() - started) * 1000
    if response.request_id != request.request_id:
        raise ValueError("VLM response request_id mismatch")
    if response.metadata.get("inference_performed") is not (not response.is_mock):
        raise ValueError("VLM inference_performed contradicts is_mock")
    if not response.is_mock:
        tokens, limit = response.metadata.get("generated_tokens"), response.metadata.get("max_new_tokens")
        if any(type(v) is not int or v <= 0 for v in (tokens, limit)):
            raise ValueError("real response requires positive token counts")
        if tokens >= limit:
            raise ValueError("token limit reached; baseline description rejected")
    if not response.description.strip() or len(response.description) > 8000:
        raise ValueError("baseline requires nonempty description of at most 8000 characters")
    # No reference captions, keywords, or model-based rewriting are added here.
    query = response.description
    started = perf_counter()
    evidence = index.search(query, k=k)
    return {"schema_version": 1, "request_id": request.request_id, "status": "completed",
            "verified": False, "is_mock": response.is_mock, "inference_performed": not response.is_mock,
            "query": query, "query_sources": [{"field": "vlm_response.description"}],
            "evidence": [asdict(item) for item in evidence], "vlm_response": response.to_dict(),
            "trace": {"image_sha256": image_hash, "prompt_version": PROMPT_VERSION,
                      "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                      "k": k, "bm25": {"k1": index.k1, "b": index.b, "tokenizer": TOKENIZER},
                      "vlm_wall_ms": elapsed, "retrieval_ms": (perf_counter() - started) * 1000}}
