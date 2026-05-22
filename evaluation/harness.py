from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from google import genai
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from memory import store
from agent.research_agent import DeepResearchAgent
from agent.research_agent import ContextSnippet
from google.genai import types
load_dotenv()

IST = ZoneInfo("Asia/Kolkata")
DB_PATH = Path("session.db")


DATASET = [
    {
        "id": "factual",
        "type": "factual",
        "query": "What is the current population of Tokyo?",
        "expected_keywords": [
            "Tokyo metropolitan population",
            "Japan capital population",
            "million residents",
            "urban population estimate",
            "latest demographic estimate",
        ],
        "follow_up": None,
    },

    {
        "id": "multihop",
        "type": "multi_hop",
        "query": "Who founded the company that makes the M1 chip and in what year?",
        "expected_keywords": [
            "Apple founders",
            "Steve Jobs",
            "Steve Wozniak",
            "Apple founded in 1976",
            "M1 chip manufacturer",
        ],
        "follow_up": "What was that company's revenue in their most recent fiscal year?",
    },

    {
        "id": "insufficient",
        "type": "insufficient_evidence",
        "query": "What will the stock price of Apple be on December 31 2030?",
        "expected_keywords": [
            "future stock prices cannot be predicted reliably",
            "financial markets are uncertain",
            "speculative forecast",
            "no definitive answer possible",
            "depends on future market conditions",
        ],
        "follow_up": None,
    },

    {
        "id": "conflict",
        "type": "conflicting_sources",
        "query": "Is coffee beneficial or harmful to health?",
        "expected_keywords": [
            "coffee has both benefits and risks",
            "mixed scientific evidence",
            "health outcomes vary",
            "moderate consumption",
            "studies show conflicting findings",
            "depends on individual health factors",
        ],
        "follow_up": None,
    },

    {
        "id": "multiturn",
        "type": "multi_turn",
        "query": "What are the main causes of inflation?",
        "expected_keywords": [
            "demand-pull inflation",
            "cost-push inflation",
            "monetary policy",
            "supply chain disruptions",
            "rising prices",
            "inflation drivers",
        ],
        "follow_up": "Which of those causes was most responsible for the 2022 US inflation spike?",
    }
]


@dataclass
class ReliabilityJudgeResult:
    reliability: float
    label: str
    reasoning: str
    strengths: list[str]
    weaknesses: list[str]


@dataclass
class TurnResult:
    item_id: str
    query_type: str
    query: str
    answer: str

    citation_count: int
    citation_coverage: float
    keyword_recall: float
    uncertainty_acknowledged: bool
    source_diversity: int
    answer_length: int
    latency_seconds: float
    snippets_used: int


    llm_reliability: float
    llm_reliability_label: str
    llm_reliability_reasoning: str

    input_tokens: int
    output_tokens: int
    total_tokens: int
    citations_per_1k_output_tokens: float
    context_utilization: float
    retrieval_efficiency: float


    followup_query: str = ""
    followup_answer: str = ""
    followup_citation_count: int = 0
    followup_references_score:  float = 0.0
    followup_latency_seconds: float = 0.0




def count_citations(answer: str) -> int:
    return len(re.findall(r"\[.+?—.+?\]\(https?://\S+?\)", answer))


def citation_coverage(answer: str) -> float:
    paragraphs = [p.strip() for p in answer.split("\n\n") if len(p.strip()) > 60]
    if not paragraphs:
        return 0.0
    cited = sum(1 for p in paragraphs if re.search(r"\[.+?—.+?\]\(https?://", p))
    return round(cited / len(paragraphs), 2)


def keyword_recall(embedding_model,answer: str, expected: list[str],threshold: float = 0.55) -> float:
    if not expected:
        return 0.0

    answer_embedding = embedding_model.encode([answer],normalize_embeddings=True)[0]

    matched = 0

    for kw in expected:
        kw_embedding = embedding_model.encode([kw],normalize_embeddings=True)[0]
        sim = cosine_similarity([answer_embedding],[kw_embedding])[0][0]
        if sim >= threshold:
            matched += 1

    return round(matched / len(expected), 2)


def uncertainty_acknowledged(embedding_model,answer: str,threshold: float = 0.45) -> bool:
    UNCERTAINTY_CONCEPTS = [
        "The answer acknowledges uncertainty",
        "The answer discusses conflicting evidence",
        "The answer avoids overclaiming",
        "The answer expresses limitations in evidence",
        "The answer uses cautious reasoning",
    ]
    answer_embedding = embedding_model.encode(
        [answer],
        normalize_embeddings=True
    )[0]

    concept_embeddings = embedding_model.encode(
        UNCERTAINTY_CONCEPTS,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        [answer_embedding],
        concept_embeddings
    )[0]

    return bool(max(similarities) >= threshold)


def source_diversity(answer: str) -> int:
    domains = re.findall(r"\[.+?—\s*(.+?)\]\(https?://", answer)
    return len(set(d.strip() for d in domains))



def citations_per_1k_output_tokens(citation_count: int,output_tokens: int) -> float:
    if output_tokens <= 0:
        return 0.0
    return round(citation_count / (output_tokens / 1000),2)


def context_utilization(used_sources: int,retrieved_sources: int) -> float:

    if retrieved_sources <= 0:
        return 0.0
    return round(used_sources / retrieved_sources,2)


def retrieval_efficiency(answer_quality: float,total_input_tokens: int) -> float:
    if total_input_tokens <= 0:
        return 0.0
    return round(answer_quality / total_input_tokens,6)

def followup_coherence(embedding_model,prior_answer: str,followup_answer: str) -> float:
    if not prior_answer.strip() or not followup_answer.strip():
        return 0.0
    embeddings = embedding_model.encode([prior_answer, followup_answer],normalize_embeddings=True)
    similarity = cosine_similarity([embeddings[0]],[embeddings[1]])[0][0]
    return round(float(similarity), 2)



def extract_snippet_text(snippets: list[ContextSnippet]) -> str:
    parts = []
    for snippet in snippets:
        block = (
            f"Title: {snippet.title}\n"
            f"Domain: {snippet.domain}\n"
            f"URL: {snippet.url}\n\n"
            f"{snippet.text}"
        )
        parts.append(block)
    return "\n\n---\n\n".join(parts)


def get_turn_token_metrics(conn, session_id: str) -> dict:
    messages = store.get_messages(conn, session_id)
    input_tokens = 0
    output_tokens = 0
    for m in messages:
        role = m.role.lower()
        tokens = int(m.total_tokens or 0)
        if role == "assistant":
            output_tokens += tokens
        else:
            input_tokens += tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
    }








RELIABILITY_SYSTEM_PROMPT = """
You are evaluating the reliability of a research answer.

Judge the answer on:
- groundedness in the provided evidence
- whether citations support the claims
- whether uncertainty is handled correctly
- whether conflicting evidence is acknowledged
- whether the answer avoids unsupported speculation

Score guidance:
- 0.90–1.00: highly reliable, tightly grounded, citations support the main claims
- 0.70–0.89: mostly reliable, minor gaps or mild overreach
- 0.50–0.69: mixed reliability, some unsupported or weakly supported claims
- 0.30–0.49: weak reliability, several unsupported claims or poor grounding
- 0.00–0.29: unreliable, largely unsupported, misleading, or hallucinated

IMPORTANT:
- Return ONLY valid JSON
- Do NOT use markdown
- Do NOT include ```json fences

Required JSON schema:
{
  "reliability": float,
  "label": "high" | "medium" | "low",
  "reasoning": string,
  "strengths": [string],
  "weaknesses": [string]
}
""".strip()



def judge_answer_reliability_llm(llm_client,model,question: str,answer: str,snippets: list[Any],) -> ReliabilityJudgeResult:

    evidence = extract_snippet_text(snippets)
    user_prompt = f"""
Question:
{question}

Answer:
{answer}

Retrieved evidence:
\"\"\"
{evidence}
\"\"\"

Rate the reliability of the answer for this question.
""".strip()

    response = llm_client.models.generate_content(model=model,contents=user_prompt,config=types.GenerateContentConfig(
            system_instruction=RELIABILITY_SYSTEM_PROMPT,
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    parsed = json.loads(response.text)

    return ReliabilityJudgeResult(
        reliability=float(parsed["reliability"]),
        label=str(parsed.get("label", "")).strip(),
        reasoning=str(parsed.get("reasoning", "")).strip(),
        strengths=list(parsed.get("strengths", [])),
        weaknesses=list(parsed.get("weaknesses", [])),
    )

def run_item(conn,embedding_model,agent: DeepResearchAgent, item: dict) -> TurnResult:
    model = "gemini-2.5-flash"
    llm_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))    
    session_id = store.new_session(conn)



    t0 = time.time()
    answer = ""
    snippets: list[Any] = []
    retrieved_snippets: list[Any] = []

    for event in agent.run(session_id, item["query"]):
        if event["step"] == "done":
            answer = event["answer"]
            snippets = event.get("snippets", [])
            retrieved_snippets = event.get("retrieved_snippets",snippets)

    latency = round(time.time() - t0, 2)

    token_metrics = get_turn_token_metrics(conn,session_id)
    input_tokens = token_metrics["input_tokens"]
    output_tokens = token_metrics["output_tokens"]
    total_tokens = token_metrics["total_tokens"]

    
    cit_count = count_citations(answer)
    cov = citation_coverage(answer)
    rec = keyword_recall(embedding_model,answer,item["expected_keywords"])
    unc = uncertainty_acknowledged(embedding_model,answer)
    div = source_diversity(answer)
    cit_per_1k = citations_per_1k_output_tokens(cit_count,output_tokens)
    used_sources = len(snippets)
    retrieved_sources = (len(retrieved_snippets)if retrieved_snippets else used_sources)
    context_util = context_utilization(used_sources,retrieved_sources)
    total_input_tokens = (input_tokens )
    retrieval_eff = 0.0
    reliability = 0.0
    reliability_label = ""
    reliability_reasoning = ""

    if llm_client is not None:
        judge = judge_answer_reliability_llm(llm_client=llm_client,model=model,question=item["query"],answer=answer,snippets=snippets,)
        reliability = judge.reliability
        reliability_label = judge.label
        reliability_reasoning = judge.reasoning
        retrieval_eff = retrieval_efficiency(answer_quality=reliability,total_input_tokens=total_input_tokens)



    result = TurnResult(
        item_id=item["id"],
        query_type=item["type"],
        query=item["query"],
        answer=answer,
        citation_count=cit_count,
        citation_coverage=cov,
        keyword_recall=rec,
        uncertainty_acknowledged=unc,
        source_diversity=div,
        answer_length=len(answer),
        latency_seconds=latency,
        snippets_used=len(snippets),
        llm_reliability=reliability,
        llm_reliability_label=reliability_label,
        llm_reliability_reasoning=reliability_reasoning,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        citations_per_1k_output_tokens=cit_per_1k,
        context_utilization=context_util,
        retrieval_efficiency=retrieval_eff,
    )



    if item.get("follow_up"):
        t1 = time.time()
        followup_answer = ""
        for event in agent.run(session_id,item["follow_up"]):
            if event["step"] == "done":
                followup_answer = event["answer"]
        result.followup_query = item["follow_up"]
        result.followup_answer = followup_answer
        result.followup_citation_count = count_citations(followup_answer)
        result.followup_references_score = (followup_coherence(embedding_model,answer,followup_answer))
        result.followup_latency_seconds = round(time.time() - t1,2)
    return result



def print_summary(results: list[TurnResult]):
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    total = len(results)
    if total == 0:
        print("No results.")
        return

    avg_cit = sum(r.citation_count for r in results) / total
    avg_cov = sum(r.citation_coverage for r in results) / total
    avg_rec = sum(r.keyword_recall for r in results) / total
    avg_lat = sum(r.latency_seconds for r in results) / total
    avg_div = sum(r.source_diversity for r in results) / total
    avg_tok = sum(r.total_tokens for r in results) / total
    avg_in = sum(r.input_tokens for r in results) / total

    avg_out = sum(r.output_tokens for r in results) / total

    avg_rel = sum(r.llm_reliability for r in results) / total
    avg_cit_1k = sum(r.citations_per_1k_output_tokens for r in results) / total
    avg_ctx_util = sum(r.context_utilization for r in results) / total
    avg_ret_eff = sum(r.retrieval_efficiency for r in results) / total
    unc_rate = sum(1 for r in results if r.uncertainty_acknowledged) / total

    multiturn = [r for r in results if r.followup_query]
    coherence = (sum(r.followup_references_score  for r in multiturn) / len(multiturn) if multiturn else 0.0)

    print(f"  Questions evaluated         : {total}")
    print(f"  Avg citations/answer        : {avg_cit:.1f}")
    print(f"  Avg citation coverage       : {avg_cov:.2f}")
    print(f"  Avg keyword recall          : {avg_rec:.2f}")
    print(f"  Avg source diversity        : {avg_div:.1f}")
    print(f"  Uncertainty rate            : {unc_rate:.2f}")
    print(f"  Multi-turn coherence        : {coherence:.2f}")
    print(f"  Avg LLM reliability         : {avg_rel:.2f}")
    print(f"  Avg citations / 1k out tok  : {avg_cit_1k:.2f}")
    print(f"  Avg context utilization      : {avg_ctx_util:.2f}")
    print(f"  Avg retrieval efficiency    : {avg_ret_eff:.6f}")
    print(f"  Avg latency                 : {avg_lat:.2f}s")
    print(f"  Avg input tokens            : {avg_in:.0f}")
    print(f"  Avg output tokens           : {avg_out:.0f}")
    print(f"  Avg total tokens/query      : {avg_tok:.0f}")


    print("\n  By question type:")
    for qtype in [
        "factual",
        "multi_hop",
        "comparison",
        "insufficient_evidence",
        "conflicting_sources",
        "multi_turn",
    ]:
        group = [r for r in results if r.query_type == qtype]
        if not group:
            continue

        g_rec = sum(r.keyword_recall for r in group) / len(group)
        g_cit = sum(r.citation_count for r in group) / len(group)
        g_unc = sum(1 for r in group if r.uncertainty_acknowledged) / len(group)
        g_rel = sum(r.llm_reliability for r in group) / len(group)

        print(
            f"    {qtype:<25} n={len(group)}  "
            f"recall={g_rec:.2f}  citations={g_cit:.1f}  "
            f"uncertainty={g_unc:.2f}  reliability={g_rel:.2f}"
        )

    print("=" * 60)



def main():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    conn = store.conn(DB_PATH)
    agent = DeepResearchAgent(conn,embedding_model)

    results: list[TurnResult] = []

    print(f"\nRunning evaluation on {len(DATASET)} questions…\n")

    for item in DATASET:
        print(f"  [{item['id']}] {item['query'][:65]}…")
        try:
            result = run_item(conn, embedding_model,agent, item)
            results.append(result)
            print(
                f"    ✓ cit={result.citation_count} "
                f"cov={result.citation_coverage} "
                f"rec={result.keyword_recall} "
                f"rel={result.llm_reliability:.2f} "
                f"lat={result.latency_seconds}s"
            )
            print("waiting 80s to avoid API rate limits...")
            time.sleep(80)
        except Exception as e:
            print(f"    ✗ ERROR: {e}")

    print_summary(results)

    output_path = Path("evaluation/results.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
    print(f"\nFull results saved → {output_path}\n")


if __name__ == "__main__":
    main()