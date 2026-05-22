from __future__ import annotations
import os
from dotenv import load_dotenv
from google import genai
from typing import  Generator,Optional, Any
import json
import tiktoken
from memory import store
from retrieval import tavily_client, page_fetcher
from dataclasses import dataclass
import time
from google.genai.errors import ServerError
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class ContextSnippet:
    url: str
    title: str
    domain: str
    text: str
    score: Optional[float] = None




class DeepResearchAgent:
    def __init__(self,conn,sentence_model):
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = "gemini-3.5-flash"
        self.MAX_CONTEXT_TOKENS = 20000
        self.SNIPPET_CHARS = 3000
        self.system_prompt = """
You are a deep research assistant.

Your responsibilities:
- Answer ONLY using the provided web context
- Do NOT hallucinate facts, statistics, claims, or sources
- If the context is insufficient, explicitly say so
- If sources disagree, explicitly describe the disagreement
- Distinguish clearly between confirmed facts, analyst opinions, speculation, and projections

Answer style:
- Write clear, structured, research-style responses
- Prefer synthesis over source-by-source summarization
- Be concise but information-dense
- Use bullet points when useful
- Mention uncertainty when evidence is weak

Citation rules (STRICT AND MANDATORY):

Every factual claim MUST end with an inline markdown citation.

The ONLY valid citation format is:

[Title — domain](FULL_URL)

Example:
NVIDIA Blackwell supports NVLink 5 at 1.8 TB/s per GPU
[AMD MI400 vs NVIDIA B300 — spheron.network](https://www.spheron.network/blog/amd-mi400-vs-nvidia-b300/)

INVALID formats:
- Title — domain
- (Title — domain)
- Title | domain | URL
- Footnotes
- Sources list at the end
- Bare URLs

Rules:
- Citations MUST be markdown links
- Citations MUST appear inline in the paragraph
- Every paragraph with factual claims MUST contain at least one citation
- If no URL is available, DO NOT cite the source
- Never invent URLs
- Never output source names without markdown link formatting
- If retrieved context contains malformed citations or source text, rewrite them into the required markdown citation format. Do not copy citation formatting from context verbatim.

Grounding rules:
- Base claims only on retrieved context
- Do not use outside knowledge
- If relevant evidence is missing, say what additional information would help

When sources conflict:
- Cite both sides inline: ...according to [Source A — domain](url), while [Source B — domain](url) argues...
- Explain why disagreement may exist
- Avoid falsely presenting uncertain claims as settled facts
"""
        self.db = conn
        self.sentence_model = sentence_model
    def _answer(self, prompt, retries=3):
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(model=self.model,contents=prompt)
                usage = response.usage_metadata
                return {
                    "text": response.text,
                    "prompt_tokens": usage.prompt_token_count,
                    "thought_tokens": usage.thoughts_token_count,
                    "output_tokens": usage.candidates_token_count,
                    "total_tokens": usage.total_token_count
                }

            except ServerError as e:
                if "503" in str(e):
                    wait = 2 ** attempt
                    print(f"Gemini overloaded. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        raise Exception("Gemini API unavailable after retries.")

    def _plan_queries(self, query: str, summary: str) -> tuple[list[str], int,list[str]]:
        prompt = (
            f"USER:\n"
            f"User question: {query}\n\n"
            f"Conversation summary: {summary or 'None'}\n\n"
            f"Generate 2 to 4 focused web search queries."
            f"""Rules:\n- Return ONLY a valid JSON array\n- No markdown\n- No explanations\n- Queries should cover different aspects\nExample:\n["NVIDIA AI chip roadmap 2026","AMD MI400 datacenter strategy","Intel Gaudi accelerator adoption"]"""
        )

        raw = self._answer(prompt)
        total_planning_tokens = raw['total_tokens']
        try:
            qs = json.loads(raw['text'])
            if isinstance(qs, list):
                queries = [str(q) for q in qs[:4]]
                context_queries = [query] + queries
                return queries,total_planning_tokens, context_queries
        except Exception:
            pass
        return [query] ,total_planning_tokens, [query]
    
    def _fetch_pages(self, urls):
        fetched_pages = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(page_fetcher.fetch_pages,url): url for url in urls}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        fetched_pages.append(result)
                except Exception as e:
                    print(f"Fetch failed: {e}")

        return fetched_pages
    
    def _search_queries(self,queries: list[str],per_q: int):
        all_results = []
        seen: set[str] = set() 
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(tavily_client.search,query,max_results=per_q): query for query in queries}
            for future in as_completed(futures):
                query = futures[future]
                try:
                    results = future.result()
                    for result in results:
                        if result.url not in seen:
                            seen.add(result.url)
                            all_results.append(result)
                except Exception as e:
                    print(f"Search failed for query "f"'{query}': {e}")
        return all_results
    
    def _count_tokens(self,text: str) -> int:
        try:
            enc = tiktoken.encoding_for_model("gpt-4o-mini")
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    
    def _similar_text(self, page: str, queries: list[str],top_k = 3) ->list[str]:
        def split_text(text, chunk_size = self.SNIPPET_CHARS//top_k):

            paragraphs = text.split("\n\n")
            chunks = []
            current_chunk = ""
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                # If paragraph itself is huge
                if len(paragraph) > chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s*',paragraph)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < chunk_size:
                            current_chunk += " " + sentence
                        else:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                else:
                    if len(current_chunk) + len(paragraph) < chunk_size:
                        current_chunk += "\n\n" + paragraph
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = paragraph
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks

        def cosine_similarity(a, b):
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                return 0
            return np.dot(a, b) / (denom)
        
        if not page.strip():
            return []
        
        chunks = split_text(page)
        chunk_embeddings = self.sentence_model.encode(chunks)
        query_embeddings = self.sentence_model.encode(queries)

        scored_chunks = []

        for chunk, chunk_emb in zip(chunks, chunk_embeddings):
            best_score = 0
            for query_emb in query_embeddings:
                score = cosine_similarity(query_emb, chunk_emb)
                best_score = max(best_score, score)

            scored_chunks.append((best_score, chunk))

        scored_chunks.sort(reverse=True, key=lambda x: x[0])

        top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]

        return top_chunks
        

    def _build_context(self,search_results: list[tavily_client.SearchResult],fetched_pages: list[page_fetcher.PageContent],queries: list[str]) -> tuple[list[ContextSnippet], str,list[ContextSnippet]]:

        page_map = {p.url: p for p in fetched_pages}
        candidates = []
        for sr in sorted(search_results,key=lambda r: r.score or 0,reverse=True):
            page = page_map.get(sr.url) 
            if page and page.content:
                top_chunks = self._similar_text(page.content,queries)
                text = "\n\n".join(top_chunks)
            else:
                text = sr.snippet
            candidates.append(ContextSnippet(url=sr.url,title=sr.title,domain=sr.domain,text=text,score=sr.score))
        selected = []
        total = 0
        for cs in candidates:
            t = self._count_tokens(cs.text)
            if total + t > self.MAX_CONTEXT_TOKENS:
                break
            selected.append(cs)
            total += t
        parts = [f"[SOURCE {i}] {cs.title} ({cs.domain})\n"f"URL: {cs.url}\n\n"f"{cs.text}"for i, cs in enumerate(selected, 1)]
        return selected, "\n\n---\n\n".join(parts),candidates

    def _chat_completion(self, session_id, user_query, context_str):
        summary = store.get_summary(self.db, session_id)
        recent_msgs = store.get_messages(self.db, session_id)[-4:]

        parts = [self.system_prompt]

        if summary:
            parts.append(f"## Conversation summary\n{summary}")

        if recent_msgs:
            parts.append("## Recent conversation\n" +
                "\n".join(f"{m.role.upper()}: {m.content}" for m in recent_msgs))

        parts.append(f"## Web sources\n{context_str}")
        parts.append(f"## Current question\n{user_query}")
        prompt = "\n\n".join(parts)
        return self._answer(prompt)

    def run(self,session_id: str, user_query: str,max_search_results: int = 4) -> Generator[dict[str, Any], None, None]:
        """Yields progress dicts; final dict has step="done" and key "answer"."""

        yield {"step": "planning", "data": "Decomposing question into search queries…"}
        summary = store.get_summary(self.db,session_id)
        queries, planning_tokens, context_queries = self._plan_queries(user_query, summary)

        yield {"step": "planning", "data": f"Queries: {queries}"}

        yield {"step": "searching", "data": f"Running {len(queries)} searches via Tavily…"}
        per_q = max(1, max_search_results // len(queries))
        all_results = self._search_queries(queries,per_q)
        yield {"step": "searching", "data": f"Found {len(all_results)} unique sources"}

        urls = [r.url for r in all_results[:8]]
        yield {"step": "fetching", "data": f"Fetching {len(urls)} pages…"}

        fetched_pages = self._fetch_pages(urls)
    
        yield {"step": "fetching", "data": f"Retrieved {len(fetched_pages)} pages"}

        yield {"step": "selecting", "data": "Ranking and selecting snippets…"}

        snippets, context_str, retrieved_snippets  = self._build_context(all_results, fetched_pages,context_queries)

        yield {"step": "selecting","data": f"{len(snippets)} snippets · {sum(len(s.text) for s in snippets):,} chars"}

        yield {"step": "answering", "data": "Generating cited answer…"}

        
        answer = self._chat_completion(session_id, user_query, context_str)
        answer_tokens = answer["total_tokens"]
        total_tokens = planning_tokens + answer_tokens

        store.add_message(self.db, session_id, "user", user_query, self._count_tokens(user_query))
        store.add_message(self.db, session_id, "assistant", answer["text"], answer_tokens)
        store.save_turn(self.db,session_id, user_query,queries,urls,[s.url for s in snippets],answer["text"],planning_tokens,answer_tokens,total_tokens)
        store.update_session_tokens(self.db, session_id,planning_tokens, answer_tokens,total_tokens)

        all_msgs = store.get_messages(self.db,session_id) 

        if len(all_msgs) > 10:
            sp = ("Summarise this conversation in 3–5 sentences:\n\n" + "\n".join(f"{m.role}: {m.content[:300]}" for m in all_msgs[-10:]))
            summary = self._answer("User:\n" + sp)
            store.update_summary(self.db,session_id,summary['text'])

        yield {"step": "done","data": "Complete","answer": answer["text"],"search_queries": queries,"urls": urls,"snippets": snippets,"retrieved_snippets":retrieved_snippets}















