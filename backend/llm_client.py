"""
LLM Client
Sends prompts to an LLM and returns grounded answers with source citations.
Supports: Anthropic Claude (default), OpenAI GPT-4, local Ollama.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ─── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are DocuChat, a precise document analysis assistant.

CRITICAL RULES:
1. Answer questions EXCLUSIVELY based on the provided document context.
2. If the answer is not in the documents, say clearly: "I cannot find this information in the uploaded documents."
3. NEVER invent, hallucinate, or extrapolate facts not present in the documents.
4. For basic language understanding or universal facts (e.g., "what does 'annual' mean?"), you may answer from general knowledge — but make it clear you are doing so.
5. Always cite your sources using the format: [Source: <document name>, page <N>]
6. Detect the language of the user's question and respond in the SAME language.
7. Be concise but complete. Do not pad your answers.
8. When comparing multiple documents, organize your answer clearly by source.
9. When summarizing, structure the summary with the most important points first.

CITATION FORMAT: After each factual statement from a source, add [Source: <filename>, page <N>].
At the end of your answer, list the unique sources used under a "**Sources:**" section.
"""


class LLMClient:
    """Wrapper for LLM providers."""

    def __init__(self):
        self.provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        self.model = os.environ.get("LLM_MODEL", self._default_model())
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            logger.warning("No LLM_API_KEY found. Set LLM_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.")

    def _default_model(self):
        p = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        if p == "openai":
            return "gpt-4o-mini"
        elif p == "ollama":
            return "llama3.2"
        return "claude-haiku-4-5-20251001"

    def answer(
        self,
        question: str,
        context: str,
        history: List[Dict],
        raw_chunks: List[Dict],
        doc_map: Dict[str, str]
    ) -> Tuple[str, List[Dict]]:
        """
        Generate a grounded answer and extract source citations.
        Returns (answer_text, sources_list)
        """
        user_content = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION: {question}

Remember: Only answer from the document context above. Cite sources."""

        messages = []
        # Add conversation history (skip system-level messages)
        for h in history[-6:]:  # last 6 messages
            if h["role"] in ("user", "assistant"):
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_content})

        try:
            if self.provider == "anthropic":
                answer = self._call_anthropic(messages)
            elif self.provider == "openai":
                answer = self._call_openai(messages)
            elif self.provider == "ollama":
                answer = self._call_ollama(messages)
            else:
                answer = f"Unknown LLM provider: {self.provider}"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer = f"Error contacting LLM: {str(e)}. Check your API key and provider settings."

        sources = self._extract_sources(answer, raw_chunks, doc_map)
        return answer, sources

    # ── Provider Implementations ───────────────────────────────────────────────

    def _call_anthropic(self, messages: List[Dict]) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages
        )
        return response.content[0].text

    def _call_openai(self, messages: List[Dict]) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        response = client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=2048,
            temperature=0.1
        )
        return response.choices[0].message.content

    def _call_ollama(self, messages: List[Dict]) -> str:
        import requests
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json={"model": self.model, "messages": full_messages, "stream": False},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    # ── Source Extraction ──────────────────────────────────────────────────────

    def _extract_sources(
        self,
        answer: str,
        raw_chunks: List[Dict],
        doc_map: Dict[str, str]
    ) -> List[Dict]:
        """Parse sources from the answer and cross-reference with chunks."""
        sources = []
        seen = set()

        # Extract from answer text: [Source: filename, page N]
        pattern = r"\[Source:\s*([^,\]]+?)(?:,\s*page\s*(\d+))?\]"
        for match in re.finditer(pattern, answer, re.IGNORECASE):
            name = match.group(1).strip()
            page = int(match.group(2)) if match.group(2) else None
            key = (name, page)
            if key not in seen:
                seen.add(key)
                sources.append({"document": name, "page": page})

        # If no citations found in text, add top chunks as sources
        if not sources:
            for chunk in raw_chunks[:3]:
                doc_name = doc_map.get(chunk.get("doc_id"), "Unknown")
                key = (doc_name, chunk.get("page"))
                if key not in seen:
                    seen.add(key)
                    sources.append({"document": doc_name, "page": chunk.get("page")})

        return sources