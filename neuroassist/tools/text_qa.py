"""Answer questions about a document (or a transcript), and say so when the answer is not there.

Guardrail: the model must return the exact sentence that supports its answer, and code checks that
sentence really is in the source. Numbers in the question that the source never mentions are a
strong sign of a trick question, so those abstain too."""
import re
from difflib import SequenceMatcher

from .. import config, llm
from ..ingest import load_document
from ..schema import Answer

NOT_FOUND = "The {source} does not say."

SYSTEM = ("You answer questions using ONLY the {source} you are given. Never use outside knowledge. "
          "If the {source} does not contain the answer, say so.")

SCHEMA = {"type": "object", "required": ["found", "quote", "answer"], "properties": {
    "found": {"type": "boolean"},
    "quote": {"type": "string"},
    "answer": {"type": "string"}}}

PROMPT = """{Source}:
<<<
{doc}
>>>

Question: {question}

If the {source} answers the question: set "found" to true, copy the exact sentence from the {source} that contains the answer into "quote", and put a short answer (a few words) in "answer".
If it does not: set "found" to false and leave "quote" and "answer" empty.
Check carefully: names, numbers and dates in the question must match the {source} exactly."""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def quote_in_source(quote: str, doc: str) -> bool:
    q, d = _norm(quote), _norm(doc)
    if len(q) < 8:
        return False
    if q in d:
        return True
    m = SequenceMatcher(None, d, q, autojunk=False).find_longest_match(0, len(d), 0, len(q))
    return m.size >= 0.85 * len(q)


def unknown_numbers(question: str, doc: str) -> list[str]:
    """Numbers in the question that never appear in the source (e.g. 1706 when the text says 1760)."""
    nums = re.findall(r"\b\d[\d,.]*\d\b|\b\d\b", question)
    d = doc.replace(",", "")
    return [n for n in nums if n.replace(",", "") not in d]


def answer_from_text(question: str, doc: str, source: str = "document") -> Answer:
    missing = unknown_numbers(question, doc)
    if missing:
        return Answer(NOT_FOUND.format(source=source), abstain=True, confidence=0.8,
                      evidence={"reason": f"question mentions {', '.join(missing)}, which the {source} never does"})
    r = llm.chat_json(config.QA_MODEL,
                      PROMPT.format(Source=source.capitalize(), source=source, doc=doc, question=question),
                      SCHEMA, system=SYSTEM.format(source=source), num_ctx=config.NUM_CTX_DOC, max_tokens=200)
    found, quote, ans = bool(r.get("found")), (r.get("quote") or "").strip(), (r.get("answer") or "").strip()
    if not found or not ans:
        return Answer(NOT_FOUND.format(source=source), abstain=True, confidence=0.7,
                      evidence={"reason": "model found no supporting text"})
    if not quote_in_source(quote, doc):
        return Answer(NOT_FOUND.format(source=source), abstain=True, confidence=0.5,
                      evidence={"reason": "model's quote is not in the source", "rejected_answer": ans,
                                "rejected_quote": quote})
    return Answer(ans, evidence={"quote": quote}, confidence=0.9 if _norm(ans) in _norm(quote) else 0.6)


def run(question, files, route, session) -> Answer:
    path = files[0]
    doc = session.memo(path, "document", lambda: load_document(path))
    if route.task == "summarize":
        text = llm.chat(config.CARD_MODEL,
                        f"Document:\n<<<\n{doc}\n>>>\n\n{question}\nAnswer in at most 5 short bullet points.",
                        num_ctx=config.NUM_CTX_DOC, max_tokens=300)
        return Answer(text.strip())
    return answer_from_text(question, doc)
