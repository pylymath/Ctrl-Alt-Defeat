"""Photos: count, read text, identify, describe."""
import re

from .. import config, llm
from ..ingest import prepare_image
from ..schema import Answer

SHORT = {"type": "object", "required": ["answer", "detail"], "properties": {
    "answer": {"type": "string"}, "detail": {"type": "string"}}}
BOXES = {"type": "object", "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "required": ["label", "box"], "properties": {
        "label": {"type": "string"},
        "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}}}}}}
COUNT = {"type": "object", "required": ["count", "detail"], "properties": {
    "count": {"type": "integer"}, "detail": {"type": "string"}}}

PLURALS = {"people": "person", "men": "man", "women": "woman", "children": "child", "mice": "mouse",
           "geese": "goose", "sheep": "sheep", "fish": "fish", "deer": "deer"}


def _target(question: str) -> tuple[str, str]:
    m = re.search(r"how many ([a-z]+)", question.lower())
    plural = m.group(1) if m and m.group(1) not in ("are", "is", "of", "times") else "objects"
    singular = PLURALS.get(plural) or (plural[:-1] if plural.endswith("s") else plural)
    return plural, singular


def _dedupe(boxes: list) -> list:
    out = []
    for b in boxes:
        if isinstance(b, list) and len(b) == 4 and not any(sum(abs(x - y) for x, y in zip(b, o)) < 20 for o in out):
            out.append(b)
    return out


def count(question: str, img: bytes) -> Answer:
    plural, singular = _target(question)
    if config.COUNT_MODE == "boxes":
        r = llm.chat_json(config.VISION_MODEL,
                          f"Find every {singular} in this image, including small, partly hidden or background ones. "
                          f"Return one item per {singular} with its bounding box [x1, y1, x2, y2] in 0-1000 coordinates.",
                          BOXES, images=[img], num_ctx=config.NUM_CTX_IMAGE, max_tokens=600)
        boxes = _dedupe([i.get("box") for i in r.get("items", []) if isinstance(i, dict)])
        n = len(boxes)
        return Answer(f"{n} {singular if n == 1 else plural}.", evidence={"boxes": boxes})
    r = llm.chat_json(config.VISION_MODEL,
                      f"{question} Count carefully, including small, partly hidden or background ones.",
                      COUNT, images=[img], num_ctx=config.NUM_CTX_IMAGE, max_tokens=150)
    n = int(r.get("count", 0) or 0)
    return Answer(f"{n} {singular if n == 1 else plural}.", evidence={"detail": r.get("detail", "")})


PROMPTS = {
    "read_text": "Read the text in this image exactly as written. In 'answer' give only the main text; "
                 "in 'detail' say where it appears.",
    "identify": "{q}\nIn 'answer' name the main object in a few words (brand or type if visible). "
                "In 'detail' give one short sentence of description.",
    "describe": "{q}\nIn 'answer' give a direct, short answer. In 'detail' give one or two sentences "
                "describing what is in the image.",
    "summarize": "{q}\nIn 'answer' give the key point in one sentence. In 'detail' list the important "
                 "things visible, separated by semicolons.",
}


def run(question, files, route, session) -> Answer:
    path = files[0]
    img = session.memo(path, "image", lambda: prepare_image(path, config.IMAGE_SIDE), config.IMAGE_SIDE)
    if route.task == "count":
        return count(question, img)
    prompt = PROMPTS.get(route.task, PROMPTS["describe"]).format(q=question)
    r = llm.chat_json(config.VISION_MODEL, prompt, SHORT, images=[img],
                      num_ctx=config.NUM_CTX_IMAGE, max_tokens=200)
    return Answer((r.get("answer") or "").strip() or "I could not tell.", evidence={"detail": r.get("detail", "")})
