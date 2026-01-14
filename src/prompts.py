from __future__ import annotations

SYSTEM_PROMPT = """
You are a spoiler classification assistant for book reviews.
Decide whether the review sentence contains spoilers.
Return ONLY a JSON object that matches the required schema.
""".strip()

USER_TEMPLATE = """
Task: decide whether the review sentence contains a spoiler.
You are given the review sentence (target), book title, and full review as context.
Use the full review only for context; the label is about the review sentence.
Definitions:
- spoiler: reveals ending, key plot twist, core character fate, or mystery answer.
- not spoiler: general opinions, vague praise, or non-essential details.

Output JSON schema:
{{
  "has_spoiler": 0 or 1,
  "confidence": 0.0-1.0,
  "spoiler_type": "movie" | "novel" | "none",
  "key_spoiler_sentence": "..." | "none"
}}

Example 1:
Review sentence: The hero dies at the end to save the world.
Book title: Unknown Book
Full review: The hero dies at the end to save the world.
Output: {{"has_spoiler":1,"confidence":0.98,"spoiler_type":"movie","key_spoiler_sentence":"The hero dies at the end to save the world."}}

Example 2:
Review sentence: The visuals are stunning and the soundtrack is great.
Book title: Unknown Book
Full review: The visuals are stunning and the soundtrack is great.
Output: {{"has_spoiler":0,"confidence":0.95,"spoiler_type":"none","key_spoiler_sentence":"none"}}

Review sentence: {review_sentence}
Book title: {book_title}
Full review: {review_text}
Content type hint: {content_type}
""".strip()


def _pick_field(data: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return ""


def _extract_inputs(input_data: str | dict) -> tuple[str, str, str]:
    if isinstance(input_data, dict):
        review_sentence = _pick_field(input_data, ("text", "review_sentence", "sentence"))
        book_title = _pick_field(input_data, ("book_titles", "book_title", "title", "book"))
        review_text = _pick_field(input_data, ("review", "full_review", "review_text"))
        if not review_text:
            review_text = review_sentence
        if not book_title:
            book_title = "Unknown Book"
        return review_sentence, book_title, review_text
    text = str(input_data).strip()
    return text, "Unknown Book", text


def build_messages(input_text: str | dict, content_type: str = "unknown") -> list[dict[str, str]]:
    # 这里的 .format 只会替换单括号的 {review_sentence}/{book_title}/{review_text}/{content_type}
    # 双括号的 {{ }} 会被自动降级为单括号的 { } 输出给 AI
    review_sentence, book_title, review_text = _extract_inputs(input_text)
    user_prompt = USER_TEMPLATE.format(
        review_sentence=review_sentence,
        book_title=book_title,
        review_text=review_text,
        content_type=content_type,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
