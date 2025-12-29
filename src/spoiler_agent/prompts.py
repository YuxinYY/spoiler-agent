from __future__ import annotations

SYSTEM_PROMPT = """
You are a spoiler classification assistant. Decide whether the input text contains spoilers.
Return ONLY a JSON object that matches the required schema.
""".strip()

USER_TEMPLATE = """
Task: classify the input as spoiler or not.
Definitions:
- spoiler: reveals ending, key plot twist, core character fate, or mystery answer.
- not spoiler: general opinions, vague praise, or non-essential details.

Output JSON schema:
{
  "has_spoiler": 0 or 1,
  "confidence": 0.0-1.0,
  "spoiler_type": "movie" | "novel" | "none",
  "key_spoiler_sentence": "..." | "none"
}

Example 1:
Text: The hero dies at the end to save the world.
Output: {"has_spoiler":1,"confidence":0.98,"spoiler_type":"movie","key_spoiler_sentence":"The hero dies at the end to save the world."}

Example 2:
Text: The visuals are stunning and the soundtrack is great.
Output: {"has_spoiler":0,"confidence":0.95,"spoiler_type":"none","key_spoiler_sentence":"none"}

Input text: {input_text}
Content type hint: {content_type}
""".strip()


def build_messages(input_text: str, content_type: str = "unknown") -> list[dict[str, str]]:
    user_prompt = USER_TEMPLATE.format(input_text=input_text, content_type=content_type)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
