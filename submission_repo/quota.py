"""How many tokens the key has left today. One tiny call (about 10 tokens).

    python quota.py

The key's quota is about 1,000,000 tokens; when it runs out every call
returns 403 "Token quota is exceeded" for many hours. Both teammates
share the key, so check before a long run.
"""
import json

from bot.llm import CHAT_DEPLOYMENT, CHAT_URL, _client

raw = _client(CHAT_URL).chat.completions.with_raw_response.create(
    model=CHAT_DEPLOYMENT, max_tokens=1, messages=[{"role": "user", "content": "hi"}])
left = json.loads(raw.text).get("remaining_quota_tokens")
print(f"remaining quota: {left:,} tokens" if left is not None else raw.headers.get("remaining-quota"))
