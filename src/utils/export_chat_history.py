import json

with open("result.json", encoding="utf-8") as f:
    data = json.load(f)

messages = data["messages"]
clean_posts = []

for msg in messages:
    # Пропускаем служебные и не-сообщения
    if msg.get("type") != "message":
        continue
    if not msg.get("text"):
        continue
        
    # Извлекаем текст из text_entities (универсально)
    if isinstance(msg["text"], str):
        text = msg["text"]
    elif isinstance(msg["text"], list):
        text = "".join(
            item["text"] if isinstance(item, dict) else item
            for item in msg["text"]
        )
    else:
        text = str(msg["text"])

    # Фильтр: минимум 20 символов и не просто эмодзи/смайлы
    if len(text.strip()) < 20:
        continue
    if all(c in "🚀💪🤓💩🥰😱🎉" for c in text.replace(" ", "")):
        continue

    clean_posts.append({
        "id": msg["id"],
        "date": msg["date"],
        "text": text.strip(),
        "edited": "edited" in msg,
        "reactions": {r["emoji"]: r["count"] for r in msg.get("reactions", [])},
        "has_media": "photo" in msg or "file" in msg,
        "author": msg.get("author", "База")
    })  
    
    
def popularity_score(reactions):
    # Пример: ❤️=1.5, 🔥=1.2, 💩=0.1, 🫡=1.0 — адаптируйте под стиль канала
    weights = {"❤": 1.5, "🔥": 1.2, "💩": 0.1, "🫡": 1.0, "🎉": 1.1}
    return sum(weights.get(emo, 1.0) * cnt for emo, cnt in reactions.items())

for post in clean_posts:
    post["metadata"] = {
        "date": post["date"],
        "edited": post["edited"],
        "has_media": post["has_media"],
        "popularity_score": popularity_score(post["reactions"]),
        "source": "telegram",
        "channel": "База"
    }
    
def detect_style(text):
    tags = []
    if "©️" in text or "гг. до н.э." in text:
        tags.append("pseudo_quote")
    if "СМУП" in text or "офицер" in text.lower():
        tags.append("military_parody")
    if "псих" in text.lower() or "шизо" in text.lower():
        tags.append("pseudo_science")
    if "💔" in text and "умер" in text.lower():
        tags.append("dramatic_eulogy")
    return tags

post["metadata"]["style_tags"] = detect_style(post["text"])

with open("data/corpus.jsonl", "w", encoding="utf-8") as f:
    for post in clean_posts:
        doc = {
            "id": f"base_{post['id']}",
            "text": post["text"],
            "metadata": post["metadata"]
        }
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")