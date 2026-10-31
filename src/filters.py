"""Build ChromaDB `where` filters from simple options."""

SENTIMENTS = ("positive", "neutral", "negative")


def build_where(topic=None, category=None, sentiment=None, strict_topic=False):
    """Return a ChromaDB `where` dict (or None when no filter is given)."""
    if strict_topic and topic is None:
        raise ValueError("strict_topic needs a topic id")
    if sentiment is not None and sentiment not in SENTIMENTS:
        raise ValueError(f"sentiment must be one of {SENTIMENTS}")

    conditions = []
    if topic is not None:
        conditions.append({"topic_id": int(topic)})
        if strict_topic:
            conditions.append({"topic_strict": True})
    if category:
        conditions.append({"primary_category": category})
    if sentiment:
        conditions.append({"sentiment": sentiment})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}