"""User feedback collection."""
from datetime import datetime, timezone
from database import db


def save_feedback(
    query: str,
    answer: str,
    rating: int,
    comment: str = "",
    helpful: bool | None = None,
):
    db.insert_feedback({
        "query": query,
        "answer": answer,
        "rating": rating,
        "comment": comment,
        "helpful": int(helpful) if helpful is not None else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
