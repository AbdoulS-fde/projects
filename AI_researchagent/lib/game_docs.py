"""Loads game entries from games_data/games.json and turns them into RAG documents."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_GAMES_PATH = Path(__file__).resolve().parent.parent / "games_data" / "games.json"


def load_games(path: Path = DEFAULT_GAMES_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def game_to_document(game: Dict[str, Any]) -> str:
    """Concatenate a game's key fields into a natural-language blob for embedding."""
    return (
        f"{game['title']} is a {game['genre']} game developed by {game['developer']} "
        f"and published by {game['publisher']}. It was released on {game['release_date']} "
        f"for {game['platform']}. {game['description']}"
    )


def build_documents(games: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Return (ids, documents, metadatas) ready to upsert into a Chroma collection, keyed by the game's slug id."""
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    for game in games:
        ids.append(game["id"])
        documents.append(game_to_document(game))
        metadatas.append(
            {
                "title": game["title"],
                "platform": game["platform"],
                "genre": game["genre"],
                "publisher": game["publisher"],
                "developer": game["developer"],
                "release_date": game["release_date"],
            }
        )
    return ids, documents, metadatas
