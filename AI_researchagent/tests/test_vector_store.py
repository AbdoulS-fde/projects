"""End-to-end test for the RAG pipeline: loads games.json, embeds it into a scratch
ChromaDB collection, and checks that semantic search surfaces the right game.

Requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL) in AI_researchagent/.env,
since it makes real embedding calls.

Run from the AI_researchagent folder with: python -m unittest tests.test_vector_store
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from lib.game_docs import build_documents, load_games
from lib.vector_store import VectorStoreManager


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set")
class TestVectorStoreManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = VectorStoreManager(
            persist_directory=self.tmp_dir,
            collection_name="games_test",
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        games = load_games()
        self.ids, self.documents, self.metadatas = build_documents(games)
        self.store.add_documents(ids=self.ids, documents=self.documents, metadatas=self.metadatas)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_documents_upserts_all_games(self):
        self.assertEqual(self.store.count(), len(self.ids))

    def test_query_returns_relevant_game(self):
        result = self.store.query("Who developed FIFA 21?", k=3)
        top_title = result["metadatas"][0][0]["title"]
        self.assertEqual(top_title, "FIFA 21")

    def test_reset_clears_collection(self):
        self.store.reset()
        self.assertEqual(self.store.count(), 0)


if __name__ == "__main__":
    unittest.main()
