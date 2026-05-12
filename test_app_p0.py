import ast
import pathlib
import unittest


APP_PATH = pathlib.Path(__file__).with_name("app.py")


class P0StaticGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = APP_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_response_level_is_parsed_in_ask_route(self):
        self.assertIn('response_level = data.get("response_level", 3)', self.source)
        self.assertIn("response_level = int(response_level)", self.source)
        self.assertIn("if response_level not in (1, 2, 3):", self.source)
        self.assertIn("response_level=response_level", self.source)

    def test_response_level_affects_prompt_and_retrieval(self):
        self.assertIn("def ask_llm(question: str, collection_name: str, response_level: int = 3)", self.source)
        self.assertIn("k_map = {1: 12, 2: 18, 3: 30}", self.source)
        self.assertIn("build_prompt(question, chunks, response_level=response_level)", self.source)
        self.assertIn("def build_prompt(question: str, chunks: list, response_level: int = 3)", self.source)

    def test_admin_lockout_configuration_exists(self):
        self.assertIn("_ADMIN_MAX_ATTEMPTS = 5", self.source)
        self.assertIn("_ADMIN_LOCKOUT_SECONDS = 600", self.source)
        self.assertIn("def _is_locked_admin_ip(", self.source)
        self.assertIn("def _record_admin_login_failure(", self.source)
        self.assertIn("Đăng nhập sai quá nhiều lần", self.source)

    def test_hashed_admin_password_support_exists(self):
        self.assertIn('ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "").strip()', self.source)
        self.assertIn("check_password_hash(ADMIN_PASSWORD_HASH, raw_password)", self.source)
        self.assertIn("if ADMIN_PASSWORD_HASH:", self.source)

    def test_admin_secret_requires_password_or_hash(self):
        self.assertIn("not ADMIN_PASSWORD and not ADMIN_PASSWORD_HASH", self.source)

    def test_free_only_mode_defaults_are_present(self):
        self.assertIn('FREE_ONLY_MODE = os.getenv("FREE_ONLY_MODE", "true")', self.source)
        self.assertIn('ALLOW_OPENROUTER_IN_FREE_MODE = os.getenv("ALLOW_OPENROUTER_IN_FREE_MODE", "false")', self.source)
        self.assertIn("if FREE_ONLY_MODE and not ALLOW_OPENROUTER_IN_FREE_MODE:", self.source)

    def test_retrieval_has_dedup_and_query_adaptive_threshold(self):
        self.assertIn("deduped = []", self.source)
        self.assertIn("seen = set()", self.source)
        self.assertIn("is_list_query = any(kw in question.lower() for kw in list_keywords)", self.source)
        self.assertIn("threshold = 0.2 if is_list_query else 0.3", self.source)

    def test_ask_llm_has_query_adaptive_k(self):
        self.assertIn("is_list_query = any(kw in question_lc for kw in", self.source)
        self.assertIn("is_procedure_query = any(kw in question_lc for kw in", self.source)
        self.assertIn("if is_list_query:", self.source)
        self.assertIn("elif is_procedure_query:", self.source)


if __name__ == "__main__":
    unittest.main()
