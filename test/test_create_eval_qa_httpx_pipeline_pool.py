import unittest

from main_create_eval_qa_httpx_pipeline_pool import parse_sample_size, select_sampled_items


class EvalQASamplingTests(unittest.TestCase):
    def test_parse_sample_size_allows_none_or_non_positive_as_all(self) -> None:
        self.assertIsNone(parse_sample_size(None))
        self.assertIsNone(parse_sample_size(0))
        self.assertIsNone(parse_sample_size(-1))
        self.assertEqual(parse_sample_size("3"), 3)

    def test_select_sampled_items_is_seeded_and_limited(self) -> None:
        items = [{"id": str(i)} for i in range(10)]

        first = select_sampled_items(items, sample_size=4, seed=42)
        second = select_sampled_items(items, sample_size=4, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertNotEqual(first, items[:4])

    def test_select_sampled_items_returns_all_when_limit_is_none_or_large(self) -> None:
        items = [{"id": str(i)} for i in range(3)]

        self.assertEqual(select_sampled_items(items, sample_size=None, seed=42), items)
        self.assertEqual(select_sampled_items(items, sample_size=10, seed=42), items)


if __name__ == "__main__":
    unittest.main()
