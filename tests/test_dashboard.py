import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class DashboardCategoryTests(unittest.TestCase):
    def test_missing_api_key_explains_configuration(self) -> None:
        statement = b"Date,Narration,Debit\n2026-01-03,SWIGGY ORDER,250\n"
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            app = AppTest.from_file(app_path, default_timeout=20).run()
            app.file_uploader[0].upload(
                "statement.csv", statement, "text/csv"
            ).run()
            next(
                button
                for button in app.button
                if button.label == "Analyze & Categorize Expenses"
            ).click().run()

        self.assertEqual(app.exception, [])
        self.assertTrue(
            any(
                "Copy .env.example to .env" in error.value
                for error in app.error
            )
        )

    def test_dashboard_categorizes_local_merchant_rules(self) -> None:
        statement = b"Date,Narration,Debit\n2026-01-03,SWIGGY ORDER,250\n"
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            app = AppTest.from_file(app_path, default_timeout=20).run()
            app.file_uploader[0].upload(
                "statement.csv", statement, "text/csv"
            ).run()
            next(
                button
                for button in app.button
                if button.label == "Analyze & Categorize Expenses"
            ).click().run()

        self.assertEqual(app.exception, [])
        self.assertIn("1 matched instantly via cache/rules", app.info[0].value)
        self.assertEqual(
            app.session_state["analyzed_output"]["Category"].tolist(),
            ["Food & Dining"],
        )

    def test_spending_chat_uses_aggregates_without_sending_raw_transaction(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.entered = False
                self.closed = False
                self.prompt = ""
                self.exit_args = None
                self.models = self

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
                self.exit_args = (_exc_type, _exc_value, _traceback)
                self.closed = True

            def generate_content_stream(self, **kwargs):
                if not self.entered or self.closed:
                    raise RuntimeError("request used a closed Gemini client")
                self.prompt = kwargs["contents"]
                yield SimpleNamespace(text="Your total spending was 250.")

        statement = (
            b"Date,Narration,Debit\n"
            b"2026-01-03,UPI-SWIGGY-ORDER-MERCHANT@UPI-IFSC-123456789-ORDER,250\n"
        )
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        client = FakeClient()

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
            patch("google.genai.Client", return_value=client),
        ):
            app = AppTest.from_file(app_path, default_timeout=20).run()
            app.file_uploader[0].upload(
                "statement.csv", statement, "text/csv"
            ).run()
            next(
                button
                for button in app.button
                if button.label == "Analyze & Categorize Expenses"
            ).click().run()
            app.chat_input[0].set_value(
                "What patterns stand out in my merchant spending?"
            ).run()

        self.assertEqual(app.exception, [])
        self.assertTrue(client.closed)
        self.assertEqual(client.exit_args, (None, None, None))
        self.assertIn("total_expenses", client.prompt)
        self.assertIn("monthly_totals", client.prompt)
        self.assertIn("category_totals", client.prompt)
        self.assertIn("monthly_category_totals", client.prompt)
        self.assertIn("top_merchant_totals", client.prompt)
        self.assertIn("SWIGGY", client.prompt)
        self.assertNotIn("MERCHANT@UPI", client.prompt)
        self.assertNotIn("123456789", client.prompt)
        self.assertIn("Your total spending was 250.", [item.value for item in app.markdown])

    def test_report_questions_are_answered_without_gemini(self) -> None:
        statement = (
            b"Date,Narration,Debit\n"
            b"2026-01-03,SWIGGY ORDER,250\n"
            b"2026-01-11,AMAZON MARKET,100\n"
            b"2026-02-01,SWIGGY ORDER,200\n"
            b"2026-02-10,ZOMATO FOOD,300\n"
            b"2026-02-11,AMAZON MARKET,75\n"
        )
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        questions_and_answers = [
            ("How much did I spend overall?", "925.00"),
            ("Which month did I spend the most?", "February 2026"),
            ("Which month did I spend the least?", "January 2026"),
            ("Show my category totals.", "Food & Dining: 750.00"),
            ("How much did I spend on Shopping?", "175.00"),
            ("Which category did I spend most on?", "Food & Dining"),
            (
                "What's your suggestion for reducing my expenses?",
                "Food & Dining is your highest-spend category",
            ),
        ]

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
            patch(
                "google.genai.Client",
                side_effect=AssertionError("Gemini should not be called"),
            ) as gemini_client,
        ):
            app = AppTest.from_file(app_path, default_timeout=20).run()
            app.file_uploader[0].upload(
                "statement.csv", statement, "text/csv"
            ).run()
            next(
                button
                for button in app.button
                if button.label == "Analyze & Categorize Expenses"
            ).click().run()
            for question, expected_answer in questions_and_answers:
                app.chat_input[0].set_value(question).run()
                self.assertEqual(app.exception, [])
                answer = app.session_state["spending_chat_messages"][-1][
                    "content"
                ]
                self.assertIn(expected_answer, answer)
                if "suggestion" in question:
                    self.assertIn("set a realistic spending limit", answer)

        self.assertEqual(app.exception, [])
        gemini_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
