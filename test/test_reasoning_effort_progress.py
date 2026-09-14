import asyncio
from contextlib import redirect_stdout
import io
import tempfile
import unittest
from unittest.mock import patch

from reasoning_effort.progress import QAProgress, use_progress_bar
from test.test_reasoning_effort_dataset import ROW, pipeline, settings


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_qa_includes_invalid_and_cache_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = settings(directory)
            for repeat in range(2):
                p = pipeline(cfg)
                output = io.StringIO()
                try:
                    with redirect_stdout(output):
                        result = await p.run([ROW, {}])
                    self.assertEqual(result['completed_qa'], 2)
                    self.assertIn('QA処理: 2/2', output.getvalue())
                    self.assertNotIn('\r', output.getvalue())
                    self.assertNotIn('\x1b', output.getvalue())
                    self.assertEqual(len(p.generator.calls), 3 if repeat == 0 else 0)
                finally:
                    await p.aclose()

    async def test_unknown_total_and_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            p = pipeline(settings(directory))
            entered = asyncio.Event()
            async def generate(*args):
                entered.set()
                await asyncio.Event().wait()
            p.generator.generate = generate
            output = io.StringIO()
            try:
                with redirect_stdout(output):
                    task = asyncio.create_task(p.run(iter([ROW])))
                    await entered.wait()
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                self.assertEqual(p.stats['completed_qa'], 0)
                self.assertIn('QA処理: 0/?', output.getvalue())
            finally:
                await p.aclose()

    def test_rate_limited_plain_logs(self):
        from collections import Counter
        output = io.StringIO()
        with redirect_stdout(output), patch('reasoning_effort.progress.time.monotonic', return_value=0) as clock:
            progress = QAProgress(2, Counter())
            progress.refresh()
            self.assertEqual(len(output.getvalue().splitlines()), 1)
            clock.return_value = 31
            progress.refresh()
            self.assertEqual(len(output.getvalue().splitlines()), 2)
            progress.close()

    def test_stderr_terminal_and_explicit_modes(self):
        with patch.dict('os.environ', {'REASONING_PROGRESS': 'auto'}), \
             patch('reasoning_effort.progress.sys.stdout.isatty', return_value=False), \
             patch('reasoning_effort.progress.sys.stderr.isatty', return_value=True):
            self.assertTrue(use_progress_bar())
        with patch('reasoning_effort.progress.sys.stderr.isatty', return_value=False):
            with patch.dict('os.environ', {'REASONING_PROGRESS': 'bar'}):
                self.assertTrue(use_progress_bar())
            with patch.dict('os.environ', {'REASONING_PROGRESS': 'auto'}):
                self.assertFalse(use_progress_bar())
        with patch.dict('os.environ', {'REASONING_PROGRESS': 'log'}):
            self.assertFalse(use_progress_bar())

    def test_terminal_bar(self):
        from collections import Counter
        with patch.dict('os.environ', {'REASONING_PROGRESS': 'bar'}), \
             patch('reasoning_effort.progress.tqdm') as bar:
            progress = QAProgress(2, Counter())
            progress.finish_qa()
            progress.close()
            bar.return_value.update.assert_called_once_with(1)
            bar.return_value.close.assert_called_once()
