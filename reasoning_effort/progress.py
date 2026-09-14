"""One QA progress bar shared by asynchronous effort workers."""
import re
import os
import sys
import time

from tqdm import tqdm


def use_progress_bar():
    mode = os.environ.get('REASONING_PROGRESS', 'auto').lower()
    if mode not in ('auto', 'bar', 'log'):
        raise ValueError('REASONING_PROGRESS must be auto, bar or log')
    return mode == 'bar' or (mode == 'auto' and sys.stderr.isatty())


def write_log(message):
    if use_progress_bar():
        # Keep normal logs on stdout, including when stdout is piped to tee.
        tqdm.write(message, file=sys.stdout)
    else:
        print(re.sub(r'\x1b\[[0-9;]*m', '', message), flush=True)


class QAProgress:
    def __init__(self, total, stats):
        self.stats = stats
        self.total = total
        self.completed = 0
        self.active = 0
        self.started = self.last_log = time.monotonic()
        self.terminal = use_progress_bar()
        if total is None:
            write_log('QA総件数は不明です。完了件数・経過時間・速度を表示します（割合・残り時間は表示しません）。')
        self.bar = tqdm(total=total, desc='QA処理', unit='QA', dynamic_ncols=True,
                        mininterval=0.5, disable=not self.terminal, file=sys.stderr)
        self.refresh(force=True)

    def finish_qa(self):
        self.completed += 1
        self.stats['completed_qa'] += 1
        self.bar.update(1)
        self.refresh()

    def refresh(self, force=False):
        hits = self.stats['canonical_cache_hits']
        success = sum(self.stats['canonical_' + e] for e in ('low', 'medium', 'high')) - hits
        failures = sum(v for k, v in self.stats.items() if k.endswith('_terminal_failures'))
        status = f'生成成功={success} cache再利用={hits} 最終失敗={failures} 実行中={self.active}'
        if self.terminal:
            self.bar.set_postfix_str(status, refresh=False)
            if force:
                self.bar.refresh()
        elif force or time.monotonic() - self.last_log >= 30:
            denominator = str(self.total) if self.total is not None else '?'
            elapsed = int(time.monotonic() - self.started)
            write_log(f'QA処理: {self.completed}/{denominator} 経過={elapsed}s {status}')
            self.last_log = time.monotonic()

    def close(self):
        self.refresh(force=True)
        self.bar.close()
