"""Download progress line and resumability (no network: a fake client)."""
import io

import pandas as pd

from pm473 import collect


class FakeClient:
    def __init__(self, n_rows, fail=()):
        self.n_rows, self.fail, self.calls = n_rows, set(fail), []

    def trades_for_condition(self, cid, page_limit=500):
        self.calls.append(cid)
        if cid in self.fail:
            raise RuntimeError("boom")
        for k in range(self.n_rows[cid]):
            yield {"proxyWallet": f"0x{k}", "price": 0.5, "size": 1.0, "side": "BUY", "timestamp": 1_700_000_000 + k, "conditionId": cid}


def make_markets():
    return pd.DataFrame({"condition_id": ["0xaa", "0xbb", "0xcc"], "volume": [1000.0, 3000.0, 6000.0]})


def test_progress_line_and_resume(tmp_path):
    mk = make_markets()
    c = FakeClient({"0xaa": 1200, "0xbb": 5, "0xcc": 3}, fail={"0xbb"})
    out = io.StringIO()
    import sys
    old, sys.stdout = sys.stdout, out
    try:
        files, failed = collect.download_trades(c, mk, tmp_path)
    finally:
        sys.stdout = old
    text = out.getvalue()
    assert failed == ["0xbb"] and len(files) == 2
    assert "markets already on disk" in text and "FAILED" in text and "ETA" in text and "mkts" in text
    log = (tmp_path / "_download_log.txt").read_text()
    assert "0xaa" in log and "0xcc" in log and "FAILED" in log
    # re-run: finished markets are skipped, only the failed one is retried
    c2 = FakeClient({"0xbb": 7})
    files2, failed2 = collect.download_trades(c2, mk, tmp_path, progress=False)
    assert c2.calls == ["0xbb"] and failed2 == [] and len(files2) == 3


def test_progress_counts_existing_files_and_volume(tmp_path):
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "0xcc.parquet")      # the biggest market is already on disk
    p = collect.DownloadProgress(3, 10000.0, 1, 6000.0, stream=io.StringIO(), tty=False)
    assert "60.0%" in p.line() and "1/3 mkts" in p.line() and "ETA ..." in p.line()
    p.vol_done, p.n_done = 10000.0, 3
    assert "ETA done" in p.line()


def test_line_fits_the_terminal_and_counts_the_unfinished_market():
    p = collect.DownloadProgress(100, 1_000_000.0, 0, 0.0, stream=io.StringIO(), tty=True, width=60)
    p.n_done, p.vol_done, p.vol_run, p.rows = 10, 100_000.0, 100_000.0, 10_000        # 0.1 rows per dollar so far
    p.start_market("0xabcdef123456", 400_000.0)
    p.cur_rows = 20_000                                                               # worth about 200,000 dollars of volume
    assert abs(p.effective_done() - 300_000.0) < 1
    assert len(p.line(60)) <= 60
    p.cur_rows = 10_000_000                                                           # never counts more than 95% of the market
    assert p.effective_done() <= 100_000.0 + 0.95 * 400_000.0 + 1
    p.show(force=True)
    out = p.stream.getvalue()
    assert out.startswith("\r") and "\n" not in out and len(out) <= 70


def test_fmt_duration():
    assert collect._fmt_dur(59) == "0m59s" and collect._fmt_dur(3725) == "1h02m"
