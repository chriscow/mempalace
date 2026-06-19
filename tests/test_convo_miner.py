import os
import tempfile
import shutil
from pathlib import Path

import chromadb
import pytest

from mempalace.convo_miner import (
    _is_ai_tool_path,
    _resolve_wing,
    mine_convos,
)
from mempalace.palace import MineAlreadyRunning, file_already_mined


def _write_convo(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _read_convo_drawers(palace_path: str, source_file: str) -> list[dict]:
    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection("mempalace_drawers")
    rows = col.get(where={"source_file": source_file}, include=["documents", "metadatas"])
    records = []
    for drawer_id, document, meta in zip(
        rows.get("ids") or [], rows.get("documents") or [], rows.get("metadatas") or []
    ):
        meta = meta or {}
        records.append(
            {
                "drawer_id": drawer_id,
                "document": document,
                "chunk_index": meta.get("chunk_index"),
                "meta": meta,
            }
        )
    return sorted(records, key=lambda item: item["chunk_index"])


def test_convo_mining():
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "chat.txt"), "w") as f:
        f.write(
            "> What is memory?\nMemory is persistence.\n\n> Why does it matter?\nIt enables continuity.\n\n> How do we build it?\nWith structured storage.\n"
        )

    palace_path = os.path.join(tmpdir, "palace")
    mine_convos(tmpdir, palace_path, wing="test_convos")

    client = chromadb.PersistentClient(path=palace_path)
    col = client.get_collection("mempalace_drawers")
    assert col.count() >= 2

    # Verify search works
    results = col.query(query_texts=["memory persistence"], n_results=1)
    assert len(results["documents"][0]) > 0

    shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_does_not_reprocess_short_files(capsys):
    """Files below MIN_CHUNK_SIZE get a sentinel so they are skipped on re-run."""
    tmpdir = tempfile.mkdtemp()
    try:
        # A file too short to produce any chunks
        with open(os.path.join(tmpdir, "tiny.txt"), "w") as f:
            f.write("hi")

        palace_path = os.path.join(tmpdir, "palace")

        # First run -- file is processed (sentinel written)
        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()  # drain output

        # Verify sentinel was written (resolve path -- macOS /var -> /private/var)
        resolved_file = str(Path(tmpdir).resolve() / "tiny.txt")
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        assert file_already_mined(col, resolved_file)

        # Second run -- file should be skipped
        mine_convos(tmpdir, palace_path, wing="test")
        out2 = capsys.readouterr().out
        assert "Files skipped (already filed): 1" in out2
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_does_not_reprocess_empty_chunk_files(capsys):
    """Files that normalize but produce 0 exchange chunks get a sentinel."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Content long enough to pass MIN_CHUNK_SIZE but with no exchange markers
        # (no "> " lines), so chunk_exchanges returns []
        with open(os.path.join(tmpdir, "no_exchanges.txt"), "w") as f:
            f.write("This is a plain paragraph without any exchange markers. " * 5)

        palace_path = os.path.join(tmpdir, "palace")

        mine_convos(tmpdir, palace_path, wing="test")
        mine_convos(tmpdir, palace_path, wing="test")
        out2 = capsys.readouterr().out
        assert "Files skipped (already filed): 1" in out2
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_allows_general_after_exchange(capsys):
    """A transcript mined as exchange can later be mined as general memories."""
    tmpdir = tempfile.mkdtemp()
    try:
        convo_path = Path(tmpdir) / "chat.txt"
        convo_path.write_text(
            "> What did we decide?\n"
            "We decided to use SQLite because it keeps the local setup simple.\n\n"
            "> What broke?\n"
            "The search failed because the old index was stale, and the fix was rebuild.\n"
        )
        palace_path = os.path.join(tmpdir, "palace")

        mine_convos(tmpdir, palace_path, wing="test", extract_mode="exchange")
        capsys.readouterr()
        mine_convos(tmpdir, palace_path, wing="test", extract_mode="general")
        out = capsys.readouterr().out

        assert "Files skipped (already filed): 0" in out

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        resolved = str(Path(tmpdir).resolve() / "chat.txt")
        rows = col.get(where={"source_file": resolved}, include=["metadatas"])
        modes = {meta.get("extract_mode") for meta in rows["metadatas"]}
        assert {"exchange", "general"} <= modes
        assert any(drawer_id.startswith("drawer_test_decision_") for drawer_id in rows["ids"])
        del col, client
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_rebuilds_stale_drawers_after_schema_bump(capsys):
    """When stored drawers have an older normalize_version, the next mine
    silently purges them and refiles — no manual erase required.

    This is what makes the strip_noise upgrade apply to existing corpora:
    users just run `mempalace mine` again and old noise-filled drawers get
    replaced with clean ones."""
    from mempalace.palace import NORMALIZE_VERSION

    tmpdir = tempfile.mkdtemp()
    try:
        convo_path = Path(tmpdir) / "chat.txt"
        convo_path.write_text(
            "> What is memory?\nMemory is persistence.\n\n"
            "> Why does it matter?\nIt enables continuity.\n\n"
            "> How do we build it?\nWith structured storage.\n"
        )
        palace_path = os.path.join(tmpdir, "palace")

        # First mine — stamps drawers with NORMALIZE_VERSION
        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        resolved = str(Path(tmpdir).resolve() / "chat.txt")
        first_pass = col.get(where={"source_file": resolved})
        first_ids = set(first_pass["ids"])
        assert first_ids, "first mine should produce drawers"
        for meta in first_pass["metadatas"]:
            assert meta.get("normalize_version") == NORMALIZE_VERSION

        # Simulate pre-v2 drawers: rewrite metadata to an older version,
        # and replace content with "noise" so we can see it get cleaned up.
        stale_metas = []
        for meta in first_pass["metadatas"]:
            stale = dict(meta)
            stale["normalize_version"] = 1
            stale_metas.append(stale)
        col.update(
            ids=list(first_pass["ids"]),
            documents=["STALE NOISE"] * len(first_pass["ids"]),
            metadatas=stale_metas,
        )
        # Add an extra orphan drawer that should also be purged.
        col.add(
            ids=["orphan_drawer"],
            documents=["OLD ORPHAN"],
            metadatas=[
                {
                    "wing": "test",
                    "room": "default",
                    "source_file": resolved,
                    "chunk_index": 999,
                    "normalize_version": 1,
                }
            ],
        )
        del col, client

        # Second mine — version gate should trigger rebuild
        mine_convos(tmpdir, palace_path, wing="test")
        out = capsys.readouterr().out
        assert "Files skipped (already filed): 0" in out, (
            "stale drawers should force a rebuild, not a skip"
        )

        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
        rebuilt = col.get(where={"source_file": resolved})
        # Orphan is gone
        assert "orphan_drawer" not in rebuilt["ids"]
        # No stale content survived
        assert all("STALE NOISE" not in d for d in rebuilt["documents"])
        assert all("OLD ORPHAN" not in d for d in rebuilt["documents"])
        # All rebuilt drawers carry the current version
        for meta in rebuilt["metadatas"]:
            assert meta.get("normalize_version") == NORMALIZE_VERSION
        del col, client
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mine_convos_default_path_remains_skipped(tmp_path, capsys):
    convo = tmp_path / "session.jsonl"
    _write_convo(
        convo,
        (
            "> What did we decide about the API?\n"
            "We decided to keep the API local because it avoids external dependencies.\n\n"
            "> What did we decide about storage?\n"
            "We decided to keep the storage append-only so re-mines stay idempotent.\n"
        ),
    )
    palace_path = str(tmp_path / "palace")

    mine_convos(str(convo), palace_path, wing="test")
    capsys.readouterr()
    before = _read_convo_drawers(palace_path, str(convo.resolve()))

    second = mine_convos(str(convo), palace_path, wing="test")
    capsys.readouterr()
    after = _read_convo_drawers(palace_path, str(convo.resolve()))

    assert second["files_skipped"] == 1
    assert [row["drawer_id"] for row in before] == [row["drawer_id"] for row in after]
    assert [row["document"] for row in before] == [row["document"] for row in after]


def test_mine_convos_incremental_tail_append_preserves_prefix(tmp_path, capsys):
    convo = tmp_path / "session.jsonl"
    initial = (
        "> What is alpha?\n"
        "Alpha is the first item and has enough detail to become a drawer.\n\n"
        "> What is beta?\n"
        "Beta is the second item and also has enough detail to become a drawer.\n"
    )
    appended = (
        initial + "\n> What is gamma?\n"
        "Gamma is the new item appended later and should only add tail drawers.\n"
    )
    _write_convo(convo, initial)
    palace_path = str(tmp_path / "palace")

    mine_convos(str(convo), palace_path, wing="test")
    capsys.readouterr()
    before = _read_convo_drawers(palace_path, str(convo.resolve()))

    _write_convo(convo, appended)
    result = mine_convos(str(convo), palace_path, wing="test", from_chunk=len(before) - 1)
    capsys.readouterr()
    after = _read_convo_drawers(palace_path, str(convo.resolve()))

    assert result["new_chunks"] == 2
    assert len(after) == 3
    assert len({row["drawer_id"] for row in after}) == 3
    assert before[0]["drawer_id"] == after[0]["drawer_id"]
    assert before[0]["document"] == after[0]["document"]
    assert after[1]["chunk_index"] == 1
    assert after[2]["chunk_index"] == 2


def test_mine_convos_overlap_rewrites_boundary_chunk(tmp_path, capsys):
    run1 = (
        "> What happened in the draft?\n"
        "The draft is mostly done and needs one more pass before shipping.\n\n"
        "> What did the assistant say?\n"
        "Part 1 of the final answer is here and it is intentionally incomplete.\n"
    )
    run2 = (
        "> What happened in the draft?\n"
        "The draft is mostly done and needs one more pass before shipping.\n\n"
        "> What did the assistant say?\n"
        "Part 1 of the final answer is here, Part 2 fills in the missing detail, and the reply continues.\n\n"
        "> What happened next?\n"
        "A follow-up exchange was added after the assistant finished the boundary turn.\n"
    )

    def _mine_case(palace_name: str, from_chunk: int):
        convo = tmp_path / f"{palace_name}.jsonl"
        palace_path = str(tmp_path / palace_name)
        _write_convo(convo, run1)
        mine_convos(str(convo), palace_path, wing="test")
        capsys.readouterr()
        _write_convo(convo, run2)
        mine_convos(str(convo), palace_path, wing="test", from_chunk=from_chunk)
        capsys.readouterr()
        return _read_convo_drawers(palace_path, str(convo.resolve()))

    with_overlap = _mine_case("palace_overlap", from_chunk=1)
    without_overlap = _mine_case("palace_no_overlap", from_chunk=2)

    boundary_with_overlap = next(row for row in with_overlap if row["chunk_index"] == 1)
    boundary_without_overlap = next(row for row in without_overlap if row["chunk_index"] == 1)

    assert "Part 2" in boundary_with_overlap["document"]
    assert "Part 2" not in boundary_without_overlap["document"]
    assert "Part 1" in boundary_without_overlap["document"]


def test_mine_convos_incremental_rerun_is_idempotent(tmp_path, capsys):
    convo = tmp_path / "session.jsonl"
    initial = (
        "> What is alpha?\n"
        "Alpha is the first item and has enough detail to become a drawer.\n\n"
        "> What is beta?\n"
        "Beta is the second item and also has enough detail to become a drawer.\n"
    )
    appended = (
        initial + "\n> What is gamma?\n"
        "Gamma is the new item appended later and should only add tail drawers.\n"
    )
    _write_convo(convo, initial)
    palace_path = str(tmp_path / "palace")

    mine_convos(str(convo), palace_path, wing="test")
    capsys.readouterr()
    _write_convo(convo, appended)
    mine_convos(str(convo), palace_path, wing="test", from_chunk=1)
    capsys.readouterr()
    first = _read_convo_drawers(palace_path, str(convo.resolve()))

    second = mine_convos(str(convo), palace_path, wing="test", from_chunk=1)
    capsys.readouterr()
    after = _read_convo_drawers(palace_path, str(convo.resolve()))

    assert second["new_chunks"] == 2
    assert [row["drawer_id"] for row in first] == [row["drawer_id"] for row in after]
    assert [row["document"] for row in first] == [row["document"] for row in after]


def test_mine_convos_safe_noop_when_from_chunk_is_past_end(tmp_path, capsys):
    convo = tmp_path / "session.jsonl"
    _write_convo(
        convo,
        (
            "> What is alpha?\n"
            "Alpha is the first item and has enough detail to become a drawer.\n\n"
            "> What is beta?\n"
            "Beta is the second item and also has enough detail to become a drawer.\n"
        ),
    )
    palace_path = str(tmp_path / "palace")

    mine_convos(str(convo), palace_path, wing="test")
    capsys.readouterr()
    before = _read_convo_drawers(palace_path, str(convo.resolve()))

    result = mine_convos(str(convo), palace_path, wing="test", from_chunk=99)
    capsys.readouterr()
    after = _read_convo_drawers(palace_path, str(convo.resolve()))

    assert result["new_chunks"] == 0
    assert [row["drawer_id"] for row in before] == [row["drawer_id"] for row in after]
    assert [row["document"] for row in before] == [row["document"] for row in after]


def test_mine_convos_falls_back_to_full_mine_when_state_is_missing(tmp_path, capsys):
    convo = tmp_path / "session.jsonl"
    _write_convo(
        convo,
        (
            "> What is alpha?\n"
            "Alpha is the first item and has enough detail to become a drawer.\n\n"
            "> What is beta?\n"
            "Beta is the second item and also has enough detail to become a drawer.\n"
        ),
    )
    palace_path = str(tmp_path / "palace")

    result = mine_convos(str(convo), palace_path, wing="test", from_chunk=1)
    capsys.readouterr()
    after = _read_convo_drawers(palace_path, str(convo.resolve()))

    assert result["new_chunks"] == 2
    assert len(after) == 2
    assert {row["chunk_index"] for row in after} == {0, 1}


def test_mine_convos_rejects_multiple_files_when_from_chunk_is_set(tmp_path):
    convo_dir = tmp_path / "convos"
    convo_dir.mkdir()
    _write_convo(
        convo_dir / "one.jsonl",
        "> What is alpha?\nAlpha is a chunk that is long enough to count.\n",
    )
    _write_convo(
        convo_dir / "two.jsonl",
        "> What is beta?\nBeta is another chunk that is also long enough to count.\n",
    )

    with pytest.raises(ValueError, match="single convo file"):
        mine_convos(str(convo_dir), str(tmp_path / "palace"), wing="test", from_chunk=1)


def _hold_palace_lock_in_child(palace_path, ready_flag, release_flag):
    """Acquire mine_palace_lock in a child process and hold until signalled.

    Cannot use threads because mine_palace_lock is intentionally re-entrant
    within a single thread (so ChromaCollection write methods can compose
    with miner.mine() without self-deadlock). The convos concurrency
    guarantee is across processes / threads, so the test has to mirror that.
    """
    import os as _os
    import time as _time

    from mempalace.palace import mine_palace_lock as _mpl

    with _mpl(palace_path):
        open(ready_flag, "w").close()
        for _ in range(500):
            if _os.path.exists(release_flag):
                return
            _time.sleep(0.01)


def test_mine_convos_refuses_concurrent_run_against_same_palace(tmp_path, monkeypatch):
    """A second `mine_convos` against a palace currently being mined must
    raise MineAlreadyRunning, not stack up as a waiter that drives parallel
    ChromaDB writes. Mirrors the guarantee already given by `miner.mine`
    (see test_palace_locks.py) for the convos code path.
    """
    import multiprocessing
    import time

    monkeypatch.setenv("HOME", str(tmp_path))
    convo_dir = tmp_path / "convos"
    convo_dir.mkdir()
    (convo_dir / "chat.txt").write_text("> q1\nshort answer.\n\n> q2\nanother short answer.\n")
    palace_path = str(tmp_path / "palace")
    ready_flag = str(tmp_path / "ready")
    release_flag = str(tmp_path / "release")

    ctx = multiprocessing.get_context("spawn")
    holder = ctx.Process(
        target=_hold_palace_lock_in_child,
        args=(palace_path, ready_flag, release_flag),
    )
    holder.start()
    try:
        # Wait for the child to actually hold the lock before we attempt
        # to acquire from this process.
        for _ in range(500):
            if os.path.exists(ready_flag):
                break
            time.sleep(0.01)
        assert os.path.exists(ready_flag), "child never acquired palace lock"

        with pytest.raises(MineAlreadyRunning):
            mine_convos(str(convo_dir), palace_path, wing="test")
    finally:
        open(release_flag, "w").close()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)


def test_mine_convos_dry_run_bypasses_palace_lock(tmp_path, monkeypatch):
    """Dry-run never writes to the palace, so it must coexist with a live
    mine instead of being blocked by the per-palace flock.
    """
    import multiprocessing
    import time

    monkeypatch.setenv("HOME", str(tmp_path))
    convo_dir = tmp_path / "convos"
    convo_dir.mkdir()
    (convo_dir / "chat.txt").write_text("> q1\nshort answer.\n\n> q2\nanother short answer.\n")
    palace_path = str(tmp_path / "palace")
    ready_flag = str(tmp_path / "ready_dry")
    release_flag = str(tmp_path / "release_dry")

    ctx = multiprocessing.get_context("spawn")
    holder = ctx.Process(
        target=_hold_palace_lock_in_child,
        args=(palace_path, ready_flag, release_flag),
    )
    holder.start()
    try:
        for _ in range(500):
            if os.path.exists(ready_flag):
                break
            time.sleep(0.01)
        assert os.path.exists(ready_flag), "child never acquired palace lock"

        # Must not raise — dry-run skips the lock entirely.
        mine_convos(str(convo_dir), palace_path, wing="test", dry_run=True)
    finally:
        open(release_flag, "w").close()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)


# ── _is_ai_tool_path / _resolve_wing — wing_api auto-routing ───────────
#
# When a user runs `mempalace mine --mode convos` against a directory
# inside a known AI-tool storage path (Claude Code's
# ~/.claude/projects/, OpenAI Codex's ~/.codex/, Google Gemini CLI's
# ~/.gemini/), the wing auto-defaults to "wing_api" rather than the
# directory basename. This keeps API-sourced conversations grouped
# under a single dedicated wing for visibility and privacy isolation.
#
# Explicit user-passed --wing always wins. Unrelated directories use
# the existing basename fallback unchanged.


def test_is_ai_tool_path_claude_projects_subdir(tmp_path):
    """A subdirectory inside ~/.claude/projects/ is an AI tool path."""
    target = tmp_path / ".claude" / "projects" / "-Users-test-myapp"
    target.mkdir(parents=True)
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_claude_projects_root(tmp_path):
    """The ~/.claude/projects/ directory itself is an AI tool path."""
    target = tmp_path / ".claude" / "projects"
    target.mkdir(parents=True)
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_codex_root(tmp_path):
    target = tmp_path / ".codex"
    target.mkdir()
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_codex_sessions(tmp_path):
    """Codex stores sessions under ~/.codex/sessions/YYYY/MM/DD/."""
    target = tmp_path / ".codex" / "sessions" / "2026" / "04" / "26"
    target.mkdir(parents=True)
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_gemini_root(tmp_path):
    target = tmp_path / ".gemini"
    target.mkdir()
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_gemini_chats(tmp_path):
    """Gemini stores sessions under ~/.gemini/tmp/<hash>/chats/."""
    target = tmp_path / ".gemini" / "tmp" / "abc123" / "chats"
    target.mkdir(parents=True)
    assert _is_ai_tool_path(target) is True


def test_is_ai_tool_path_dotclaude_without_projects_not_matched(tmp_path):
    """`.claude/` alone (without `/projects`) is the settings dir, not a
    conversation source — it MUST NOT auto-route to wing_api."""
    target = tmp_path / ".claude"
    target.mkdir()
    assert _is_ai_tool_path(target) is False


def test_is_ai_tool_path_unrelated_directory(tmp_path):
    target = tmp_path / "Documents" / "myproject"
    target.mkdir(parents=True)
    assert _is_ai_tool_path(target) is False


def test_is_ai_tool_path_substring_no_false_positive(tmp_path):
    """A directory NAMED like `.gemini-backup` or `.codex-archive` is NOT
    a real AI tool path. We use exact-segment match, not substring."""
    a = tmp_path / ".gemini-backup"
    a.mkdir()
    b = tmp_path / ".codex-archive"
    b.mkdir()
    assert _is_ai_tool_path(a) is False
    assert _is_ai_tool_path(b) is False


def test_resolve_wing_explicit_wins_over_auto_detection(tmp_path):
    """User-passed --wing always wins, even on an AI tool path."""
    target = tmp_path / ".claude" / "projects" / "-Users-x"
    target.mkdir(parents=True)
    assert _resolve_wing(target, wing="my_custom_wing") == "my_custom_wing"


def test_resolve_wing_claude_projects_auto_routes_to_wing_api(tmp_path):
    target = tmp_path / ".claude" / "projects" / "-Users-test-myapp"
    target.mkdir(parents=True)
    assert _resolve_wing(target, wing=None) == "wing_api"


def test_resolve_wing_codex_auto_routes_to_wing_api(tmp_path):
    target = tmp_path / ".codex" / "sessions" / "2026"
    target.mkdir(parents=True)
    assert _resolve_wing(target, wing=None) == "wing_api"


def test_resolve_wing_gemini_auto_routes_to_wing_api(tmp_path):
    target = tmp_path / ".gemini" / "tmp" / "abc" / "chats"
    target.mkdir(parents=True)
    assert _resolve_wing(target, wing=None) == "wing_api"


def test_resolve_wing_unrelated_dir_uses_basename_fallback(tmp_path):
    """Existing behavior preserved: arbitrary directories use the
    sanitized basename as the wing."""
    target = tmp_path / "MyProject Folder"
    target.mkdir()
    # Spaces become underscores, hyphens become underscores, lowercased.
    assert _resolve_wing(target, wing=None) == "myproject_folder"


def test_resolve_wing_empty_string_treated_as_no_wing(tmp_path):
    """An empty string for wing should behave like None — fall through to
    auto-detection / basename. Mirrors the original `if not wing:` guard."""
    target = tmp_path / ".gemini" / "tmp"
    target.mkdir(parents=True)
    assert _resolve_wing(target, wing="") == "wing_api"


def test_mine_convos_limit_skips_already_mined(capsys):
    """--limit N counts only new work, not already-mined skips (#1535)."""
    tmpdir = tempfile.mkdtemp()
    try:
        convo_text = (
            "> What is topic {i}?\n"
            "Topic {i} is about something important and interesting enough "
            "to produce at least one exchange chunk for the test.\n\n"
            "> Tell me more about topic {i}.\n"
            "Sure, topic {i} has many facets worth exploring in detail.\n"
        )
        for i in range(4):
            with open(os.path.join(tmpdir, f"chat_{i}.txt"), "w") as f:
                f.write(convo_text.format(i=i))

        palace_path = os.path.join(tmpdir, "palace")

        mine_convos(tmpdir, palace_path, wing="test")
        capsys.readouterr()

        for i in range(4, 7):
            with open(os.path.join(tmpdir, f"chat_{i}.txt"), "w") as f:
                f.write(convo_text.format(i=i))

        mine_convos(tmpdir, palace_path, wing="test", limit=2)
        out = capsys.readouterr().out

        assert "Files processed: 2" in out
        assert "Drawers filed:" in out
        for line in out.split("\n"):
            if "Drawers filed:" in line:
                filed = int(line.split(":")[1].strip())
                assert filed > 0, f"limit=2 should mine new files, got {filed}"
                break
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
