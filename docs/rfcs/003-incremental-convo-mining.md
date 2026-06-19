# RFC 003 — Incremental Conversation Mining (`--from-chunk`)

- **Status:** Implemented (branch `crew-chief/incremental-mining`)
- **Motivation:** Long-running Claude Code sessions (days/weeks without closing)
  miss all but their first compaction's worth of content when `Stop` never fires.

## Problem

The conversation miner's `file_already_mined()` check returns `True` as soon as
any drawer exists for a source file. For short sessions this is correct — mine
once at `Stop`, done. For a long-running session that is never closed, the `Stop`
hook never fires and `PreCompact` is the only mining opportunity. But
`PreCompact` fires repeatedly as the context window fills; after the first call
the file is already in `mined_set` and all subsequent calls silently skip it.

The result: only the content up to the first compaction event is ever mined.
Everything after is lost until the session eventually closes.

## Solution

A new `--from-chunk N` flag on `mine --mode convos` that enables append-only
incremental mining of a growing transcript file.

### Behavior

- **`from_chunk == 0` (default):** existing behavior, unchanged. The `mined_set`
  shortcut, `file_already_mined()` guard, and purge-before-insert path are all
  active. Normal short sessions are structurally unaffected.

- **`from_chunk > 0`:** append-only path.
  - Skips the `mined_set` shortcut and `file_already_mined()` recheck.
  - Does **not** purge existing drawers (`_source_file_delete_ids` is bypassed).
  - Normalizes and chunks the full file, then mines only `chunks[N:]`.
  - Chunk indices are absolute (assigned as `len(chunks)` during chunking), so
    slicing preserves drawer IDs: chunk index K always produces the same
    deterministic ID, and `upsert` rewrites it in place if it exists.
  - Reports `total_chunks` (pre-slice) so the caller can persist the next offset.

- `--json` flag emits a machine-readable result dict with `total_chunks`,
  `new_chunks`, and per-file metadata, for use by automated callers.

- `scan_convos()` now accepts a single file path in addition to a directory,
  enabling the hook to pass a transcript path directly.

### The boundary-chunk overlap rule

The Claude Code JSONL parser merges consecutive assistant turns. If a compaction
fires while an assistant turn is still in progress, that turn's content grows
between runs — the *text* of the last chunk changes while its `chunk_index`
stays the same. A strict `from_chunk = last_count` would skip that chunk,
leaving a stale drawer.

The fix: callers must pass `from_chunk = max(0, last_mined_chunk_count - 1)`.
The overlap-by-one lets `upsert` idempotently rewrite the boundary chunk with
its final content while skipping the already-stable prefix.

### Constraints

- `--from-chunk > 0` is only valid with `--mode convos`.
- If the target resolves to more than one file, `ValueError` is raised — a single
  offset is meaningless across multiple files.
- If `from_chunk >= total_chunks`, the call is a safe no-op (0 writes, exit 0).
- If `from_chunk > 0` but no existing drawers are found for the file, falls back
  to a full mine and emits a warning. Prefer idempotent over-mining to leaving
  chunks 0..N-1 missing.

## Implementation

Changes confined to `mempalace/convo_miner.py` and `mempalace/cli.py`.

- `_file_chunks_locked()` gains `purge: bool = True` and
  `respect_already_mined: bool = True` keyword-only parameters. When
  `from_chunk > 0`, called with `purge=False, respect_already_mined=False`.
- `mine_convos()` and `_mine_convos_impl()` gain `from_chunk: int = 0` and
  `quiet: bool = False` parameters.
- `mine_convos()` returns a structured result dict (was `None`).
- `_has_existing_convo_drawers()` helper for the defensive-fallback check.

## Tests

Nine regression and correctness tests added in `tests/test_convo_miner.py` and
`tests/test_cli.py`. Key tests:

1. Default path unchanged — second mine with no flag is skipped; IDs identical.
2. Incremental tail append — `0..N-2` drawers untouched; tail appended without
   duplicates.
3. **Boundary-chunk overlap** — the test runs both `from_chunk=N-1` (overlap)
   and `from_chunk=N` (no overlap) and asserts the boundary drawer contains the
   new content with overlap and is **stale without it**. The test fails if the
   overlap rule is removed.
4. Idempotent rerun — same incremental call twice; drawer count and IDs unchanged.
5. Safe no-op — `from_chunk >= total_chunks`; exits 0, nothing written.
6. Guard-bypass proof — fully-mined file still gets new drawers on incremental
   call, proving both skip points were bypassed.
7. Single-file constraint — `ValueError` on multi-file target.
8. Defensive fallback — `from_chunk > 0` with no existing drawers; full mine.
9. `total_chunks` reporting — `--json` output matches actual chunk count.

## Reference caller (Crew Chief PreCompact hook)

```python
# crew-chief/.claude/hooks/precompact_flush.py (abridged)
_MEMPALACE_REPO = Path("/Users/chris/dev/mempalace")
_venv_python = _MEMPALACE_REPO / ".venv" / "bin" / "python"

last_mined_chunk_count = int(state.get("last_mined_chunk_count") or 0)
from_chunk = max(0, last_mined_chunk_count - 1)  # overlap-by-one

proc = subprocess.run(
    [str(_venv_python), "-m", "mempalace.cli",
     "--palace", str(palace),
     "mine", transcript_path,
     "--mode", "convos",
     "--from-chunk", str(from_chunk),
     "--json"],
    cwd=str(_MEMPALACE_REPO), timeout=60, ...
)
payload = json.loads(proc.stdout)
state["last_mined_chunk_count"] = payload["total_chunks"]
```

The hook calls the fork's venv interpreter by absolute path — no dependency on
`uv` or the interactive shell PATH, both of which are unreliable in hook
environments.
