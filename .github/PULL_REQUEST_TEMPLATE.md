## Summary

<!-- 1–3 sentences: what changed and why. Focus on the "why". -->

## Test plan

<!-- Checklist of how this PR was verified. -->

- [ ] `uv run ruff check src tests benches`
- [ ] `uv run ruff format --check src tests benches`
- [ ] `uv run pytest -q`
- [ ] Relevant benchmarks (if touching hot paths): `uv run pytest benches/ --benchmark-only -m "not full"`

## Docs touched

<!-- Check every doc whose contract this PR changes. Leave unchecked if N/A. -->

- [ ] `CLAUDE.md`
- [ ] `README.md`
- [ ] `docs/concepts.md`
- [ ] `docs/architecture.md`
- [ ] `docs/api.md`
- [ ] `docs/filters.md`
- [ ] `docs/testing.md`

## Notes

<!-- Anything reviewers should know: tradeoffs, follow-ups, LanceDB quirks hit, etc. -->
