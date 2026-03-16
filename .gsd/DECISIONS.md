# Decisions

<!-- Append-only register of architectural and pattern decisions -->

| ID | Decision | Rationale | Date |
|----|----------|-----------|------|
| D001 | Center track and robot at (0,0,0) in environment configuration. | Resolves clipping and sizing issues caused by large track offsets. Alignment with `TrackManager` waypoint sampling which centers the path at (0,0). | 2026-03-15 |
