# Answer key: xbotgo.mp4, 10 s window

- File: `xbotgo_10s_mot.txt`, MOT 1.1 (frame, id, x, y, w, h, 1, 1, 1.0)
- Window: original frames 307–546 (12.82–22.82 s), every 6th frame → 40 labeled frames. The frame map is in `data/frames/task003_xbotgo_10s/export_sidecar.json`.
- 289 boxes, 9 players:

| ID | Player |
|---|---|
| 1 | light-blue jersey |
| 2 | blue pinnie, black shorts |
| 3 | #61 white |
| 4 | #8 white, orange shoes |
| 5 | #90 white/red/blue |
| 6 | blue pinnie, black leggings |
| 7 | navy jersey / blue pinnie, white tee |
| 8 | #32 white |
| 9 | #2 white, red shorts |

- Removed: the adult in black by the bench, and the spectator in a navy tracksuit by the doors. Neither is a player.

## Known gaps

Built from the tracker's boxes, with IDs corrected and spectators removed. No new boxes were drawn, so:
- a few players are unlabeled in frames where the tracker missed them, or where its box was on the wrong person and was dropped (e.g. player 2 in frames 38–40, player 8 in frames 18–20)
- unlabeled players will count as tracker "false positives" when scoring

Fixing these in CVAT and re-exporting would make the key stricter.
