# Project Plan

## Before recording any footage
- [ ] Confirm the department's AI-use policy for senior design and what disclosure it requires
- [ ] Ask the advisor whether IRB/ethics approval is needed for filming minors; start it early if so
- [ ] Written permission from the school/program; parent consent process for players
- [ ] Read the university IP policy for senior design work (matters if this becomes a company)
- [ ] Decide camera setup: elevated position, 60fps if possible, fast shutter to reduce blur
- [ ] Set up GitHub repo (private) and GPU access (Colab / cluster / lab)

## Semester 1: core pipeline
| # | Milestone | Status |
|---|---|---|
| 1.1 | Record 3+ games; hand-chart shots for each (ground truth) | Not started |
| 1.2 | Frame extraction + labeling workflow (CVAT or Roboflow) | Not started |
| 1.3 | Player detection fine-tuned on our footage | Not started |
| 1.4 | Player tracking with stable IDs (ByteTrack) | Not started |
| 1.5 | Court keypoints + homography to court coordinates (feet) | Not started |
| 1.6 | Ball detection + shot detection (attempt → make/miss) | Not started |
| **1.7** | **Milestone: accurate shot chart for one full held-out game** | Not started |

## Semester 2: product and evidence
| # | Milestone | Status |
|---|---|---|
| 2.1 | Shot → player attribution (manual tagging first; jersey OCR is a stretch goal) | Not started |
| 2.2 | Coach-facing dashboard (Streamlit) | Not started |
| 2.3 | Pilot with a real team for part of their season | Not started |
| 2.4 | Full accuracy evaluation vs hand-charted stats | Not started |
| 2.5 | Stretch: referee positioning / court coverage analysis | Not started |
| 2.6 | Final report + presentation | Not started |

## After graduation
NBA Launchpad 2028 cohort applications are expected around September–October 2027. Apply with the pilot results and accuracy numbers.
