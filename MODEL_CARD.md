# OCR candidate and validation evidence

The evaluated configuration is DINOv2 ViT-S/14, a single-field detector head, the CRNN labelled “+5” (fold 0 for the fixed candidate), and dot-span refinement. DINOv2 supplies features, the local head proposes a box, and the CRNN scores up to 36 expansions. A six-row geometric check can extend the crop before the final reading. An invalid box returns a review flag. Upstream DINOv2 alone cannot read the field.

## Training and checkpoint provenance

| Artifact | SHA-256 | Evidence |
|---|---|---|
| Local detector head, fold 0 | `6559b2c300fbcad79b9db30c10a372074119281fe70df06fe9b3fa60ef2d31c5` | Trained on the training fold of a 25-page owner-reviewed snapshot. The dataset register records authorization for local training. Evidence covering commercial supply of the trained head is outstanding. |
| Local CRNN +5, fold 0 | `92cc64d081cd53f649e48da9229e0f2dce0ddbde9da7ccc447eb75533a4a3d29` | All 68 state tensors equal the archived CRNN. Epoch zero won selection. This checkpoint selects grayscale input, while the older file defaults to Otsu binarization. |
| Archived project CRNN | `9941eb2ac1398b78d4142eea3362e4d7dea555f158a961a0ff13a5aa58a7bc1a` | Same bytes as the local original model_sequence.pth. The original model card declares MIT at repository level and describes training on 80,000 synthetic rows. The generation and training logs independently record 80,000. |

The legacy generator reads an editable bitmap font and has a fallback font. A local font editor, its exported grids, and code describing measurements of physical characters are present. The two extant custom JSON files agree cell-for-cell on all 22 characters used by this CRNN. Twelve of their 43 glyphs differ, all outside this model's alphabet. The logs do not identify which font source the generator loaded or bind the training run to the final checkpoint by hash. The legacy dataset and reader are marked blocked for release in the project register.

Fine-tuning experiments also used a reviewed synthetic snapshot of 19,542 distinct labels after 458 exclusions (15,636 train, 1,957 validation, 1,949 test). Checkpoint selection used real crops. On fixed fold 0 the selected tensors stayed equal to the archived reader. The validated preprocessing and weights must be used together.

On the same nine labelled diagnostic photos, the archived reader and the selected fold 0 reader each read eight full strings and 71 of 72 characters correctly using DINOv2 and dot-span refinement. Both missed the same photo. For the selected reader, crop-margin selection alone gave six correct strings and adding dot-span refinement gave eight. The latter comparison holds the detector and reader fixed. These experiments do not establish the accuracy of the original YOLO pipeline on these photos.

## Exact match of all eight characters

| Evaluation | Correct | Interpretation |
|---|---:|---|
| 25 original pages, five document-disjoint folds | 24/25 (96%) | Each page used its fold's weights. These validation folds also informed checkpoint selection. |
| Four perspective transforms per original page | 69/100 (69%) | Stress test using the same original documents. |
| Newer diagnostic photos, fixed fold 0 | 8/9 (88.9%) | Ten photos, one without a reliable transcription. Seven of eight distinct labelled documents were correct on every available view. These images informed refinement development. |

The figures describe internal evaluation. Production acceptance still requires a locked model evaluated on fresh, independently labelled documents. The fixed candidate's task-specific weights should be released once the rights evidence for the local detector training and the legacy reader is resolved, or replaced by checkpoints trained on documented inputs and evaluated anew.

Handoff verification on November 19, 2026 used a fresh clone of the external bundle, the pinned upstream checkout, the local Python environment, and the two separately stored checkpoints. All ten diagnostic predictions, final boxes and refinement statuses matched the archived records. One photo was also processed through the command-line entry point and its JSON output checked.
