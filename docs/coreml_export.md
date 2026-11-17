# Core ML export

How Veto's selected model is exported as a single Core ML artifact, how that artifact is checked
against Python inference, and the contract a consumer (the future `veto-ios` repository) relies
on. **No Swift code lives in this repository**; this milestone ends at a validated artifact
boundary.

## Why this exists

The training pipeline produces a PyTorch/scikit-learn model that only runs in Python. A phone app
needs a compact, self-contained model plus an exact description of how to feed it. The export
turns the selected model into:

```text
Veto internals ──▶ veto export coreml ──▶ dist/coreml/ ──▶ veto-ios
```

```text
dist/coreml/
├── veto.mlpackage/            the model (Core ML ML Program, FP16)
├── metadata.json              threshold, labels, shapes, tokenizer contract, versions, validation
└── tokenizer/
    ├── tokenizer.json         full tokenizer definition (Hugging Face "tokenizers" format)
    ├── tokenizer_config.json
    ├── vocab.txt              WordPiece vocabulary, one token per line, line number = id
    └── golden_tokenization.json   example texts with their exact token ids (for testing)
```

`dist/` is gitignored and enforced by CI: the package contains weights trained on private
messages, so it is never committed (see the README on why weights are not published).

## What gets exported

The export follows the model selected by Veto's training (`models/veto/metadata.json`). The
current v1 is the **baseline**:

```text
Python (training/inference)               Core ML (one graph)
─────────────────────────────             ───────────────────────────────────────────
text                                      input_ids, attention_mask   (from the tokenizer)
 └─ tokenizer (BERT uncased, 256)          └─ MiniLM encoder (6 layers, 384-d)
     └─ MiniLM encoder                          └─ mean pooling over attention_mask
         └─ mean pooling                             └─ L2 normalization
             └─ L2 normalization                          └─ linear layer (logistic regression)
                 └─ logistic regression                        └─ sigmoid
                     └─ probability                                 └─ political_solicitation_probability
```

The learned logistic-regression coefficients (384 weights and the intercept) are copied into a
linear layer, so the app does not reimplement pooling or regression. If the fine-tuned model is
selected instead, the exporter reproduces that path (encoder, pooler, 2-class head, softmax); that
path is covered by tests but has not been exported from real data.

**Nothing is retrained.** The wrapper (`src/veto/export/wrapper.py`) loads the saved parameters
and `verify` checks it against the existing Python inference (differences of about 4e-7 for the
baseline).

### Why the encoder forward is written out explicitly

Tracing Hugging Face's stock `BertModel` forward with transformers 5.x fails in Core ML
conversion (it builds attention masks with Python `int()` casts on tensors). Rather than change
the learned behavior or pin an old library, the wrapper runs the same trained `Linear` and
`LayerNorm` layers through an explicit forward pass (plain matmul, softmax, layer norm, GELU).
No operation in the model is unsupported by Core ML.

## Tokenization contract

Tokenization happens **outside** the Core ML model: the app converts text to two integer tensors.

| item | value |
|---|---|
| Tokenizer | BERT WordPiece, `all-MiniLM-L6-v2` vocabulary (30,522 tokens) |
| Casing | **Lower-cased.** Capital letters are lost before the model sees the text |
| Accents | Stripped (follows lower-casing in the Hugging Face BERT normalizer) |
| Cleaning | Control characters removed; CJK characters split into single tokens |
| Unknown characters | Map to `[UNK]` (id 100). **Emoji are `[UNK]`.** |
| Template | `[CLS] tokens… [SEP]` (ids 101 … 102) |
| Max length | 256 tokens, including `[CLS]` and `[SEP]` |
| Truncation | Longest-first, right side |
| Padding | Right-pad to exactly 256 with `[PAD]` (id 0) |
| `attention_mask` | 1 for real tokens (including `[CLS]`/`[SEP]`), 0 for padding |
| `token_type_ids` | Not an input; the model assumes a single segment |
| dtype | `int32` |

`tokenizer/golden_tokenization.json` lists the committed, invented fixture texts with the exact
unpadded `input_ids` Python produces. A Swift tokenizer is correct only if it reproduces those ids
for all examples, including the emoji, accent and punctuation cases. (The Hugging Face
`swift-transformers` package can read `tokenizer.json`; otherwise implement BERT basic
tokenization plus WordPiece over `vocab.txt`.)

## Model interface

| | name | shape | dtype |
|---|---|---|---|
| input | `input_ids` | `[1, 256]` | int32 |
| input | `attention_mask` | `[1, 256]` | int32 |
| output | `political_solicitation_probability` | `[1]` | float |

Batch size is fixed at 1 (one message at a time) and the sequence length is fixed at 256, which
keeps shapes static for Core ML. Messages shorter than 256 tokens are padded and masked, so the
result matches unpadded Python inference.

The output is a probability in [0, 1] of `political_solicitation`.

## Threshold

The decision threshold is **not** part of the graph. It lives in `metadata.json`:

```text
should_filter = political_solicitation_probability >= metadata.threshold
```

The threshold is chosen on the validation set (maximum recall subject to precision >= 0.95). For
the plain FP16 export it is 0.8700; for a compressed export it is **re-derived** from the
compressed model's own validation scores (see "Compression" below), because compression shifts
probabilities. Changing the threshold does not require re-exporting the model.

```text
String
  │
  ▼
tokenizer (see contract above)
  │
  ├── input_ids
  └── attention_mask
          │
          ▼
    veto.mlpackage
          │
          ▼
   probability
          │
          ▼
   metadata.threshold
          │
          ▼
    should_filter
```

**Heads-up on the FP16 threshold.** Veto's training-time threshold selection places the threshold
exactly at one validation positive's own score (a tie of zero margin). Any reduced-precision
runtime can flip that single message. The training-side rule is unchanged, but the compressed
export applies a safety margin: the threshold moves to the midpoint between that score and the
next lower validation score, which gives identical validation decisions with the largest margin.
The plain FP16 export still carries the zero-margin threshold.

## Precision and deployment target

- Format: Core ML **ML Program** (`mlprogram`), **FP16** compute precision.
- Minimum deployment target: **iOS 17** (default; `--deployment-target iOS16|iOS17|iOS18`).
- The plain export applies no quantization or palettization: FP16 comes to 45.2 MB. That is too
  large for the Message Filter extension; see "Compression" below.

## Compression (for the Message Filter extension)

> **Superseded on a device.** The 4-bit export below was still killed at 24 MB on an iPhone 14.
> Use the static-embedding model (next section) for the extension. Compression is kept as an
> option for the full app.

**Why.** iOS kills a Message Filter extension that exceeds a **24 MB** memory limit. On a real
iPhone 14 the FP16 model (about 45 MB, memory-mapped) produced:

```text
memorystatus: VetoMessageFilter exceeded mem limit: ActiveHard 24 MB (fatal)
Process VetoMessageFilter killed by jetsam reason per-process-limit
```

The extension died shortly after starting, so every message passed through unfiltered. The model
needs to be much smaller.

**How.** `--compression {none,int8,palettize4,palettize6}` applies coremltools weight compression
(`coremltools.optimize.coreml`) to the converted FP16 model. Only the weights change:
input/output names, shapes, dtypes and the tokenizer are identical, so the iOS side needs no
code change. `metadata.json` records the method (`precision` becomes e.g. `palettize4`,
`compute_precision` stays `fp16`, and a `compression` block gives method, mode and bits).

```bash
uv run veto export coreml --compression palettize4 --benchmark
```

Compression needs the private labeled corpus: the export compares the compressed model with the
FP16 export on the full validation set, re-derives the threshold from the compressed model's
validation scores (rule: max recall subject to precision >= 0.95, then moved to the midpoint of
the gap to the next lower validation score), and writes the new threshold to `metadata.json`
(`threshold_fp16_reference` keeps the old one).

### Measured on v1 (M2 Pro; validation and test each hold 1,807 messages, 24 positives)

| variant | model file | validation precision / recall / PR-AUC | held-out test precision / recall / PR-AUC | threshold | fixture decision flips (of 70) |
|---|---|---|---|---|---|
| FP16 (reference) | 45.2 MB | 1.000 / 1.000 / 1.000 | 1.000 / 0.792 / 0.949 | 0.8700 | 0 |
| **palettize4 (4-bit)** | **11.4 MB** | 0.960 / 1.000 / 0.997 | 0.852 / 0.958 / 0.966 | 0.6770 | 6 |
| palettize6 (6-bit) | 17.1 MB | 1.000 / 1.000 / 1.000 | 0.913 / 0.875 / 0.954 | 0.8245 | 1 |
| int8 | 22.8 MB | 1.000 / 1.000 / 1.000 | 0.913 / 0.875 / 0.953 | 0.8303 | 1 |

Each model is evaluated at its own threshold. Score deviation of 4-bit from FP16 on validation:
max 0.337, mean 0.008, p95 0.034 (FP16 vs FP32 is max 0.004).

**Choice: `palettize4`.** It is the only variant near the ~10 MB target (11.4 MB model, 12.4 MB
with tokenizer assets); 6-bit and int8 are 17 and 23 MB, too close to the 24 MB limit once
runtime memory is added. It passes the validation gates (precision 0.96 >= 0.95; recall equal to
FP16). It does **not** meet every acceptance criterion cleanly, and these exceptions are
deliberate and recorded in `metadata.json` (`validation.status = PASS_WITH_EXCEPTIONS`):

- **Fixtures: 6 of 70 decisions differ** from the FP16/Python reference. Every flip is a message
  the model scored between roughly 0.67 and 0.88, straddling the old and new thresholds, so the
  parity fixtures (which deliberately include borderline messages) cannot agree on all decisions
  once the threshold moves. The tolerance on the probability difference is therefore loose (0.5)
  and "within precision noise" is vacuous for this variant; the decision-based gates above carry
  the acceptance, not the fixture comparison.
- **Held-out test precision is 0.852** (4 false positives, 1 false negative), below the 0.95
  target that the threshold was tuned for on validation. FP16 at its threshold has no test false
  positives but misses 5 of 24 positives (recall 0.792). The compressed model's test PR-AUC
  (0.966) is not lower than FP16's (0.949), so ranking quality is intact; the operating point
  moved to a lower threshold and trades precision for recall. With 24 positives per split, one
  message is about 4 points of recall, so treat these numbers as rough.
- The model file is 11.4 MB against a target of "about 10 MB".

**Alternatives if this is not good enough:** `--compression palettize6` (17 MB, better held-out
fidelity but little headroom); shrinking the embedding table by pruning the vocabulary to tokens
seen in training, with an id remap inside the graph (keeps the iOS contract, but unseen tokens
would map to `[UNK]`, a modeling change); or a smaller distilled encoder. None of these were done.
Per-group 4-bit palettization was tried and rejected: it needs iOS 18 and produced a 29 MB model.

### Memory and latency

- File size is only a proxy for memory. `veto export coreml --benchmark` also loads the *compiled*
  model CPU-only in a fresh process (as the extension does) and records growth in resident memory
  and `phys_footprint` over a load plus 50 predictions.
- **On macOS this probe does not discriminate between the variants**: peak RSS growth was about
  34 MB for both FP16 and 4-bit, and `phys_footprint` growth was 5-6 MB for both, because Core ML
  memory-maps weights. The macOS numbers do not reproduce the iPhone jetsam limit, so whether the
  4-bit model fits in 24 MB has to be confirmed on a device.
- CPU-only latency (median / p95): 5.34 / 5.87 ms for 4-bit against 5.61 / 6.32 ms for FP16;
  2.20 ms median with all compute units. Mac measurements, not iPhone.

## Static-embedding model (for the Message Filter extension)

**Why compression was not enough.** On an iPhone 14 (iOS 27.0.1) both MiniLM exports were killed
inside the extension: the 45 MB FP16 model and the 11.4 MB 4-bit model. The extension's
`phys_footprint` was 2.4 MB before loading, 11.2 MB while the weight file was mapped, and then it
jumped past 24 MB in under a millisecond. In the full app (no limit), the 4-bit model on `cpuOnly`
cost about 10 MB to load and 13.5 MB after the first prediction. The likely cause is that Core ML
on the CPU expands compressed weights to 16-bit when loading them. That is not proven, but if it
is right, any compressed MiniLM variant fails the same way. The macOS memory probe showed only
about 5 MB, so **neither the Mac figure nor the file size shows whether a model fits.**

**Design.** `veto distill` builds a model with no transformer:

1. **Token vectors.** Every vocabulary token `t` gets MiniLM's mean-pooled output for
   `[CLS] t [SEP]`, using the trained v1 encoder. The vectors are reduced to 256 dimensions with
   PCA.
2. **Vocabulary.** Only the 6,367 tokens that occur in the training split are kept. The pruning is
   an id remap stored *inside the model*: a 30,522-entry table maps each token id to a row, and
   unseen tokens and `[PAD]`/`[CLS]`/`[SEP]`/`[MASK]` map to an all-zero row, so they are ignored.
3. **Message vector.** The sum of its token vectors (masked), L2-normalized. This equals the
   normalized mean.
4. **Classifier.** Logistic regression (`C = 10`, balanced classes), refit on the training split.

**Contract.** Nothing changes for `veto-ios`:
- Inputs, output, shapes and dtypes are identical.
- The tokenizer assets are byte-for-byte identical to the MiniLM export (`vocab.txt`,
  `tokenizer.json`, `tokenizer_config.json`, `golden_tokenization.json`).
- Token ids are unchanged, because the remap happens in the graph.
- Only the `threshold` value in `metadata.json` changes.

**Size and precision.**
- 1,630,465 parameters; the model file is 3.30 MB and the bundle 4.28 MB with the tokenizer.
- The embedding table is stored in FP16 (rounded to FP16 at training time) and only the lookup runs
  in FP16. The sum, normalization and classifier run in FP32 (`compute_precision`: "fp16 table
  lookup, fp32 arithmetic"). With FP16 everywhere the parity error reached 0.026 and one fixture
  flipped.
- The weights are not palettized or quantized, so there is nothing for Core ML to expand.

**How the configuration was chosen.** Variants (vocabulary, dimensions, token weighting, `C`)
were compared on validation. The rule, fixed before looking at the test split, was: the smallest
table with validation precision >= 0.95 and validation recall within 2 points of the MiniLM
baseline. The test split was only reported. Other variants seen:
- Full vocabulary, 128 dimensions with frequency (SIF) weighting: 7.8 MB, test recall 0.875, but
  validation recall only 0.917.
- Full vocabulary, 256 dimensions: 15.6 MB.

**Threshold.** The same rule as before: maximum recall subject to validation precision >= 0.95,
placed at the midpoint of the gap to the next lower validation score. The result is **0.8898**,
with a margin of 0.0138.

### Quality against the MiniLM baseline (each at its own threshold; 1,807 messages, 24 positives per split)

| split | model | precision | recall | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|
| validation | MiniLM baseline | 1.000 | 1.000 | 1.000 | 0 | 0 |
| validation | **static** | 1.000 | 1.000 | 1.000 | 0 | 0 |
| test (held-out) | MiniLM baseline | 1.000 | 0.792 | 0.951 | 0 | 5 |
| test (held-out) | **static** | 1.000 | 0.708 | 0.958 | 0 | 7 |

- On validation, the static model matches the baseline.
- On the held-out test split it catches 2 fewer of the 24 positives (recall 0.708 against 0.792)
  with no false positives. That is more than the 2-point target, and with 24 positives it is 2
  messages.
- PR-AUC is not lower, so ranking quality held; the threshold, tuned on validation, is
  conservative on test.
- The trade-off for size: tokens never seen in training are ignored, so new names and wording
  contribute nothing until the model is retrained.

### Parity and performance (M2 Pro)

| comparison | n | max diff | disagreements |
|---|---|---|---|
| Fixtures, Core ML vs Python (ALL and CPU only) | 70 | 0.00024 | 0 |
| Local corpus, Core ML vs Python | 3,614 | 0.00024 | 0 |
| FP32 Core ML reference | 3,614 | < 1e-6 | 0 |

- **Verdict:** PASS, with a tolerance of 0.01.
- **CPU-only inference:** 0.065 ms median, or 0.20 ms including Python tokenization. On the Mac,
  memory grew by 8 MB RSS and 3 MB footprint.
- **iPhone footprint: not measured yet.** That needs the device test in `veto-ios`.

## How to run

```bash
uv sync --extra coreml                  # once: installs coremltools
uv run veto export coreml               # exports models/veto -> dist/coreml and verifies it
```

Options:

| flag | default | meaning |
|---|---|---|
| `--model-dir` | `models/veto` | the trained model to export (the selected one) |
| `--output` | `dist/coreml` | output directory (replaced if it exists) |
| `--precision` | `fp16` | `fp16` or `fp32` |
| `--model-dir models/veto_static` | | export the static model made by `veto distill` (for the extension) |
| `--compression` | `none` | `none`, `int8`, `palettize4` or `palettize6` (see Compression) |
| `--deployment-target` | `iOS17` | `iOS16`, `iOS17` or `iOS18` |
| `--no-verify` | off | skip the parity check |
| `--local-sample N` | all | number of local validation/test messages to compare |
| `--benchmark` | off | measure latency on this Mac |

Core ML export and prediction require macOS. The export takes about 100 seconds (the parity check
on 3,614 messages dominates). `uv run pytest` runs the export tests only where `coremltools` is
installed.

## How parity is verified

The reference is Veto's own Python inference (`VetoClassifier`). Core ML is judged as a
deployment-equivalent copy, not as a new model. `veto export coreml` compares them on:

1. **Committed fixtures** (`src/veto/export/parity_fixtures.json`): 70 invented messages covering
   campaign solicitation, fundraising, advocacy, election reminders, candidate updates, ballot
   measures, polls, personal conversation, personal political conversation, transactional,
   commercial, spam, short and long messages, URLs, and STOP/END language.
2. **Threshold bands**: invented messages whose *Python* probability is well below, near and above
   the threshold (25 each), generated deterministically at run time.
3. **Local corpus** (private, never committed): all 3,614 validation and test messages.

Each comparison reports max, mean and p95 absolute probability difference, and the number of
`should_filter` disagreements. The command runs it for Core ML FP16 on all compute units, FP16 on
CPU only, and an **FP32 Core ML reference** (to separate precision effects from conversion bugs).
Only aggregates are written to `artifacts/coreml_export_report.json`; no message text.

### Acceptance criteria

- **Committed fixtures: 0 classification disagreements**, and max probability difference ≤ **0.01**
  (FP16) or ≤ 1e-4 (FP32).
- **Tolerance rationale:** observed FP16 maxima are 0.0071 (fixtures), 0.0061 (3,614 local
  messages) and 0.0036 (near threshold). 0.01 leaves about 1.4x headroom and is roughly 100x
  tighter than a real conversion bug (wrong pooling, mask or weights) would produce.
- **Elsewhere:** a disagreement is accepted ("explained") only if the message sits closer to the
  threshold than the largest probability deviation observed. Anything else fails.

### Results (current v1, M2 Pro, FP16)

| comparison | n | max diff | mean diff | p95 diff | disagreements |
|---|---|---|---|---|---|
| Fixtures (ALL units) | 70 | 0.0071 | 0.0011 | 0.0033 | **0** |
| Fixtures (CPU only) | 70 | 0.0061 | 0.0009 | 0.0036 | **0** |
| Band: below threshold | 25 | 0.0047 | 0.0022 | 0.0045 | 0 |
| Band: near threshold | 25 | 0.0036 | 0.0018 | 0.0035 | 2 (explained) |
| Band: above threshold | 25 | 0.0008 | 0.0003 | 0.0006 | 0 |
| Local corpus (ALL units) | 3,614 | 0.0061 | 0.00037 | 0.0015 | 1 (explained) |
| Local corpus (CPU only) | 3,614 | 0.0043 | 0.00017 | 0.0007 | 0 |
| **FP32 Core ML reference** | 3,614 | 1e-6 | 0.0 | 0.0 | **0** |

Findings:

- The FP32 Core ML model matches Python to about 1e-6 with no disagreements anywhere, so the
  wrapper and the conversion are exact. All deviations come from FP16 rounding.
- The near-threshold flips are messages whose Python probability is within 0.0002 of the
  threshold; the single flip among the 3,614 real messages is the validation message whose own
  score defines the threshold (see "Heads-up" above). No flip involves a message farther from the
  threshold than the FP16 noise.
- False positives and false negatives against the labels are unchanged (0 and 5) on the local
  set in FP32 and in FP16 on CPU. In FP16 on all compute units one extra false negative appears
  (the threshold-defining message).
- The verdict is therefore **PASS_WITH_NOTES**.

## Artifact size

| part | size |
|---|---|
| `veto.mlpackage` | 45.21 MB (45,207,168 bytes) |
| `tokenizer/` (4 files) | 0.98 MB |
| `metadata.json` | 3.7 KB |
| **Total deployment footprint** | **46.19 MB** |

Most of the size is the embedding table: about 11.9M of the model's 22.7M parameters.

## Performance (development Mac only)

Apple M2 Pro, 200 timed runs after 20 warm-up runs, batch size 1. **These are Mac measurements,
not iPhone measurements.**

| step | median | p95 |
|---|---|---|
| Tokenization (Python, Hugging Face fast tokenizer) | 0.15 ms | 0.19 ms |
| Core ML inference, all compute units | 2.24 ms | 2.56 ms |
| Tokenization + inference, all compute units | 2.43 ms | 2.82 ms |
| Core ML inference, CPU only | 5.61 ms | 6.32 ms |

Swift tokenization speed will differ from the Python figure.

## Current limitations

- **Not measured on an iPhone.** Latency, memory and Neural Engine behavior on device are untested.
  The extension's 24 MB limit is the reason for compression (above); the macOS memory probe
  cannot confirm that the compressed model fits, so check it on a device.
- **MiniLM does not fit the Message Filter extension on a device**, not even 4-bit. Use the static
  model (`veto distill`); its device footprint is still to be measured.
- **The compressed threshold is tuned on 24 validation positives**, so it is noisy; see the
  held-out test numbers above.
- **Fixed shape** `[1, 256]`: every message is padded to 256 tokens, so short messages cost the
  same as long ones. Enumerated sequence lengths are a possible later optimization.
- **FP16 resolution near the threshold**: messages within about 0.004 of the threshold can change
  decision versus Python. See the threshold note above.
- **Tokenizer fidelity is the consumer's job**: lower-casing, accent stripping and emoji-to-`[UNK]`
  behavior must be reproduced exactly (see the golden file).
- **Toolchain**: `coremltools` 9.0 officially supports up to torch 2.7; this project uses 2.14.1
  and works, with warnings. `torch.jit.trace` is deprecated in torch 2.14, so a future torch
  release may require switching the export path. `coremltools` also prints an internal
  "overflow encountered in cast" warning during conversion that does not affect results (parity
  was verified).
- **Fine-tuned model path** is implemented and tested on a tiny model but has not been exported
  from a trained fine-tuned model of real data.
- **Determinism**: exports of the same model give identical predictions; the `.mlpackage` files
  themselves are not byte-identical (they embed identifiers).
