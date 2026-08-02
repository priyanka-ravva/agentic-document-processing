# Evaluation

Run the evaluation suite from the project root:

```bash
python evaluation/evaluate.py
```

The script reads `evaluation/scenarios.json`, runs each document through the
LangGraph workflow, saves run traces under `logs/runs/`, and writes aggregate
results to `evaluation/evaluation_results.json`.

Current scenarios:

- `invoice_simple`: searchable invoice PDF
- `medical_discharge`: searchable medical discharge PDF
- `contract_nda`: searchable contract PDF
- `invoice_image_ocr`: image invoice requiring OCR or vision fallback
- `unknown_test_sample`: unknown-category PDF sample
- `invoice_image_ocr_0007`: additional image invoice requiring OCR or vision fallback

All current evaluation scenarios use files from `sample_docs/`.
