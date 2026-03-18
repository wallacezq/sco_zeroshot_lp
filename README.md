# SmartCart — Zero-Shot Self-Checkout Fraud Detection

A real-time self-checkout monitoring system that detects product substitution fraud using a two-stage zero-shot vision pipeline. No model fine-tuning or retraining is required — new products can be added to the vocabulary at runtime.

## Architecture

```
Camera Frame
    │
    ▼
┌──────────────────────────────┐
│  Stage 1: YOLO-World         │   Detects any "object_held_at_hand"
│  (open-vocabulary detector)  │   in the frame (single class prompt)
└──────────┬───────────────────┘
           │  bounding boxes
           ▼
┌──────────────────────────────┐
│  Stage 2: OpenCLIP DFN5B     │   Zero-shot classifies each detected
│  ViT-H-14-378               │   object against the product vocabulary
└──────────┬───────────────────┘
           │  top-k predictions
           ▼
┌──────────────────────────────┐
│  Fraud Verdict Engine        │   Compares scanned barcode to CLIP
│  (top-1 or top-5 matching)   │   predictions → match / substitution
└──────────────────────────────┘
```

### Detection (Stage 1)

**YOLO-World** runs with a single open-vocabulary class — `"object_held_at_hand"` — so the detector fires on any hand-held object regardless of product category. Bounding boxes are tracked across frames using **BoxMOT ByteTrack** for stable IDs in the live view.

### Classification (Stage 2)

For each detected bounding box, a clean copy of the frame is annotated with the box (red rectangle) and submitted to **OpenCLIP DFN5B-CLIP-ViT-H-14-378** for zero-shot classification. Text embeddings for the full product vocabulary are computed once (using 8 prompt templates) and cached; only image encoding runs per frame.

### Fraud Verdict

When a barcode is scanned, the system captures a frame, runs the two-stage pipeline, and compares the scanned product name against the CLIP predictions:

| Mode   | Behaviour |
|--------|-----------|
| `top1` | Scanned product must be the #1 CLIP prediction (strict) |
| `top5` | Scanned product accepted if it appears anywhere in the top-5 (lenient) |

Verdicts: **match**, **substitution**, **no_item**, **verify_miss**, **verify_clear**.

## Features

- **Zero-shot** — no training data or fine-tuning required for new products
- **Runtime vocabulary** — add products via API; text embeddings rebuild automatically
- **Live MJPEG stream** — real-time camera feed with YOLO-World + ByteTrack overlays (coloured per track ID)
- **Barcode scanning** — scans trigger snapshot capture, classification, and fraud evaluation
- **Bagging area verification** — detects unscanned items visible in the frame
- **Shopping cart** — full cart management with quantity controls and totals
- **Fraud audit log** — timestamped event log of all scan verdicts
- **Configurable match mode** — switch between top-1 and top-5 at runtime

## Product Database

81 products across 4 categories:

| Category   | Examples |
|------------|----------|
| Fruit (28) | Apples (5 varieties), Banana, Avocado, Mango, Melon (4 types), … |
| Juice (10) | Tropicana, God Morgon, Bravo (apple/orange/grapefruit) |
| Dairy (21) | Milk, Yoghurt, Sour Cream, Oat Milk, Soy Milk, Soyghurt |
| Vegetables (22) | Bell Peppers (4 colours), Potatoes (3 types), Tomatoes (3 types), … |

## Requirements

- Python 3.10+
- Webcam (or video capture device)
- ~3 GB disk for OpenCLIP DFN5B weights (auto-downloaded on first run)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The server starts at **http://0.0.0.0:5000**. Open a browser to access the UI.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/start` | Start camera + model pipeline |
| `POST` | `/api/stop` | Stop pipeline and release camera |
| `GET`  | `/video_feed` | MJPEG live stream |
| `POST` | `/api/scan` | Scan barcode → capture + classify + verdict |
| `POST` | `/api/verify` | Verify bagging area for unscanned items |
| `GET`  | `/api/cart` | Get current cart contents |
| `POST` | `/api/cart/remove` | Decrement / remove item from cart |
| `POST` | `/api/cart/clear` | Clear entire cart |
| `GET`  | `/api/fraud_log` | Get fraud audit log (last 30 events) |
| `POST` | `/api/fraud_log/clear` | Clear fraud log |
| `GET`  | `/api/vocab` | Get current product vocabulary |
| `POST` | `/api/vocab/add` | Add product to zero-shot vocabulary |
| `GET`  | `/api/status` | System status (models, camera, config) |
| `GET`  | `/api/match_mode` | Get current match mode |
| `POST` | `/api/match_mode` | Set match mode (`top1` / `top5`) |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CONF_THRESHOLD` | `0.25` | YOLO-World detection confidence threshold |
| `TAX_RATE` | `0.06` | Tax rate applied at checkout |
| `MATCH_MODE` | `"top1"` | Fraud matching mode (`top1` or `top5`) |

## Models

| Component | Model | Source |
|-----------|-------|--------|
| Detector | YOLO-World v2 (Medium) | `yolov8m-worldv2.pt` via ultralytics |
| Classifier | DFN5B-CLIP-ViT-H-14-378 | `apple/DFN5B-CLIP-ViT-H-14-378` via HuggingFace |
| Tracker | ByteTrack | BoxMOT |
| Inference | OpenVINO (FP16 / INT8) | optimum-intel |
