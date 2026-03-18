"""
SmartCart — Self-Checkout Flask Application  v6
================================================
Detection engine : YOLO-World (zero-shot open-vocabulary)
Classifier       : OpenCLIP  DFN5B-CLIP_ViT-H-14-378  (zero-shot)
─────────────────────────────────────────────────────────────────
Two-stage pipeline
──────────────────
Stage 1 – YOLO-World detects every "object_held_at_hand" in the frame.
           A single open-vocabulary class is used so the detector fires on
           any hand-held object regardless of category.

Stage 2 – For EACH detected bounding box:
           a. Copy the original frame.
           b. Draw the YOLO bounding box (red) onto that copy.
           c. Submit the full annotated frame to OpenCLIP for zero-shot
              classification against the product vocabulary text prompts.
              "A photo of a <product name>" is computed once per product
              and cached as text embeddings; only the image encoding changes
              per frame.
           d. Replace the raw YOLO label with the top-1 CLIP match.

The annotated frame is then base64-encoded and returned to the UI.

OpenCLIP model
──────────────
  Model  : DFN5B-CLIP_ViT-H-14-378
  Source : open_clip  (pip install open_clip_torch)
  Weights: auto-downloaded from HuggingFace on first use (~3 GB)
  Input  : 378 × 378 px images
  Why    : DFN5B data-filtered training gives state-of-the-art zero-shot
           accuracy on diverse visual domains without any fine-tuning.
"""

import base64
import time
import threading
import random
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from optimum.intel.openvino import OVModelOpenCLIPForZeroShotImageClassification, OVWeightQuantizationConfig
from optimum.intel.openvino import OVModelOpenCLIPText, OVModelOpenCLIPVisual
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from tqdm import tqdm
import torch
import torch.nn.functional as F

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify, request, stream_with_context

# ── OpenCLIP  DFN5B-CLIP_ViT-H-14-378  zero-shot classifier ──────────────────
#
# Model card  : https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378
# open_clip id: model_name="ViT-H-14-378-quickgelu"
#               pretrained="dfn5b"
#
# Weights (~3 GB) are auto-downloaded to ~/.cache/huggingface on first use.
#
# Zero-shot protocol
# ──────────────────
#   1. Encode all candidate class names once as text embeddings (cached).
#      Template: "a photo of a {label}"
#   2. Per frame: encode the image → cosine-similarity against text embeddings
#      → softmax → top-1 label + probability.
#   3. The candidate label list is rebuilt whenever ALL_PRODUCT_NAMES changes
#      (e.g. via /api/vocab/add).  Call rebuild_clip_text_features() after any
#      such change.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import open_clip
    from PIL import Image as PILImage

    _CLIP_MODEL_NAME = "ViT-H-14-378-quickgelu"
    _CLIP_MODEL_ID   = "apple/DFN5B-CLIP-ViT-H-14-378"
    _CLIP_PRETRAINED = "dfn5b"
    _CLIP_device     = "GPU"
    _CLIP_quantized  = False
    _CLIP_base_dir   = Path(f"{_CLIP_MODEL_ID.split('/')[-1]}-openclip")
    
    if _CLIP_quantized:
        model_dir = _CLIP_base_dir / "INT8"
        if not model_dir.exists():
            OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(_CLIP_MODEL_ID,
                quantization_config=OVWeightQuantizationConfig(bits=8)
            ).save_pretrained(model_dir)
    else:
        model_dir = _CLIP_base_dir / "FP16"
        if not model_dir.exists():
            OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(_CLIP_MODEL_ID).save_pretrained(model_dir)
 
    _clip_model_transformer = CLIPModel.from_pretrained(_CLIP_MODEL_ID) 
    _clip_model = OVModelOpenCLIPVisual.from_pretrained(model_dir, device=_CLIP_device)
    _clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
    _clip_tokenizer = CLIPTokenizer.from_pretrained(_CLIP_MODEL_ID)

    #_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    #    _CLIP_MODEL_NAME, pretrained=_CLIP_PRETRAINED
    #)
    #_clip_model.eval()
    #_clip_tokenizer = open_clip.get_tokenizer(_CLIP_MODEL_NAME)

    # Text feature cache — populated by rebuild_clip_text_features()
    _clip_text_features: torch.Tensor | None = None   # (C, D) normalised
    _clip_text_labels:   list[str]           = []     # matches row order

    CLIP_AVAILABLE = True
    print(f"[INFO] OpenCLIP {_CLIP_MODEL_NAME} / {_CLIP_PRETRAINED} loaded.")

except Exception as _clip_err:
    CLIP_AVAILABLE = False
    _clip_text_features = None
    _clip_text_labels   = []
    print(f"[WARN] OpenCLIP unavailable – classifier disabled. ({_clip_err})")


def rebuild_clip_text_features(labels: list[str]) -> None:
    """
    Encode `labels` as CLIP text embeddings and cache them.
    Template: "a photo of a {label}"
    Call this once at startup and again whenever the label list changes.
    """
    global _clip_text_features, _clip_text_labels
    zeroshot_weights = []
    class_templates =  [
          "a photo of a {label}.",
          "a product photo of a {label}.",
          "a retail image of a {label}.",
          "a picture of a {label}.",
          "an image of a {label}.",
          "{label} for sale.",
          "a shelf photo of a {label}.",
          "a close-up photo of a {label}."]
      
    if not CLIP_AVAILABLE or not labels:
        return
        
    zeroshot_weights_pth = Path("clip_zeroshot_cls.pth")
    
    if zeroshot_weights_pth.exists():
        _clip_text_features = torch.load("clip_zeroshot_cls.pth", map_location="cpu")
        _clip_text_labels   = list(labels)
    else:
        try:
            for label in tqdm(labels):
            #prompts = [f"a photo of a {lbl}" for lbl in labels]
                prompts = [template.format(label=label) for template in class_templates]
                texts = _clip_processor(text=prompts, return_tensors="pt", padding=True)
                with torch.no_grad():
                    # openvino acceleration for openclip text-encoding not working, fallback to transformer for now
                    class_embeddings = _clip_model_transformer.get_text_features(**texts)
                    class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1)        
            _clip_text_features = zeroshot_weights
            _clip_text_labels   = list(labels)
            torch.save(_clip_text_features, zeroshot_weights_pth)
            print(f"[INFO] CLIP text features built for {len(labels)} labels: {labels}")
        except Exception as exc:
            print(f"[WARN] rebuild_clip_text_features error: {exc}")

        print(f"[INFO] zeroshot_weights's shape: {zeroshot_weights.shape}")
        print(f"[INFO] zeroshot_weights: {zeroshot_weights}")

def classify_frame(bgr_frame, top_k: int = 5) -> list[dict]:
    """
    Zero-shot classify a full BGR OpenCV frame using
    OpenCLIP DFN5B-CLIP_ViT-H-14-378.

    The frame should already have the YOLO bounding box drawn on it so the
    model sees the highlighted object in its full scene context.

    Returns a list of up to `top_k` dicts sorted by probability descending:
      [{"label": str, "conf": float, "rank": int}, ...]

    Returns [{"label": "unknown", "conf": 0.0, "rank": 1}] on any failure.
    """
    if (not CLIP_AVAILABLE
            or _clip_text_features is None
            or bgr_frame is None
            or bgr_frame.size == 0):
        return [{"label": "unknown", "conf": 0.0, "rank": 1}]
    try:
        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil    = PILImage.fromarray(rgb)
        tensor = _clip_preprocess(pil).unsqueeze(0)    # (1, 3, 378, 378)

        with torch.no_grad():
            img_feat = _clip_model.encode_image(tensor)            # (1, D)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logit_scale = _clip_model.logit_scale.exp()
            logits      = logit_scale * img_feat @ _clip_text_features.T  # (1, C)
            probs       = torch.softmax(logits[0], dim=0)

        k       = min(top_k, len(_clip_text_labels))
        top_v, top_i = probs.topk(k)
        results = []
        for rank, (idx, prob) in enumerate(zip(top_i.tolist(), top_v.tolist()), start=1):
            label = _clip_text_labels[idx] if idx < len(_clip_text_labels) else "unknown"
            results.append({"label": label, "conf": round(prob, 3), "rank": rank})
        return results

    except Exception as exc:
        print(f"[WARN] OpenCLIP classify_frame error: {exc}")
        return [{"label": "unknown", "conf": 0.0, "rank": 1}]

# ── BoxMOT ByteTrack ──────────────────────────────────────────────────────────
try:
    from boxmot import ByteTrack
    import numpy as _np
    BYTETRACK_AVAILABLE = True
    print("[INFO] BoxMOT ByteTrack available.")
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("[WARN] boxmot not installed – tracking IDs disabled in live view.")

# ── YOLO-World import ─────────────────────────────────────────────────────────
try:
    from ultralytics import YOLOWorld
    YOLO_AVAILABLE = True
    print("[INFO] ultralytics YOLOWorld available.")
except ImportError:
    try:
        # Older ultralytics versions expose it as YOLO with a world checkpoint
        from ultralytics import YOLO as YOLOWorld   # type: ignore
        YOLO_AVAILABLE = True
        print("[INFO] ultralytics YOLO (world compat) available.")
    except ImportError:
        YOLO_AVAILABLE = False
        print("[WARN] ultralytics not installed – running in DEMO mode.")

app = Flask(__name__)

from products import PRODUCT_DB, ALL_PRODUCT_NAMES, PRODUCT_DISPLAY_NAMES, _db_entry

# Build CLIP text embeddings for the initial product vocabulary.
# This is a no-op when CLIP is unavailable; safe to call at import time.
rebuild_clip_text_features(ALL_PRODUCT_NAMES)

# Confidence threshold for YOLO-World detections
CONF_THRESHOLD = 0.25   # lower than standard YOLO because zero-shot scores are softer

TAX_RATE = 0.06

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SnapResult:
    """Result of YOLO-World inference on one captured frame."""
    detections:   list[dict]   # [{label, conf, box}] sorted by conf desc
    top_class:    str          # highest-conf detected name, or ""
    top_conf:     float
    classes_used: list[str]    # the text classes passed to YOLO-World
    snapshot_b64: str          # annotated JPEG, base64-encoded
    ts:           float = field(default_factory=time.time)


@dataclass
class FraudEvent:
    kind:         str    # "match" | "substitution" | "no_item" | "verify_miss" | "verify_clear"
    scanned_name: str
    detected_top: str
    barcode:      str
    confidence:   float
    classes_used: list[str]
    snapshot_b64: str
    ts:           float = field(default_factory=time.time)

    def to_dict(self):
        d = asdict(self)
        d["age"] = round(time.time() - self.ts, 1)
        return d

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
camera_lock  = threading.Lock()
state_lock   = threading.Lock()
# Serialise all YOLO-World calls — set_classes + predict must be atomic
model_lock   = threading.Lock()

camera:     Optional[cv2.VideoCapture] = None
model       = None           # YOLOWorld instance
is_running  = False

cart_items: dict[str, dict] = {}          # name → {count, unit_price, barcode}
fraud_log:  list[FraudEvent] = []         # audit log, newest first
last_snap:  Optional[SnapResult] = None
active_alert: dict = {"kind": None, "msg": "", "ts": 0}

# ── Fraud detection match mode ────────────────────────────────────────────────
# "top1"  → the barcode must match the single highest-confidence CLIP prediction
# "top5"  → the barcode is accepted if it matches ANY of the top-5 predictions
#           (more lenient; useful when classifier is uncertain between close items)
MATCH_MODE: str = "top1"   # default; toggled at runtime via /api/match_mode

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def open_camera(index: int = 0) -> bool:
    global camera
    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        camera = cap
    return True


def release_camera():
    global camera
    with camera_lock:
        if camera:
            camera.release()
            camera = None


def grab_frame():
    """Grab one raw frame (thread-safe)."""
    with camera_lock:
        if camera is None or not camera.isOpened():
            return None
        ret, frame = camera.read()
    return frame if ret else None

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_model(weights: str = "yolov8m-worldv2.pt") -> bool:
    """
    Load a YOLO-World checkpoint.
    The model is always queried with the single open-vocab class
    "object_held_at_hand"; OpenCLIP DFN5B handles zero-shot classification.
    """
    global model
    if not YOLO_AVAILABLE:
        return False
    try:
        with model_lock:
            m = YOLOWorld(weights)
            m.set_classes(["object_held_at_hand"])   # single detection class
            model = m
        print(f"[INFO] YOLO-World loaded: {weights}")
        print(f"[INFO] Detection class: 'object_held_at_hand'")
        print(f"[INFO] Classifier: OpenCLIP {_CLIP_MODEL_NAME}/{_CLIP_PRETRAINED}"
              f"  (available={CLIP_AVAILABLE})")
        return True
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# ZERO-SHOT INFERENCE  (called ONCE per scan event)
# ─────────────────────────────────────────────────────────────────────────────
_demo_labels = list(ALL_PRODUCT_NAMES) or [
    "apple", "banana", "water bottle", "donut", "hot dog"
]

def _fake_single(frame, classes: list[str]) -> list[dict]:
    """Deterministic fake detections for demo mode (no model installed)."""
    h, w = frame.shape[:2]
    rng   = random.Random(int(time.time() * 2))
    label = rng.choice(classes)
    conf  = round(rng.uniform(0.52, 0.94), 2)
    x1 = rng.randint(80, w // 2)
    y1 = rng.randint(80, h // 2)
    x2 = min(x1 + rng.randint(130, 260), w - 20)
    y2 = min(y1 + rng.randint(130, 260), h - 20)
    return [{"label": label, "conf": conf, "box": [x1, y1, x2, y2]}]


def run_yolo_world(frame, classes: list[str] | None = None) -> list[dict]:
    """
    Two-stage pipeline
    ──────────────────
    Stage 1 – YOLO-World detects every "object_held_at_hand" in the frame.
              `classes` parameter is IGNORED – we always use the single open-
              vocabulary class so the detector fires on ANY hand-held object.

    Stage 2 – For each detected box, a clean copy of the original frame is
              made, the red bounding box is drawn on it, and the full
              annotated frame is submitted to EfficientNet-B0. The ImageNet
              top-1 prediction replaces the raw YOLO placeholder label.

    Returns list of dicts:
      {label, conf, yolo_conf, effnet_conf, box}
    """
    DETECTION_CLASS = ["object_held_at_hand"]   # YOLO-World open-vocab prompt

    # ── Stage 1: YOLO-World detection ────────────────────────────────────────
    if not YOLO_AVAILABLE or model is None:
        raw_boxes = _fake_single(frame, DETECTION_CLASS)
    else:
        with model_lock:
            model.set_classes(DETECTION_CLASS)
            results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

        result    = results[0]
        raw_boxes = []
        for box in result.boxes:
            yolo_conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            raw_boxes.append({
                "label":      "object_held_at_hand",   # placeholder
                "conf":       round(yolo_conf, 2),
                "yolo_conf":  round(yolo_conf, 2),
                "box":        [x1, y1, x2, y2],
            })

    # ── Stage 2: OpenCLIP zero-shot classification on box-annotated full frame ─
    # For EACH detected bounding box:
    #   a. Make a clean copy of the original frame.
    #   b. Draw the YOLO bounding box (red) onto that copy.
    #   c. Submit the full annotated frame to OpenCLIP DFN5B-CLIP_ViT-H-14-378
    #      for zero-shot classification against ALL_PRODUCT_NAMES text prompts.
    #      Top-5 predictions are stored; the active MATCH_MODE decides which
    #      ranks are used for fraud verdict in capture_and_evaluate().
    # This gives the classifier full scene context with the object highlighted.
    BOX_COLOR = (0, 0, 220)   # red (BGR) — same as annotation step

    refined: list[dict] = []
    for det in raw_boxes:
        x1, y1, x2, y2 = det["box"]

        annotated_frame = frame.copy()
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, 3)

        top_preds = classify_frame(annotated_frame, top_k=5)   # always get top-5
        top1      = top_preds[0] if top_preds else {"label": "unknown", "conf": 0.0, "rank": 1}

        refined.append({
            "label":       top1["label"],      # top-1 label (primary display)
            "conf":        top1["conf"],        # top-1 confidence
            "yolo_conf":   det.get("yolo_conf", det["conf"]),
            "clip_conf":   top1["conf"],
            "top_preds":   top_preds,           # full top-5 list for UI + verdict
            "box":         det["box"],
        })

    refined.sort(key=lambda d: d["clip_conf"], reverse=True)
    return refined

# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOT ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────
_VERDICT_COLORS = {
    "match":        (0, 230, 118),    # green
    "substitution": (30,  30, 220),   # red
    "no_item":      (0,  165, 255),   # amber
    "verify_miss":  (0,  165, 255),   # amber
    "verify_clear": (0, 230, 118),    # green
}

def annotate_snapshot(frame, detections: list[dict],
                      verdict: str, verdict_msg: str,
                      highlight_name: str = "") -> str:
    """
    Draw RED bounding boxes for every detection.
    Label shows top-1 CLIP class + confidence scores.
    Returns base64 JPEG string.
    """
    BOX_COLOR   = (0, 0, 220)
    LABEL_BG    = (0, 0, 180)
    LABEL_TEXT  = (255, 255, 255)

    h, w = frame.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["box"]

        clip_label = det.get("label", "unknown")
        clip_conf  = det.get("clip_conf", det.get("conf", 0.0))
        yolo_conf  = det.get("yolo_conf", det.get("conf", 0.0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        tag_top = f"#{1} {clip_label}"
        tag_bot = f"clip:{clip_conf:.0%}  det:{yolo_conf:.0%}"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness  = 1
        pad        = 4

        (tw1, th1), _ = cv2.getTextSize(tag_top, font, font_scale, thickness)
        (tw2, th2), _ = cv2.getTextSize(tag_bot, font, font_scale, thickness)
        lw = max(tw1, tw2) + 2 * pad
        lh = th1 + th2 + 3 * pad

        lx1, ly1 = x1, max(0, y1 - lh)
        lx2, ly2 = x1 + lw, y1
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), LABEL_BG, -1)

        cv2.putText(frame, tag_top, (lx1 + pad, ly1 + pad + th1),
                    font, font_scale, LABEL_TEXT, thickness, cv2.LINE_AA)
        cv2.putText(frame, tag_bot, (lx1 + pad, ly2 - pad),
                    font, font_scale, LABEL_TEXT, thickness, cv2.LINE_AA)

    # Verdict banner
    v_color = _VERDICT_COLORS.get(verdict, (80, 80, 80))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-52), (w, h), v_color, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.putText(frame, verdict_msg, (10, h-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    # Watermark
    cv2.putText(frame,
                f"YOLO-World → OpenCLIP DFN5B  |  {time.strftime('%H:%M:%S')}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (180, 180, 180), 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""

# ─────────────────────────────────────────────────────────────────────────────
# CORE: CAPTURE SNAPSHOT + YOLO-WORLD + EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def capture_and_evaluate(expected_name: str, barcode: str) -> dict:
    """
    Called on every barcode scan.

    Detection:      YOLO-World with class "object_held_at_hand"
    Classification: OpenCLIP DFN5B-CLIP_ViT-H-14-378 zero-shot (top-5 stored)
    Verdict:        controlled by MATCH_MODE
      "top1" → expected_name must match the #1 CLIP prediction
      "top5" → expected_name must appear anywhere in the top-5 predictions
    """
    global last_snap, active_alert

    frame = grab_frame()
    if frame is None:
        return {"verdict": "error", "msg": "Camera unavailable"}

    detections = run_yolo_world(frame)

    top1_label   = detections[0]["label"]              if detections else ""
    top1_conf    = detections[0]["conf"]               if detections else 0.0
    top_preds    = detections[0].get("top_preds", [])  if detections else []
    classes      = ["object_held_at_hand"]

    # Display-friendly name for verdict messages (last path segment)
    display_name = expected_name.split("/")[-1]

    # ── Verdict — respects MATCH_MODE ────────────────────────────────────────
    def _name_matches(label: str) -> bool:
        """
        Match expected_name (full path) against a CLIP label (also full path).
        Checks both full-path equality and last-segment substring match so that
        e.g. "Fruit/Apple/Granny-Smith" matches a label of "Granny-Smith".
        """
        exp_seg = expected_name.split("/")[-1].lower().replace("-", " ")
        lbl_seg = label.split("/")[-1].lower().replace("-", " ")
        return (expected_name.lower() == label.lower()
                or exp_seg in lbl_seg or lbl_seg in exp_seg)

    if not detections:
        verdict     = "no_item"
        verdict_msg = f"⚠ NO ITEM IN FRAME  (scanned: {display_name})"

    elif MATCH_MODE == "top5":
        matched_rank = next(
            (p["rank"] for p in top_preds if _name_matches(p["label"])), None
        )
        if matched_rank is not None:
            matched_pred = top_preds[matched_rank - 1]
            mp_disp      = matched_pred["label"].split("/")[-1]
            verdict      = "match"
            verdict_msg  = (f"✓ MATCH (Top-{matched_rank})  {display_name}  "
                            f"→ CLIP #{matched_rank}: {mp_disp}"
                            f"  [{matched_pred['conf']:.0%}]")
        else:
            top5_disp   = ", ".join(p["label"].split("/")[-1] for p in top_preds)
            verdict     = "substitution"
            verdict_msg = (f"🚨 SUBSTITUTION  scanned '{display_name}'"
                           f"  CLIP top-5: {top5_disp}")
    else:
        top1_disp = top1_label.split("/")[-1]
        if _name_matches(top1_label):
            verdict     = "match"
            verdict_msg = (f"✓ MATCH (Top-1)  {display_name}  "
                           f"→ CLIP: {top1_disp}  [{top1_conf:.0%}]")
        else:
            verdict     = "substitution"
            verdict_msg = (f"🚨 SUBSTITUTION  scanned '{display_name}'"
                           f"  CLIP saw '{top1_disp}'  [{top1_conf:.0%}]")

    snap_b64 = annotate_snapshot(
        frame.copy(), detections, verdict, verdict_msg, expected_name
    )

    snap = SnapResult(
        detections=detections, top_class=top1_label, top_conf=top1_conf,
        classes_used=classes, snapshot_b64=snap_b64
    )
    evt = FraudEvent(
        kind=verdict, scanned_name=expected_name, detected_top=top1_label,
        barcode=barcode, confidence=top1_conf,
        classes_used=classes, snapshot_b64=snap_b64
    )

    with state_lock:
        last_snap = snap
        fraud_log.insert(0, evt)
        if len(fraud_log) > 100:
            del fraud_log[100:]
        if verdict in ("substitution", "no_item"):
            active_alert.update({"kind": verdict, "msg": verdict_msg, "ts": time.time()})
        else:
            active_alert.update({"kind": None, "msg": "", "ts": 0})

    return {
        "verdict":      verdict,
        "msg":          verdict_msg,
        "match_mode":   MATCH_MODE,
        "display_name": display_name,
        "top_class":    top1_label,
        "top_conf":     top1_conf,
        "top_preds":    top_preds,
        "classes_used": classes,
        "snapshot_b64": snap_b64,
        "detections":   detections,
    }


def verify_bagging_area() -> dict:
    """
    Manual verify: detect ALL hand-held objects with YOLO-World
    ("object_held_at_hand"), classify each with EfficientNet-B0,
    and flag anything whose EfficientNet label is not already in the cart.
    """
    global last_snap, active_alert

    frame = grab_frame()
    if frame is None:
        return {"verdict": "error", "msg": "Camera unavailable"}

    detections = run_yolo_world(frame)   # uses "object_held_at_hand"
    classes    = ["object_held_at_hand"]

    # Which product full-path names are already in the cart?
    with state_lock:
        in_cart = set(cart_items.keys())

    # An item is "unscanned" when its CLIP label doesn't match any cart item.
    # Matching compares the last segment of both the CLIP label and cart names
    # (since CLIP returns full-path labels from the product vocab).
    def _in_cart(label: str) -> bool:
        lbl_seg = label.split("/")[-1].lower().replace("-", " ")
        for cart_name in in_cart:
            cart_seg = cart_name.split("/")[-1].lower().replace("-", " ")
            if cart_name.lower() == label.lower() or lbl_seg in cart_seg or cart_seg in lbl_seg:
                return True
        return False

    unscanned = [d for d in detections if not _in_cart(d["label"])]

    if not unscanned:
        verdict     = "verify_clear"
        verdict_msg = "✓ VERIFY OK — all visible items accounted for"
    else:
        # Show display names (last segment) in the verdict message
        disp_labels = ", ".join(sorted({d["label"].split("/")[-1] for d in unscanned}))
        verdict     = "verify_miss"
        verdict_msg = f"⚠ UNSCANNED ITEM(S): {disp_labels}"

    snap_b64 = annotate_snapshot(
        frame.copy(), detections, verdict, verdict_msg
    )
    snap = SnapResult(
        detections=detections,
        top_class=unscanned[0]["label"] if unscanned else "",
        top_conf=unscanned[0]["conf"]   if unscanned else 0.0,
        classes_used=classes, snapshot_b64=snap_b64
    )

    evts = [
        FraudEvent(
            kind="verify_miss", scanned_name="", detected_top=d["label"],
            barcode="", confidence=d["conf"],
            classes_used=classes, snapshot_b64=snap_b64
        )
        for d in unscanned
    ]

    with state_lock:
        last_snap = snap
        for e in evts:
            fraud_log.insert(0, e)
        if len(fraud_log) > 100:
            del fraud_log[100:]
        if verdict == "verify_miss":
            active_alert.update({"kind": verdict, "msg": verdict_msg, "ts": time.time()})
        else:
            active_alert.update({"kind": None, "msg": "", "ts": 0})

    return {
        "verdict":      verdict,
        "msg":          verdict_msg,
        "unscanned":    [d["label"] for d in unscanned],
        "classes_used": classes,
        "snapshot_b64": snap_b64,
        "detections":   detections,
    }

# ─────────────────────────────────────────────────────────────────────────────
# LIVE DETECTION + TRACKING STATE
# Tracks from the most recent YOLO-World + ByteTrack pass, shared between
# the background detection thread and the MJPEG stream generator.
# Each entry: {box: [x1,y1,x2,y2], yolo_conf: float, track_id: int}
# ─────────────────────────────────────────────────────────────────────────────
live_boxes: list[dict] = []
live_boxes_lock = threading.Lock()

# ByteTrack instance — created once and reused across frames so it can
# maintain internal Kalman state and assign stable IDs across frames.
_tracker = ByteTrack() if BYTETRACK_AVAILABLE else None


def _live_detection_loop():
    """
    Background thread: runs YOLO-World + ByteTrack on the camera feed and
    publishes tracked bounding boxes (with stable track IDs) to `live_boxes`.

    Pipeline per frame
    ──────────────────
    1. YOLO-World detects every "object_held_at_hand" → raw boxes + confidences.
    2. Detections are formatted as an (N, 6) numpy array [x1,y1,x2,y2,conf,cls]
       and passed to ByteTrack.update().
    3. ByteTrack returns active tracks as (M, 7) [x1,y1,x2,y2,track_id,conf,cls].
       It interpolates missing detections via Kalman prediction so IDs survive
       brief occlusions / missed frames.
    4. The resulting tracks are written to `live_boxes` (under lock).

    EfficientNet is NOT called here — classification only happens on explicit
    scan/verify events to keep the stream fast.
    """
    DETECTION_CLASS = ["object_held_at_hand"]
    LIVE_CONF       = 0.20   # slightly lower threshold for live view
    INFERENCE_DELAY = 0.08   # ~12 fps inference cadence

    while is_running:
        frame = grab_frame()
        if frame is None or not (YOLO_AVAILABLE and model is not None):
            time.sleep(INFERENCE_DELAY)
            continue

        try:
            # ── Stage 1: YOLO-World detection ────────────────────────────────
            with model_lock:
                model.set_classes(DETECTION_CLASS)
                results = model.predict(frame, conf=LIVE_CONF, verbose=False)

            # Build (N, 6) array: [x1, y1, x2, y2, conf, cls_id]
            raw = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                raw.append([x1, y1, x2, y2, conf, 0])   # cls_id=0 (single class)

            # ── Stage 2: ByteTrack update ────────────────────────────────────
            if _tracker is not None:
                dets_np = _np.array(raw, dtype=float) if raw else _np.empty((0, 6))
                # tracks shape: (M, 7) → [x1, y1, x2, y2, track_id, conf, cls]
                tracks = _tracker.update(dets_np, frame)
            else:
                # Fallback: no tracker — assign a dummy track_id of -1
                tracks = [
                    [r[0], r[1], r[2], r[3], -1, r[4], r[5]] for r in raw
                ]

            tracked_boxes = []
            for t in tracks:
                x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                track_id       = int(t[4])
                conf           = round(float(t[5]), 2)
                tracked_boxes.append({
                    "box":       [x1, y1, x2, y2],
                    "yolo_conf": conf,
                    "track_id":  track_id,
                })

            with live_boxes_lock:
                live_boxes.clear()
                live_boxes.extend(tracked_boxes)

        except Exception as exc:
            print(f"[WARN] Live detection/tracking error: {exc}")

        time.sleep(INFERENCE_DELAY)


# ─────────────────────────────────────────────────────────────────────────────
# MJPEG STREAM  — draws live YOLO-World + ByteTrack boxes on every frame
# ─────────────────────────────────────────────────────────────────────────────
_live_det_thread: threading.Thread | None = None

# Distinct BGR colours for up to 20 simultaneous track IDs (cycles if more)
_TRACK_COLOURS = [
    (0,   0,   220),   # red
    (220, 0,   0  ),   # blue
    (0,   180, 0  ),   # green
    (0,   180, 220),   # yellow
    (180, 0,   180),   # magenta
    (0,   140, 255),   # orange
    (255, 0,   140),   # pink
    (140, 255, 0  ),   # lime
    (0,   255, 200),   # teal-yellow
    (200, 100, 0  ),   # indigo
]


def _track_colour(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR colour for a given track ID."""
    return _TRACK_COLOURS[abs(track_id) % len(_TRACK_COLOURS)]


def frame_generator():
    global _live_det_thread

    # Start the background detection+tracking thread on first stream open
    if _live_det_thread is None or not _live_det_thread.is_alive():
        _live_det_thread = threading.Thread(
            target=_live_detection_loop, daemon=True, name="live-det"
        )
        _live_det_thread.start()

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    while is_running:
        frame = grab_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]

        # ── Overlay ByteTrack results ─────────────────────────────────────────
        with live_boxes_lock:
            current_boxes = list(live_boxes)

        for det in current_boxes:
            x1, y1, x2, y2 = det["box"]
            conf     = det["yolo_conf"]
            track_id = det.get("track_id", -1)
            color    = _track_colour(track_id)

            # Bounding box in track colour
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label: "ID 3 · object_held_at_hand  82%"
            tid_str = f"ID {track_id}" if track_id >= 0 else "untracked"
            tag     = f"{tid_str}  \u00b7  object_held_at_hand  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.46, 1)
            pad  = 3
            lx1  = x1
            ly1  = max(0, y1 - th - 2 * pad)
            cv2.rectangle(frame, (lx1, ly1), (lx1 + tw + 2 * pad, y1), color, -1)

            # Dark text for readability on coloured bg
            lum = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
            txt_color = (10, 10, 10) if lum > 128 else (255, 255, 255)
            cv2.putText(frame, tag, (lx1 + pad, y1 - pad),
                        FONT, 0.46, txt_color, 1, cv2.LINE_AA)

        # ── Alert banner ─────────────────────────────────────────────────────
        with state_lock:
            alert = dict(active_alert)

        if alert["kind"]:
            if time.time() - alert["ts"] < 8.0:
                col = (30, 30, 200) if alert["kind"] == "substitution" else (0, 100, 220)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h-54), (w, h), col, -1)
                cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
                cv2.putText(frame, alert["msg"], (10, h-16),
                            FONT, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                with state_lock:
                    active_alert.update({"kind": None, "msg": "", "ts": 0})

        # ── Watermark ─────────────────────────────────────────────────────────
        n   = len(current_boxes)
        trk = "ByteTrack" if BYTETRACK_AVAILABLE else "no tracker"
        wm  = (f"SMARTCART  |  YOLO-World + {trk} → OpenCLIP DFN5B  |  "
               f"LIVE  [{n} track{'s' if n != 1 else ''}]")
        cv2.putText(frame, wm, (10, 24), FONT, 0.46, (180, 180, 180), 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_pipeline():
    global is_running
    data    = request.get_json(silent=True) or {}
    cam_idx = int(data.get("camera_index", 0))
    if not open_camera(cam_idx):
        return jsonify({"ok": False, "error": f"Cannot open camera {cam_idx}"}), 400

    if YOLO_AVAILABLE and model is None:
        weights = data.get("weights", "yolov8m-worldv2.pt")
        load_model(weights)

    is_running = True
    return jsonify({
        "ok":               True,
        "yolo_world":       YOLO_AVAILABLE,
        "model_loaded":     model is not None,
        "openclip":         CLIP_AVAILABLE,
        "clip_model":       f"{_CLIP_MODEL_NAME}/{_CLIP_PRETRAINED}" if CLIP_AVAILABLE else None,
        "clip_labels":      len(_clip_text_labels),
        "detection_class":  "object_held_at_hand",
        "product_vocab":    ALL_PRODUCT_NAMES,
        "vocab_size":       len(ALL_PRODUCT_NAMES),
    })


@app.route("/api/stop", methods=["POST"])
def stop_pipeline():
    global is_running
    is_running = False
    time.sleep(0.15)
    release_camera()
    with live_boxes_lock:
        live_boxes.clear()
    return jsonify({"ok": True})


@app.route("/video_feed")
def video_feed():
    if not is_running:
        return Response("Pipeline not running", status=503)
    return Response(stream_with_context(frame_generator()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── Barcode scan → snapshot → zero-shot YOLO-World → verdict ─────────────────
@app.route("/api/scan", methods=["POST"])
def scan_barcode():
    data    = request.get_json(silent=True) or {}
    barcode = data.get("barcode", "").strip()

    if not barcode:
        return jsonify({"ok": False, "error": "Empty barcode"}), 400

    product = PRODUCT_DB.get(barcode)
    if not product:
        return jsonify({"ok": False, "error": f"Unknown barcode: '{barcode}'"}), 404

    name         = product["name"]           # full path → fed to CLIP matching
    display_name = product["display_name"]   # last segment → shown in UI
    price        = product["price"]

    # Add to cart optimistically (keyed by full name for uniqueness)
    with state_lock:
        if name not in cart_items:
            cart_items[name] = {"count": 0, "unit_price": price,
                                "barcode": barcode, "display_name": display_name}
        cart_items[name]["count"] += 1

    # Capture + zero-shot evaluate
    result = capture_and_evaluate(name, barcode)

    return jsonify({"ok": True, "name": name, "display_name": display_name,
                    "price": price, "barcode": barcode, **result})


@app.route("/api/verify", methods=["POST"])
def verify():
    result = verify_bagging_area()
    return jsonify({"ok": True, **result})


@app.route("/api/last_snap", methods=["GET"])
def last_snapshot():
    with state_lock:
        if last_snap is None:
            return jsonify({"ok": False})
        return jsonify({
            "ok":           True,
            "snapshot_b64": last_snap.snapshot_b64,
            "top_class":    last_snap.top_class,
            "top_conf":     last_snap.top_conf,
            "classes_used": last_snap.classes_used,
            "detections":   last_snap.detections,
            "ts":           last_snap.ts,
        })


@app.route("/api/cart", methods=["GET"])
def get_cart():
    with state_lock:
        items = [
            {"name":         k,
             "display_name": v.get("display_name", k.split("/")[-1]),
             "count":        v["count"],
             "unit_price":   v["unit_price"],
             "subtotal":     round(v["count"] * v["unit_price"], 2),
             "barcode":      v.get("barcode", "")}
            for k, v in cart_items.items()
        ]
    return jsonify({"items": items,
                    "total": round(sum(i["subtotal"] for i in items), 2)})


@app.route("/api/cart/remove", methods=["POST"])
def remove_from_cart():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    with state_lock:
        if name in cart_items:
            cart_items[name]["count"] -= 1
            if cart_items[name]["count"] <= 0:
                del cart_items[name]
    return jsonify({"ok": True})


@app.route("/api/cart/clear", methods=["POST"])
def clear_cart():
    with state_lock:
        cart_items.clear()
        active_alert.update({"kind": None, "msg": "", "ts": 0})
    return jsonify({"ok": True})


@app.route("/api/fraud_log", methods=["GET"])
def get_fraud_log():
    with state_lock:
        log   = [e.to_dict() for e in fraud_log[:30]]
        alert = dict(active_alert)
    return jsonify({"log": log, "active_alert": alert})


@app.route("/api/fraud_log/clear", methods=["POST"])
def clear_fraud_log():
    with state_lock:
        fraud_log.clear()
    return jsonify({"ok": True})


@app.route("/api/vocab", methods=["GET"])
def get_vocab():
    """Return the current zero-shot product vocabulary."""
    return jsonify({"vocab": ALL_PRODUCT_NAMES, "count": len(ALL_PRODUCT_NAMES)})


@app.route("/api/vocab/add", methods=["POST"])
def add_to_vocab():
    """
    Dynamically add a new product name to the zero-shot vocabulary at runtime.
    No retraining needed — both YOLO-World and OpenCLIP DFN5B will recognise
    the new label on the next scan: YOLO-World via set_classes (already using
    the fixed "object_held_at_hand" class, so no change needed there), and
    OpenCLIP via a fresh rebuild of the cached text embeddings.
    """
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip().lower()
    if not name:
        return jsonify({"ok": False, "error": "Missing name"}), 400
    if name not in ALL_PRODUCT_NAMES:
        ALL_PRODUCT_NAMES.append(name)
        ALL_PRODUCT_NAMES.sort()
        # Rebuild CLIP text embeddings to include the new label
        rebuild_clip_text_features(ALL_PRODUCT_NAMES)
    return jsonify({"ok": True, "vocab": ALL_PRODUCT_NAMES,
                    "clip_labels": len(_clip_text_labels)})


@app.route("/api/status")
def get_status():
    with camera_lock:
        cam_ok = camera is not None and camera.isOpened()
    return jsonify({
        "running":          is_running,
        "camera_open":      cam_ok,
        "yolo_world":       YOLO_AVAILABLE,
        "model_loaded":     model is not None,
        "openclip":         CLIP_AVAILABLE,
        "clip_model":       f"{_CLIP_MODEL_NAME}/{_CLIP_PRETRAINED}" if CLIP_AVAILABLE else None,
        "clip_labels":      len(_clip_text_labels),
        "bytetrack":        BYTETRACK_AVAILABLE,
        "detection_class":  "object_held_at_hand",
        "match_mode":       MATCH_MODE,
        "vocab_size":       len(ALL_PRODUCT_NAMES),
        "conf_threshold":   CONF_THRESHOLD,
    })


@app.route("/api/match_mode", methods=["GET"])
def get_match_mode():
    """Return the current fraud-detection match mode."""
    return jsonify({"match_mode": MATCH_MODE})


@app.route("/api/match_mode", methods=["POST"])
def set_match_mode():
    """
    Set the fraud-detection match mode.
    Body: {"mode": "top1"} or {"mode": "top5"}

    top1 – the scanned product must be the single highest-confidence CLIP
            prediction (strictest, lowest false-acceptance rate).
    top5 – the scanned product is accepted if it appears anywhere in the
            top-5 CLIP predictions (more lenient; tolerates classifier
            uncertainty between visually similar items).
    """
    global MATCH_MODE
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "").strip().lower()
    if mode not in ("top1", "top5"):
        return jsonify({"ok": False, "error": "mode must be 'top1' or 'top5'"}), 400
    MATCH_MODE = mode
    print(f"[INFO] Fraud match mode set to: {MATCH_MODE}")
    return jsonify({"ok": True, "match_mode": MATCH_MODE})


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS PAGE + PRODUCT DB MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/settings")
def settings_page():
    return render_template("settings.html")


@app.route("/api/products", methods=["GET"])
def list_products():
    """Return full product database."""
    items = [
        {"barcode": bc, **info}
        for bc, info in sorted(PRODUCT_DB.items())
    ]
    return jsonify({"ok": True, "products": items, "count": len(items)})


@app.route("/api/products", methods=["POST"])
def create_product():
    """Add a new product to the database."""
    data = request.get_json(silent=True) or {}
    barcode = data.get("barcode", "").strip()
    name    = data.get("name", "").strip()
    price   = data.get("price")

    if not barcode or not name or price is None:
        return jsonify({"ok": False, "error": "barcode, name, and price are required"}), 400
    try:
        price = round(float(price), 2)
    except (ValueError, TypeError):
        return jsonify({"ok": False, "error": "price must be a number"}), 400
    if barcode in PRODUCT_DB:
        return jsonify({"ok": False, "error": f"Barcode '{barcode}' already exists"}), 409

    PRODUCT_DB[barcode] = _db_entry(name, price)

    # Keep derived lists in sync
    full_name = PRODUCT_DB[barcode]["name"]
    if full_name not in ALL_PRODUCT_NAMES:
        ALL_PRODUCT_NAMES.append(full_name)
        ALL_PRODUCT_NAMES.sort()
    PRODUCT_DISPLAY_NAMES[full_name] = PRODUCT_DB[barcode]["display_name"]

    return jsonify({"ok": True, "product": {"barcode": barcode, **PRODUCT_DB[barcode]}})


@app.route("/api/products/<barcode>", methods=["PUT"])
def update_product(barcode):
    """Modify an existing product (name and/or price)."""
    if barcode not in PRODUCT_DB:
        return jsonify({"ok": False, "error": f"Barcode '{barcode}' not found"}), 404

    data = request.get_json(silent=True) or {}
    old_name = PRODUCT_DB[barcode]["name"]

    new_name  = data.get("name", old_name).strip()
    new_price = data.get("price", PRODUCT_DB[barcode]["price"])
    try:
        new_price = round(float(new_price), 2)
    except (ValueError, TypeError):
        return jsonify({"ok": False, "error": "price must be a number"}), 400

    # Update DB entry
    PRODUCT_DB[barcode] = _db_entry(new_name, new_price)

    # Sync derived lists
    if old_name != new_name:
        if old_name in ALL_PRODUCT_NAMES:
            ALL_PRODUCT_NAMES.remove(old_name)
        PRODUCT_DISPLAY_NAMES.pop(old_name, None)
    full_name = PRODUCT_DB[barcode]["name"]
    if full_name not in ALL_PRODUCT_NAMES:
        ALL_PRODUCT_NAMES.append(full_name)
        ALL_PRODUCT_NAMES.sort()
    PRODUCT_DISPLAY_NAMES[full_name] = PRODUCT_DB[barcode]["display_name"]

    return jsonify({"ok": True, "product": {"barcode": barcode, **PRODUCT_DB[barcode]}})


@app.route("/api/products/<barcode>", methods=["DELETE"])
def delete_product(barcode):
    """Remove a product from the database."""
    if barcode not in PRODUCT_DB:
        return jsonify({"ok": False, "error": f"Barcode '{barcode}' not found"}), 404

    removed = PRODUCT_DB.pop(barcode)
    removed_name = removed["name"]

    # Only remove from vocab if no other barcode maps to the same name
    still_used = any(p["name"] == removed_name for p in PRODUCT_DB.values())
    if not still_used:
        if removed_name in ALL_PRODUCT_NAMES:
            ALL_PRODUCT_NAMES.remove(removed_name)
        PRODUCT_DISPLAY_NAMES.pop(removed_name, None)

    return jsonify({"ok": True, "removed": {"barcode": barcode, **removed}})


@app.route("/api/embeddings/regenerate", methods=["POST"])
def regenerate_embeddings():
    """
    Delete the cached clip_zeroshot_cls.pth and rebuild CLIP text embeddings
    from the current ALL_PRODUCT_NAMES vocabulary.
    """
    pth = Path("clip_zeroshot_cls.pth")
    if pth.exists():
        pth.unlink()
        print("[INFO] Deleted cached clip_zeroshot_cls.pth")

    rebuild_clip_text_features(ALL_PRODUCT_NAMES)

    return jsonify({
        "ok": True,
        "clip_labels": len(_clip_text_labels),
        "vocab_size": len(ALL_PRODUCT_NAMES),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
