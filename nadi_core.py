"""
NadiRakshak - core logic (AI detection, rules, evidence).
The Streamlit user interface is in app.py.
"""
import csv
import hashlib
import io
import json
import os
from datetime import datetime, time

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)
ALERTS_FILE = os.path.join(DATA, "alerts.json")
REPORTS_FILE = os.path.join(DATA, "citizen_reports.json")
REGISTER_FILE = os.path.join(BASE, "canoe_register.csv")

BOAT_CLASS = 8     # "boat" in the COCO dataset used by YOLOv8
PERSON_CLASS = 0   # "person" (only used to BLUR faces/people - never to identify)

# ---------------------------------------------------------------------------
# 1. Demo geography (ILLUSTRATIVE - not official lease boundaries)
# ---------------------------------------------------------------------------
CAMERA_SITES = {
    "CAM-01 Chapora @ Colvale":  {"lat": 15.6335, "lon": 73.8210, "river": "Chapora"},
    "CAM-02 Mandovi @ Old Goa":  {"lat": 15.5065, "lon": 73.9120, "river": "Mandovi"},
    "CAM-03 Zuari @ Cortalim":   {"lat": 15.4050, "lon": 73.9080, "river": "Zuari"},
    "CAM-04 Terekhol @ Keri":    {"lat": 15.7150, "lon": 73.7300, "river": "Terekhol"},
}

# Approved (legal) extraction zones - polygons as (lon, lat)
APPROVED_ZONES = {
    "Zone M-3 (Mandovi, approved)": [(73.905, 15.500), (73.920, 15.500),
                                     (73.920, 15.512), (73.905, 15.512)],
    "Zone Z-5 (Zuari, approved)":   [(73.915, 15.395), (73.930, 15.395),
                                     (73.930, 15.405), (73.915, 15.405)],
}

PERMITTED_START = time(7, 0)   # legal extraction allowed only in daylight
PERMITTED_END = time(18, 0)


def zone_for_point(lat, lon):
    """Return the approved zone name that contains this point, or None."""
    p = Point(lon, lat)
    for name, coords in APPROVED_ZONES.items():
        if Polygon(coords).contains(p):
            return name
    return None


# ---------------------------------------------------------------------------
# 2. Canoe register (synthetic)
# ---------------------------------------------------------------------------
def load_register():
    if not os.path.exists(REGISTER_FILE):
        return {}
    with open(REGISTER_FILE, newline="", encoding="utf-8") as f:
        return {r["reg_no"].strip().upper(): r for r in csv.DictReader(f)}


def check_registration(reg_no):
    """Returns (status, details). status: REGISTERED / UNREGISTERED / NOT VISIBLE"""
    reg_no = (reg_no or "").strip().upper()
    if not reg_no:
        return "NOT VISIBLE", None
    rec = load_register().get(reg_no)
    if rec is None:
        return "UNREGISTERED", None
    if rec.get("licence_status", "").lower() != "active":
        return "UNREGISTERED", rec          # cancelled/suspended counts as unregistered
    return "REGISTERED", rec


# ---------------------------------------------------------------------------
# 3. Image helpers
# ---------------------------------------------------------------------------
def enhance_night(frame):
    """Brighten dark/night footage: CLAHE on the lightness channel + gamma."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    gamma = 1.6
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(out, table)


def blur_boxes(frame, boxes):
    """Blur people so nobody can be identified (privacy by design)."""
    out = frame.copy()
    h, w = out.shape[:2]
    for (x1, y1, x2, y2) in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (51, 51), 30)
    return out


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# 4. Canoe detection + tracking + behaviour (mining vs moving)
# ---------------------------------------------------------------------------
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO("yolov8n.pt")   # downloads automatically (~6 MB) the first time
    return _MODEL


def analyse_video(path, night_mode=False, time_scale=60.0, stationary_minutes=10,
                  frame_stride=3, conf=0.25, progress_cb=None, preview_cb=None):
    """
    Detect and track boats in a video, measure how long each one stays still.

    time_scale: how many REAL seconds one VIDEO second represents.
                60 = demo mode (1 video second = 1 minute), 1 = real time.
    Returns a dict with per-track results and the best evidence frame.
    """
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")        # fresh model = fresh tracker for every video
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    tracks = {}          # id -> info
    best_frame, best_score = None, -1
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % frame_stride:
            continue
        # resize for speed
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)))
            h, w = frame.shape[:2]
        if night_mode:
            frame = enhance_night(frame)

        t_real = (idx / fps) * time_scale            # "real world" seconds
        res = model.track(frame, persist=True, conf=conf, classes=[BOAT_CLASS, PERSON_CLASS],
                          tracker="bytetrack.yaml", verbose=False)[0]

        people, annotated, frame_score = [], frame.copy(), -1
        radius = 0.06 * (w ** 2 + h ** 2) ** 0.5      # "not moved" if centre stays within 6% of diagonal
        if res.boxes is not None and len(res.boxes):
            ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [None] * len(res.boxes)
            for box, cls, cf, tid in zip(res.boxes.xyxy.tolist(), res.boxes.cls.int().tolist(),
                                         res.boxes.conf.tolist(), ids):
                if cls == PERSON_CLASS:
                    people.append(box)
                    continue
                if tid is None:
                    continue
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                tr = tracks.setdefault(tid, {"first": t_real, "anchor": (cx, cy), "still_since": t_real,
                                             "max_still": 0.0, "max_conf": 0.0, "last": t_real, "seen": 0})
                tr["seen"] += 1
                if ((cx - tr["anchor"][0]) ** 2 + (cy - tr["anchor"][1]) ** 2) ** 0.5 > radius:
                    tr["anchor"], tr["still_since"] = (cx, cy), t_real     # it moved: restart clock
                still = t_real - tr["still_since"]
                tr["max_still"] = max(tr["max_still"], still)
                tr["max_conf"] = max(tr["max_conf"], cf)
                tr["last"] = t_real

                flagged = still >= stationary_minutes * 60
                colour = (0, 0, 255) if flagged else (0, 200, 0)
                label = f"Canoe #{tid} {cf:.2f} | still {still/60:.0f} min"
                if flagged:
                    label += " | POSSIBLE EXTRACTION"
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 3)
                tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
                cv2.rectangle(annotated, (x1, max(0, y1 - 26)), (x1 + tw + 8, y1), colour, -1)
                cv2.putText(annotated, label, (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 2)
                frame_score = max(frame_score, still + cf)
            annotated = blur_boxes(annotated, people)   # privacy: blur people

        stamp = f"NadiRakshak | sim. time +{t_real/60:.1f} min"
        cv2.putText(annotated, stamp, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if frame_score > best_score:                    # keep the most convincing frame as evidence
            best_score, best_frame = frame_score, annotated.copy()
        if preview_cb:
            preview_cb(annotated)
        if progress_cb:
            progress_cb(min(idx / total, 1.0))
    cap.release()

    results = []
    for tid, tr in tracks.items():
        if tr["seen"] < 3:          # ignore flickering one-off detections
            continue
        results.append({"track_id": tid, "max_still_min": round(tr["max_still"] / 60, 1),
                         "confidence": round(tr["max_conf"], 2),
                         "stationary_flag": tr["max_still"] >= stationary_minutes * 60})
    results.sort(key=lambda r: -r["max_still_min"])
    return {"tracks": results, "evidence_frame": best_frame}


# ---------------------------------------------------------------------------
# 5. Decision layer - LEGAL / SUSPICIOUS / ILLEGAL
# ---------------------------------------------------------------------------
def decide(stationary, site_name, obs_time, reg_status):
    site = CAMERA_SITES[site_name]
    zone = zone_for_point(site["lat"], site["lon"])
    in_hours = PERMITTED_START <= obs_time <= PERMITTED_END
    reasons = []
    if not stationary:
        return "NO ACTION", ["Canoe kept moving (transit / fishing pattern) - logged only"], zone

    reasons.append("Canoe stayed stationary for a long time (extraction pattern)")
    violations = 0
    if zone is None:
        violations += 1
        reasons.append("Location is OUTSIDE all approved extraction zones")
    else:
        reasons.append(f"Location is inside {zone}")
    if not in_hours:
        violations += 1
        reasons.append(f"Time {obs_time.strftime('%H:%M')} is OUTSIDE permitted hours "
                       f"({PERMITTED_START.strftime('%H:%M')}-{PERMITTED_END.strftime('%H:%M')})")
    else:
        reasons.append("Within permitted hours")
    if reg_status != "REGISTERED":
        violations += 1
        reasons.append(f"Canoe registration: {reg_status}")
    else:
        reasons.append("Canoe is registered with an active licence")

    if violations == 0:
        return "LEGAL", reasons, zone
    if violations == 1:
        return "SUSPICIOUS", reasons, zone
    return "ILLEGAL", reasons, zone


# ---------------------------------------------------------------------------
# 6. Alerts storage
# ---------------------------------------------------------------------------
def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json(path, items):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def add_alert(alert):
    alerts = load_json(ALERTS_FILE)
    alert["alert_id"] = f"NR-{datetime.now().strftime('%Y%m%d')}-{len(alerts) + 1:03d}"
    alerts.append(alert)
    save_json(ALERTS_FILE, alerts)
    return alert


# ---------------------------------------------------------------------------
# 7. Evidence pack PDF
# ---------------------------------------------------------------------------
def build_evidence_pdf(alert, frame_bgr, sat_image_bgr=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    frame_png = cv2.imencode(".png", frame_bgr)[1].tobytes()
    frame_hash = sha256_bytes(frame_png)
    meta_hash = sha256_bytes(json.dumps(alert, sort_keys=True, default=str).encode())

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    c.setFillColor(colors.HexColor("#1f2a52"))
    c.rect(0, H - 2.4 * cm, W, 2.4 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(1.5 * cm, H - 1.5 * cm, "NadiRakshak - Evidence Pack")
    c.setFont("Helvetica", 10)
    c.drawRightString(W - 1.5 * cm, H - 1.5 * cm, f"Alert ID: {alert['alert_id']}")

    colour = {"ILLEGAL": "#c62828", "SUSPICIOUS": "#ef6c00", "LEGAL": "#2e7d32"}.get(alert["decision"], "#555")
    c.setFillColor(colors.HexColor(colour))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1.5 * cm, H - 3.4 * cm, f"Decision: {alert['decision']}")

    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    y = H - 4.2 * cm
    rows = [("Generated at", alert["generated_at"]), ("Camera site", alert["site"]),
            ("River", alert["river"]), ("GPS", f"{alert['lat']}, {alert['lon']}"),
            ("Observation time", alert["obs_time"]), ("Approved zone", alert["zone"] or "None"),
            ("Canoe registration no.", alert["reg_no"] or "Not visible"),
            ("Registration status", alert["reg_status"]),
            ("Max stationary time", f"{alert['max_still_min']} min"),
            ("AI detection confidence", str(alert["confidence"]))]
    for k, v in rows:
        c.setFont("Helvetica-Bold", 10); c.drawString(1.5 * cm, y, k + ":")
        c.setFont("Helvetica", 10); c.drawString(6 * cm, y, str(v))
        y -= 0.55 * cm
    y -= 0.2 * cm
    c.setFont("Helvetica-Bold", 10); c.drawString(1.5 * cm, y, "Reasons:"); y -= 0.5 * cm
    c.setFont("Helvetica", 9)
    for r in alert["reasons"]:
        c.drawString(2 * cm, y, "- " + r); y -= 0.45 * cm

    img = ImageReader(io.BytesIO(frame_png))
    iw, ih = img.getSize()
    dw = W - 3 * cm
    dh = dw * ih / iw
    if dh > 9 * cm:
        dh = 9 * cm; dw = dh * iw / ih
    y -= dh + 0.3 * cm
    c.drawImage(img, 1.5 * cm, y, dw, dh)
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(1.5 * cm, y - 0.4 * cm, "Evidence frame (people automatically blurred - no face recognition used)")

    y -= 1.2 * cm
    c.setFont("Helvetica-Bold", 9); c.drawString(1.5 * cm, y, "Digital fingerprints (SHA-256):")
    c.setFont("Courier", 7.5)
    c.drawString(1.5 * cm, y - 0.45 * cm, f"Frame:    {frame_hash}")
    c.drawString(1.5 * cm, y - 0.85 * cm, f"Metadata: {meta_hash}")
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(1.5 * cm, 1.5 * cm, "AI-generated alert. Must be verified by an authorised officer before any FIR/action. "
                                     "Prototype - synthetic data.")
    c.showPage()

    if sat_image_bgr is not None:
        sat_png = cv2.imencode(".png", sat_image_bgr)[1].tobytes()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1.5 * cm, H - 2 * cm, "Satellite change evidence (before vs after)")
        si = ImageReader(io.BytesIO(sat_png)); sw, sh = si.getSize()
        dw = W - 3 * cm; dh = dw * sh / sw
        c.drawImage(si, 1.5 * cm, H - 3 * cm - dh, dw, dh)
        c.setFont("Courier", 7.5)
        c.drawString(1.5 * cm, H - 3.6 * cm - dh, f"SHA-256: {sha256_bytes(sat_png)}")
        c.showPage()
    c.save()
    pdf = buf.getvalue()
    return pdf, sha256_bytes(pdf), frame_hash


# ---------------------------------------------------------------------------
# 8. Satellite change detection (before vs after image)
# ---------------------------------------------------------------------------
def satellite_change(before_bgr, after_bgr, sensitivity=35):
    """Pixel-level change detection between two images of the same area."""
    h, w = before_bgr.shape[:2]
    after_bgr = cv2.resize(after_bgr, (w, h))
    b = cv2.GaussianBlur(cv2.cvtColor(before_bgr, cv2.COLOR_BGR2LAB), (7, 7), 0).astype(np.int16)
    a = cv2.GaussianBlur(cv2.cvtColor(after_bgr, cv2.COLOR_BGR2LAB), (7, 7), 0).astype(np.int16)
    diff = np.sqrt(((a - b) ** 2).sum(axis=2))
    mask = (diff > sensitivity).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # keep only reasonably large changed areas
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    min_area = 0.0005 * w * h
    boxes = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[lab == i] = 255
            x, y, bw, bh = stats[i, :4]
            boxes.append((x, y, bw, bh))
    overlay = after_bgr.copy()
    overlay[clean > 0] = (0, 0, 255)
    out = cv2.addWeighted(after_bgr, 0.6, overlay, 0.4, 0)
    for (x, y, bw, bh) in boxes:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
    pct = 100.0 * (clean > 0).sum() / (w * h)
    side = np.hstack([before_bgr, out])
    cv2.putText(side, "BEFORE", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.putText(side, "AFTER + detected change", (w + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return side, pct, len(boxes)


# ---------------------------------------------------------------------------
# 9. Citizen report checking (AI verification + duplicate detection)
# ---------------------------------------------------------------------------
def check_citizen_photo(img_bgr):
    """Returns (has_boat, confidence, annotated, perceptual_hash)."""
    import imagehash
    from PIL import Image
    model = get_model()
    res = model.predict(img_bgr, conf=0.25, classes=[BOAT_CLASS, PERSON_CLASS], verbose=False)[0]
    annotated, people, best = img_bgr.copy(), [], 0.0
    for box, cls, cf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.int().tolist(), res.boxes.conf.tolist()):
        if cls == PERSON_CLASS:
            people.append(box); continue
        best = max(best, cf)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(annotated, f"boat {cf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    annotated = blur_boxes(annotated, people)
    phash = str(imagehash.phash(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))))
    return best > 0, round(best, 2), annotated, phash


def is_duplicate(phash, max_distance=8):
    import imagehash
    h = imagehash.hex_to_hash(phash)
    for r in load_json(REPORTS_FILE):
        if h - imagehash.hex_to_hash(r["phash"]) <= max_distance:
            return r["report_id"]
    return None
