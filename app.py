"""
NadiRakshak - Protecting Goa's Rivers from Illegal Sand Mining with AI
Prototype dashboard. Run with:   streamlit run app.py
"""
import os
import tempfile
from datetime import datetime, time

import cv2
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import nadi_core as core

st.set_page_config(page_title="NadiRakshak", page_icon="🌊", layout="wide")

# ---------- small helpers ----------
def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_upload(file):
    data = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


BADGE = {"ILLEGAL": "🔴 ILLEGAL", "SUSPICIOUS": "🟠 SUSPICIOUS", "LEGAL": "🟢 LEGAL",
         "NO ACTION": "⚪ NO ACTION"}

# ---------- header ----------
st.markdown(
    """
    <div style="background:#1f2a52;padding:18px 24px;border-radius:12px;margin-bottom:12px">
      <h1 style="color:white;margin:0">NadiRakshak</h1>
      <p style="color:#e6c77a;margin:0">Protecting Goa's Rivers from Illegal Sand Mining with AI
      &nbsp;|&nbsp; Team AquaVigil</p>
    </div>
    """, unsafe_allow_html=True)

alerts = core.load_json(core.ALERTS_FILE)
reports = core.load_json(core.REPORTS_FILE)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rivers monitored", "4")
c2.metric("Illegal alerts", sum(a["decision"] == "ILLEGAL" for a in alerts))
c3.metric("Suspicious (needs review)", sum(a["decision"] == "SUSPICIOUS" for a in alerts))
c4.metric("Citizen reports", len(reports))

tab_map, tab_cam, tab_sat, tab_cit, tab_log = st.tabs(
    ["🗺️ Command Map", "🎥 Smart Camera AI", "🛰️ Satellite Change", "📱 Citizen Report", "📁 Alerts & Evidence"])

# =====================================================================
# TAB 1 - MAP
# =====================================================================
with tab_map:
    st.subheader("Live command map - Goa rivers")
    m = folium.Map(location=[15.55, 73.85], zoom_start=10, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite").add_to(m)
    for name, coords in core.APPROVED_ZONES.items():
        folium.Polygon([(lat, lon) for lon, lat in coords], color="green", fill=True, fill_opacity=0.3,
                       tooltip=name + " (illustrative)").add_to(m)
    for name, s in core.CAMERA_SITES.items():
        folium.Marker([s["lat"], s["lon"]], tooltip=name,
                      icon=folium.Icon(color="blue", icon="video-camera", prefix="fa")).add_to(m)
    colour = {"ILLEGAL": "red", "SUSPICIOUS": "orange", "LEGAL": "green"}
    for a in alerts:
        folium.CircleMarker([a["lat"], a["lon"]], radius=14, color=colour.get(a["decision"], "gray"),
                            fill=True, fill_opacity=0.7,
                            popup=f"{a['alert_id']}<br>{a['decision']}<br>{a['obs_time']}").add_to(m)
    for r in reports:
        folium.Marker([r["lat"], r["lon"]], tooltip=f"Citizen report {r['report_id']}",
                      icon=folium.Icon(color="purple", icon="user-secret", prefix="fa")).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, height=520, use_container_width=True)
    st.caption("Blue = AI camera sites · Green = approved extraction zones (illustrative, not official) · "
               "Red/Orange circles = AI alerts · Purple = anonymous citizen reports")

# =====================================================================
# TAB 2 - CAMERA AI
# =====================================================================
with tab_cam:
    st.subheader("Smart night camera: detect → track → classify → decide")
    left, right = st.columns([1, 2])
    with left:
        video = st.file_uploader("Upload CCTV / river video (mp4, avi, mov)", type=["mp4", "avi", "mov", "mkv"])
        site = st.selectbox("Camera site", list(core.CAMERA_SITES.keys()))
        obs_time = st.time_input("Time of recording", value=time(23, 40))
        reg_no = st.text_input("Canoe registration no. seen (leave empty if not visible)",
                               placeholder="e.g. GA-CAN-2045")
        night = st.toggle("Night enhancement (low-light)", value=True)
        demo = st.toggle("Demo time scale: 1 video second = 1 real minute", value=True)
        still_min = st.slider("Flag if canoe stays still for at least (minutes)", 5, 60, 10)
        run = st.button("▶ Run AI analysis", type="primary", disabled=video is None)
    with right:
        preview = st.empty()
        bar = st.empty()

    if run and video is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.name)[1])
        tmp.write(video.read()); tmp.close()
        prog = bar.progress(0.0, text="AI is watching the video...")
        counter = {"n": 0}

        def show(frame):
            counter["n"] += 1
            if counter["n"] % 3 == 0:
                preview.image(to_rgb(frame), channels="RGB", use_container_width=True)

        result = core.analyse_video(tmp.name, night_mode=night, time_scale=60.0 if demo else 1.0,
                                    stationary_minutes=still_min,
                                    progress_cb=lambda p: prog.progress(p, text=f"Analysing... {int(p*100)}%"),
                                    preview_cb=show)
        prog.progress(1.0, text="Analysis complete")
        os.unlink(tmp.name)
        st.session_state["cam_result"] = dict(result=result, site=site, obs_time=obs_time,
                                              reg_no=reg_no, still_min=still_min)

    if "cam_result" in st.session_state:
        cr = st.session_state["cam_result"]
        res = cr["result"]
        if cr["result"]["evidence_frame"] is not None:
            preview.image(to_rgb(res["evidence_frame"]), caption="Best evidence frame (people auto-blurred)",
                          use_container_width=True)
        if not res["tracks"]:
            st.warning("No canoe/boat detected in this video.")
        else:
            st.markdown("#### Canoes tracked")
            st.dataframe(pd.DataFrame(res["tracks"]).rename(columns={
                "track_id": "Canoe ID", "max_still_min": "Max time stationary (min)",
                "confidence": "AI confidence", "stationary_flag": "Extraction pattern?"}),
                hide_index=True, use_container_width=True)
            top = res["tracks"][0]
            reg_status, rec = core.check_registration(cr["reg_no"])
            decision, reasons, zone = core.decide(top["stationary_flag"], cr["site"], cr["obs_time"], reg_status)

            st.markdown(f"### Decision: {BADGE[decision]}")
            for r in reasons:
                st.write("• " + r)
            if rec:
                st.caption(f"Register entry: {rec}")

            if decision in ("ILLEGAL", "SUSPICIOUS", "LEGAL"):
                if st.button("🚨 Raise alert & generate evidence pack" if decision != "LEGAL"
                             else "📝 Log legal operation"):
                    s = core.CAMERA_SITES[cr["site"]]
                    alert = core.add_alert({
                        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "site": cr["site"], "river": s["river"], "lat": s["lat"], "lon": s["lon"],
                        "obs_time": cr["obs_time"].strftime("%H:%M"), "zone": zone,
                        "reg_no": cr["reg_no"].upper(), "reg_status": reg_status,
                        "max_still_min": top["max_still_min"], "confidence": top["confidence"],
                        "decision": decision, "reasons": reasons})
                    sat = st.session_state.get("sat_image")
                    pdf, pdf_hash, frame_hash = core.build_evidence_pdf(alert, res["evidence_frame"], sat)
                    path = os.path.join(core.DATA, f"{alert['alert_id']}.pdf")
                    with open(path, "wb") as f:
                        f.write(pdf)
                    alerts_all = core.load_json(core.ALERTS_FILE)
                    alerts_all[-1]["pdf_sha256"] = pdf_hash
                    alerts_all[-1]["pdf_path"] = path
                    core.save_json(core.ALERTS_FILE, alerts_all)
                    if decision == "ILLEGAL":
                        st.error(f"🚨 ALERT {alert['alert_id']} sent to Flying Squad ({s['river']} river sector) "
                                 f"and Coastal Police (simulated SMS/WhatsApp).")
                    elif decision == "SUSPICIOUS":
                        st.warning(f"Alert {alert['alert_id']} queued for human review.")
                    else:
                        st.success(f"Logged {alert['alert_id']} as legal operation.")
                    st.download_button("⬇ Download evidence pack (PDF)", pdf, file_name=f"{alert['alert_id']}.pdf",
                                       mime="application/pdf", on_click="ignore")
                    st.code(f"Evidence PDF SHA-256: {pdf_hash}", language=None)

# =====================================================================
# TAB 3 - SATELLITE CHANGE
# =====================================================================
with tab_sat:
    st.subheader("Weekly satellite watch: before vs after change detection")
    st.caption("Upload two images of the SAME river stretch from different dates "
               "(e.g. Google Earth Pro historical imagery or Copernicus Browser Sentinel-2).")
    a, b = st.columns(2)
    before = a.file_uploader("BEFORE image", type=["png", "jpg", "jpeg"], key="before")
    after = b.file_uploader("AFTER image", type=["png", "jpg", "jpeg"], key="after")
    sens = st.slider("Sensitivity (lower = detects smaller changes)", 10, 80, 35)
    if before and after and st.button("🛰️ Detect change", type="primary"):
        bi, ai = read_upload(before), read_upload(after)
        side, pct, n = core.satellite_change(bi, ai, sens)
        st.session_state["sat_image"] = side
        st.image(to_rgb(side), use_container_width=True)
        k1, k2, k3 = st.columns(3)
        k1.metric("Area changed", f"{pct:.2f} %")
        k2.metric("Change hotspots", n)
        risk = "HIGH" if pct > 5 else "MEDIUM" if pct > 1 else "LOW"
        k3.metric("Risk level for this stretch", risk)
        st.info("High-risk stretches are prioritised for camera placement and flying-squad patrols. "
                "This satellite image is automatically attached to the next evidence pack.")

# =====================================================================
# TAB 4 - CITIZEN REPORT
# =====================================================================
with tab_cit:
    st.subheader("Anonymous citizen report")
    st.caption("🔒 No name or phone number is collected. Photo metadata (EXIF/GPS of the phone) is stripped; "
               "only the river location is kept.")
    photo = st.file_uploader("Photo of suspected sand mining", type=["png", "jpg", "jpeg"], key="cit")
    river = st.selectbox("Nearest river stretch", list(core.CAMERA_SITES.keys()), key="cit_site")
    note = st.text_area("What did you see? (optional)", max_chars=300)
    if photo and st.button("📤 Submit anonymously", type="primary"):
        img = read_upload(photo)              # re-encoding drops all EXIF metadata
        has_boat, conf, annotated, phash = core.check_citizen_photo(img)
        dup = core.is_duplicate(phash)
        st.image(to_rgb(annotated), caption="AI verification (people auto-blurred)", width=600)
        if dup:
            st.warning(f"Duplicate of earlier report {dup}. Merged - priority of that location increased.")
        elif not has_boat:
            st.error("AI could not find a canoe/boat in this photo. Report kept as LOW priority for manual check.")
        else:
            s = core.CAMERA_SITES[river]
            items = core.load_json(core.REPORTS_FILE)
            rid = f"CR-{len(items) + 1:03d}"
            items.append({"report_id": rid, "lat": s["lat"] + 0.004, "lon": s["lon"] + 0.004,
                          "site": river, "ai_confidence": conf, "phash": phash, "note": note,
                          "time": datetime.now().strftime("%Y-%m-%d %H:%M")})
            core.save_json(core.REPORTS_FILE, items)
            st.success(f"Report {rid} verified by AI (boat confidence {conf}) and forwarded to the "
                       f"nodal officer. Your identity was never recorded.")

# =====================================================================
# TAB 5 - LOG
# =====================================================================
with tab_log:
    st.subheader("Alerts & evidence register")
    if not alerts:
        st.info("No alerts yet. Run the Smart Camera AI first.")
    else:
        df = pd.DataFrame(alerts)
        cols = [c for c in ["alert_id", "generated_at", "decision", "site", "obs_time", "reg_no",
                            "reg_status", "max_still_min", "pdf_sha256"] if c in df.columns]
        st.dataframe(df[cols], hide_index=True, use_container_width=True)
        for a in reversed(alerts):
            p = a.get("pdf_path")
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    st.download_button(f"⬇ {a['alert_id']} ({a['decision']})", f.read(),
                                       file_name=os.path.basename(p), mime="application/pdf", key=p, on_click="ignore")
    st.markdown("#### Verify evidence integrity")
    up = st.file_uploader("Upload an evidence PDF to check it has not been tampered with", type=["pdf"], key="verify")
    if up:
        h = core.sha256_bytes(up.read())
        match = [a["alert_id"] for a in alerts if a.get("pdf_sha256") == h]
        if match:
            st.success(f"✅ Authentic - fingerprint matches {match[0]}")
        else:
            st.error("❌ Fingerprint does not match any stored evidence - file may be altered.")
    if st.button("🗑 Reset demo data"):
        for f in [core.ALERTS_FILE, core.REPORTS_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.rerun()

st.divider()
# st.caption("Prototype · synthetic register & illustrative zones · YOLOv8 (Ultralytics, COCO 'boat' class) + ByteTrack · "
        #    "no face recognition · every AI alert must be verified by an authorised officer.")
