# NadiRakshak Prototype: Step-by-Step Guide (5 Hours)

Team AquaVigil · Sankalp Setu 2026 · PCCE Internal Round

https://nadirakshak.streamlit.app/

This folder is a complete, working prototype. You don't need to write any code; you only install, run and practise. All the code was tested before being shared with you.

## What the prototype shows

| Tab | What it does | Slide it proves |
|---|---|---|
| Command Map | Goa map with AI camera sites, approved zones (illustrative), alerts and citizen reports | Solution / Decision layer |
| Smart Camera AI | Upload a video: YOLOv8 detects canoes, ByteTrack tracks them, measures how long each canoe stays still, night enhancement, then decides LEGAL / SUSPICIOUS / ILLEGAL | Smart night cameras |
| Satellite Change | Upload before and after images of a river stretch and see changed areas highlighted in red, with a risk level | Satellite watch |
| Citizen Report | Anonymous photo upload: AI checks for a boat, detects duplicates, strips phone metadata, blurs people | Anonymous citizen reports |
| Alerts & Evidence | Alert register, evidence PDF download, tamper check with SHA-256 fingerprints | Tamper-proof evidence pack |

## Files in this folder

```
NadiRakshak_Prototype/
├── app.py               <- the dashboard (Streamlit)
├── nadi_core.py         <- the AI + rules + PDF logic
├── make_demo_video.py   <- creates test videos if you have no footage
├── canoe_register.csv   <- synthetic canoe registration register
├── sample_canoe.jpg     <- photo of a canoe in Colvale, Goa (Wikimedia Commons)
├── yolov8n.pt           <- pre-trained YOLOv8 nano model (6 MB)
├── requirements.txt     <- list of Python libraries
└── README.md            <- this guide
```

---

## HOUR 1: Install everything (one laptop, ~45 min)

### Step 1: Install Python 3.11 or 3.12
1. Go to https://www.python.org/downloads/ and download **Python 3.12** (3.11 also works; avoid 3.13 or newer for now, because some libraries may not install).
2. Windows: during installation, **tick the box "Add python.exe to PATH"**, then click Install Now.
3. Check: open **Command Prompt** (press Windows key, type `cmd`, press Enter) and type:
   ```
   python --version
   ```
   It should say `Python 3.12.x`.

### Step 2: Install VS Code (optional but helpful)
Download from https://code.visualstudio.com/. You can open the folder there and use its built-in terminal (Terminal → New Terminal).

### Step 3: Put the project folder somewhere simple
Unzip `NadiRakshak_Prototype.zip` to, for example, `C:\NadiRakshak_Prototype` (Windows) or your Desktop (Mac).

### Step 4: Open a terminal in that folder
Windows:
```
cd C:\NadiRakshak_Prototype
```
Mac / Linux:
```
cd ~/Desktop/NadiRakshak_Prototype
```

### Step 5: Create a virtual environment (a clean box for libraries)
Windows:
```
python -m venv venv
venv\Scripts\activate
```
Mac / Linux:
```
python3 -m venv venv
source venv/bin/activate
```
You should now see `(venv)` at the start of the line.

> If Windows PowerShell says "running scripts is disabled", run this once and try again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or just use Command Prompt (cmd) instead of PowerShell.

### Step 6: Install the libraries (about 10–20 minutes, ~1 GB download, needs good Wi-Fi)
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This installs YOLOv8 (Ultralytics), PyTorch, OpenCV, Streamlit, maps and PDF tools.

---

## HOUR 2: First run + create demo videos (~30 min)

### Step 7: Create the test videos
```
python make_demo_video.py
```
This creates two files:
- `demo_stationary_night.mp4`: a dark, night-style CCTV clip where the canoe stays in one place (should be flagged)
- `demo_moving_day.mp4`: a daytime clip where the canoe keeps moving (should NOT be flagged)

### Step 8: Start the app
```
streamlit run app.py
```
Your browser opens automatically at **http://localhost:8501**. If it doesn't, copy that address into Chrome.
(The first time, Streamlit may ask for your email; just press Enter to skip.)

To stop the app: click the terminal and press `Ctrl + C`.
To start it again later: open the terminal in the folder, activate venv (Step 5, second line), then run `streamlit run app.py`.

### Step 9: Test the Smart Camera AI
1. Open the **Smart Camera AI** tab.
2. Upload `demo_stationary_night.mp4`.
3. Keep: Camera site = **CAM-01 Chapora @ Colvale**, Time = **23:40**, registration number empty, Night enhancement ON, Demo time scale ON, threshold 10 min.
4. Click **Run AI analysis**. You'll see the video being analysed live, with boxes around canoes (about 20–60 seconds on a normal laptop).
5. Result: **ILLEGAL**, with 3 reasons. Click **Raise alert & generate evidence pack**, then **Download evidence pack (PDF)**.

Now test the other outcomes (to show the system is fair):

| Test | Settings | Expected |
|---|---|---|
| Legal operation | CAM-02 Mandovi @ Old Goa, time 10:00, reg no `GA-CAN-2045`, stationary video | LEGAL |
| Suspicious | CAM-02 Mandovi @ Old Goa, time 21:00, reg no `GA-CAN-2045` | SUSPICIOUS (only 1 violation, so a person reviews it) |
| Cancelled licence | CAM-02, 10:00, reg no `GA-CAN-2077` | SUSPICIOUS (licence cancelled) |
| Fishing / moving | Any site, upload `demo_moving_day.mp4`, Night OFF | NO ACTION (canoe kept moving) |

---

## HOUR 3: Get real-looking data (~60 min, split the work)

### Member A: real river / boat video (Pexels, free, no login)
1. Go to https://www.pexels.com/videos/ and search: `canoe river`, `fishing boat river`, `boat night river`, `wooden boat India`.
2. Pick 1–2 clips (15–40 seconds) where a boat is clearly visible and mostly **still**, plus one where a boat **moves**.
3. Click Free Download, choose a small size (HD 1280×720 is enough), and save into the project folder.
4. Test them in the Smart Camera AI tab. If a boat isn't detected, lower the model confidence in `nadi_core.py` (`conf=0.25` → `conf=0.15`) and restart the app.

> The prototype uses the pre-trained YOLOv8 **"boat"** class from the COCO dataset, so no training is needed. Tell the judges: "In the next phase we fine-tune YOLOv8 on Goan canoe images from local jetties."

### Member B: real satellite before/after images of a Goa river
Easiest option: **Google Earth Pro (free desktop app)**
1. Download from https://www.google.com/earth/about/versions/ → Google Earth Pro on desktop.
2. Search for a river stretch, e.g. `Colvale, Goa` (Chapora), `Tivim`, `Old Goa` (Mandovi), `Cortalim` (Zuari).
3. Click the **clock icon (Historical Imagery)** in the top toolbar. A time slider appears.
4. Choose an older date, zoom so the riverbank fills the screen, then File → Save → Save Image, and name it `before.jpg`.
5. Without moving the map, slide to a recent date and save `after.jpg`.
6. In the app, open **Satellite Change**, upload both and click **Detect change**. Adjust sensitivity until the riverbank or sandbar changes show clearly.

Alternative: **Copernicus Browser** (real Sentinel-2 images, free account): https://browser.dataspace.copernicus.eu → search the location → pick two dates with low cloud cover → Download image (PNG).

> Tip: satellite changes happen in the dry season (Nov–May). Pick dates like January vs May of the same or next year. Monsoon images are cloudy.

Run the satellite detection **before** raising a camera alert, and the satellite image is added to the evidence PDF as page 2.

### Member C: citizen report photo
Use `sample_canoe.jpg` or any boat photo. Submit it once (verified), then submit the **same photo again**, and the AI flags it as a **duplicate**.

---

### Team members
| Member |
|---|
| Simmi (lead) |
| Seisha |
| Skyla |
| Manasvi |
---

## Credits / disclosure
- YOLOv8 by Ultralytics (AGPL-3.0), pre-trained on the COCO dataset; ByteTrack tracker
- Streamlit, OpenCV, Folium/Leaflet, OpenStreetMap and Esri World Imagery tiles, ReportLab, Shapely, ImageHash
- Sample image: "Colvale village in Goa, India - Canoe near the river", Wikimedia Commons
- Synthetic data: canoe register, approved zones, camera sites
- Generative AI assistance was used to help write the prototype code and documentation (disclose as the rules require)
