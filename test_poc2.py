import cv2, time, numpy as np, os, csv, platform, datetime
from PIL import Image, ImageDraw, ImageFont
import os

# 日本語フォントを自動検出
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\meiryo.ttc",
    r"C:\Windows\Fonts\YuGothM.ttc",
    r"C:\Windows\Fonts\msgothic.ttc",
    r"C:\Windows\Fonts\MSMINCHO.TTC",
]
def _load_jp_font(size=28):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    # 最後の手段（日本語は出ないがクラッシュは避ける）
    return ImageFont.load_default()

def jp_put_text(bgr_img, text, xy=(30,40), size=28, color=(0,255,0)):
    """BGR(OpenCV)画像に日本語テキストを描画"""
    font = _load_jp_font(size)
    # OpenCV(BGR) -> PIL(RGB)
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    # アウトライン付きで視認性UP（stroke_widthはPillow>=8系）
    draw.text(xy, text, font=font, fill=(color[2], color[1], color[0]),
              stroke_width=2, stroke_fill=(0,0,0))
    # PIL(RGB) -> OpenCV(BGR)
    bgr_img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return bgr_img

if platform.system() == "Windows":
    import winsound

PREFERRED_DEVICE_NAMES = ["Phone Link", "Phone Link Camera", "スマートフォン連携", "Link to Windows"]

# ---- 調整パラメータ（現場で微調整） ----
DIFF_THRESH   = 8.0   # 安定判定（下げる=敏感、上げる=鈍感）
STABLE_N      = 4     # 安定フレーム連続数
ENTER_THRESH  = 12.0  # ベースラインとの差→入室判定
AREA_MIN      = 1200  # ▼count用：輪郭の最小面積（[ と ]で変更可能）
AREA_MAX      = 999999  # 必要なら ; と ' で調整
BINARY_INV    = True  # 黒地/白地で反転が必要なとき切替

# 理由タグ（triage用）
REASONS = ["角スレ(F1)", "糸ほつれ(F2)", "汚れ(F3)", "金具変色(F4)", "メッキ剥がれ(F5)", "箱潰れ(F6)", "金具変形(F7)"]
REASON_KEYS = {ord('1'): "再販可", ord('2'): "要修理", ord('3'): "廃棄"}

# 出力フォルダ
os.makedirs("captures/triage_unlabeled", exist_ok=True)
os.makedirs("captures/triage_labeled", exist_ok=True)
os.makedirs("captures/count", exist_ok=True)
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/poc_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts","mode","count","class","reasons","mean_diff","base_diff","latency_ms","img_path"])

def beep_ok(): 
    try: winsound.Beep(1200, 100)
    except: pass
def beep_ng():
    try: winsound.Beep(600, 160)
    except: pass

def open_phone_link_camera():
    for name in PREFERRED_DEVICE_NAMES:
        cap = cv2.VideoCapture(f"video={name}", cv2.CAP_DSHOW)
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"[INFO] Opened by name: {name}")
            return cap
        cap.release()
    for idx in range(0, 10):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"[INFO] Opened by index: {idx}")
            return cap
        cap.release()
    raise RuntimeError("Phone Linkカメラが見つかりません。設定をご確認ください。")

def get_roi_rect(frame):
    h, w = frame.shape[:2]
    x1, y1 = int(w*0.15), int(h*0.15)
    x2, y2 = int(w*0.85), int(h*0.85)
    return x1, y1, x2, y2

def prep_gray_small(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    return g

def to_binary(gray):
    # 二値化：画面条件で adaptive / Otsu 切り替えも可
    if BINARY_INV:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=1)
    return th

def count_objects(roi):
    gray = prep_gray_small(roi)
    th = to_binary(gray)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        if AREA_MIN <= a <= AREA_MAX:
            x,y,w,h = cv2.boundingRect(c)
            # 画面サイズ差補正（縮小→元座標へ）
            scale_x = roi.shape[1]/gray.shape[1]
            scale_y = roi.shape[0]/gray.shape[0]
            boxes.append((
                int(x*scale_x),
                int(y*scale_y),
                int(w*scale_x),
                int(h*scale_y)
            ))
    return len(boxes), boxes

def save_log(mode, count, klass, reasons, mean_diff, base_diff, latency_ms, img_path):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            mode, count, klass, "|".join(reasons) if reasons else "",
            f"{mean_diff:.1f}", f"{base_diff:.1f}", f"{latency_ms:.0f}", img_path or ""
        ])

def main():
    global AREA_MIN, AREA_MAX
    cap = open_phone_link_camera()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    prev_small = None
    base_small = None
    stable_cnt = 0
    last_trigger_t = 0
    mode = "count"   # "count" or "triage"
    awaiting_label = False
    pending_img = None
    pending_info = None
    selected_reasons = set()

    print("[操作] 空トレイで 'b' → ベースライン登録。'm'：モード切替（count/triage）。Esc：終了。")
    print("[count] '['/']'：AREA_MIN 調整, ';'/'\''：AREA_MAX 調整, 'v'：反転切替")
    print("[triage] 1=再販可, 2=要修理, 3=廃棄, F1..F5=理由タグ, Enter=確定保存")

    while True:
        ok, frame = cap.read()
        if not ok: break

        x1, y1, x2, y2 = get_roi_rect(frame)
        roi = frame[y1:y2, x1:x2]
        small = cv2.resize(prep_gray_small(roi), (320, 180), interpolation=cv2.INTER_AREA)

        preview = frame.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0,255,0), 2)

        mean_diff = 0.0
        base_diff = 0.0
        entered = False

        if prev_small is not None:
            d = cv2.absdiff(small, prev_small).mean()
            mean_diff = float(d)
            if mean_diff < DIFF_THRESH:
                stable_cnt += 1
            else:
                stable_cnt = 0

        if base_small is not None:
            base_diff = float(cv2.absdiff(small, base_small).mean())
            entered = base_diff > ENTER_THRESH

        now = time.time()
        triggered = False
        if (not awaiting_label) and entered and stable_cnt >= STABLE_N and now - last_trigger_t > 0.5:
            last_trigger_t = now
            triggered = True

        latency_ms = 0.0
        overlay_text = f"mode={mode}  mean_diff={mean_diff:.1f}  stable={stable_cnt}/{STABLE_N}  base_diff={base_diff:.1f}  AREA_MIN={AREA_MIN}"

        if triggered:
            t0 = time.time()
            if mode == "count":
                n, boxes = count_objects(roi)
                latency_ms = (time.time() - t0) * 1000
                # 表示
                for (bx,by,bw,bh) in boxes:
                    cv2.rectangle(roi, (bx,by), (bx+bw,by+bh), (0,0,255), 2)
                #cv2.putText(preview, f"COUNT: {n}", (x1+20, y1+60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,255), 4)
                jp_put_text(preview, f"COUNT: {n}", xy=(x1+20, y1+60), size=24, color=(0,0,255))
                beep_ok()
                # 保存＆ログ
                img_path = f"captures/count/{int(time.time())}_{n}.jpg"
                cv2.imwrite(img_path, roi)
                save_log("count", n, "", [], mean_diff, base_diff, latency_ms, img_path)
            else:  # triage = 収集＆ラベリング待ち
                latency_ms = (time.time() - t0) * 1000
                pending_img = roi.copy()
                awaiting_label = True
                selected_reasons = set()
                beep_ok()

        # triage ラベリングUI
        if awaiting_label and pending_img is not None:
            #cv2.putText(preview, "TRIAGE LABEL: 1=再販可  2=要修理  3=廃棄   F1..F5=理由  Enter=保存",
            jp_put_text(preview, "TRIAGE LABEL: 1=再販可  2=要修理  3=廃棄   F1..F5=理由  Enter=保存",
                        xy=(40, y2+40 if y2+40 < preview.shape[0]-10 else preview.shape[0]-10),
                        size=24,color=(0,0,255))

        # 情報表示
        #cv2.putText(preview, overlay_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        jp_put_text(preview, overlay_text, xy=(30, 40), size=24,color=(0,255,0))
        cv2.imshow("Phone Link Preview (PoC v2)", preview)

        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
        elif k == ord('m'):
            mode = "triage" if mode == "count" else "count"
            print(f"[INFO] mode: {mode}")
        elif k == ord('b'):
            base_small = small.copy()
            print("[INFO] ベースライン登録")
        elif k == ord('['):
            AREA_MIN = max(100, AREA_MIN - 200); print("AREA_MIN:", AREA_MIN)
        elif k == ord(']'):
            AREA_MIN += 200; print("AREA_MIN:", AREA_MIN)
        elif k == ord(';'):
            AREA_MAX = max(AREA_MIN+100, AREA_MAX - 1000); print("AREA_MAX:", AREA_MAX)
        elif k == ord('\''):
            AREA_MAX += 1000; print("AREA_MAX:", AREA_MAX)
        elif k == ord('v'):
            global BINARY_INV; BINARY_INV = not BINARY_INV
            print("BINARY_INV:", BINARY_INV)

        # triage キー処理
        if awaiting_label:
            if k in (ord('1'), ord('2'), ord('3')):
                pending_class = REASON_KEYS[k]
                pending_info = {"class": pending_class}
                print("[TRIAGE] class:", pending_class)
            # 理由（F1..F5）
            if k == 0x70: selected_reasons.symmetric_difference_update(["角スレ(F1)"])       # F1
            if k == 0x71: selected_reasons.symmetric_difference_update(["糸ほつれ(F2)"])     # F2
            if k == 0x72: selected_reasons.symmetric_difference_update(["汚れ(F3)"])         # F3
            if k == 0x73: selected_reasons.symmetric_difference_update(["金具変色(F4)"])     # F4
            if k == 0x74: selected_reasons.symmetric_difference_update(["メッキ剥がれ(F5)"]) # F5
            if k == 0x75: selected_reasons.symmetric_difference_update(["箱潰れ(F6)"]) # F6
            if k == 0x76: selected_reasons.symmetric_difference_update(["金具変形(F7)"]) # F7

            if k == 13:  # Enterで保存
                if pending_img is not None and pending_info is not None:
                    fn = f"{int(time.time())}_{pending_info['class']}.jpg"
                    out_path = os.path.join("captures/triage_labeled", fn)
                    cv2.imwrite(out_path, pending_img)
                    save_log("triage", "", pending_info["class"], sorted(selected_reasons), mean_diff, base_diff, 0.0, out_path)
                    print("[TRIAGE] saved:", out_path, "reasons:", selected_reasons)
                    pending_img = None; pending_info = None; selected_reasons = set()
                    awaiting_label = False
                else:
                    beep_ng()

        prev_small = small

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
