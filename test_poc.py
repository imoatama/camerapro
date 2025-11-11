import cv2, time, numpy as np
import platform
if platform.system() == "Windows":
    import winsound

PREFERRED_DEVICE_NAMES = ["Phone Link", "Phone Link Camera", "スマートフォン連携", "Link to Windows"]

DIFF_THRESH = 8.0     # ← 安定判定のしきい値（まずは 8〜12 で調整）
STABLE_N    = 4       # ← 連続何フレームで安定とみなすか
ENTER_THRESH = 10.0   # ← ベースラインとの差で「物が入った」を判定（要調整）

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
    raise RuntimeError("Phone Linkカメラが見つかりません。Phone Linkの設定をご確認ください。")

def get_roi_rect(frame):
    h, w = frame.shape[:2]
    x1, y1 = int(w*0.15), int(h*0.15)
    x2, y2 = int(w*0.85), int(h*0.85)
    return x1, y1, x2, y2

def prep_gray_small(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    # 計算負荷とノイズを下げるため縮小
    g = cv2.resize(g, (320, 180), interpolation=cv2.INTER_AREA)
    return g

def beep():
    try:
        winsound.Beep(1000, 120)  # 周波数, 長さ(ms)
    except Exception:
        pass

def main():
    cap = open_phone_link_camera()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    prev = None
    base = None   # 空トレイのベースライン
    stable_cnt = 0
    last_trigger_t = 0

    print("[操作] 空のトレイ状態でウィンドウを選択し、キーボードで 'b' を押してベースライン登録。Esc で終了。")

    while True:
        ok, frame = cap.read()
        if not ok: break

        x1, y1, x2, y2 = get_roi_rect(frame)
        roi = frame[y1:y2, x1:x2]
        g = prep_gray_small(roi)

        preview = frame.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0,255,0), 2)

        mean_diff = 0.0
        if prev is not None:
            diff = cv2.absdiff(g, prev)
            mean_diff = float(diff.mean())
            if mean_diff < DIFF_THRESH:
                stable_cnt += 1
            else:
                stable_cnt = 0

        # 入室検知（空→物が入った）
        entered = False
        base_diff = 0.0
        if base is not None:
            base_diff = float(cv2.absdiff(g, base).mean())
            entered = base_diff > ENTER_THRESH

        # 反応トリガ：入室あり & 安定 N フレーム
        now = time.time()
        if entered and stable_cnt >= STABLE_N and now - last_trigger_t > 0.5:
            last_trigger_t = now
            # ▼ここで本来は infer_api(...) を呼ぶ
            beep()
            cv2.putText(preview, "TRIGGERED!", (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 4)
            stable_cnt = 0  # 次の品のためにリセット

        # デバッグ表示
        cv2.putText(preview, f"mean_diff={mean_diff:.1f}  stable={stable_cnt}/{STABLE_N}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        if base is None:
            cv2.putText(preview, "Press 'b' to set BASELINE (empty tray).",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        else:
            cv2.putText(preview, f"base_diff={base_diff:.1f}  ENTER>{ENTER_THRESH}",
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

        cv2.imshow("Phone Link Preview (PoC Debug)", preview)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
        elif k == ord('b'):
            base = g.copy()
            print("[INFO] ベースラインを登録しました（空トレイ）。")

        prev = g

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
