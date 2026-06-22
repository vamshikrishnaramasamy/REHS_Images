import argparse
import os
import time

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Live stop-sign detector on Raspberry Pi (Picamera2).")
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--save", default="latest_detection.jpg",
                        help="Path to write the latest annotated frame (headless viewing).")
    parser.add_argument("--frames", type=int, default=0,
                        help="Stop after N frames (0 = run forever).")
    args = parser.parse_args()

    width, height = args.width, args.height
    model = YOLO(args.model)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (width, height)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)  # let auto-exposure settle

    have_display = bool(os.environ.get("DISPLAY"))
    last_time = time.perf_counter()
    fps = 0.0
    count = 0

    print(f"Live detector running at {width}x{height} on Pi camera. Ctrl+C to quit.")
    try:
        while True:
            frame = picam2.capture_array()  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = model.predict(frame, imgsz=(height, width), conf=args.conf, verbose=False)
            annotated = results[0].plot()

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
            last_time = now

            n = len(results[0].boxes)
            if n:
                confs = [f"{c:.2f}" for c in results[0].boxes.conf.tolist()]
                print(f"[{count}] {n} stop sign(s)  conf={confs}  {fps:.1f} FPS")

            cv2.putText(annotated, f"{width}x{height}  {fps:.1f} FPS",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            if args.save:
                cv2.imwrite(args.save, annotated)

            if have_display:
                cv2.imshow("Stop sign detector (Pi)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            count += 1
            if args.frames and count >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        if have_display:
            cv2.destroyAllWindows()
        print(f"Stopped after {count} frames. Last annotated frame: {args.save}")


if __name__ == "__main__":
    main()
