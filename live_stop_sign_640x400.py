import argparse
import time

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Live stop-sign detector at 640x400 camera input.")
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    width, height = 640, 400
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    last_time = time.perf_counter()
    fps = 0.0

    print("Live detector running at 640x400. Press q in the video window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        results = model.predict(frame, imgsz=(height, width), conf=args.conf, verbose=False)
        annotated = results[0].plot()

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
        last_time = now
        cv2.putText(
            annotated,
            f"640x400  {fps:.1f} FPS",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Stop sign detector 640x400", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
