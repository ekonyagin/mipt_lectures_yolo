import argparse
import cv2
from config import camera_cfg, calibration_cfg

from src.ximea_processor import xi_camera_
from src.results_postprocessing import make_measurement_results
from src.yolo import YOLO


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--network', default="tiny", help='Network Type: normal / tiny / prn')
    ap.add_argument('-d', '--device', default=0, help='Device to use')
    ap.add_argument('-s', '--size', default=256, help='Size for yolo')
    ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
    args = ap.parse_args()

    if args.network == "normal":
        print("loading yolo...")
        yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
    elif args.network == "prn":
        print("loading yolo-tiny-prn...")
        yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
    else:
        print("loading yolo-tiny...")
        yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

    yolo.size = int(args.size)
    yolo.confidence = float(args.confidence)

    print("starting XIMEA...")
    cv2.namedWindow("preview")

    cam = xi_camera_(config=camera_cfg)
    cam.start()
    
    widths = []
    heights = []
    rotations = []

    while 1:
        ret, frame = cam.read()
        
        frame = cv2.resize(frame, (camera_cfg.PREPROCESSING["downsample"],camera_cfg.PREPROCESSING["downsample"]))
        width, height, inference_time, results = yolo.inference(frame)
        for detection in results:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)

            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            dist  = "dist: %f" % (w*(calibration_cfg.CALIBRATIONS["width_coef"]) + \
            calibration_cfg.CALIBRATIONS["width_offset"])
            rotation = "rot: %f" % (1.*w/h)
            cv2.putText(frame, dist,(x, y - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            cv2.putText(frame, rotation,(x, y - 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            widths.append(w)
            heights.append(h)
            #areas.append(w * h)
            rotations.append(1.*w/h)

        cv2.imshow("preview", frame)

        key = cv2.waitKey(20)
        if key == 27:
            break
    cam.stop()
    cv2.destroyWindow("preview")
    make_measurement_results(widths, heights, rotations)
