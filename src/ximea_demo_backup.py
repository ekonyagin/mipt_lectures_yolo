import argparse
import cv2

from ximea import xiapi
from config import camera_cfg

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
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

cam = xiapi.Camera(dev_id=0)
img = xiapi.Image()
try:
    cam.open_device()

    cam.set_imgdataformat(camera_cfg.CAMERA_PARAMS['img_format'])
    cam.set_exposure(camera_cfg.CAMERA_PARAMS['exposure'])
    cam.set_param("gain", camera_cfg.CAMERA_PARAMS['gain'])
    cam.set_param("auto_wb", camera_cfg.CAMERA_PARAMS['auto_wb'])
    cam.set_param("framerate", camera_cfg.CAMERA_PARAMS['framerate'])

    cam.start_acquisition()
except Exception as e:
    print(f"Unable to launch camera! Error is: {e}. Aborting")
    exit()

rval = True

while rval:
    cam.get_image(img)
    frame = img.get_image_data_numpy()[camera_cfg.CAMERA_PARAMS["offset_Y"]: camera_cfg.CAMERA_PARAMS["offset_Y"]+ camera_cfg.CAMERA_PARAMS["height"],
                camera_cfg.CAMERA_PARAMS["offset_X"]: camera_cfg.CAMERA_PARAMS["offset_X"]+ camera_cfg.CAMERA_PARAMS["width"]]
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

    cv2.imshow("preview", frame)

    key = cv2.waitKey(20)
    if key == 27:
        break

cv2.destroyWindow("preview")
cam.stop_acquisition()
cam.close_device()
