from ximea import xiapi
import numpy as np
import cv2

class xi_camera_():
    def __init__(self, config, dev_id=0):
        self.config=config
        try:
            self.cam = xiapi.Camera(dev_id=dev_id)
            self.cam.open_device()
            self.cam.set_imgdataformat(self.config.CAMERA_PARAMS['img_format'])
            self.cam.set_exposure(self.config.CAMERA_PARAMS['exposure'])
            self.cam.set_param("gain", self.config.CAMERA_PARAMS['gain'])
            self.cam.set_param("auto_wb", self.config.CAMERA_PARAMS['auto_wb'])
            self.cam.set_param("framerate", self.config.CAMERA_PARAMS['framerate'])
            self.status = 0
        except Exception as e:
            print(f"XI_PROCESSOR: Unable to launch camera! Error is: {e}. Aborting")
            exit(-1)
        print("XI_PROCESSOR: Camera configuration finished successfully!\n")
        self.img = xiapi.Image()
    
    def start(self):
        print("XI_PROCESSOR: Starting acquisition...\n")
        self.cam.start_acquisition()
        self.status = 1
    
    def stop(self):
        print("XI_PROCESSOR: Stopping acquisition...\n")
        self.cam.stop_acquisition()
        self.status = 0
    
    def __del__(self):
        if self.status == 1:
    	    self.cam.stop_acquisition()
        self.cam.close_device()

    def get(self, val):
        if val == cv2.CAP_PROP_FRAME_WIDTH:
            return self.config.CAMERA_PARAMS["width"]
        if val == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.config.CAMERA_PARAMS["height"]

    def read(self):
        self.cam.get_image(self.img)
        frame = self.img.get_image_data_numpy()[self.config.CAMERA_PARAMS["offset_Y"]: \
        self.config.CAMERA_PARAMS["offset_Y"]+ self.config.CAMERA_PARAMS["height"],
                    self.config.CAMERA_PARAMS["offset_X"]: \
                    self.config.CAMERA_PARAMS["offset_X"]+ self.config.CAMERA_PARAMS["width"]]
        return True, frame