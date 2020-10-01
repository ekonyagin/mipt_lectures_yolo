import numpy as np
import cv2
import json

def define_region(center_x, line_marker):
	if center_x < line_marker:
		return 0
	if center_x < 2*line_marker:
		return 1
	return 2

def define_bounding_rect(region_id):
	if region_id == 0:
		return 0
	if region_id == 2:
		return 3840 - 1920
	return 960

def draw_frame(img, obj, frame_nr):
    if obj.get(str(frame_nr+1)) != None:
        top_left = (obj[str(frame_nr+1)]["center_x"]-obj[str(frame_nr+1)]["width"]/2,
                   obj[str(frame_nr+1)]["center_y"]-obj[str(frame_nr+1)]["height"]/2)
        #print(top_left)
        bottom_right = (obj[str(frame_nr+1)]["center_x"]+obj[str(frame_nr+1)]["width"]/2,
                       obj[str(frame_nr+1)]["center_y"]+obj[str(frame_nr+1)]["height"]/2)
        cv2.rectangle(img, (int(top_left[0]), int(top_left[1])), 
                      (int(bottom_right[0]), int(bottom_right[1])),(0,255,0),3)
        return (obj[str(frame_nr+1)]["center_x"], obj[str(frame_nr+1)]["center_y"])
    return None

if __name__ == "__main__":
	cap = cv2.VideoCapture("./data/Clip0166.MXF")
	ret, frame = cap.read()
	frame1 = frame.copy()
	with open("result.json") as f:
	    a = json.load(f)

	obj = {}
	for i in range(len(a)):
	    for j in a[i]['objects']:
	        if j['name'] == "person":
	            #print(i, j['relative_coordinates'])
	            obj[str(i)] = j['relative_coordinates']
	            obj[str(i)]['center_x'] *=frame.shape[1]
	            obj[str(i)]['center_y'] *=frame.shape[0]
	            obj[str(i)]['width'] *=frame.shape[1]
	            obj[str(i)]['height'] *=frame.shape[0]

	i = 1
	line_marker 		   = frame.shape[1]//3
	border_marker 		   = frame.shape[0]

	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (10,500)
	fontScale              = 1
	fontColor              = (255,0,0)
	lineType               = 2

	Y_TOP = 370 ### WILL BE DEFINED BY BOARD DETECTOR

	region_ids = np.array([1 for _ in range(30)])
	region_id = 1
	region_id_ = 1
	region_switch_delay = 50
	delay = 0

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output1.mp4', fourcc, 30.0, (1920,2160))

	while (cv2.waitKey(1) < 0 and ret == True):
	    for j in range(1,3):
	    	frame = cv2.line(frame, (j*line_marker, 0), 
	    				(j*line_marker,border_marker), (0,0,255), 10)
	    c_ = draw_frame(frame, obj, i) ##### PREDICTIONS WILL BE PARSED HERE
	    if c_ is not None:
	    	region_id_ = define_region(c_[0], line_marker)
	    	region_ids = np.roll(region_ids, -1)
	    	region_ids[-1] = region_id_
	    	
	    	cv2.putText(frame, str(region_id), 
			    bottomLeftCornerOfText, 
			    font, 
			    fontScale,
			    fontColor,
			    lineType)
	    
	    if (region_id != region_id_) and (delay > region_switch_delay):
	    	region_id = region_id_
	    	delay = 0
	    if (region_id != region_id_) and (delay <= region_switch_delay):
	    	delay +=1
	    if (region_id == region_id_):
	    	delay = 0

	    cv2.rectangle(frame, (define_bounding_rect(region_id), Y_TOP), 
                      (define_bounding_rect(region_id) + 1920, Y_TOP + 1080),(0,255,0),3)
	    frame_cropped = frame1[Y_TOP:Y_TOP + 1080, define_bounding_rect(region_id): define_bounding_rect(region_id)+1920,:]
	    frame = cv2.resize(frame, (1920, 1080))
	    
	    final_img = cv2.vconcat([frame, frame_cropped])
	    cv2.imshow("demo", final_img)
	    out.write(final_img)
	    ret, frame = cap.read()
	    frame1 = frame.copy()
	    i += 1
