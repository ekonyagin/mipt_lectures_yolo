CXX=g++
CFLAGS+=-I/usr/local/include/opencv4/opencv -I/usr/local/include/opencv4
LDFLAGS+= -L/usr/local/lib -lopencv_gapi -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired \
			 -lopencv_ccalib -lopencv_cudabgsegm -lopencv_cudafeatures2d -lopencv_cudaobjdetect -lopencv_cudastereo \
			 -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype \
			 -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_quality -lopencv_reg \
			 -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres \
			 -lopencv_cudacodec -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot \
			 -lopencv_videostab -lopencv_cudaoptflow -lopencv_optflow -lopencv_cudalegacy -lopencv_videoio -lopencv_cudawarping\
			 -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect \
			 -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo \
			 -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_imgproc -lopencv_cudaarithm -lopencv_core -lopencv_cudev \
			 -lpthread
SOURCES=yolo_opencv.cpp
OBJECTS=$(SOURCES:.cpp=.o)
PROGRAM=speaker_tracker

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
	$(CXX) -g   $(OBJECTS) -O3 -o $@ $(LDFLAGS)

.cpp.o: $(patsubst %.cpp,%.o,$(wildcard *.cpp))
	$(CXX) -g  -c -O3 $(CFLAGS)  $< -o $@

clean:
	rm -f $(PROGRAM) $(OBJECTS)

