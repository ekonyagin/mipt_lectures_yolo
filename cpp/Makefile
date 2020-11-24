CXX=g++
CFLAGS+=-I/usr/local/include/opencv4/opencv -I/usr/local/include/opencv4
LDFLAGS+= -L/usr/local/lib \
			 -lopencv_gapi \
			 -lopencv_highgui \
			 -lopencv_dnn \
			 -lopencv_videoio \
			 -lopencv_video \
			 -lopencv_imgproc -lopencv_core \
			 -lpthread
SOURCES=yolo_opencv_sequential.cpp
OBJECTS=$(SOURCES:.cpp=.o)
PROGRAM=speaker_tracker

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
	$(CXX) -g   $(OBJECTS) -O3 -o $@ $(LDFLAGS)

.cpp.o: $(patsubst %.cpp,%.o,$(wildcard *.cpp))
	$(CXX) -g  -c -O3 $(CFLAGS)  $< -o $@

clean:
	rm -f $(PROGRAM) $(OBJECTS)

