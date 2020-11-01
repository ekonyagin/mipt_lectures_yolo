#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

#ifdef CV_CXX11
#include <mutex>
#include <thread>
#include <queue>
#endif

//#include "common.hpp"

using namespace cv;
using namespace dnn;

float confThreshold, nmsThreshold;
std::vector<std::string> classes;

inline void preprocess(const Mat& frame, Net& net, Size inpSize, float scale,
                       const Scalar& mean, bool swapRB);

void postprocess(Mat& frame, const std::vector<Mat>& out, Net& net, int backend, int& region);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

void callback(int pos, void* userdata);

bool AddPoint(const Point& point, const std::vector<Point>& points);

#ifdef CV_CXX11
template <typename T>
class QueueFPS : public std::queue<T>
{
public:
    QueueFPS() : counter(0) {}

    void push(const T& entry)
    {
        std::lock_guard<std::mutex> lock(mutex);

        std::queue<T>::push(entry);
        counter += 1;
        if (counter == 1)
        {
            // Start counting from a second frame (warmup).
            tm.reset();
            tm.start();
        }
    }

    T get()
    {
        std::lock_guard<std::mutex> lock(mutex);
        T entry = this->front();
        this->pop();
        return entry;
    }

    float getFPS()
    {
        tm.stop();
        double fps = counter / tm.getTimeSec();
        tm.start();
        return static_cast<float>(fps);
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex);
        while (!this->empty())
            this->pop();
    }

    unsigned int counter;

private:
    TickMeter tm;
    std::mutex mutex;
};
#endif  // CV_CXX11

int main(int argc, char** argv)
{
    
    confThreshold = 0.5;
    nmsThreshold = 0.4;
    float scale = 0.005;
    Scalar mean = 0;
    bool swapRB = true;
    int inpWidth = 416;
    int inpHeight = 416;
    size_t asyncNumReq = 0;
    
    std::string modelPath = "yolov3.weights";
    std::string configPath = "yolov3.cfg";
  
    int region = 0;

    // Open file with classes names.
    if(1)
    {
        std::string file = "classes.txt";
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    // Load a model.
    Net net = readNet(modelPath, configPath);
    int backend = cv::dnn::DNN_BACKEND_CUDA;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    // Create a window
    static const std::string kWinName = "Original video preview";
    namedWindow(kWinName, WINDOW_NORMAL);
    //int initialConf = (int)(confThreshold * 100);
    //createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;

    cap.open("Clip0166.MXF");

//#ifdef CV_CXX11
    bool process = true;

    // Frames capturing thread
    QueueFPS<Mat> framesQueue;
    std::thread framesThread([&](){
        Mat frame1, frame;
        while (process)
        {
            cap >> frame1;
            
            if (!frame1.empty()){
                resize(frame1, frame, cv::Size(), 0.2, 0.2);
                framesQueue.push(frame.clone());
            }
                
            else
                break;
        }
    });

    // Frames processing thread
    QueueFPS<Mat> processedFramesQueue;
    QueueFPS<std::vector<Mat> > predictionsQueue;
    std::thread processingThread([&](){
        std::queue<AsyncArray> futureOutputs;
        Mat blob;
        while (process)
        {
            // Get a next frame
            Mat frame;
            {
                if (!framesQueue.empty())
                {
                    frame = framesQueue.get();
                    if (asyncNumReq)
                    {
                        if (futureOutputs.size() == asyncNumReq)
                            frame = Mat();
                    }
                    else
                        framesQueue.clear();  // Skip the rest of frames
                }
            }

            // Process the frame
            if (!frame.empty())
            {
                preprocess(frame, net, Size(inpWidth, inpHeight), scale, mean, swapRB);
                processedFramesQueue.push(frame);

                std::vector<Mat> outs;
                net.forward(outs, outNames);
                predictionsQueue.push(outs);
            }

            while (!futureOutputs.empty() &&
                   futureOutputs.front().wait_for(std::chrono::seconds(0)))
            {
                AsyncArray async_out = futureOutputs.front();
                futureOutputs.pop();
                Mat out;
                async_out.get(out);
                predictionsQueue.push({out});
            }
        }
    });

    // Postprocessing and rendering loop
    while (waitKey(1) < 0)
    {
        if (predictionsQueue.empty())
            continue;

        std::vector<Mat> outs = predictionsQueue.get();
        Mat frame = processedFramesQueue.get();

        postprocess(frame, outs, net, backend, region);

        if (predictionsQueue.counter > 1)
        {
            std::string label = format("Camera: %.2f FPS", framesQueue.getFPS());
            putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

            label = format("Network: %.2f FPS", predictionsQueue.getFPS());
            putText(frame, label, Point(0, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

            label = format("Skipped frames: %d", framesQueue.counter - predictionsQueue.counter);
            putText(frame, label, Point(0, 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }
        imshow(kWinName, frame);
    }

    process = false;
    framesThread.join();
    processingThread.join();
/*
#else  // CV_CXX11
    if (asyncNumReq)
        CV_Error(Error::StsNotImplemented, "Asynchronous forward is supported only with Inference Engine backend.");

    // Process frames.
    Mat frame, blob;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        preprocess(frame, net, Size(inpWidth, inpHeight), scale, mean, swapRB);

        std::vector<Mat> outs;
        net.forward(outs, outNames);

        postprocess(frame, outs, net, backend);

        // Put efficiency information.
        std::vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = format("Inference time: %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        imshow(kWinName, frame);
    }
#endif  // CV_CXX11
*/
    return 0;
}

inline void preprocess(const Mat& frame, Net& net, Size inpSize, float scale,
                       const Scalar& mean, bool swapRB)
{
    static Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0) inpSize.width = frame.cols;
    if (inpSize.height <= 0) inpSize.height = frame.rows;
    blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);

    // Run a model.
    net.setInput(blob, "", scale, mean);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        resize(frame, frame, inpSize);
        Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
}

void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net, int backend, int& region)
{
    int coords[2] = {0,0};
  
    Size s = frame.size();
    int Width = s.height;
    int Height = s.width;

    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;
    printf("%s\n", "hello from postprocess 1\n");
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    std::vector<Point> centers;
    
    if (outLayerType == "Region")
    {
        printf("%s\n", "hello from postprocess 2\n");
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (classIdPoint.x == 0){
                    if (confidence > confThreshold)
                    {
                        int centerX = (int)(data[0] * frame.cols);
                        int centerY = (int)(data[1] * frame.rows);
                        int width = (int)(data[2] * frame.cols);
                        int height = (int)(data[3] * frame.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        
                        if (AddPoint(Point(centerX, centerY), centers)){
                            centers.push_back(Point(centerX, centerY));
                            classIds.push_back(classIdPoint.x);
                            confidences.push_back((float)confidence);
                            boxes.push_back(Rect(left, top, width, height));
                        }
                        
                        //break;
                    }

                }
                
            }
        }
    }

    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
    //printf("Confidence size is %d\n", (int)confidences.size());
    
    if(centers.size()==1){
        int loc = centers[0].x;
        region = 3 - Width / loc;
        if (region<1)
            region = 1;
        printf("region is %d\n", region);
    }
    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        Rect box = boxes[idx];
        if (classIds[idx] == 0)
            drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    std::string label = format("%.2f", conf);
    
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
              Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    //putText(frame, std::to_string(classId), Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void callback(int pos, void*)
{
    confThreshold = pos * 0.01f;
}

bool AddPoint(const Point& point, const std::vector<Point>& points)
{
    if (points.empty())
        return true;
    for(size_t j=0; j<points.size(); j++){
        int diff_x = point.x - points[j].x;
        int diff_y = point.y - points[j].y;
        printf("%d %d\n", diff_x, diff_y);
        if (diff_x*diff_x + diff_y*diff_y > 10000)
            return false;
    }
    return false;
}
