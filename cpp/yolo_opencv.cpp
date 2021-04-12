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


float confThreshold, nmsThreshold;
std::vector<std::string> classes;

inline void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale,
                       const cv::Scalar& mean, bool swapRB);

void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& out, cv::dnn::Net& net, int backend, int& region);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame);

void callback(int pos, void* userdata);

bool AddPoint(const cv::Point& point, const std::vector<cv::Point>& points);

inline int define_bounding_rect(const int& region_id);

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
    cv::TickMeter tm;
    std::mutex mutex;
};

int main(int argc, char** argv)
{
    
    confThreshold = 0.5;
    nmsThreshold = 0.4;
    float scale = 0.005;
    cv::Scalar mean = 0;
    bool swapRB = true;
    int inpWidth = 416;
    int inpHeight = 416;
    size_t asyncNumReq = 0;
    
    std::string modelPath = "yolov3.weights";
    std::string configPath = "yolov3.cfg";
  
    int region = 0;

    // Open file with classes names.
    
    std::string file = "classes.txt";
    std::ifstream ifs(file.c_str());
    if (!ifs.is_open())
        CV_Error(cv::Error::StsError, "File " + file + " not found");
    std::string line;
    while (std::getline(ifs, line))
    {
        classes.push_back(line);
    }
    

    // Load a model.
    cv::dnn::Net net = cv::dnn::readNet(modelPath, configPath);
    int backend = cv::dnn::DNN_BACKEND_CUDA;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();

    // Create a window
    static const std::string kWinName = "Original video preview";
    static const std::string speaker_window = "Speaker view";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
    cv::namedWindow(speaker_window, cv::WINDOW_NORMAL);

    //int initialConf = (int)(confThreshold * 100);
    //createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback);

    // Open a video file or an image file or a camera stream.
    cv::VideoCapture cap;

    cap.open("Clip0004.MXF");

    bool process = true;

    // Frames capturing thread
    QueueFPS<cv::Mat> framesQueue;
    QueueFPS<cv::Mat> originalFrames;
    cv::Mat original_frame;

    std::thread framesThread([&](){
        cv::Mat frame, frame1;
        while (process)
        {
            cap >> frame1;
            
            if (!frame1.empty()){
                cv::resize(frame1, frame, cv::Size(), 0.2, 0.2);
                
                framesQueue.push(frame);
                originalFrames.push(frame1.clone());
            }
                
            else
                break;
        }
    });

    // Frames processing thread
    QueueFPS<cv::Mat> processedFramesQueue;
    QueueFPS<std::vector<cv::Mat> > predictionsQueue;
    std::thread processingThread([&](){
        //std::queue<AsyncArray> futureOutputs;
        cv::Mat blob;
        while (process)
        {
            // Get a next frame
            cv::Mat frame;
            {
                if (!framesQueue.empty())
                {
                    frame = framesQueue.get();
                    framesQueue.clear();  // Skip the rest of frames
                }
            }

            // Process the frame
            if (!frame.empty())
            {
                preprocess(frame, net, cv::Size(inpWidth, inpHeight), scale, mean, swapRB);
                processedFramesQueue.push(frame);

                std::vector<cv::Mat> outs;
                net.forward(outs, outNames);
                predictionsQueue.push(outs);
            }

    
        }
    });

    // Postprocessing and rendering loop
    while (cv::waitKey(1) < 0)
    {
        if (predictionsQueue.empty())
            continue;

        std::vector<cv::Mat> outs = predictionsQueue.get();
        cv::Mat frame = processedFramesQueue.get();
        
        if (!originalFrames.empty())
        {
            original_frame = originalFrames.get();
            //originalFrames.pop();
        }
        
        postprocess(frame, outs, net, backend, region);


        if (predictionsQueue.counter > 1)
        {
            std::string label = cv::format("Camera: %.2f FPS", framesQueue.getFPS());
            cv::putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

            label = cv::format("Network: %.2f FPS", predictionsQueue.getFPS());
            cv::putText(frame, label, cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));

            label = cv::format("Skipped frames: %d", framesQueue.counter - predictionsQueue.counter);
            cv::putText(frame, label, cv::Point(0, 45), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        }
        cv::imshow(kWinName, frame);
        //printf("Detecting rect..\n");
        cv::Mat crop = original_frame(cv::Rect(define_bounding_rect(region), 
                                                370, 
                                                1920,
                                                1080));
        
        cv::imshow(speaker_window, crop);
    }

    process = false;
    framesThread.join();
    processingThread.join();  
    return 0;
}

inline void preprocess(const cv::Mat& frame, cv::dnn::Net& net, cv::Size inpSize, float scale,
                       const cv::Scalar& mean, bool swapRB)
{
    static cv::Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0) inpSize.width = frame.cols;
    if (inpSize.height <= 0) inpSize.height = frame.rows;
    cv::dnn::blobFromImage(frame, blob, 1.0, inpSize, cv::Scalar(), swapRB, false, CV_8U);

    // Run a model.
    net.setInput(blob, "", scale, mean);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
    {
        cv::resize(frame, frame, inpSize);
        cv::Mat imInfo = (cv::Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
}

void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net& net, int backend, int& region)
{
    int coords[2] = {0,0};
  
    cv::Size s = frame.size();
    int Width = s.height;
    int Height = s.width;

    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Point> centers;
    
    if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                
                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
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

                        
                        if (AddPoint(cv::Point(centerX, centerY), centers)){
                            centers.push_back(cv::Point(centerX, centerY));
                            classIds.push_back(classIdPoint.x);
                            confidences.push_back((float)confidence);
                            boxes.push_back(cv::Rect(left, top, width, height));
                        }
                        
                        //break;
                    }

                }
                
            }
        }
    }

    else
        CV_Error(cv::Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
    //printf("Confidence size is %d\n", (int)confidences.size());
    
    if(centers.size()==1){
        int loc = centers[0].x;
        region = 3 - Width / loc;
        printf("Width is %d\n", Width);
        if (region<1)
            region = 1;
        //printf("region is %d\n", region);
    }
    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        cv::Rect box = boxes[idx];
        if (classIds[idx] == 0)
            drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));

    std::string label = cv::format("%.2f", conf);
    
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - labelSize.height),
              cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
    //putText(frame, std::to_string(classId), Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void callback(int pos, void*)
{
    confThreshold = pos * 0.01f;
}

bool AddPoint(const cv::Point& point, const std::vector<cv::Point>& points)
{
    if (points.empty())
        return true;
    for(size_t j=0; j<points.size(); j++)
    {
        int diff_x = point.x - points[j].x, diff_y = point.y - points[j].y;
        if (diff_x*diff_x + diff_y*diff_y > 10000)
            return true;
    }
    return false;
}

inline int define_bounding_rect(const int& region_id)
{
    if (region_id == 1)
        return 0;
    if (region_id == 3)
        return 3840 - 1920;
    return 960;
}
