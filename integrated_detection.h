#pragma once

#include "pose_detection.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

// Forward declarations for your existing YOLONAS structures
struct Bbox {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

struct PREPROCESS_DATA {
    float scale_x, scale_y;
    int pad_x, pad_y;
    cv::Size original_size;
};

// Measure class for timing (assuming you have this)
class Measure {
public:
    void start();
    void stop();
    int64_t get() const;
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
};

// Integrated detection result
struct IntegratedDetectionResult {
    std::vector<Bbox> detections;
    std::vector<PoseResult> poses;
    int64_t detection_prep_time;
    int64_t detection_inf_time;
    int64_t detection_post_time;
    int64_t pose_prep_time;
    int64_t pose_inf_time;
    int64_t pose_post_time;
    int64_t total_time;
};

// Integrated detector class
class IntegratedDetector {
public:
    IntegratedDetector(
        const std::string& yolo_model_path,
        const std::string& pose_model_path,
        bool use_gpu = false
    );
    
    ~IntegratedDetector();
    
    // Initialize both detection and pose models
    bool initialize();
    
    // Process single frame with both object detection and pose estimation
    IntegratedDetectionResult processFrame(
        cv::Mat& frame,
        const std::vector<std::string>& classes,
        float conf_threshold = 0.5f,
        float iou_threshold = 0.4f,
        float pose_conf_threshold = 0.5f
    );
    
    // Draw both detections and poses on frame
    void drawResults(cv::Mat& frame, const IntegratedDetectionResult& result, const std::vector<std::string>& classes);
    
    // Get detection model info
    std::vector<int64_t> getDetectionInputShape() const;
    std::vector<std::string> getDetectionClasses() const;
    
    // Get pose model info
    cv::Size getPoseInputSize() const;

private:
    // YOLONAS detection functions (you'll need to implement these based on your existing code)
    std::vector<float> preprocessDetection(const cv::Mat& frame, const cv::Size& input_size, PREPROCESS_DATA& metadata);
    std::pair<std::vector<float>, std::vector<float>> runDetectionInference(
        const std::vector<float>& input_tensor, 
        const std::vector<int64_t>& input_shape
    );
    std::vector<Bbox> postprocessDetection(
        const std::vector<float>& boxes_output,
        const std::vector<float>& scores_output,
        const PREPROCESS_DATA& metadata,
        float conf_threshold,
        float iou_threshold
    );
    void drawDetections(cv::Mat& frame, const std::vector<Bbox>& detections, const std::vector<std::string>& classes);
    
    // Extract person crops from detections
    std::vector<cv::Mat> extractPersonCrops(const cv::Mat& frame, const std::vector<Bbox>& detections, const std::vector<std::string>& classes);
    
    // Filter detections for person class
    std::vector<Bbox> filterPersonDetections(const std::vector<Bbox>& detections, const std::vector<std::string>& classes);

private:
    // YOLONAS detection model
    std::string yolo_model_path_;
    std::unique_ptr<Ort::Env> detection_env_;
    std::unique_ptr<Ort::Session> detection_session_;
    std::unique_ptr<Ort::SessionOptions> detection_session_options_;
    std::unique_ptr<Ort::MemoryInfo> detection_memory_info_;
    std::vector<int64_t> detection_input_shape_;
    std::string detection_input_name_;
    std::string detection_boxes_output_name_;
    std::string detection_scores_output_name_;
    
    // Pose detection model
    std::unique_ptr<PoseDetector> pose_detector_;
    
    // Configuration
    bool use_gpu_;
    bool initialized_;
};