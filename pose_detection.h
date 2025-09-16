#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

// Pose landmark structure
struct PoseLandmark {
    float x, y, z;  // 3D coordinates
    float visibility;  // visibility score
    float presence;    // presence score
};

// Pose detection result
struct PoseResult {
    std::vector<PoseLandmark> landmarks;
    float confidence;
    cv::Rect bounding_box;
};

// Pose detection class
class PoseDetector {
public:
    PoseDetector(const std::string& model_path, bool use_gpu = false);
    ~PoseDetector();
    
    // Initialize the pose detection model
    bool initialize();
    
    // Detect pose landmarks for a person crop
    PoseResult detectPose(const cv::Mat& person_crop);
    
    // Detect poses for multiple person crops
    std::vector<PoseResult> detectPoses(const std::vector<cv::Mat>& person_crops);
    
    // Get model input size
    cv::Size getInputSize() const { return input_size_; }
    
    // Get model input shape
    std::vector<int64_t> getInputShape() const { return input_shape_; }

private:
    // Preprocess image for pose detection
    std::vector<float> preprocessPoseImage(const cv::Mat& image);
    
    // Postprocess pose detection results
    PoseResult postprocessPose(const std::vector<float>& output, const cv::Rect& crop_rect);
    
    // Draw pose landmarks on image
    void drawPoseLandmarks(cv::Mat& image, const PoseResult& pose_result);
    
    // MediaPipe pose landmark connections for drawing
    static const std::vector<std::pair<int, int>> POSE_CONNECTIONS;
    
    // MediaPipe pose landmark names
    static const std::vector<std::string> POSE_LANDMARK_NAMES;

private:
    std::string model_path_;
    bool use_gpu_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    
    // Model input/output info
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    cv::Size input_size_;
    std::string input_name_;
    std::string output_name_;
    
    // Preprocessing parameters
    float scale_factor_;
    cv::Point2f offset_;
};

// Utility functions for pose detection
namespace PoseUtils {
    // Convert MediaPipe landmark to OpenCV point
    cv::Point2f landmarkToPoint(const PoseLandmark& landmark, const cv::Size& image_size);
    
    // Calculate pose bounding box from landmarks
    cv::Rect calculatePoseBoundingBox(const std::vector<PoseLandmark>& landmarks, const cv::Size& image_size);
    
    // Filter landmarks by visibility threshold
    std::vector<PoseLandmark> filterLandmarksByVisibility(const std::vector<PoseLandmark>& landmarks, float threshold = 0.5f);
    
    // Calculate pose key angles (e.g., elbow, knee angles)
    std::vector<float> calculatePoseAngles(const std::vector<PoseLandmark>& landmarks);
}