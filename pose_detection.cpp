#include "pose_detection.h"
#include <iostream>
#include <algorithm>
#include <cmath>

// MediaPipe pose landmark connections for drawing skeleton
const std::vector<std::pair<int, int>> PoseDetector::POSE_CONNECTIONS = {
    // Face
    {0, 1}, {1, 2}, {2, 3}, {3, 7},
    {0, 4}, {4, 5}, {5, 6}, {6, 8},
    // Torso
    {9, 10}, {11, 12}, {11, 13}, {13, 15}, {15, 17}, {15, 19}, {15, 21}, {17, 19},
    {12, 14}, {14, 16}, {16, 18}, {16, 20}, {16, 22}, {18, 20},
    // Arms
    {11, 23}, {12, 24}, {23, 24},
    // Left arm
    {11, 13}, {13, 15},
    // Right arm  
    {12, 14}, {14, 16},
    // Left leg
    {23, 25}, {25, 27}, {27, 29}, {29, 31}, {27, 31},
    // Right leg
    {24, 26}, {26, 28}, {28, 30}, {30, 32}, {28, 32}
};

// MediaPipe pose landmark names
const std::vector<std::string> PoseDetector::POSE_LANDMARK_NAMES = {
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
};

PoseDetector::PoseDetector(const std::string& model_path, bool use_gpu)
    : model_path_(model_path), use_gpu_(use_gpu) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PoseDetector");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
}

PoseDetector::~PoseDetector() = default;

bool PoseDetector::initialize() {
    try {
        // Configure session options
        if (use_gpu_) {
            // Try to use GPU if available
            try {
                OrtCUDAProviderOptions cuda_options{};
                session_options_->AppendExecutionProvider_CUDA(cuda_options);
            } catch (const std::exception& e) {
                std::cout << "CUDA not available, falling back to CPU: " << e.what() << std::endl;
            }
        }
        
        // Set optimization level
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_options_->SetIntraOpNumThreads(4);
        
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, model_path_.c_str(), *session_options_);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t num_input_nodes = session_->GetInputCount();
        if (num_input_nodes != 1) {
            std::cerr << "Expected 1 input node, got " << num_input_nodes << std::endl;
            return false;
        }
        
        input_name_ = session_->GetInputName(0, allocator);
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = input_tensor_info.GetShape();
        
        // MediaPipe pose model typically expects [1, 3, 256, 256] input
        if (input_shape_.size() != 4 || input_shape_[0] != 1 || input_shape_[1] != 3) {
            std::cerr << "Unexpected input shape: expected [1, 3, H, W], got [";
            for (size_t i = 0; i < input_shape_.size(); ++i) {
                std::cerr << input_shape_[i];
                if (i < input_shape_.size() - 1) std::cerr << ", ";
            }
            std::cerr << "]" << std::endl;
            return false;
        }
        
        input_size_ = cv::Size(input_shape_[3], input_shape_[2]); // W, H
        
        // Output info
        size_t num_output_nodes = session_->GetOutputCount();
        if (num_output_nodes != 1) {
            std::cerr << "Expected 1 output node, got " << num_output_nodes << std::endl;
            return false;
        }
        
        output_name_ = session_->GetOutputName(0, allocator);
        auto output_type_info = session_->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_shape_ = output_tensor_info.GetShape();
        
        std::cout << "Pose detector initialized successfully" << std::endl;
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            std::cout << input_shape_[i];
            if (i < input_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Input size: " << input_size_.width << "x" << input_size_.height << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize pose detector: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> PoseDetector::preprocessPoseImage(const cv::Mat& image) {
    cv::Mat resized, rgb_image;
    
    // Resize image to model input size
    cv::resize(image, resized, input_size_);
    
    // Convert BGR to RGB
    cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);
    
    // Normalize to [0, 1]
    rgb_image.convertTo(rgb_image, CV_32F, 1.0 / 255.0);
    
    // Convert to CHW format
    std::vector<cv::Mat> channels;
    cv::split(rgb_image, channels);
    
    std::vector<float> input_tensor;
    input_tensor.reserve(input_shape_[1] * input_shape_[2] * input_shape_[3]);
    
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_shape_[2]; ++h) {
            for (int w = 0; w < input_shape_[3]; ++w) {
                input_tensor.push_back(channels[c].at<float>(h, w));
            }
        }
    }
    
    return input_tensor;
}

PoseResult PoseDetector::postprocessPose(const std::vector<float>& output, const cv::Rect& crop_rect) {
    PoseResult result;
    result.bounding_box = crop_rect;
    
    // MediaPipe pose model outputs landmarks in format [x, y, z, visibility, presence]
    // Assuming output shape is [1, 1, 33, 5] for 33 landmarks with 5 values each
    const int num_landmarks = 33;
    const int values_per_landmark = 5;
    
    if (output.size() < num_landmarks * values_per_landmark) {
        std::cerr << "Unexpected output size: expected at least " 
                  << num_landmarks * values_per_landmark 
                  << ", got " << output.size() << std::endl;
        return result;
    }
    
    result.landmarks.reserve(num_landmarks);
    
    for (int i = 0; i < num_landmarks; ++i) {
        PoseLandmark landmark;
        int base_idx = i * values_per_landmark;
        
        // Convert normalized coordinates to crop coordinates
        landmark.x = output[base_idx] * crop_rect.width + crop_rect.x;
        landmark.y = output[base_idx + 1] * crop_rect.height + crop_rect.y;
        landmark.z = output[base_idx + 2];
        landmark.visibility = output[base_idx + 3];
        landmark.presence = output[base_idx + 4];
        
        result.landmarks.push_back(landmark);
    }
    
    // Calculate overall confidence as average of visible landmarks
    float total_visibility = 0.0f;
    int visible_count = 0;
    for (const auto& landmark : result.landmarks) {
        if (landmark.visibility > 0.5f) {
            total_visibility += landmark.visibility;
            visible_count++;
        }
    }
    
    result.confidence = visible_count > 0 ? total_visibility / visible_count : 0.0f;
    
    return result;
}

PoseResult PoseDetector::detectPose(const cv::Mat& person_crop) {
    if (!session_) {
        std::cerr << "Pose detector not initialized" << std::endl;
        return PoseResult{};
    }
    
    try {
        // Preprocess image
        std::vector<float> input_tensor = preprocessPoseImage(person_crop);
        
        // Create input tensor
        auto input_tensor_ptr = Ort::Value::CreateTensor<float>(
            *memory_info_, input_tensor.data(), input_tensor.size(),
            input_shape_.data(), input_shape_.size());
        
        // Run inference
        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                          input_names, &input_tensor_ptr, 1,
                                          output_names, 1);
        
        // Extract output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        std::vector<float> output(output_data, output_data + output_size);
        
        // Postprocess results
        return postprocessPose(output, cv::Rect(0, 0, person_crop.cols, person_crop.rows));
        
    } catch (const std::exception& e) {
        std::cerr << "Pose detection failed: " << e.what() << std::endl;
        return PoseResult{};
    }
}

std::vector<PoseResult> PoseDetector::detectPoses(const std::vector<cv::Mat>& person_crops) {
    std::vector<PoseResult> results;
    results.reserve(person_crops.size());
    
    for (const auto& crop : person_crops) {
        results.push_back(detectPose(crop));
    }
    
    return results;
}

void PoseDetector::drawPoseLandmarks(cv::Mat& image, const PoseResult& pose_result) {
    if (pose_result.landmarks.empty()) return;
    
    // Draw landmarks
    for (const auto& landmark : pose_result.landmarks) {
        if (landmark.visibility > 0.5f) {
            cv::Point2f point(landmark.x, landmark.y);
            cv::circle(image, point, 3, cv::Scalar(0, 255, 0), -1);
        }
    }
    
    // Draw connections
    for (const auto& connection : POSE_CONNECTIONS) {
        int idx1 = connection.first;
        int idx2 = connection.second;
        
        if (idx1 < pose_result.landmarks.size() && idx2 < pose_result.landmarks.size()) {
            const auto& landmark1 = pose_result.landmarks[idx1];
            const auto& landmark2 = pose_result.landmarks[idx2];
            
            if (landmark1.visibility > 0.5f && landmark2.visibility > 0.5f) {
                cv::Point2f point1(landmark1.x, landmark1.y);
                cv::Point2f point2(landmark2.x, landmark2.y);
                cv::line(image, point1, point2, cv::Scalar(255, 0, 0), 2);
            }
        }
    }
}

// Utility functions implementation
namespace PoseUtils {
    cv::Point2f landmarkToPoint(const PoseLandmark& landmark, const cv::Size& image_size) {
        return cv::Point2f(landmark.x * image_size.width, landmark.y * image_size.height);
    }
    
    cv::Rect calculatePoseBoundingBox(const std::vector<PoseLandmark>& landmarks, const cv::Size& image_size) {
        if (landmarks.empty()) return cv::Rect();
        
        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();
        
        for (const auto& landmark : landmarks) {
            if (landmark.visibility > 0.5f) {
                min_x = std::min(min_x, landmark.x);
                min_y = std::min(min_y, landmark.y);
                max_x = std::max(max_x, landmark.x);
                max_y = std::max(max_y, landmark.y);
            }
        }
        
        if (min_x == std::numeric_limits<float>::max()) return cv::Rect();
        
        // Add some padding
        float padding = 20.0f;
        min_x = std::max(0.0f, min_x - padding);
        min_y = std::max(0.0f, min_y - padding);
        max_x = std::min(static_cast<float>(image_size.width), max_x + padding);
        max_y = std::min(static_cast<float>(image_size.height), max_y + padding);
        
        return cv::Rect(static_cast<int>(min_x), static_cast<int>(min_y),
                       static_cast<int>(max_x - min_x), static_cast<int>(max_y - min_y));
    }
    
    std::vector<PoseLandmark> filterLandmarksByVisibility(const std::vector<PoseLandmark>& landmarks, float threshold) {
        std::vector<PoseLandmark> filtered;
        for (const auto& landmark : landmarks) {
            if (landmark.visibility > threshold) {
                filtered.push_back(landmark);
            }
        }
        return filtered;
    }
    
    std::vector<float> calculatePoseAngles(const std::vector<PoseLandmark>& landmarks) {
        std::vector<float> angles;
        
        // Calculate key joint angles
        // Left elbow angle (shoulder-elbow-wrist)
        if (landmarks.size() > 15) {
            angles.push_back(calculateAngle(landmarks[11], landmarks[13], landmarks[15])); // Left elbow
            angles.push_back(calculateAngle(landmarks[12], landmarks[14], landmarks[16])); // Right elbow
            angles.push_back(calculateAngle(landmarks[23], landmarks[25], landmarks[27])); // Left knee
            angles.push_back(calculateAngle(landmarks[24], landmarks[26], landmarks[28])); // Right knee
        }
        
        return angles;
    }
    
    float calculateAngle(const PoseLandmark& p1, const PoseLandmark& p2, const PoseLandmark& p3) {
        // Calculate angle at p2 between p1-p2-p3
        float v1x = p1.x - p2.x;
        float v1y = p1.y - p2.y;
        float v2x = p3.x - p2.x;
        float v2y = p3.y - p2.y;
        
        float dot = v1x * v2x + v1y * v2y;
        float mag1 = std::sqrt(v1x * v1x + v1y * v1y);
        float mag2 = std::sqrt(v2x * v2x + v2y * v2y);
        
        if (mag1 == 0 || mag2 == 0) return 0.0f;
        
        float cos_angle = dot / (mag1 * mag2);
        cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle)); // Clamp to [-1, 1]
        
        return std::acos(cos_angle) * 180.0f / M_PI; // Convert to degrees
    }
}