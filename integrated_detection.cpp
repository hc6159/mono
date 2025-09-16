#include "integrated_detection.h"
#include <chrono>
#include <algorithm>
#include <numeric>

// Measure class implementation
void Measure::start() {
    start_time_ = std::chrono::high_resolution_clock::now();
}

void Measure::stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
}

int64_t Measure::get() const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_);
    return duration.count();
}

IntegratedDetector::IntegratedDetector(
    const std::string& yolo_model_path,
    const std::string& pose_model_path,
    bool use_gpu)
    : yolo_model_path_(yolo_model_path), use_gpu_(use_gpu), initialized_(false) {
    
    // Initialize detection environment
    detection_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "IntegratedDetector");
    detection_session_options_ = std::make_unique<Ort::SessionOptions>();
    detection_memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    
    // Initialize pose detector
    pose_detector_ = std::make_unique<PoseDetector>(pose_model_path, use_gpu);
}

IntegratedDetector::~IntegratedDetector() = default;

bool IntegratedDetector::initialize() {
    try {
        // Initialize detection model
        if (use_gpu_) {
            try {
                OrtCUDAProviderOptions cuda_options{};
                detection_session_options_->AppendExecutionProvider_CUDA(cuda_options);
            } catch (const std::exception& e) {
                std::cout << "CUDA not available for detection, falling back to CPU: " << e.what() << std::endl;
            }
        }
        
        detection_session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        detection_session_options_->SetIntraOpNumThreads(4);
        
        detection_session_ = std::make_unique<Ort::Session>(*detection_env_, yolo_model_path_.c_str(), *detection_session_options_);
        
        // Get detection model input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t num_input_nodes = detection_session_->GetInputCount();
        if (num_input_nodes != 1) {
            std::cerr << "Expected 1 detection input node, got " << num_input_nodes << std::endl;
            return false;
        }
        
        detection_input_name_ = detection_session_->GetInputName(0, allocator);
        auto input_type_info = detection_session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        detection_input_shape_ = input_tensor_info.GetShape();
        
        // Output info
        size_t num_output_nodes = detection_session_->GetOutputCount();
        if (num_output_nodes != 2) {
            std::cerr << "Expected 2 detection output nodes, got " << num_output_nodes << std::endl;
            return false;
        }
        
        detection_boxes_output_name_ = detection_session_->GetOutputName(0, allocator);
        detection_scores_output_name_ = detection_session_->GetOutputName(1, allocator);
        
        // Initialize pose detector
        if (!pose_detector_->initialize()) {
            std::cerr << "Failed to initialize pose detector" << std::endl;
            return false;
        }
        
        initialized_ = true;
        std::cout << "Integrated detector initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize integrated detector: " << e.what() << std::endl;
        return false;
    }
}

IntegratedDetectionResult IntegratedDetector::processFrame(
    cv::Mat& frame,
    const std::vector<std::string>& classes,
    float conf_threshold,
    float iou_threshold,
    float pose_conf_threshold) {
    
    IntegratedDetectionResult result;
    Measure timer;
    
    if (!initialized_) {
        std::cerr << "Detector not initialized" << std::endl;
        return result;
    }
    
    // Start total timing
    timer.start();
    
    // Step 1: Object Detection
    cv::Size input_size(detection_input_shape_[3], detection_input_shape_[2]);
    PREPROCESS_DATA metadata;
    
    // Detection preprocessing
    timer.start();
    std::vector<float> input_tensor = preprocessDetection(frame, input_size, metadata);
    timer.stop();
    result.detection_prep_time = timer.get();
    
    // Detection inference
    timer.start();
    auto detection_outputs = runDetectionInference(input_tensor, detection_input_shape_);
    timer.stop();
    result.detection_inf_time = timer.get();
    
    // Detection postprocessing
    timer.start();
    result.detections = postprocessDetection(
        detection_outputs.first, detection_outputs.second, 
        metadata, conf_threshold, iou_threshold);
    timer.stop();
    result.detection_post_time = timer.get();
    
    // Step 2: Filter person detections
    std::vector<Bbox> person_detections = filterPersonDetections(result.detections, classes);
    
    // Step 3: Pose Detection for each person
    if (!person_detections.empty()) {
        // Extract person crops
        std::vector<cv::Mat> person_crops = extractPersonCrops(frame, person_detections, classes);
        
        // Pose preprocessing timing
        timer.start();
        // Pose preprocessing is done inside detectPoses
        timer.stop();
        result.pose_prep_time = timer.get();
        
        // Pose inference
        timer.start();
        result.poses = pose_detector_->detectPoses(person_crops);
        timer.stop();
        result.pose_inf_time = timer.get();
        
        // Pose postprocessing timing
        timer.start();
        // Filter poses by confidence
        std::vector<PoseResult> filtered_poses;
        for (const auto& pose : result.poses) {
            if (pose.confidence >= pose_conf_threshold) {
                filtered_poses.push_back(pose);
            }
        }
        result.poses = filtered_poses;
        timer.stop();
        result.pose_post_time = timer.get();
    } else {
        result.pose_prep_time = 0;
        result.pose_inf_time = 0;
        result.pose_post_time = 0;
    }
    
    // End total timing
    timer.stop();
    result.total_time = timer.get();
    
    return result;
}

void IntegratedDetector::drawResults(cv::Mat& frame, const IntegratedDetectionResult& result, const std::vector<std::string>& classes) {
    // Draw object detections
    drawDetections(frame, result.detections, classes);
    
    // Draw pose landmarks for each person
    for (const auto& pose : result.poses) {
        pose_detector_->drawPoseLandmarks(frame, pose);
    }
}

std::vector<int64_t> IntegratedDetector::getDetectionInputShape() const {
    return detection_input_shape_;
}

std::vector<std::string> IntegratedDetector::getDetectionClasses() const {
    // You should return your actual class names here
    return {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"};
}

cv::Size IntegratedDetector::getPoseInputSize() const {
    return pose_detector_->getInputSize();
}

// YOLONAS detection functions implementation
std::vector<float> IntegratedDetector::preprocessDetection(const cv::Mat& frame, const cv::Size& input_size, PREPROCESS_DATA& metadata) {
    // Store original size
    metadata.original_size = frame.size();
    
    // Calculate scaling factors
    float scale_x = static_cast<float>(input_size.width) / frame.cols;
    float scale_y = static_cast<float>(input_size.height) / frame.rows;
    float scale = std::min(scale_x, scale_y);
    
    // Calculate padding
    int new_width = static_cast<int>(frame.cols * scale);
    int new_height = static_cast<int>(frame.rows * scale);
    int pad_x = (input_size.width - new_width) / 2;
    int pad_y = (input_size.height - new_height) / 2;
    
    metadata.scale_x = scale;
    metadata.scale_y = scale;
    metadata.pad_x = pad_x;
    metadata.pad_y = pad_y;
    
    // Resize and pad image
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_width, new_height));
    
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_y, input_size.height - new_height - pad_y,
                      pad_x, input_size.width - new_width - pad_x, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // Convert to RGB and normalize
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    
    // Convert to CHW format
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);
    
    std::vector<float> input_tensor;
    input_tensor.reserve(3 * input_size.height * input_size.width);
    
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < input_size.height; ++h) {
            for (int w = 0; w < input_size.width; ++w) {
                input_tensor.push_back(channels[c].at<float>(h, w));
            }
        }
    }
    
    return input_tensor;
}

std::pair<std::vector<float>, std::vector<float>> IntegratedDetector::runDetectionInference(
    const std::vector<float>& input_tensor, 
    const std::vector<int64_t>& input_shape) {
    
    // Create input tensor
    auto input_tensor_ptr = Ort::Value::CreateTensor<float>(
        *detection_memory_info_, input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size());
    
    // Run inference
    const char* input_names[] = {detection_input_name_.c_str()};
    const char* output_names[] = {detection_boxes_output_name_.c_str(), detection_scores_output_name_.c_str()};
    
    auto output_tensors = detection_session_->Run(Ort::RunOptions{nullptr},
                                                input_names, &input_tensor_ptr, 1,
                                                output_names, 2);
    
    // Extract outputs
    float* boxes_data = output_tensors[0].GetTensorMutableData<float>();
    float* scores_data = output_tensors[1].GetTensorMutableData<float>();
    
    size_t boxes_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    size_t scores_size = output_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount();
    
    std::vector<float> boxes_output(boxes_data, boxes_data + boxes_size);
    std::vector<float> scores_output(scores_data, scores_data + scores_size);
    
    return {boxes_output, scores_output};
}

std::vector<Bbox> IntegratedDetector::postprocessDetection(
    const std::vector<float>& boxes_output,
    const std::vector<float>& scores_output,
    const PREPROCESS_DATA& metadata,
    float conf_threshold,
    float iou_threshold) {
    
    std::vector<Bbox> detections;
    
    // This is a simplified postprocessing - you should implement the full YOLONAS postprocessing
    // including NMS (Non-Maximum Suppression) based on your specific model output format
    
    // For now, return empty detections - you need to implement this based on your YOLONAS model
    // The actual implementation would depend on your model's output format
    
    return detections;
}

void IntegratedDetector::drawDetections(cv::Mat& frame, const std::vector<Bbox>& detections, const std::vector<std::string>& classes) {
    for (const auto& detection : detections) {
        // Draw bounding box
        cv::Rect rect(static_cast<int>(detection.x1), static_cast<int>(detection.y1),
                     static_cast<int>(detection.x2 - detection.x1), static_cast<int>(detection.y2 - detection.y1));
        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
        
        // Draw label
        std::string label = classes[detection.class_id] + ": " + std::to_string(detection.confidence);
        cv::putText(frame, label, cv::Point(rect.x, rect.y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

std::vector<cv::Mat> IntegratedDetector::extractPersonCrops(const cv::Mat& frame, const std::vector<Bbox>& detections, const std::vector<std::string>& classes) {
    std::vector<cv::Mat> crops;
    
    for (const auto& detection : detections) {
        // Check if this is a person detection
        if (detection.class_id < classes.size() && classes[detection.class_id] == "person") {
            // Extract crop with some padding
            int padding = 20;
            int x1 = std::max(0, static_cast<int>(detection.x1) - padding);
            int y1 = std::max(0, static_cast<int>(detection.y1) - padding);
            int x2 = std::min(frame.cols, static_cast<int>(detection.x2) + padding);
            int y2 = std::min(frame.rows, static_cast<int>(detection.y2) + padding);
            
            cv::Rect crop_rect(x1, y1, x2 - x1, y2 - y1);
            cv::Mat crop = frame(crop_rect);
            crops.push_back(crop);
        }
    }
    
    return crops;
}

std::vector<Bbox> IntegratedDetector::filterPersonDetections(const std::vector<Bbox>& detections, const std::vector<std::string>& classes) {
    std::vector<Bbox> person_detections;
    
    for (const auto& detection : detections) {
        if (detection.class_id < classes.size() && classes[detection.class_id] == "person") {
            person_detections.push_back(detection);
        }
    }
    
    return person_detections;
}