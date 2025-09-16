// Updated processDetectionFrame function with integrated pose detection
// This replaces your original function and adds pose detection for person objects

#include "integrated_detection.h"

// Updated processDetectionFrame function with pose detection integration
void processDetectionFrameWithPose(
    cv::Mat& frame, 
    Ort::Session& detection_session,
    const std::vector<int64_t>& input_shape,
    const std::vector<std::string>& classes,
    float conf_threshold,
    float iou_threshold,
    PoseDetector& pose_detector,  // Add pose detector parameter
    int64_t& prep_time,
    int64_t& inf_time,
    int64_t& post_time,
    int64_t& pose_time)  // Add pose timing parameter
{
    Measure timer;
    cv::Size input_size(input_shape[3], input_shape[2]);
    PREPROCESS_DATA metadata;
    
    // Preprocessing timing
    timer.start();
    std::vector<float> input_tensor = Preprocess(frame, input_size, metadata);
    timer.stop();
    prep_time = timer.get();
    
    // Inference timing
    timer.start();
    auto outputs = RunInference(input_tensor, detection_session, input_shape);
    timer.stop();
    inf_time = timer.get();
    
    // Postprocessing timing
    timer.start();
    std::vector<Bbox> results = Postprocess(
        outputs.first, outputs.second, metadata, conf_threshold, iou_threshold);
    timer.stop();
    post_time = timer.get();
    
    // NEW: Pose detection for person objects
    timer.start();
    std::vector<Bbox> person_detections;
    
    // Filter for person detections
    for (const auto& detection : results) {
        if (detection.class_id < classes.size() && classes[detection.class_id] == "person") {
            person_detections.push_back(detection);
        }
    }
    
    // Extract person crops and detect poses
    if (!person_detections.empty()) {
        std::vector<cv::Mat> person_crops;
        
        // Extract crops for each person detection
        for (const auto& person : person_detections) {
            // Add some padding around the detection
            int padding = 20;
            int x1 = std::max(0, static_cast<int>(person.x1) - padding);
            int y1 = std::max(0, static_cast<int>(person.y1) - padding);
            int x2 = std::min(frame.cols, static_cast<int>(person.x2) + padding);
            int y2 = std::min(frame.rows, static_cast<int>(person.y2) + padding);
            
            cv::Rect crop_rect(x1, y1, x2 - x1, y2 - y1);
            cv::Mat person_crop = frame(crop_rect);
            person_crops.push_back(person_crop);
        }
        
        // Detect poses for all person crops
        std::vector<PoseResult> pose_results = pose_detector.detectPoses(person_crops);
        
        // Draw pose landmarks on the original frame
        for (size_t i = 0; i < pose_results.size() && i < person_detections.size(); ++i) {
            const auto& pose = pose_results[i];
            const auto& person = person_detections[i];
            
            // Adjust pose coordinates back to original frame coordinates
            int padding = 20;
            int offset_x = std::max(0, static_cast<int>(person.x1) - padding);
            int offset_y = std::max(0, static_cast<int>(person.y1) - padding);
            
            // Draw pose landmarks
            for (const auto& landmark : pose.landmarks) {
                if (landmark.visibility > 0.5f) {
                    cv::Point2f point(landmark.x + offset_x, landmark.y + offset_y);
                    cv::circle(frame, point, 3, cv::Scalar(0, 255, 0), -1);
                }
            }
            
            // Draw pose connections (skeleton)
            for (const auto& connection : PoseDetector::POSE_CONNECTIONS) {
                int idx1 = connection.first;
                int idx2 = connection.second;
                
                if (idx1 < pose.landmarks.size() && idx2 < pose.landmarks.size()) {
                    const auto& landmark1 = pose.landmarks[idx1];
                    const auto& landmark2 = pose.landmarks[idx2];
                    
                    if (landmark1.visibility > 0.5f && landmark2.visibility > 0.5f) {
                        cv::Point2f point1(landmark1.x + offset_x, landmark1.y + offset_y);
                        cv::Point2f point2(landmark2.x + offset_x, landmark2.y + offset_y);
                        cv::line(frame, point1, point2, cv::Scalar(255, 0, 0), 2);
                    }
                }
            }
        }
    }
    
    timer.stop();
    pose_time = timer.get();
    
    // Draw original detections
    drawDetections(frame, results, classes);
}

// Alternative: Complete integrated approach using the IntegratedDetector class
void processFrameIntegrated(
    cv::Mat& frame,
    IntegratedDetector& detector,
    const std::vector<std::string>& classes,
    float conf_threshold = 0.5f,
    float iou_threshold = 0.4f,
    float pose_conf_threshold = 0.5f)
{
    // Process frame with both detection and pose estimation
    IntegratedDetectionResult result = detector.processFrame(
        frame, classes, conf_threshold, iou_threshold, pose_conf_threshold);
    
    // Draw all results (detections + poses)
    detector.drawResults(frame, result, classes);
    
    // Print timing information
    std::cout << "Detection - Prep: " << result.detection_prep_time / 1000.0 << "ms, "
              << "Inf: " << result.detection_inf_time / 1000.0 << "ms, "
              << "Post: " << result.detection_post_time / 1000.0 << "ms" << std::endl;
    
    std::cout << "Pose - Prep: " << result.pose_prep_time / 1000.0 << "ms, "
              << "Inf: " << result.pose_inf_time / 1000.0 << "ms, "
              << "Post: " << result.pose_post_time / 1000.0 << "ms" << std::endl;
    
    std::cout << "Total time: " << result.total_time / 1000.0 << "ms" << std::endl;
    std::cout << "Detections: " << result.detections.size() 
              << ", Poses: " << result.poses.size() << std::endl;
}

// Example usage in your main loop
void exampleUsage() {
    // Initialize pose detector
    PoseDetector pose_detector("pose_landmarker.onnx", false);
    if (!pose_detector.initialize()) {
        std::cerr << "Failed to initialize pose detector" << std::endl;
        return;
    }
    
    // Your existing detection session setup
    // Ort::Session detection_session = ...;
    // std::vector<int64_t> input_shape = ...;
    // std::vector<std::string> classes = ...;
    
    cv::Mat frame;
    int64_t prep_time, inf_time, post_time, pose_time;
    
    // Process each frame
    while (true) {  // Your frame reading loop
        // Read frame
        // cap >> frame;
        
        // Process with pose detection
        processDetectionFrameWithPose(
            frame, detection_session, input_shape, classes,
            0.5f, 0.4f, pose_detector,
            prep_time, inf_time, post_time, pose_time
        );
        
        // Display or save frame
        cv::imshow("Detection + Pose", frame);
        
        // Print timing
        std::cout << "Frame processing times (ms):" << std::endl;
        std::cout << "  Detection prep: " << prep_time / 1000.0 << std::endl;
        std::cout << "  Detection inf: " << inf_time / 1000.0 << std::endl;
        std::cout << "  Detection post: " << post_time / 1000.0 << std::endl;
        std::cout << "  Pose total: " << pose_time / 1000.0 << std::endl;
        std::cout << "  Total: " << (prep_time + inf_time + post_time + pose_time) / 1000.0 << std::endl;
        
        if (cv::waitKey(1) == 'q') break;
    }
}