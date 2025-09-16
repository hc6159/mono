#include "integrated_detection.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    // Configuration
    std::string yolo_model_path = "yolonas.onnx";  // Your YOLONAS model path
    std::string pose_model_path = "pose_landmarker.onnx";  // MediaPipe pose model path
    std::string input_video = "input.mp4";  // Input video or use camera
    std::string output_video = "output_with_poses.mp4";
    
    // Detection parameters
    float conf_threshold = 0.5f;
    float iou_threshold = 0.4f;
    float pose_conf_threshold = 0.5f;
    bool use_gpu = false;  // Set to true if you want to use GPU
    
    // Parse command line arguments
    if (argc > 1) input_video = argv[1];
    if (argc > 2) output_video = argv[2];
    if (argc > 3) conf_threshold = std::stof(argv[3]);
    if (argc > 4) pose_conf_threshold = std::stof(argv[4]);
    
    // Initialize integrated detector
    IntegratedDetector detector(yolo_model_path, pose_model_path, use_gpu);
    
    if (!detector.initialize()) {
        std::cerr << "Failed to initialize integrated detector" << std::endl;
        return -1;
    }
    
    // Open input video
    cv::VideoCapture cap;
    if (input_video == "camera" || input_video == "0") {
        cap.open(0);  // Use default camera
    } else {
        cap.open(input_video);
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input video: " << input_video << std::endl;
        return -1;
    }
    
    // Get video properties
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    // Setup output video writer
    cv::VideoWriter writer;
    if (output_video != "display") {
        writer.open(output_video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
                   fps, cv::Size(frame_width, frame_height));
        if (!writer.isOpened()) {
            std::cerr << "Failed to open output video writer" << std::endl;
            return -1;
        }
    }
    
    // Get class names
    std::vector<std::string> classes = detector.getDetectionClasses();
    
    // Processing loop
    cv::Mat frame;
    int frame_count = 0;
    int64_t total_processing_time = 0;
    
    std::cout << "Starting integrated detection and pose estimation..." << std::endl;
    std::cout << "Press 'q' to quit, 's' to save current frame" << std::endl;
    
    while (cap.read(frame)) {
        if (frame.empty()) break;
        
        frame_count++;
        
        // Process frame
        auto start_time = std::chrono::high_resolution_clock::now();
        IntegratedDetectionResult result = detector.processFrame(
            frame, classes, conf_threshold, iou_threshold, pose_conf_threshold);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        int64_t frame_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        total_processing_time += frame_time;
        
        // Draw results
        detector.drawResults(frame, result, classes);
        
        // Draw performance info
        std::string perf_text = "Frame: " + std::to_string(frame_count) + 
                               " | FPS: " + std::to_string(1000000.0 / frame_time) +
                               " | Detections: " + std::to_string(result.detections.size()) +
                               " | Poses: " + std::to_string(result.poses.size());
        
        cv::putText(frame, perf_text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // Draw timing breakdown
        std::string timing_text = "Detection: " + std::to_string(result.detection_inf_time / 1000.0) + "ms | " +
                                 "Pose: " + std::to_string(result.pose_inf_time / 1000.0) + "ms | " +
                                 "Total: " + std::to_string(result.total_time / 1000.0) + "ms";
        
        cv::putText(frame, timing_text, cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
        
        // Display or save frame
        if (output_video == "display") {
            cv::imshow("Integrated Detection + Pose Estimation", frame);
            
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q') break;
            if (key == 's') {
                std::string filename = "frame_" + std::to_string(frame_count) + ".jpg";
                cv::imwrite(filename, frame);
                std::cout << "Saved frame: " << filename << std::endl;
            }
        } else {
            writer.write(frame);
        }
        
        // Print progress every 100 frames
        if (frame_count % 100 == 0) {
            double avg_fps = 1000000.0 * frame_count / total_processing_time;
            std::cout << "Processed " << frame_count << " frames, Average FPS: " 
                      << std::fixed << std::setprecision(2) << avg_fps << std::endl;
        }
    }
    
    // Cleanup
    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    cv::destroyAllWindows();
    
    // Print final statistics
    if (frame_count > 0) {
        double avg_fps = 1000000.0 * frame_count / total_processing_time;
        std::cout << "\nProcessing completed!" << std::endl;
        std::cout << "Total frames processed: " << frame_count << std::endl;
        std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << avg_fps << std::endl;
        std::cout << "Average processing time per frame: " << total_processing_time / frame_count / 1000.0 << "ms" << std::endl;
    }
    
    return 0;
}