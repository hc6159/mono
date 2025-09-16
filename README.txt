Integrated Object Detection and Pose Estimation
==============================================

This project integrates YOLONAS object detection with MediaPipe pose landmark detection for person objects using ONNX Runtime on CPU.

Features:
- YOLONAS object detection for multiple classes
- MediaPipe pose landmark detection for detected person objects
- Real-time processing optimized for CPU inference
- Detailed timing breakdown for performance monitoring

Requirements:
- OpenCV 4.x
- ONNX Runtime 1.15+
- CMake 3.16+
- C++17 compatible compiler

Build:
mkdir build && cd build
cmake .. && make -j$(nproc)

Usage:
./integrated_detection input_video.mp4 output_video.mp4

Models needed:
- yolonas.onnx (YOLONAS object detection model)
- pose_landmarker.onnx (MediaPipe pose landmark model)