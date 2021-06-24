#include <vector>
#include <string>
#include <iostream>
#include <chrono>

#include "priorbox.hpp"

#include "opencv2/opencv.hpp"

#include "facedetection.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_file_name>\n";
        return -1;
    }

    // Build blob
    cv::Mat im = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (im.empty()) {
        std::cerr << "Cannot load the image file " << argv[1] << ".\n";
        return -1;
    }

    // Face detection
    FaceDetectionNet facedetectionnet;
    facedetectionnet.load("./");
    int w = im.cols;
    int h = im.rows;
    facedetectionnet.calculate_net_size(320, &w, &h);
    
    std::string save_fpath = "./result.jpg";

    std::vector<Face> dets;
    
    float point_dst[10] = {
        30.2946f + 8.0f, 51.6963f,
        65.5318f + 8.0f, 51.5014f,
        48.0252f + 8.0f, 71.7366f,
        33.5493f + 8.0f, 92.3655f,
        62.7299f + 8.0f, 92.2041f,
    };

    facedetectionnet.detect(im, dets);

    for (auto i = 0; i < dets.size(); ++i)
	{
        // get current time for filename
        std::string time = std::to_string((std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())).count());
        
        cv::Mat cropped_face(112, 112, CV_8UC3);

        float tm_inv[6];

        float point_src[10] = {
            dets[i].landmarks.left_eye.x, dets[i].landmarks.left_eye.y,
            dets[i].landmarks.right_eye.x, dets[i].landmarks.right_eye.y,
            dets[i].landmarks.nose_tip.x, dets[i].landmarks.nose_tip.y,
            dets[i].landmarks.mouth_left.x, dets[i].landmarks.mouth_left.y,
            dets[i].landmarks.mouth_right.x, dets[i].landmarks.mouth_right.y,
        };

        ncnn::get_affine_transform(point_dst, point_src, 5, tm_inv);
        ncnn::warpaffine_bilinear_c3(im.data, im.cols, im.rows,
                                     cropped_face.data, cropped_face.cols, cropped_face.rows,
                                     tm_inv);

        std::string filename = time + ".jpg";
		cv::imwrite(filename, cropped_face);
    }

    return 0;
}
