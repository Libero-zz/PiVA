#ifndef _FACE_DETECTION_H_
#define _FACE_DETECTION_H_

#include <vector>

#include "opencv2/opencv.hpp"
#include "net.h"
#include "priorbox.hpp"

class FaceDetectionNet{
public:
    FaceDetectionNet();
    ~FaceDetectionNet();

    int load(const char* root_path);
    void calculate_net_size(int target_size, int *input_w, int *input_h);
    int detect(const cv::Mat& img, std::vector<Face> &faces);
    void draw(cv::Mat& img,
              const std::vector<Face>& faces,
              bool draw_bbox,
              bool draw_landmark);

private:
    void nms(std::vector<Face>& dets, const float thresh = 0.3);
    void scale_faces(std::vector<Face>& faces, float x_scale, float y_scale);

    ncnn::Net* fd_net_;
    PriorBox *pb_;

    bool initialized_;
    int net_w = 128;
    int net_h = 96;
    float conf_thresh = 0.6;
    float nms_thresh = 0.3;
    int keep_top_k = 750;

};

#endif

