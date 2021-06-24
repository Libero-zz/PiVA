#include "facedetection.hpp"
#include <iostream>
#include <string>

FaceDetectionNet::FaceDetectionNet() {
    fd_net_ = new ncnn::Net();
    initialized_ = false;
    pb_ = new PriorBox(cv::Size(net_w, net_h), cv::Size(net_w, net_h));
}

FaceDetectionNet::~FaceDetectionNet() {
    delete pb_;
    fd_net_->clear();
}

int FaceDetectionNet::load(const char * root_path) {
    if (initialized_) {
        return 0;
    }
    std::string fd_param = std::string(root_path) + "/fd_opt.param";
    std::string fd_bin = std::string(root_path) + "/fd_opt.bin";
    if (fd_net_->load_param(fd_param.c_str()) == -1 ||
        fd_net_->load_model(fd_bin.c_str()) == -1) {
        std::cout << "load face recognize model failed." << std::endl;
	return 10000;
    }
    fd_net_->opt.num_threads = 1;

    initialized_ = true;
    return 0;
}

void FaceDetectionNet::calculate_net_size(int target_size, int *input_w, int *input_h) {
    std::cout << "presize " << *input_w << " " << *input_h << std::endl;
    int *long_side = (*input_w) > (*input_h) ? input_w : input_h;
    int *short_side = (*input_w) > (*input_h) ? input_h : input_w;

    float scale = target_size * 1.f / (*long_side);
    *long_side = target_size;
    *short_side = (*short_side) * 1.f * scale;
    std::cout << "aftsize " << *input_w << " " << *input_h << std::endl;
    if (net_w != *input_w || net_h != *input_h) {
        net_w = *input_w;
        net_h = *input_h;
        delete pb_;
        pb_ = new PriorBox(cv::Size(net_w, net_h), cv::Size(net_w, net_h));
    }
    std::cout << "netsize " << net_w << " " << net_h << std::endl;

}

int FaceDetectionNet::detect(const cv::Mat & img, std::vector<Face> &faces) {
    faces.clear();
    if (!initialized_) {
        std::cout << "facedetection model uninitialized." << std::endl;
        return 10000;
    }
    if (img.empty()) {
        std::cout << "input empty." << std::endl;
        return 10001;
    }

    float x_scale = img.cols * 1.f / net_w;
    float y_scale = img.rows * 1.f / net_h;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, net_w, net_h);
    ncnn::Extractor ex = fd_net_->create_extractor();
    ex.input("input", in);
    ncnn::Mat output_blobs[3];
    ex.extract("loc", output_blobs[0]);
    ex.extract("conf", output_blobs[1]);
    ex.extract("iou", output_blobs[2]);

    // Decode bboxes, landmarks and scores
    faces = pb_->decode(output_blobs[0], output_blobs[1], output_blobs[2], conf_thresh);

    // NMS
    if (faces.size() > 1) {
        nms(faces, nms_thresh);
        if (faces.size() > keep_top_k) { faces.erase(faces.begin()+keep_top_k, faces.end()); }
    }
   
    scale_faces(faces, x_scale, y_scale);

    return 0;
}

// dets is of dimension [num, 15], which is 
// num * [x1, y1, x2, y2, x_re, y_re, x_le, y_le, x_ml, y_ml, x_n, y_n, x_mr, y_ml, label]
void FaceDetectionNet::nms(std::vector<Face>& dets,
                           const float thresh) {
    std::sort(dets.begin(), dets.end(), [](const Face& a, const Face& b) { return a.score > b.score; });

    // std::vector<Face> post_nms;
    std::vector<bool> isSuppressed(dets.size(), false);
    for (auto i = 0; i < dets.size(); ++i) {
        if (isSuppressed[i]) { continue; }

        // area of i bbox
        float area_i = dets[i].bbox.area();
        for (auto j = i + 1; j < dets.size(); ++j) {
            if (isSuppressed[j]) { continue; }

            // area of intersection
            float ix1 = std::max(dets[i].bbox.top_left.x, dets[j].bbox.top_left.x);
            float iy1 = std::max(dets[i].bbox.top_left.y, dets[j].bbox.top_left.y);
            float ix2 = std::min(dets[i].bbox.bottom_right.x, dets[j].bbox.bottom_right.x);
            float iy2 = std::min(dets[i].bbox.bottom_right.y, dets[j].bbox.bottom_right.y);

            float iw = ix2 - ix1 + 1;
            float ih = iy2 - iy1 + 1;
            if (iw <= 0 || ih <= 0) { continue; }
            float inter = iw * ih;

            // area of j bbox
            float area_j = dets[j].bbox.area();

            // iou
            float iou = inter / (area_i + area_j - inter);
            if (iou > thresh) { isSuppressed[j] = true; }
        }
        // post_nms.push_back(dets[i]);
    }
    // return post_nms;
    int idx_t = 0;
    dets.erase(
        std::remove_if(dets.begin(), dets.end(), [&idx_t, &isSuppressed](const Face& f) { return isSuppressed[idx_t++]; }),
        dets.end()
    );
}

void FaceDetectionNet::draw(cv::Mat& img,
                            const std::vector<Face>& faces,
                            bool draw_bbox,
                            bool draw_landmark) {

    const int thickness = 2;
    const cv::Scalar bbox_color = {  0, 255,   0};
    const cv::Scalar text_color = {255, 255, 255};
    const std::vector<cv::Scalar> landmarks_color = {
        {255,   0,   0}, // left eye
        {  0,   0, 255}, // right eye
        {  0, 255, 255}, // mouth left
        {255, 255,   0}, // nose
        {  0, 255,   0}  // mouth right
    };

    auto point2f2point = [](cv::Point2f p, bool shift = false) {
        return shift ? cv::Point(int(p.x), int(p.y)+12) : cv::Point(int(p.x), int(p.y));
    };
    for (auto i = 0; i < faces.size(); ++i) {
        if (!faces[i].name.empty()) {
            cv::putText(img,
                        faces[i].name + " " + std::to_string(faces[i].fr_score),
                        point2f2point(faces[i].bbox.top_left-cv::Point2f(0, 40), true),
                        cv::FONT_HERSHEY_DUPLEX,
                        0.5, // Font scale
                        text_color);
        }
        std::string face_type_string;
        if (faces[i].face_type == FACETYPE::REAL) {
            face_type_string = "REAL";
        } else if (faces[i].face_type == FACETYPE::FAKE) {
            face_type_string = "FAKE";
        }
        if (!face_type_string.empty()) {
            cv::putText(img,
                        face_type_string,
                        point2f2point(faces[i].bbox.top_left-cv::Point2f(0, 20), true),
                        cv::FONT_HERSHEY_DUPLEX,
                        0.5, // Font scale
                        text_color);
        }
        if (draw_bbox) {
            // draw bbox
            cv::rectangle(img,
                          point2f2point(faces[i].bbox.top_left),
                          point2f2point(faces[i].bbox.bottom_right),
                          bbox_color,
                          thickness);
            // put score by the corner of bbox
            std::string str_score = std::to_string(faces[i].score);
            
            if (str_score.size() > 6) {
                str_score.erase(6);
            }
            cv::putText(img,
                        str_score,
                        point2f2point(faces[i].bbox.top_left, true),
                        cv::FONT_HERSHEY_DUPLEX,
                        0.5, // Font scale
                        text_color);

        }
        if (draw_landmark) {
            // draw landmarks
            const int radius = 2;
            cv::circle(img, point2f2point(faces[i].landmarks.left_eye),    radius, landmarks_color[0], thickness);
            cv::circle(img, point2f2point(faces[i].landmarks.right_eye),   radius, landmarks_color[1], thickness);
            cv::circle(img, point2f2point(faces[i].landmarks.mouth_left),  radius, landmarks_color[2], thickness);
            cv::circle(img, point2f2point(faces[i].landmarks.nose_tip),    radius, landmarks_color[3], thickness);
            cv::circle(img, point2f2point(faces[i].landmarks.mouth_right), radius, landmarks_color[4], thickness);
        }
    }
}

void FaceDetectionNet::scale_faces(std::vector<Face> &faces, float x_scale, float y_scale) {
    for (auto i = 0; i < faces.size(); ++i) {
        faces[i].bbox.top_left.x *= x_scale;
        faces[i].bbox.top_left.y *= y_scale;
        faces[i].bbox.bottom_right.x *= x_scale;
        faces[i].bbox.bottom_right.y *= y_scale;
        faces[i].landmarks.right_eye.x *= x_scale;
        faces[i].landmarks.right_eye.y *= y_scale;
        faces[i].landmarks.left_eye.x *= x_scale;
        faces[i].landmarks.left_eye.y *= y_scale;
        faces[i].landmarks.mouth_left.x *= x_scale;
        faces[i].landmarks.mouth_left.y *= y_scale;
        faces[i].landmarks.nose_tip.x *= x_scale;
        faces[i].landmarks.nose_tip.y *= y_scale;
        faces[i].landmarks.mouth_right.x *= x_scale;
        faces[i].landmarks.mouth_right.y *= y_scale;
    }
}
