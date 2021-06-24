#ifndef _FACE_MOBILEFACENET_H_
#define _FACE_MOBILEFACENET_H_

#include <vector>

#include "opencv2/opencv.hpp"
#include "net.h"

class Mobilefacenet{
public:
	Mobilefacenet();
	~Mobilefacenet();

	int LoadModel(const char* root_path);
	int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature);

private:
	ncnn::Net* mobileface_net_;
	bool initialized_;
	int kFaceFeatureDim;
};

#endif // !_FACE_MOBILEFACENET_H_

