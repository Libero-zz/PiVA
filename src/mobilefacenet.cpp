#include "mobilefacenet.hpp"
#include <iostream>
#include <string>

Mobilefacenet::Mobilefacenet() {
	mobileface_net_ = new ncnn::Net();
	initialized_ = false;
	kFaceFeatureDim = 128;
}

Mobilefacenet::~Mobilefacenet() {
	mobileface_net_->clear();
}

int Mobilefacenet::LoadModel(const char * root_path) {
	std::string fr_param = std::string(root_path) + "/fr_opt.param";
	std::string fr_bin = std::string(root_path) + "/fr_opt.bin";
	if (mobileface_net_->load_param(fr_param.c_str()) == -1 ||
		mobileface_net_->load_model(fr_bin.c_str()) == -1) {
		std::cout << "load face recognize model failed." << std::endl;
		return 10000;
	}

	initialized_ = true;
	return 0;
}

int Mobilefacenet::ExtractFeature(const cv::Mat & img_face,
	std::vector<float>* feature) {
	//std::cout << "start extract feature." << std::endl;
	feature->clear();
	if (!initialized_) {
		std::cout << "mobilefacenet model uninitialized." << std::endl;
		return 10000;
	}
	if (img_face.empty()) {
		std::cout << "input empty." << std::endl;
		return 10001;
	}

	ncnn::Mat in = ncnn::Mat::from_pixels(img_face.data,
		ncnn::Mat::PIXEL_BGR2RGB, img_face.cols, img_face.rows);
	feature->resize(kFaceFeatureDim);
	ncnn::Extractor ex = mobileface_net_->create_extractor();
	ex.input("data", in);
	ncnn::Mat out;
	ex.extract("fc1", out);
	for (int i = 0; i < kFaceFeatureDim; ++i) {
		feature->at(i) = out[i];
	}

//	std::cout << "end extract feature." << std::endl;

	return 0;
}
