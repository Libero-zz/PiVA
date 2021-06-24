#ifndef _FACE_DATABASE_H_
#define _FACE_DATABASE_H_

#include <vector>

#include "opencv2/opencv.hpp"

class FaceDatabase{
public:
    FaceDatabase();
    ~FaceDatabase();
    
    void load();
    void save(const cv::Mat &feature_mat, std::vector<std::string> names);
    void search(const cv::Mat &input, std::string &name, float &confidence);
    void search(std::vector<float> &input, std::string &name, float &confidence);

private:
    const std::string FEATURE_FILE = "feature.bin";
    const std::string TREE_FILE = "tree.bin";
    const std::string NAME_FILE = "name.txt";
    
    cv::Mat feature_mat;
    cv::flann::Index kdtree;
    std::vector<std::string> names;
    bool initialized = false;
};

#endif

