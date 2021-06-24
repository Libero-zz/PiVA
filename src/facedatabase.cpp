#include "facedatabase.hpp"

FaceDatabase::FaceDatabase() {
}

FaceDatabase::~FaceDatabase() {
}
    
void FaceDatabase::load() {
    // Face recognition database
    cv::FileStorage fs(FEATURE_FILE, cv::FileStorage::READ);
    fs["feature_mat"] >> feature_mat;
    kdtree.load(feature_mat, TREE_FILE);
    fs.release();
    
    std::ifstream namefile(NAME_FILE);
    std::string name;
    while (namefile >> name) {
	    names.push_back(name);
    }
    
    if (names.size() > 0) {
	initialized = true;
    }
}

void FaceDatabase::save(const cv::Mat &feature_mat, std::vector<std::string> names) {
    // txt file to save names
    std::ofstream namefile;
    namefile.open("name.txt", std::ios_base::out);

    for (int i = 0; i < names.size(); ++i) {
        namefile << names[i] << "\n";
    }

    // save feature mat into file
    cv::FileStorage fs(FEATURE_FILE, cv::FileStorage::WRITE);
    fs << "feature_mat" << feature_mat;
    fs.release();

    // build knn search tree and save it for booting application faster
    cv::flann::Index kdtree(feature_mat, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_L2);
    kdtree.save(TREE_FILE);
}

void FaceDatabase::search(const cv::Mat &input, std::string &name, float &confidence) {
    if (!initialized) {
	std::cout << "No faces in database!" << std::endl;
	return;
    }
    // search feature in database
    cv::Mat matches;
    cv::Mat distances;
    kdtree.knnSearch(input, matches, distances, 1, cv::flann::SearchParams(-1));
    confidence = 1 - distances.at<float>(0) / 2;
    name = names[matches.at<int>(0)];
}

void FaceDatabase::search(std::vector<float> &input, std::string &name, float &confidence) {
    cv::Mat column(1, input.size(), cv::DataType<float>::type, input.data());
    cv::Mat norm_column;
    cv::normalize(column, norm_column);

    search(norm_column, name, confidence);
}
