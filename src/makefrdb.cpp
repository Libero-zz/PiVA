#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "mobilefacenet.hpp"
#include "facedatabase.hpp"

std::vector<std::string> get_file_in_dir(const std::string &path, bool getfolder) {
    std::vector<std::string> subfolders;
    struct dirent *entry;
    DIR *dp;
    
    dp = opendir(path.c_str());
    if (dp == NULL) {
		perror("opendir: Path does not exist or could not be read.");
		return subfolders;
	}
	
	while ((entry = readdir(dp))) {
	    if (entry->d_name[0] == '.')
	        continue;
		if ((getfolder && entry->d_type == DT_DIR) ||
		    !getfolder && entry->d_type == DT_REG) {
            subfolders.push_back(entry->d_name);
		}
	}

    closedir(dp);
	
	return subfolders;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <db_folder>\n";
        return -1;
    }

    // Face recognition net
    Mobilefacenet mobilefacenet;
    mobilefacenet.LoadModel("models/");
    
    FaceDatabase facedb;
    
    std::vector<std::string> names;

    // feature mat to save all face features
    cv::Mat feature_mat(0, 128, cv::DataType<float>::type);

    // traverse through whole folder
    /*
       fr folder should be organized like this
       root_dir
       |-person1
         |-photo1.jpg
         |-photo2.jpg
       |-person2
         |-photo1.jpg
         |-photo2.jpg
       |-person3
         |-photo1.jpg
         |-photo2.jpg
    */

    std::string root_folder = std::string(argv[1]);
    std::vector<std::string> sub_folder = get_file_in_dir(argv[1], true);
    for (int i = 0; i < sub_folder.size(); ++i) {
		std::vector<std::string> file = get_file_in_dir(root_folder+"/"+sub_folder[i], false);
		std::cout << "Processing " << sub_folder[i] << std::endl;
		
		for (int j = 0; j < file.size(); ++j) {
    		std::string img_path = root_folder+"/"+sub_folder[i]+"/"+file[j];
            cv::Mat img = cv::imread(img_path);
            std::vector<float> feature;
            mobilefacenet.ExtractFeature(img, &feature);

            cv::Mat column(1, feature.size(), cv::DataType<float>::type, feature.data());
            cv::Mat norm_column;
            cv::normalize(column, norm_column);
            feature_mat.push_back(norm_column.clone());
            names.push_back(sub_folder[i]);
	    }
	}


    facedb.save(feature_mat, names);

    return 0;
}
