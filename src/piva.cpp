#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>


#include "opencv2/opencv.hpp"
#include "net.h"

#include "facedatabase.hpp"
#include "cameracontrol.hpp"
#include "priorbox.hpp"
#include "facedetection.hpp"
#include "mobilefacenet.hpp"
#include "live.hpp"

int main(int argc, char* argv[]) {
    if(argc != 2) {
        printf("Usage: %s <camera index>\n", argv[0]);
        return -1;
    }

    // used for receiving control command
    char *shm = createShm();    
    bool command[SHM_SIZE];

    cv::VideoCapture cap;
    cv::Mat im;
    cv::TickMeter cvtm;

    // Face detection
    FaceDetectionNet facedetectionnet;
    facedetectionnet.load("models/");

    // Face recognition
    Mobilefacenet mobilefacenet;
    mobilefacenet.LoadModel("models/");
    
    // Face recognition database
    FaceDatabase facedb;
    facedb.load();

    // Anti face spoofing
    ModelConfig config1 = {2.7f, 0.0f, 0.0f, 80, 80, "models/model_1_opt", false};
//    ModelConfig config2 = {4.0f, 0.0f, 0.0f, 80, 80, "models/model_2_opt", false};
    Live live;
    std::vector<ModelConfig> configs;
    configs.emplace_back(config1);
//    configs.emplace_back(config2);
    live.LoadModel(configs);

    cv::String title = cv::String("Detection Results on") + cv::String(argv[1]);

    if( isdigit(argv[1][0])) {
        cap.open(argv[1][0]-'0');
        if(! cap.isOpened()) {
            std::cerr << "Cannot open the camera." << std::endl;
            return 0;
        }
    }

    cap >> im;

    std::vector<Face> dets;

    float point_dst[10] = {
        30.2946f + 8.0f, 51.6963f,
	65.5318f + 8.0f, 51.5014f,
        48.0252f + 8.0f, 71.7366f,
        33.5493f + 8.0f, 92.3655f,
        62.7299f + 8.0f, 92.2041f,
    };

    if (cap.isOpened()) {
        while(true) {
           // read command from shm
           for (int i = 0; i < SHM_SIZE; ++i) {
                if (shm[i] == 0x1) {
                    command[i] = true;
                    shm[i] = 0;
		} else if (shm[i] == 0x2) {
                    command[i] = false;
		    shm[i] = 0;
                }
	    }

            cap >> im;
            cvtm.start();

            dets.clear();
	    if (command[static_cast<int>(CONTROL::FACE_DETECTION)] || 
	        command[static_cast<int>(CONTROL::FACE_RECOGNITION)] || 
		command[static_cast<int>(CONTROL::ANTI_SPOOFING)] || 
		command[static_cast<int>(CONTROL::CROP_FACE)]) {
	        facedetectionnet.detect(im, dets);
	    }

	    // warp face
	    if (command[static_cast<int>(CONTROL::FACE_RECOGNITION)] || 
	        command[static_cast<int>(CONTROL::CROP_FACE)]) {
	        for (auto i = 0; i < dets.size(); ++i) {
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

		    if (command[static_cast<int>(CONTROL::CROP_FACE)]) {
                        std::string time = std::to_string((std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())).count());
			cv::imwrite(time+".jpg", cropped_face);
			continue;
		    }

		    // calculate face feature
                    std::vector<float> feature;
                    mobilefacenet.ExtractFeature(cropped_face, &feature);
	    
                    // search feature in database
		    facedb.search(feature, dets[i].name, dets[i].fr_score);
                }
	    }
	    if (command[static_cast<int>(CONTROL::CROP_FACE)]) {
		command[static_cast<int>(CONTROL::CROP_FACE)] = false;
	    }

            if (command[static_cast<int>(CONTROL::ANTI_SPOOFING)]) {
		for (auto i = 0; i < dets.size(); ++i) {
		    // live
		    LiveFaceBox live_box = {dets[i].bbox.top_left.x, dets[i].bbox.top_left.y,
		                            dets[i].bbox.bottom_right.x, dets[i].bbox.bottom_right.y};
		    float confidence = live.Detect(im, live_box);
		    if (confidence <= 0.89) {
			dets[i].face_type = FACETYPE::FAKE;
		    } else {
			dets[i].face_type = FACETYPE::REAL;
		    }
		}
	    }
            facedetectionnet.draw(im, dets, command[static_cast<int>(CONTROL::BOUNDING_BOX)], command[static_cast<int>(CONTROL::LANDMARK)]);
	    
            cvtm.stop();

            std::string timeLabel = cv::format("Inference time: %.2f ms", cvtm.getTimeMilli());
            cv::putText(im, timeLabel, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            cvtm.reset();
            cv::imshow(title, im);
            if((cv::waitKey(1)& 0xFF) == 27)
                break;
        }
    }

    return 0;
}
