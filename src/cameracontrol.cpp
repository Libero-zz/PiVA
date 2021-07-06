#include <string>
#include <iostream>
#include <map>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "cameracontrol.hpp"

int main(int argc, char* argv[]) {
    char *shm = getShm();
    
	std::string arg1, arg2;
	if (argc > 1) arg1 = argv[1];
	if (argc > 2) arg2 = argv[2];

	std::map<std::string, CONTROL> index_map{{"fd", CONTROL::FACE_DETECTION},
		                                     {"fr", CONTROL::FACE_RECOGNITION},
		                                     {"bbox", CONTROL::BOUNDING_BOX},
		                                     {"landmark", CONTROL::LANDMARK},
		                                     {"realface", CONTROL::ANTI_SPOOFING},
		                                     {"cropface", CONTROL::CROP_FACE},
		                                     {"track", CONTROL::TRACK}};

	char command = 0x0;
	if (arg2 == "open")
	{
		command = 0x1;
	}
	else if (arg2 == "close")
	{
		command = 0x2;
	}
		
	auto it = index_map.find(arg1);
	if (it == index_map.end())
	{
		std::cout << "Cannot find command " << arg1 << std::endl;
		return 0;
	}
	
	shm[static_cast<int>(it->second)] = command;
	return 0;
}
