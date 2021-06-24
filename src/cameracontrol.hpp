#ifndef _CAMERA_CONTROL_H_
#define _CAMERA_CONTROL_H_

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <cstring>

#define SHM_KEY (0x1234)
#define SHM_SIZE (64)

enum class CONTROL {
    FACE_DETECTION = 0,
    FACE_RECOGNITION = 1,
    BOUNDING_BOX = 2,
    LANDMARK = 3,
    ANTI_SPOOFING = 4,
    CROP_FACE = 5,
};

char *createShm() {
    int shm_id = shmget(SHM_KEY, sizeof(char)*SHM_SIZE, 0666 | IPC_CREAT);
    char *shm = (char *)shmat(shm_id, NULL, 0);
    memset(shm, 0, sizeof(char)*SHM_SIZE);
    return shm;
}

char* getShm() {
    int shm_id = shmget(SHM_KEY, sizeof(char)*SHM_SIZE, 0666);
    char *shm = (char *)shmat(shm_id, NULL, 0);
    return shm;
}


#endif

