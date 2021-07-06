#ifndef _AUTO_TRACKER_H_
#define _AUTO_TRACKER_H_

#include <vector>

#include "pigpio.h"
#include "opencv2/opencv.hpp"

#include "face_def.hpp"

#define PWM_CONTROL_PIN_PAN 18
#define PWM_CONTROL_PIN_TILT 13
#define PWM_FREQ 50

class AutoTracker{
public:
    AutoTracker();
    ~AutoTracker();
    
    void track(std::vector<Face> input, int mid_x, int mid_y);

private:
    int angle_to_duty_cycle(int angle=0);
    bool initialized;
    int current_angle_pan = 90;
    int current_angle_tilt = 90;

};

#endif

