#include "autotracker.hpp"

AutoTracker::AutoTracker() {
	initialized = false;
	if (gpioInitialise() < 0) {
        return;
    }
    
    initialized = true;

    gpioSetMode(PWM_CONTROL_PIN_PAN, PI_OUTPUT);
    gpioSetMode(PWM_CONTROL_PIN_TILT, PI_OUTPUT);
    
    int dc = angle_to_duty_cycle(90);
    gpioHardwarePWM(PWM_CONTROL_PIN_PAN, PWM_FREQ, dc);
    gpioHardwarePWM(PWM_CONTROL_PIN_TILT, PWM_FREQ, dc);
}

AutoTracker::~AutoTracker() {
    gpioHardwarePWM(PWM_CONTROL_PIN_PAN, PWM_FREQ, 0);
    gpioHardwarePWM(PWM_CONTROL_PIN_TILT, PWM_FREQ, 0);
    
    gpioTerminate();
}

int AutoTracker::angle_to_duty_cycle(int angle) {
    int duty_cycle = (int)(500 * PWM_FREQ + (1900 * PWM_FREQ * angle / 180));
    return duty_cycle;
}

void AutoTracker::track(std::vector<Face> input, int mid_x, int mid_y) {
	if (!initialized) {
		return;
	}
	if (input.size() <= 0) {
		return;
	}
	
	float max_area = 0;
	int max_id = 0;
	// choose biggest face
	for (int i = 0; i < input.size(); ++i) {
		if (input[i].bbox.area() > max_area) {
			max_id = i;
			max_area = input[i].bbox.area();
		}
	}
	cv::Point2f mid_point = (input[max_id].bbox.top_left + input[max_id].bbox.bottom_right)/2;
	int dc;
	float buffer = 0.2;
	int step_x = 2;
	int step_y = 2;
	if (std::abs(mid_point.y - mid_y)/mid_y < 0.5)
	    step_y = 1;
	if (std::abs(mid_point.x - mid_x)/mid_x < 0.5)
	    step_x = 1;
	
	if (mid_point.y - mid_y > mid_y * buffer) {
        if (current_angle_tilt < 180-step_y) {
            current_angle_tilt = current_angle_tilt + step_y;
        }
        dc = angle_to_duty_cycle(current_angle_tilt);
        gpioHardwarePWM(PWM_CONTROL_PIN_TILT, PWM_FREQ, dc);
	} else if (mid_y - mid_point.y > mid_y * buffer) {
        if (current_angle_tilt > step_y) {
            current_angle_tilt = current_angle_tilt - step_y;
        }
        dc = angle_to_duty_cycle(current_angle_tilt);
        gpioHardwarePWM(PWM_CONTROL_PIN_TILT, PWM_FREQ, dc);
	}
	if (mid_point.x - mid_x > mid_x * buffer) {
        if (current_angle_pan > step_x) {
            current_angle_pan = current_angle_pan - step_x;
        }
        dc = angle_to_duty_cycle(current_angle_pan);
        gpioHardwarePWM(PWM_CONTROL_PIN_PAN, PWM_FREQ, dc);            
	} else if (mid_x - mid_point.x > mid_x * buffer) {
		if (current_angle_pan < 180-step_x) {
            current_angle_pan = current_angle_pan + step_x;
        }
        dc = angle_to_duty_cycle(current_angle_pan);
        gpioHardwarePWM(PWM_CONTROL_PIN_PAN, PWM_FREQ, dc);  
	}
}
