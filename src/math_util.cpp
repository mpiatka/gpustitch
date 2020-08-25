#include <cmath>
#include "math_util.hpp"

namespace gpustitch{

double toRadian(double deg){
	return (deg / 180) * 3.14159265;
}

void getRotationMat(double yaw, double pitch, double roll, float *res){
	yaw = -yaw;
	res[0] = cos(roll) * cos(yaw) - sin(roll) * sin(pitch) * sin(yaw);
	res[1] = -sin(roll) * cos(pitch);
	res[2] = cos(roll) * sin(yaw) + sin(roll) * sin(pitch) * cos(yaw);
	res[3] = sin(roll) * cos(yaw) + cos(roll) * sin(pitch) * sin(yaw);
	res[4] = cos(roll) * cos(pitch);
	res[5] = sin(roll) * sin(yaw) - cos(roll) * sin(pitch) * cos(yaw);
	res[6] = -cos(pitch) * sin(yaw);
	res[7] = sin(pitch);
	res[8] = cos(pitch) * cos(yaw);


	return;
}

}
