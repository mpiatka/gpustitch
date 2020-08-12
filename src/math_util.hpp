#ifndef MATH_UTIL_HPP
#define MATH_UTIL_HPP

namespace gpustitch{

double toRadian(double deg);
void getRotationMat(double yaw, double pitch, double roll, double *res);

}

#endif
