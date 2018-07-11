#pragma once

#include "time.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "stdio.h"
#include <cmath>
#include "math_constants.h"
#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
#include <caffe/caffe.hpp>
using namespace caffe;
using namespace cv;
using namespace std;


void match_reverse_PCA(float *device_dataa, float *device_datab, int channela, int heighta, int widtha, int heightb, int widthb, int patchsize, unsigned* &ann, int ah_half, int aw_half, int tt, float alpha, float lambda, int *f, int ntot);

void blend_content(float *a, float *ori, int heighta, int widtha, int channela, float weight);