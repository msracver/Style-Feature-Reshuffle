

#include "GeneralizedPatchMatch.cuh"
#include "curand_kernel.h"

__host__ __device__ int clamp(int x, int x_max, int x_min) {//assume x_max >= x_min
	if (x > x_max)
	{
		return x_max;
	}
	else if (x < x_min)
	{
		return x_min;
	}
	else
	{
		return x;
	}
}

__host__ __device__ float clamp_f(float x, float x_max, float x_min) {//assume x_max >= x_min
	if (x > x_max)
	{
		return x_max;
	}
	else if (x < x_min)
	{
		return x_min;
	}
	else
	{
		return x;
	}
}

__host__ __device__ unsigned int XY_TO_INT(int x, int y) {//r represent the number of 10 degree, x,y - 11 bits, max = 2047, r - max = 36, 6 bits
	return (((y) << 11) | (x));
}
__host__ __device__ int INT_TO_X(unsigned int v) {
	return (v)&((1 << 11) - 1);
}
__host__ __device__ int INT_TO_Y(unsigned int v) {
	return (v >> 11)&((1 << 11) - 1);
}

__host__ __device__ int cuMax(int a, int b) {
	if (a > b) {
		return a;
	}
	else {
		return b;
	}
}
__host__ __device__ int cuMin(int a, int b) {
	if (a < b) {
		return a;
	}
	else {
		return b;
	}
}

__device__ float MycuRand(curandState &state) {//random number in cuda, between 0 and 1
	
	 return curand_uniform(&state);

}
__device__ void InitcuRand(curandState &state) {//random number in cuda, between 0 and 1
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(i, 0, 0, &state);

}



__host__ __device__ float dist_compute(float * a, float * b, int channels, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int bx, int by, int patch_w) {//this is the average number of all matched pixel
	//suppose patch_w is an odd number
	double pixel_sum = 0, pixel_no = 0, pixel_dist = 0;//number of pixels realy counted
	double pixel_sum1 = 0;
	int a_slice = a_rows*a_cols, b_slice = b_rows*b_cols;
	int a_pitch = a_cols, b_pitch = b_cols;
	double dp_tmp;

	for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {

			if (
				(ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
				&&
				(by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
				)//the pixel in a should exist and pixel in b should exist
			{
				if (channels == 3)
				{
					for (int dc = 0; dc < channels; dc++)
					{
						dp_tmp = a[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] - b[dc * b_slice + (by + dy) * b_pitch + (bx + dx)];
						pixel_sum += dp_tmp * dp_tmp;
					}
				}
				else
				{
					double dp_tmp = 0;
					for (int dc = 0; dc < channels; dc++)
					{
						dp_tmp += a[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] * b[dc * b_slice + (by + dy) * b_pitch + (bx + dx)];
					}

					pixel_sum -= dp_tmp;
				}


				pixel_no += 1;
			}
		}

	}


	if (pixel_no>0) pixel_dist = (pixel_sum + pixel_sum1) / pixel_no;
	else pixel_dist = 2.;
	//printf("dist:: ar:%d aw:%d br:%d bw:%d ax:%d ay:%d bx:%d by:%d dist:%.5lf\n", a_rows, a_cols, b_rows, b_cols, ax, ay, bx, by, pixel_dist);
	return pixel_dist;

}



__host__ __device__ float dist(float *a, float * b, int channels, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int xp, int yp, int patch_w) {
	double d, x_diff, y_diff;
	d = dist_compute(a, b, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w);
	return d;
}



__global__ void reInitialAnn_kernel(unsigned int * ann, int * params, float *local_cor_map) {

	//just use 7 of 9 parameters
	int ah = params[1];
	int aw = params[2];
	int range = params[9]-1;

	bool whether_local_corr = params[8];

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	if (ax < aw && ay < ah) {

		if (whether_local_corr)
		{
			int bx = local_cor_map[0 * ah*aw + ay*aw + ax];
			int by = local_cor_map[1 * ah*aw + ay*aw + ax];

			if (bx>=0&&by>=0)
			{

				unsigned int vp = ann[ay*aw + ax];
				int xp = INT_TO_X(vp);
				int yp = INT_TO_Y(yp);
				if (xp < bx - range)
				{
					xp = bx - range;
				}

				if (xp > bx + range)
				{
					xp = bx + range;
				}

				if (yp < by - range)
				{
					yp = by - range;
				}

				if (yp > by + range)
				{
					yp = by + range;
				}

				ann[ay*aw + ax] = XY_TO_INT(xp, yp);

			}

		}

	}
}


__global__ void upSample_kernel(unsigned int * ann, unsigned int * ann_tmp,int * params, int aw_half,int ah_half) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];
	
	
	float aw_ratio = (float)aw / (float)aw_half;
	float ah_ratio = (float)ah / (float)ah_half;
	int ax_half = (ax+0.5) / aw_ratio;
	int ay_half = (ay+0.5) / ah_ratio;
	ax_half = clamp(ax_half, aw_half - 1, 0);
	ay_half = clamp(ay_half, ah_half - 1, 0);
	

	if (ax < aw&&ay < ah) {

		unsigned int v_half = ann[ay_half*aw_half + ax_half];
		int bx_half = INT_TO_X(v_half);
		int by_half = INT_TO_Y(v_half);

		int bx = ax + (bx_half - ax_half)*aw_ratio + 0.5;
		int by = ay + (by_half - ay_half)*ah_ratio + 0.5;

		bx = clamp(bx, bw-1, 0);
		by = clamp(by, bh-1, 0);

		ann_tmp[ay*aw + ax] = XY_TO_INT(bx, by);
	}

}








// ********** VOTE ***********

__global__ void center_vote(unsigned int * ann, double * pb, double * pc, int * params) {//pc is for recon

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];

	int slice_a = ah * aw;
	int pitch_a = aw;
	int slice_b = bh * bw;
	int pitch_b = bw;

	if (ax < aw&&ay < ah)
	{

		unsigned int vp = ann[ay*aw + ax];
		int xp = INT_TO_X(vp);
		int yp = INT_TO_Y(vp);
		if (yp < bh && xp < bw)
		{
			for (int i = 0; i < ch; i++)
			{

				pc[i*slice_a + ay*pitch_a + ax] = pb[i*slice_b + yp*pitch_b + xp];
			}
		}
	}
}


__global__ void center_vote(unsigned int * ann, float * pb, float * pc, int * params) {//pc is for recon

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];

	int slice_a = ah * aw;
	int pitch_a = aw;
	int slice_b = bh * bw;
	int pitch_b = bw;

	if (ax < aw&&ay < ah)
	{

		unsigned int vp = ann[ay*aw + ax];
		int xp = INT_TO_X(vp);
		int yp = INT_TO_Y(vp);
		if (yp < bh && xp < bw)
		{
			for (int i = 0; i < ch; i++)
			{

				pc[i*slice_a + ay*pitch_a + ax] = pb[i*slice_b + yp*pitch_b + xp];
			}
		}
	}
}



__global__ void center_vote(unsigned int * ann, float * pb, float * pc, int * params, int *maskc) {//pc is for recon

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];

	int slice_a = ah * aw;
	int pitch_a = aw;
	int slice_b = bh * bw;
	int pitch_b = bw;

	if (ax < aw&&ay < ah)
	{
	    if (maskc[ay*aw+ax] == 1)
		{
		unsigned int vp = ann[ay*aw + ax];
		int xp = INT_TO_X(vp);
		int yp = INT_TO_Y(vp);
		if (yp < bh && xp < bw)
		{
			for (int i = 0; i < ch; i++)
			{

				pc[i*slice_a + ay*pitch_a + ax] = pb[i*slice_b + yp*pitch_b + xp];
			}
		}
		}
	}
}

__global__ void avg_vote(unsigned int * ann, float * pb, float * pc, int * params) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];
	int patch_w = params[5];

	int slice_a = ah * aw;
	int pitch_a = aw;
	int slice_b = bh * bw;
	int pitch_b = bw;

	int count = 0;

	if (ax < aw&&ay < ah)
	{

		//set zero for all the channels at (ax,ay)
		for (int i = 0; i < ch; i++)
		{
			pc[i*slice_a + ay*pitch_a + ax] = 0;

		}

		//count the sum of all the possible value of (ax,ay)
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
			for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++)
			{

				if ((ax + dx) < aw && (ax + dx) >= 0 && (ay + dy) < ah && (ay + dy) >= 0)
				{
					unsigned int vp = ann[(ay + dy)*aw + ax + dx];
					
					int xp = INT_TO_X(vp);
					int yp = INT_TO_Y(vp);

					if ((xp - dx) < bw && (xp - dx) >= 0 && (yp - dy) < bh && (yp - dy) >= 0)
					{
						count++;
						for (int dc = 0; dc < ch; dc++)
						{
							pc[dc*slice_a + ay*pitch_a + ax] += pb[dc*slice_b + (yp - dy)*pitch_b + xp - dx];
						}
					}
				}

			}
		}

		//count average value
		for (int i = 0; i < ch; i++)
		{
			pc[i*slice_a + ay*pitch_a + ax] /= count;
		}

	}
}













void norm(float* &dst, float* src, int channel, int height, int width){

	int count = channel*height*width;
	float* x = src;
	float* x2;
	cudaMalloc(&x2, count*sizeof(float));
	caffe_gpu_mul(count, x, x, x2);

	//caculate dis
	float*sum;
	float* ones;
	cudaMalloc(&sum, height*width*sizeof(float));
	cudaMalloc(&ones, channel*sizeof(float));
	caffe_gpu_set(channel, 1.0f, ones);
	caffe_gpu_gemv(CblasTrans, channel, height*width, 1.0f, x2, ones, 0.0f, sum);

	float *dis;
	cudaMalloc(&dis, height*width*sizeof(float));
	caffe_gpu_powx(height*width, sum, 0.5f, dis);

	//norm	
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, channel, width*height, 1, 1.0f, ones, dis, 0.0f, x2);
	caffe_gpu_div(count, src, x2, dst);

	cudaFree(x2);
	cudaFree(ones);
	cudaFree(dis);
	cudaFree(sum);
}


__global__ void blend_cont(float *a, float *ori, int tota, float weight)
{
	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	if (ax < tota)
	{

		a[ax] = ori[ax] * weight + a[ax] * (1.0 - weight);
	}
}

void blend_content(float *a, float *ori, int heighta, int widtha, int channela, float weight)
{
	dim3 blocksPerGridAB(heighta*widtha*channela / 400, 1, 1);
	dim3 threadsPerBlockAB(400, 1, 1);
	blend_cont << <blocksPerGridAB, threadsPerBlockAB >> >(a, ori, heighta*widtha*channela, weight);
}


__global__ void initialAnn_kernel(unsigned int * ann, int * params) {
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	curandState state;
	InitcuRand(state);

	if (ax < aw && ay < ah) {
		int bx = (int)(MycuRand(state)*bw);
		int by = (int)(MycuRand(state)*bh);
		ann[ay*aw + ax] = XY_TO_INT(bx, by);

	}
}


__device__ void improve_guess(float * a, float * b, int channels, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int &xbest, int &ybest, float &dbest, int xp, int yp, int patch_w, float rr) {
	float d;
	d = dist(a, b, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w);
	if (d + rr < dbest) {
		xbest = xp;
		ybest = yp;
		dbest = d;
	}
	//}
}

__global__ void initialAnn_kernel(unsigned int * ann, int * params, unsigned * mask, int total) {

	//just use 7 of 9 parameters
	int ah = params[1];
	int aw = params[2];
	curandState state;
	InitcuRand(state);
	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	if (ax < aw && ay < ah) {
		int index = (int)(total*MycuRand(state));
		//printf("%d %d %d %d\n", ax, ay, index, total);
		ann[ay*aw + ax] = mask[index];
	}
}




__global__ void patchmatch_reverse(float * a, float * b,unsigned int *ann, float *annd, int * params, unsigned *mask, unsigned int * index, int total, unsigned *rank) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	//assign params
	int ch = params[0];
	int a_rows = params[1];
	int a_cols = params[2];
	int b_rows = params[3];
	int b_cols = params[4];
	int patch_w = params[5];
	int pm_iters = params[6];
	int rs_max = params[7];

	if (ax < a_cols && ay < a_rows) {

		// for random number
		curandState state;
		InitcuRand(state);

		unsigned int v, vp;

		int xp, yp, xbest, ybest;

		int xmin, xmax, ymin, ymax;

		float dbest;
		v = ann[ay*a_cols + ax];
		xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
		annd[ay*a_cols + ax] = dist(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, patch_w);

		for (int iter = 0; iter < pm_iters; iter++) {

			/* Current (best) guess. */
			v = ann[ay*a_cols + ax];
			xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
			dbest = annd[ay*a_cols + ax];

			/* In each iteration, improve the NNF, by jumping flooding. */
			for (int jump = 8; jump > 0; jump /= 2) {

				/* Propagation: Improve current guess by trying instead correspondences from left, right, up and downs. */
				if ((ax - jump) < a_cols && (ax - jump) >= 0)//left
				{
					vp = ann[ay*a_cols + ax - jump];//the pixel coordinates in image b

					xp = INT_TO_X(vp) + jump, yp = INT_TO_Y(vp);//the propagated match from vp, the center of the patch, which should be in the image

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols && mask[yp*b_cols + xp] == 0)
					{
						//improve guess
						improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, 0);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				if ((ax + jump) < a_cols)//right
				{
					vp = ann[ay*a_cols + ax + jump];//the pixel coordinates in image b

					xp = INT_TO_X(vp) - jump, yp = INT_TO_Y(vp);

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols && mask[yp*b_cols + xp] == 0)
					{
						//improve guess
						improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, FLT_MIN);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				if ((ay - jump) < a_rows && (ay - jump) >= 0)//up
				{
					vp = ann[(ay - jump)*a_cols + ax];//the pixel coordinates in image b
					xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + jump;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols && mask[yp*b_cols + xp] == 0)
					{

						//improve guess
						improve_guess(a, b,ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, FLT_MIN);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				if ((ay + jump) < a_rows)//down
				{
					vp = ann[(ay + jump)*a_cols + ax];//the pixel coordinates in image b	
					xp = INT_TO_X(vp), yp = INT_TO_Y(vp) - jump;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols && mask[yp*b_cols + xp] == 0)
					{
						//improve guess
						improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, FLT_MIN);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

			}

			/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
			int rs_start = rs_max;
			if (rs_start > total) {
				rs_start = total;
			}
			for (int mag = rs_start; mag >= 1; mag /= 2) {
				/* Sampling window */
				int tt = rank[ybest*b_cols + xbest];
				int tmin = cuMax(tt - mag, 0), tmax = cuMin(tt + mag + 1, total);
				int tp = tmin + (int)(MycuRand(state)*(tmax - tmin)) % (tmax - tmin);
				xp = INT_TO_X(index[tp]);
				yp = INT_TO_Y(index[tp]);

				//improve guess
				if (mask[yp*b_cols + xp] == 0)
					improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, FLT_MIN);

			}

			ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
			annd[ay*a_cols + ax] = dbest;
			__syncthreads();
		}
	}
}





__global__ void patchmatch_M(float * a, float * b, unsigned int *ann, float *annd, int * params, unsigned *mask) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	//assign params
	int ch = params[0];
	int a_rows = params[1];
	int a_cols = params[2];
	int b_rows = params[3];
	int b_cols = params[4];
	int patch_w = params[5];
	int pm_iters = params[6];
	int rs_max = params[7];

	if (ax < a_cols && ay < a_rows) {

		// for random number
		curandState state;
		InitcuRand(state);

		unsigned int v, vp;

		int xp, yp, xbest, ybest;

		int xmin, xmax, ymin, ymax;

		float dbest;
		v = ann[ay*a_cols + ax];
		xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
		annd[ay*a_cols + ax] = dist(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, patch_w);

		for (int iter = 0; iter < pm_iters; iter++) {
			if (mask[ay*a_cols + ax] == 0)
			{
				/* Current (best) guess. */
				v = ann[ay*a_cols + ax];
				xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
				dbest = annd[ay*a_cols + ax];

				/* In each iteration, improve the NNF, by jumping flooding. */
				for (int jump = 8; jump > 0; jump /= 2) {

					/* Propagation: Improve current guess by trying instead correspondences from left, right, up and downs. */
					if ((ax - jump) < a_cols && (ax - jump) >= 0)//left
					{
						vp = ann[ay*a_cols + ax - jump];//the pixel coordinates in image b

						xp = INT_TO_X(vp) + jump, yp = INT_TO_Y(vp);//the propagated match from vp, the center of the patch, which should be in the image

						if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
						{
							//improve guess
							improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, 0);
							ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
							annd[ay*a_cols + ax] = dbest;
						}
					}

					if ((ax + jump) < a_cols)//right
					{
						vp = ann[ay*a_cols + ax + jump];//the pixel coordinates in image b

						xp = INT_TO_X(vp) - jump, yp = INT_TO_Y(vp);

						if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
						{
							//improve guess
							improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, FLT_MIN);
							ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
							annd[ay*a_cols + ax] = dbest;
						}
					}

					if ((ay - jump) < a_rows && (ay - jump) >= 0)//up
					{
						vp = ann[(ay - jump)*a_cols + ax];//the pixel coordinates in image b
						xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + jump;

						if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
						{

							//improve guess
							improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, FLT_MIN);
							ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
							annd[ay*a_cols + ax] = dbest;
						}
					}

					if ((ay + jump) < a_rows)//down
					{
						vp = ann[(ay + jump)*a_cols + ax];//the pixel coordinates in image b	
						xp = INT_TO_X(vp), yp = INT_TO_Y(vp) - jump;

						if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
						{
							//improve guess
							improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, FLT_MIN);
							ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
							annd[ay*a_cols + ax] = dbest;
						}
					}

				}

				int rs_start = rs_max;
				if (rs_start > cuMax(b_cols, b_rows)) {
					rs_start = cuMax(b_cols, b_rows);
				}
				for (int mag = rs_start; mag >= 1; mag /= 2) {
					/* Sampling window */
					xmin = cuMax(xbest - mag, 0), xmax = cuMin(xbest + mag + 1, b_cols);
					ymin = cuMax(ybest - mag, 0), ymax = cuMin(ybest + mag + 1, b_rows);
					xp = xmin + (int)(MycuRand(state)*(xmax - xmin)) % (xmax - xmin);
					yp = ymin + (int)(MycuRand(state)*(ymax - ymin)) % (ymax - ymin);
					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, FLT_MIN);
					}

				}


				ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
				annd[ay*a_cols + ax] = dbest;
			}
			__syncthreads();
		}
	}
}


__global__ void pca(float *ori, float* pca, double * vec,int height,int width, int ori_c, int pca_c)
{
	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;
	if (ax < width && ay < height)
	{
		for (int i = 0; i < pca_c; i++)
		{
			double tmp = 0;
			for (int j = 0; j < ori_c; j++)
				tmp += vec[(ori_c - i - 1)*ori_c + j] * ori[j*height*width + ay*width + ax];
			pca[i*height*width + ay*width + ax] = tmp;
		}
	}
}
void match_reverse_PCA(float *device_dataa, float *device_datab, int channela, int heighta, int widtha, int heightb, int widthb, int patchsize, unsigned* &ann, int ah_half, int aw_half, int tt, float alpha, float lambda, int *f, int ntot)
{
	dim3 blocksPerGridAB(widtha / 20 + 1, heighta / 20 + 1, 1);
	dim3 threadsPerBlockAB(20, 20, 1);


	dim3 blocksPerGridBA(widthb / 20 + 1, heightb / 20 + 1, 1);
	dim3 threadsPerBlockBA(20, 20, 1);


	int params[10];
	params[0] = channela;
	params[1] = heighta;
	params[2] = widtha;
	params[3] = heightb;
	params[4] = widthb;
	params[5] = patchsize;
	params[6] = 11;
	params[7] = 6;  //72 




	int *paramsAB_device_match, *paramsBA_device_match,*paramsAB_device_vote;
	cudaMalloc(&paramsAB_device_vote, 10 * sizeof(int));
	cudaMemcpy(paramsAB_device_vote, params, 10 * sizeof(int), cudaMemcpyHostToDevice);

	

	if (tt == 2)
	{
		params[0] = 20;
	}
	else
	if (tt == 3)
	{
		params[0] = 40;
	}
	else
	if (tt == 4)
	{
		params[0] = 80;
	}
	else
	if (tt == 5)
	{
		params[0] = channela;
	}
	else params[0] = channela;


	cudaMalloc(&paramsAB_device_match, 10 * sizeof(int));
	cudaMemcpy(paramsAB_device_match, params, 10 * sizeof(int), cudaMemcpyHostToDevice);


	params[1] = heightb;
	params[2] = widthb;
	params[3] = heighta;
	params[4] = widtha;
	params[7] = 36;  //72 
	cudaMalloc(&paramsBA_device_match, 10 * sizeof(int));
	cudaMemcpy(paramsBA_device_match, params, 10 * sizeof(int), cudaMemcpyHostToDevice);


	int tmpa = heighta*widtha;
	int tmpb = heightb*widthb;

	float *data_a_ori;
	cudaMalloc(&data_a_ori, heighta*widtha*channela*sizeof(float));
	cudaMemcpy(data_a_ori, device_dataa, heighta*channela*widtha*sizeof(float), cudaMemcpyDeviceToDevice);

	

	unsigned *ann_device_AB, *ann_host_AB;
	float *annd_device_AB, *annd_host_AB;

	cudaMalloc(&ann_device_AB, tmpa*sizeof(unsigned));
	cudaMalloc(&annd_device_AB, tmpa*sizeof(float));
	ann_host_AB = (unsigned*)malloc(tmpa*sizeof(unsigned));
	annd_host_AB = (float*)malloc(tmpa*sizeof(float));


	unsigned *ann_device_BA, *ann_host_BA;
	float *annd_device_BA, *annd_host_BA;

	cudaMalloc(&ann_device_BA, tmpb*sizeof(unsigned));
	cudaMalloc(&annd_device_BA, tmpb*sizeof(float));
	ann_host_BA = (unsigned*)malloc(tmpb*sizeof(unsigned));
	annd_host_BA = (float*)malloc(tmpb*sizeof(float));

	unsigned *mask_host_A, *mask_device_A;
	unsigned *index_host_A, *index_device_A;
	unsigned *rank_host, *rank_device;
	rank_host = (unsigned*)malloc(tmpa*sizeof(unsigned));
	cudaMalloc(&rank_device, tmpa*sizeof(unsigned));
	mask_host_A = (unsigned*)malloc(tmpa*sizeof(unsigned));
	cudaMalloc(&mask_device_A, tmpa*sizeof(unsigned));
	index_host_A = (unsigned*)malloc(tmpa*sizeof(unsigned));
	cudaMalloc(&index_device_A, tmpa*sizeof(unsigned));
	char layer[100];
	sprintf(layer, "models/PCA/conv%d_1", tt);
	FILE *fin = fopen(layer, "rb");
	double * vec_host = new double[channela*channela];
	fread(vec_host, sizeof(double), channela*channela, fin);
	fread(vec_host, sizeof(double), channela*channela, fin);
	fclose(fin);

	double * vec_device;
	cudaMalloc(&vec_device, channela*channela*sizeof(double));
	cudaMemcpy(vec_device, vec_host, channela*channela*sizeof(double),cudaMemcpyHostToDevice);




	int channelaa = channela;
	channela = params[0];

	float *data_a_N, *data_b_N,*data_a_pca,*data_b_pca;
	cudaMalloc(&data_a_N, tmpa*channela*sizeof(float));
	cudaMalloc(&data_b_N, tmpb*channela*sizeof(float));
	cudaMalloc(&data_a_pca, tmpa*channela*sizeof(float));
	cudaMalloc(&data_b_pca, tmpb*channela*sizeof(float));

	float *sort_array;
	sort_array = (float*)malloc(tmpb*sizeof(float));
	unsigned *queue;
	queue = (unsigned*)malloc(ntot*ntot*sizeof(unsigned));
	queue[0] = XY_TO_INT(1, 1);
	for (int s = 0, t = 0; s <= t; s++)
	{
		int x = INT_TO_X(queue[s]);
		int y = INT_TO_Y(queue[s]);
		int tmp = f[x*ntot + y];
		if (x - 1 > 0 && f[(x - 1)*ntot + y] == tmp + 1)
		{
			t++;
			queue[t] = XY_TO_INT(x - 1, y);
		}
		else
		if (x + 1 <= ntot && f[(x + 1)*ntot + y] == tmp + 1)
		{
			t++;
			queue[t] = XY_TO_INT(x + 1, y);
		}
		else
		if (y - 1>0 && f[x*ntot + y - 1] == tmp + 1)
		{
			t++;
			queue[t] = XY_TO_INT(x, y - 1);
		}
		else
		if (y + 1 <= ntot && f[x*ntot + y + 1] == tmp + 1)
		{
			t++;
			queue[t] = XY_TO_INT(x, y + 1);
		}
	}
	ntot = ntot*ntot;
	pca << <blocksPerGridBA, threadsPerBlockBA >> >(device_datab, data_b_pca, vec_device, heightb, widthb, channelaa, channela);
	norm(data_b_N, data_b_pca, channela, heightb, widthb);
	if (tt != 1)
	{
		for (int turn = 0; turn < 5; turn++)
		{
			cerr << turn << endl;
			pca << <blocksPerGridAB, threadsPerBlockAB >> >(device_dataa, data_a_pca, vec_device, heighta, widtha, channelaa, channela);
			norm(data_a_N, data_a_pca, channela, heighta, widtha);
			initialAnn_kernel << <blocksPerGridAB, threadsPerBlockAB >> >(ann_device_AB, paramsAB_device_match);
			cudaMemcpy(ann_host_AB, ann_device_AB, tmpa*sizeof(unsigned), cudaMemcpyDeviceToHost);
			memset(mask_host_A, 0, sizeof(unsigned)*tmpa);
			{
				int tot = tmpa;
				int flag = 0, changed = tot;
				float T = 0, TT = 0, cut = 0;
				while (tot)
				{
					int tmp = 0;
					for (int t = 0; t < ntot; t++)
					{
						int i = INT_TO_Y(queue[t]);
						int j = INT_TO_X(queue[t]);
						i--, j--;
						if (i<heighta && j<widtha && mask_host_A[i*widtha + j] == 0)
						{
							rank_host[i*widtha + j] = tmp;
							index_host_A[tmp++] = XY_TO_INT(j, i);
						}
					}
					cudaMemcpy(index_device_A, index_host_A, tmp*sizeof(unsigned), cudaMemcpyHostToDevice);
					cudaMemcpy(mask_device_A, mask_host_A, tmpa*sizeof(unsigned), cudaMemcpyHostToDevice);
					cudaMemcpy(rank_device, rank_host, tmpa*sizeof(unsigned), cudaMemcpyHostToDevice);

					if (changed <= tmpa*0.005 || tot < 0.01*tmpa) {
						 break;
					}
					changed = 0;
					size_t f1, t1;

					initialAnn_kernel << <blocksPerGridBA, threadsPerBlockBA >> > (ann_device_BA, paramsBA_device_match, index_device_A, tmp);

					patchmatch_reverse << <blocksPerGridBA, threadsPerBlockBA >> >(data_b_N, data_a_N, ann_device_BA, annd_device_BA, paramsBA_device_match, mask_device_A, index_device_A, tmp, rank_device);
					
					cudaMemcpy(ann_host_BA, ann_device_BA, tmpb*sizeof(unsigned), cudaMemcpyDeviceToHost);
					cudaMemcpy(annd_host_BA, annd_device_BA, tmpb*sizeof(float), cudaMemcpyDeviceToHost);

					memcpy(sort_array, annd_host_BA, tmpb*sizeof(float));
					sort(sort_array, sort_array + tmpb);
					float ymax = 1/*sort_array[tmpb - 1]*/, ymin = -1;
					if (flag == 0)
					{
						flag = 1;
						for (int i = 0; i < tmpb; i++)
						{
							sort_array[i] = (sort_array[i] - ymin) / (ymax - ymin);
						}
						float x_p = 0, y_p = 0, B = 0, A = 0, B1 = 0, B2 = 0;
						float ptot = tmpb;
						for (int i = 0; i < tmpb; i++)
						{
							x_p += i / ptot;
							if (sort_array[i] < 1e-5) sort_array[i] = 1e-5;
							y_p += 1 / sort_array[i];
						}
						x_p /= ptot;
						y_p /= ptot;
						for (int i = 0; i < tmpb; i++)
						{
							B1 += (i / ptot - x_p)*(1 / sort_array[i] - y_p);
							B2 += (i / ptot - x_p)*(i / ptot - x_p);
						}

						B = B1 / B2;
						A = y_p - B*x_p;
						B = -B;
						float cut1 = sort_array[min(tmpb - 1, (int)round((-sqrt(1 / B) + A / B)*ptot))];
						T = 0;
						for (int i = 0; i < tmpb; i++)
						{
							if (sort_array[i]>cut1) break;
							T += (sort_array[i] * (ymax - ymin));
						}
						memcpy(sort_array, annd_host_BA, tmpb*sizeof(float));
						sort(sort_array, sort_array + tmpb);
					}
					TT = 0;
					for (int i = 0; i < tmpb; i++)
					{
						TT += sort_array[i] - ymin;
						if (TT>T) { cut = sort_array[i]; break; }
					}

					if (TT <= T) cut = ymax;
					for (int i = 0; i < heightb; i++)
					for (int j = 0; j < widthb; j++)
					{
						if (annd_host_BA[i*widthb + j] <= cut)
						{
							int x = INT_TO_X(ann_host_BA[i*widthb + j]);
							int y = INT_TO_Y(ann_host_BA[i*widthb + j]);
							float tmp = annd_host_BA[i*widthb + j];
							if (mask_host_A[y*widtha + x] == 0)
							{
								mask_host_A[y*widtha + x] = 1;
								ann_host_AB[y*widtha + x] = XY_TO_INT(j, i);
								annd_host_AB[y*widtha + x] = tmp;
								tot--;
								changed++;
							}
							else
							if (tmp < annd_host_AB[y*widtha + x])
							{
								ann_host_AB[y*widtha + x] = XY_TO_INT(j, i);
								annd_host_AB[y*widtha + x] = tmp;
							}
						}
					}


				}
			}
			cudaMemcpy(ann_device_AB, ann_host_AB, tmpa*sizeof(unsigned), cudaMemcpyHostToDevice);
			cudaMemcpy(annd_device_AB, annd_host_AB, tmpa*sizeof(unsigned), cudaMemcpyHostToDevice);
			patchmatch_M << <blocksPerGridAB, threadsPerBlockAB >> >(data_a_N, data_b_N, ann_device_AB, annd_device_AB, paramsAB_device_match, mask_device_A);
			avg_vote << <blocksPerGridAB, threadsPerBlockAB >> >(ann_device_AB, device_datab, device_dataa, paramsAB_device_vote);
			blend_content(device_dataa, data_a_ori, heighta, widtha, channelaa, alpha);
		}
	}


	if (ann != NULL) cudaFree(ann);
	cudaMalloc(&ann, tmpa*sizeof(unsigned));
	cudaMemcpy(ann, ann_device_AB, tmpa*sizeof(unsigned), cudaMemcpyDeviceToDevice);
	cudaFree(ann_device_AB);
	cudaFree(annd_device_AB);
	cudaFree(ann_device_BA);
	cudaFree(annd_device_BA);
	cudaFree(paramsAB_device_match);
	cudaFree(paramsBA_device_match);
	cudaFree(data_a_N);
	cudaFree(data_b_N);
	cudaFree(paramsAB_device_vote);
	cudaFree(data_a_ori);
	cudaFree(rank_device);
	cudaFree(mask_device_A);
	cudaFree(index_device_A);
	cudaFree(vec_device);
	cudaFree(data_a_pca);
	cudaFree(data_b_pca);





	free(ann_host_AB);
	free(ann_host_BA);
	free(annd_host_AB);
	free(annd_host_BA);
	free(rank_host);
	free(mask_host_A);
	free(index_host_A);
	free(sort_array);
	free(queue);

	delete vec_host;
}