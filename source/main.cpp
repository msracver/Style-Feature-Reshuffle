#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <cmath>
#include <ctime>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "cublas_v2.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include <caffe/caffe.hpp>
#include "GeneralizedPatchMatch.cuh"

using namespace std;
using namespace cv;
using namespace caffe;

struct Dim
{
	Dim(int c, int h, int w)
	{
		channel = c;
		height = h;
		width = w;
	}
	int channel, height, width;
};

void deconvnet(Net<double>* net, string layername1, string dataname1, double* d_y, Dim dim1, string layername2, string dataname2, double* d_x, Dim dim2)
{

	int num1 = dim1.channel*dim1.height*dim1.width;
	int num2 = dim2.channel*dim2.height*dim2.width;
	int id1;
	int id2;

	vector<string> layer_names = net->layer_names();
	for (int i = 0; i < layer_names.size(); i++)
	{
		if (layer_names[i] == layername1)
			id1 = i;
		if (layer_names[i] == layername2)
			id2 = i;
	}
	cudaError_t cudaStat;
	cudaStat = cudaMemcpy(net->blob_by_name(dataname1)->mutable_gpu_data(), d_y, num1*sizeof(double), cudaMemcpyHostToDevice);
	net->ForwardFromTo(id1 + 1, id2);
	cudaMemcpy(d_x, net->blob_by_name(dataname2)->gpu_data(), num2*sizeof(double), cudaMemcpyDeviceToHost);

}
int * f, ntot = 0, tot = 0;
void dfs(int x1, int y1, int x2, int y2, int n)
{
	if (n == 1)
	{
		tot++;
		f[x1*ntot + y1] = tot;
		return;
	}
	if (x1 == x2)
	{

		if (x1%n == 1)
		{
			n /= 2;
			if (y1 < y2)
			{

				dfs(x1, y1, x1 + n - 1, y1, n);
				dfs(x1 + n, y1, x1 + n, (y1 + y2) / 2, n);
				dfs(x1 + n, (y1 + y2) / 2 + 1, x1 + n, y2, n);
				dfs(x1 + n - 1, y2, x2, y2, n);
			}
			else
			{
				dfs(x1, y1, x1 + n - 1, y1, n);
				dfs(x1 + n, y1, x1 + n, (y1 + y2) / 2 + 1, n);
				dfs(x1 + n, (y1 + y2) / 2, x1 + n, y2, n);
				dfs(x1 + n - 1, y2, x2, y2, n);
			}
		}
		else
		{
			n /= 2;
			if (y1 < y2)
			{

				dfs(x1, y1, x1 - n + 1, y1, n);
				dfs(x1 - n, y1, x1 - n, (y1 + y2) / 2, n);
				dfs(x1 - n, (y1 + y2) / 2 + 1, x1 - n, y2, n);
				dfs(x1 - n + 1, y2, x2, y2, n);
			}
			else
			{
				dfs(x1, y1, x1 - n + 1, y1, n);
				dfs(x1 - n, y1, x1 - n, (y1 + y2) / 2 + 1, n);
				dfs(x1 - n, (y1 + y2) / 2, x1 - n, y2, n);
				dfs(x1 - n + 1, y2, x2, y2, n);
			}
		}
	}
	else
	{

		if (y1%n == 1)
		{
			n /= 2;
			if (x1 < x2)
			{
				dfs(x1, y1, x1, y1 + n - 1, n);
				dfs(x1, y1 + n, (x1 + x2) / 2, y1 + n, n);
				dfs((x1 + x2) / 2 + 1, y1 + n, x2, y1 + n, n);
				dfs(x2, y1 + n - 1, x2, y1, n);
			}
			else
			{
				dfs(x1, y1, x1, y1 + n - 1, n);
				dfs(x1, y1 + n, (x1 + x2) / 2 + 1, y1 + n, n);
				dfs((x1 + x2) / 2, y1 + n, x2, y1 + n, n);
				dfs(x2, y1 + n - 1, x2, y1, n);
			}
		}
		else
		{
			n /= 2;
			if (x1 < x2)
			{

				dfs(x1, y1, x1, y1 - n + 1, n);
				dfs(x1, y1 - n, (x1 + x2) / 2, y1 - n, n);
				dfs((x1 + x2) / 2 + 1, y1 - n, x2, y1 - n, n);
				dfs(x2, y1 - n + 1, x2, y1, n);
			}
			else
			{
				dfs(x1, y1, x1, y1 - n + 1, n);
				dfs(x1, y1 - n, (x1 + x2) / 2 + 1, y1 - n, n);
				dfs((x1 + x2) / 2, y1 - n, x2, y1 - n, n);
				dfs(x2, y1 - n + 1, x2, y1, n);
			}
		}
	}

}
int main(int argc, char * argv[])
{
	::google::InitGoogleLogging(argv[0]);
	Mat imga = imread(argv[1]);
	Mat imgb = imread(argv[2]);
	int rowa = imga.rows, cola = imga.cols, rowb = imgb.rows, colb = imgb.cols;
	float alpha = 0.6, lambda = 0.05;
	double scalea = 1, scaleb = 1;
	if (rowa > cola)
	{
		scalea = 512. / rowa;
	}
	else scalea = 512. / cola;
	if (rowb > colb) scaleb = 512. / rowb; else scaleb = 512. / colb;
	rowa *= scalea, rowb *= scaleb, cola *= scalea, colb *= scaleb;
	resize(imga, imga, Size(cola, rowa));
	resize(imgb, imgb, Size(colb, rowb));
	rowa = imga.rows, cola = imga.cols, rowb = imgb.rows, colb = imgb.cols;
	int g;
	sscanf(argv[4], "%d", &g);



	
	ntot = max(rowa, cola);
	int length = 0, k = ntot;
	for (; k != 0; k /= 2, length++);
	if (ntot > (1 << (length - 1))) ntot = (1 << length); else ntot = (1 << (length - 1));
	f = (int *)malloc(sizeof(int)*(ntot + 1)*(ntot + 1));
	dfs(1, 1, 1, ntot, ntot);




	cudaSetDevice(g);
	Caffe::set_mode(Caffe::GPU);
	Net<float>*neta, *netb;
	neta = new Net<float>("models/deconv_allsize/deploy.prototxt", TEST);
	neta->CopyTrainedLayersFrom("models/deconv_allsize/5_1_to_4_1_iter_50000.caffemodel");

	netb = new Net<float>("models/deconv_allsize/deploy.prototxt", TEST);
	netb->CopyTrainedLayersFrom("models/deconv_allsize/5_1_to_4_1_iter_50000.caffemodel");

	neta->blob_by_name("data")->Reshape(1, 3, imga.rows, imga.cols);
	neta->Reshape();

	netb->blob_by_name("data")->Reshape(1, 3, imgb.rows, imgb.cols);
	netb->Reshape();

	float *inputa, *inputb;
	inputa = new float[imga.rows*imga.cols * 3];
	inputb = new float[imgb.rows*imgb.cols * 3];
	Mat output_img(imga.rows, imga.cols, CV_32FC3);
	for (int i = 0; i < imga.rows; i++)
	for (int j = 0; j < imga.cols; j++)
	for (int k = 0; k <= 2; k++)
		output_img.at<Vec3f>(i, j)[k] = imga.at<Vec3b>(i, j)[k];
	for (int i = 0; i < imgb.rows; i++)
	{
		for (int j = 0; j < imgb.cols; j++)
		{
			inputb[0 * imgb.rows*imgb.cols + i*imgb.cols + j] = (imgb.at<Vec3b>(i, j)[0] - 103.939);
			inputb[1 * imgb.rows*imgb.cols + i*imgb.cols + j] = (imgb.at<Vec3b>(i, j)[1] - 116.779);
			inputb[2 * imgb.rows*imgb.cols + i*imgb.cols + j] = (imgb.at<Vec3b>(i, j)[2] - 123.68);
		}
	}
	cudaMemcpy(netb->blob_by_name("data")->mutable_gpu_data(), inputb, 3 * imgb.rows*imgb.cols*sizeof(float), cudaMemcpyHostToDevice);
	cudaError_t cudaStat;
	double ti = clock();
	netb->Forward();
	cerr << "Forward B: " << (clock() - ti) / 1000 << "sec.\n";

	float sums = 0;
	for (int i = 0; i < imga.rows; i++)
	{
		for (int j = 0; j < imga.cols; j++)
		{
			inputa[0 * imga.rows*imga.cols + i*imga.cols + j] = (output_img.at<Vec3f>(i, j)[0] - 103.939);
			inputa[1 * imga.rows*imga.cols + i*imga.cols + j] = (output_img.at<Vec3f>(i, j)[1] - 116.779);
			inputa[2 * imga.rows*imga.cols + i*imga.cols + j] = (output_img.at<Vec3f>(i, j)[2] - 123.68);
		}
	}

	cudaMemcpy(neta->blob_by_name("data")->mutable_gpu_data(), inputa, 3 * imga.rows*imga.cols*sizeof(float), cudaMemcpyHostToDevice);
	neta->Forward();

	unsigned *ann = NULL;
	int start = 4, end = 2;
	for (int tt = start; tt >= end; tt--)
	{
		char s[100], sa[100];
		sprintf(s, "conv%d_1", tt);
		if (tt == start) sprintf(sa, "conv%d_1", tt); else sprintf(sa, "deconv%d_1", tt);

		int heightb = netb->blob_by_name(s)->height();
		int widthb = netb->blob_by_name(s)->width();
		int channelb = netb->blob_by_name(s)->channels();
		int heighta = neta->blob_by_name(s)->height();
		int widtha = neta->blob_by_name(s)->width();
		int channela = neta->blob_by_name(s)->channels();
		
        float *device_dataa, *device_datab;
		cudaStat = cudaMalloc(&device_dataa, heighta*widtha*channela*sizeof(float));
		cudaStat = cudaMalloc(&device_datab, heightb*widthb*channelb*sizeof(float));
		int tmpa = heighta*widtha, tmpb = widthb*heightb;
		cudaMemcpy(device_dataa, neta->blob_by_name(sa)->gpu_data(), tmpa*channela*sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(device_datab, netb->blob_by_name(s)->gpu_data(), tmpb*channelb*sizeof(float), cudaMemcpyDeviceToDevice);

		char sup[100];
		sprintf(sup, "conv%d_1", tt + 1);


		if (tt == 5)
			match_reverse_PCA(device_dataa, device_datab, channela, heighta, widtha, heightb, widthb, 3, ann, 0, 0, tt, alpha, lambda, f, ntot);
		else
		if (tt >= 4) match_reverse_PCA(device_dataa, device_datab, channela, heighta, widtha, heightb, widthb, 3, ann, neta->blob_by_name(sup)->height(), neta->blob_by_name(sup)->width(), tt, alpha, lambda, f, ntot);
		else match_reverse_PCA(device_dataa, device_datab, channela, heighta, widtha, heightb, widthb, 5, ann, neta->blob_by_name(sup)->height(), neta->blob_by_name(sup)->width(), tt, alpha, lambda, f, ntot);

		char s1[1000];

		if (tt == 5) sprintf(s1, "relu5_1");
		else sprintf(s1, "relu-deconv%d_1", tt);

		if (tt == 5) sprintf(s, "conv5_1");
		else sprintf(s, "deconv%d_1", tt);

		int id1;

		vector<string> layer_names = neta->layer_names();
		for (int i = 0; i < layer_names.size(); i++)
		{
			if (layer_names[i] == s1)
				id1 = i;
		}
		cudaMemcpy(neta->blob_by_name(s)->mutable_gpu_data(), device_dataa, channela*heighta*widtha*sizeof(float), cudaMemcpyHostToDevice);
		neta->ForwardFrom(id1 + 1);
		cudaFree(device_dataa);
		cudaFree(device_datab);
	}


	if (ann) cudaFree(ann);
	float *output_t = new float[3 * imga.rows*imga.cols];
	cudaMemcpy(output_t, neta->blob_by_name("deconv_data")->gpu_data(), 3 * imga.cols*imga.rows*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < imga.rows; i++)
	for (int j = 0; j < imga.cols; j++)
	{
		output_t[0 * imga.rows*imga.cols + i*imga.cols + j] = output_t[0 * imga.rows*imga.cols + i*imga.cols + j] + 103.939;
		output_t[1 * imga.rows*imga.cols + i*imga.cols + j] = output_t[1 * imga.rows*imga.cols + i*imga.cols + j] + 116.779;
		output_t[2 * imga.rows*imga.cols + i*imga.cols + j] = output_t[2 * imga.rows*imga.cols + i*imga.cols + j] + 123.68;
	}
	for (int i = 0; i < imga.rows; i++)
	for (int j = 0; j < imga.cols; j++)
	for (int k = 0; k <= 2; k++)
		output_img.at<Vec3f>(i, j)[k] = output_t[k*imga.rows*imga.cols + i*imga.cols + j];
	char fname[100];
	free(output_t);
	string output_path = "";
	imwrite(output_path + argv[3], output_img);

	return 0;
}

