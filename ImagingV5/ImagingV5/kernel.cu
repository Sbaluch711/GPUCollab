

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include "HPT.h"
#include <vector>


using namespace cv;
using namespace std;

typedef unsigned char uByte;

Mat image;
Mat postImage;
int trackbarSize;
uByte* ImageModified;


//kernal takes in two arrays and size
__global__ void thresholdKernel(unsigned char* data, unsigned char* data2, int size, int thresholdSlider) {


	int j = (blockIdx.x *blockDim.x) + threadIdx.x;

	if (j < size) {
		if (data[j] > thresholdSlider) {
			data2[j] = 255;
		}
		else {
			data2[j] = 0;
		}
	}
}
__global__ void boxKernel(uByte* data, uByte* data2, int size) {
	
	int PlaceX = ((blockIdx.x *blockDim.x) + threadIdx.x);
	int PlaceY=(blockIdx.y *blockDim.y) + threadIdx.y;
	
	if (PlaceX *PlaceY < size) {
		int currentPixelSum = data[PlaceX*PlaceY];
		//w-1,h+1
		currentPixelSum += data[(PlaceY + 1)*PlaceX - 1];
		//h+1, w+1
		currentPixelSum += data[(PlaceY + 1) * PlaceX + 1];
		//h+1
		currentPixelSum += data[(PlaceY + 1) * PlaceX];
		//w-1
		currentPixelSum += data[(PlaceX * PlaceX) - 1];
		//w+1
		currentPixelSum += data[(PlaceY*PlaceX) + 1];
		//w-1,h-1
		currentPixelSum += data[((PlaceY - 1) * PlaceX) - 1];
		//h-1
		currentPixelSum += data[(PlaceY - 1)*PlaceX];
		//w+1,h-1
		currentPixelSum += data[(PlaceY - 1) * PlaceX + 1];

		data2[PlaceY * PlaceY] = uByte(currentPixelSum / 9.0f);
	}
}
//threshold change in cpu
void threshold(int threshold, int width, int height, unsigned char* data);
//threshold change in gpu
bool initializeImageGPU(int width, int height, Mat image);
//creates trackbar for image
void on_trackbar(int thresholdSlider, void*);
void BoxFilter(uByte* s, uByte* d, int w, int h);
//void BoxFilter(uByte* s, uByte* d, int w, int h, uByte* k, int kW, int kH, uByte* Temp);
double ConvolveCPU(uByte* src, uByte * dst, int W, int H, uByte* kernel, HighPrecisionTime timer);
double ConvolveGPU(uByte* src, uByte * dst, int W, int H, uByte* kernel, HighPrecisionTime timer);




int main(int argc, char** argv) {
	if (argc != 2) {
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}


	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	if (!image.data) {
		cout << "Could not open or find image" << endl;
		return -1;
	}
	postImage = image.clone();
	cvtColor(image, image, COLOR_RGB2GRAY);

	HighPrecisionTime Timer;
	threshold(128, image.rows, image.cols, image.data);
	uByte Kernel[9] = { 1,1,1,1,1,1,1,1,1 };
	cout << ConvolveCPU(postImage.data, image.data, image.cols, image.rows, Kernel, Timer);

	/*if (initializeImageGPU(image.rows, image.cols, image)) {
	cout << "We worked with the GPU" << endl;
	}
	else {
	cout << "It failed." << endl;
	}*/

	namedWindow("Display Window", WINDOW_NORMAL);
	//createTrackbar("Threshold", "Display Window", &threshold_slider, THRESHOLD_SLIDER_MAX, on_tracker(int, void *, Image, unsigned char* data2, size,threshold_slider));
	imshow("Display Window", image);
	namedWindow("Blur", WINDOW_NORMAL);
	imshow("Blur", postImage);

	waitKey(0);
	return 0;
}

void threshold(int threshold, int width, int height, unsigned char * data)
{
	HighPrecisionTime timeTheModification;
	double currentTime;
	timeTheModification.TimeSinceLastCall();
	for (int i = 0; i < height *width; i++) {
		if (data[i] > threshold) {
			data[i] = 255;
		}
		else {
			data[i] = 0;
		}
	}
	currentTime = timeTheModification.TimeSinceLastCall();
	cout << "CPU Threshold: " << currentTime << endl;
}

bool initializeImageGPU(int width, int height, Mat image)
{
	HighPrecisionTime timeTheModification;
	double currentTime;

	bool temp = true;
	unsigned char* ImageOriginal = nullptr;
	ImageModified = nullptr;
	int size = width*height * sizeof(char);
	trackbarSize = size;

	cudaError_t cudaTest;

	cudaTest = cudaSetDevice(0);
	if (cudaTest != cudaSuccess) {
		cout << "Error with device" << endl;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaMalloc(&ImageOriginal, size);
	if (cudaTest != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaMalloc(&ImageModified, size);
	if (cudaTest != cudaSuccess) {
		cout << "cudaMalloc2 failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaDeviceSynchronize();
	if (cudaTest != cudaSuccess) {
		cout << "cudaSync failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaMemcpy(ImageOriginal, image.data, size, cudaMemcpyHostToDevice);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy failed!" << endl;
		temp = false;
	}
	else {
		//cout << "suscsess" << endl;
	}

	int blocksNeeded = (size + 1023) / 1024;

	timeTheModification.TimeSinceLastCall();
	thresholdKernel << <blocksNeeded, 1024 >> > (ImageOriginal, ImageModified, size, 128);
	currentTime = timeTheModification.TimeSinceLastCall();
	cout << "GPU Threshold: " << currentTime << endl;

	cudaTest = cudaMemcpy(image.data, ImageModified, size, cudaMemcpyDeviceToHost);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy2 failed!" << endl;
		temp = false;
	}

	int thresholdSlider = 50;
	namedWindow("Blurred Image", WINDOW_NORMAL);
	//createTrackbar("Threshold", "BlurredImage", &thresholdSlider, 255, on_trackbar, &ImageOriginal);
	cout << "Created Trackbar";
	uByte Kernel[9] = { 1,1,1,1,1,1,1,1,1 };
	ConvolveCPU(postImage.data, image.data, image.cols, image.rows, Kernel, timeTheModification);
	//BoxFilter(image.data, postImage.data, image.cols, image.rows);
	imshow("BlurredImage", image);

	cudaTest = cudaMemcpy(image.data, ImageModified, size, cudaMemcpyDeviceToHost);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy2 failed!" << endl;
		temp = false;
	}
	//on_trackbar(thresholdSlider, 0);

	waitKey(0);

	return temp;
}

void on_trackbar(int thresholdSlider, void*)
{

	HighPrecisionTime T;
	double currentTime;
	trackbarSize = image.cols *image.rows;
	int blocksNeeded = (trackbarSize + 1023) / 1024;
	cudaDeviceSynchronize();

	T.TimeSinceLastCall();
	thresholdKernel << < blocksNeeded, 1024 >> > (image.data, ImageModified, (image.rows*image.cols), thresholdSlider);
	//BoxFilter(image.data, ImageModified, image.cols, image.rows);
	uByte Kernel[9] = { 1,1,1,1,1,1,1,1,1 };
	ConvolveCPU(postImage.data, image.data, image.cols, image.rows, Kernel, T);
	currentTime = T.TimeSinceLastCall();
	cout << "CurrentTime: " << currentTime << endl;


	if (cudaMemcpy(image.data, ImageModified, trackbarSize, cudaMemcpyDeviceToHost) != cudaSuccess) {
		cout << "Error copying." << endl;
	}

	imshow("Display Window", image);
}

void BoxFilter(uByte * s, uByte * d, int w, int h)
{
	float currentPixelSum = 0.0f;
	for (int i = 1; i < w - 1; i++) {
		for (int j = 1; j < h - 1; j++) {
			//set current pixel
			//c
			currentPixelSum = s[i* j];
			//w-1,h+1
			currentPixelSum += s[(j + 1)*i - 1];
			//h+1, w+1
			currentPixelSum += s[(j + 1) * i + 1];
			//h+1
			currentPixelSum += s[(j + 1) * i];
			//w-1
			currentPixelSum += s[(j * i) - 1];
			//w+1
			currentPixelSum += s[(j*i) + 1];
			//w-1,h-1
			currentPixelSum += s[((j - 1) * i) - 1];
			//h-1
			currentPixelSum += s[(j - 1)*i];
			//w+1,h-1
			currentPixelSum += s[(j - 1) * i + 1];

			d[i * j] = uByte(currentPixelSum / 9.0f);
			s[i*j] = d[i*j];
		}
	}
}
double ConvolveCPU(uByte* src, uByte * dst, int W, int H, uByte* kernel, HighPrecisionTime timer)
{
	int kernel_size, half_kernel_size;
	kernel_size = 3;// (getTrackbarPos(kernel_size_name, window_name_cpu) + 1) * 2 + 1;
	half_kernel_size = kernel_size / 2;
	float divisor = 9.0f;//float(SumKernel(kernel, kernel_size));

	timer.TimeSinceLastCall();
	for (int y = 0; y < H; y++)
	{
		for (int x = 0; x < W; x++)
		{
			// Initialize with 0,0
			int sum = 0; // *(src + y * W + x) * kernel[half_kernel_size * kernel_size + half_kernel_size];
			uByte * kp = kernel + half_kernel_size;

			for (int y_offset = -half_kernel_size; y_offset <= half_kernel_size; y_offset++, kp += kernel_size)
			{
				if (y_offset + y < 0 || y_offset + y >= H)
					continue;

				sum += *(src + (y_offset + y) * W + x) * *kp;
				for (int x_offset = 1; x_offset <= half_kernel_size; x_offset++)
				{
					if (x - x_offset >= 0)
						sum += *(src + (y_offset + y) * W - x_offset + x) * *(kp - x_offset);
					if (x + x_offset < W)
						sum += *(src + (y_offset + y) * W + x_offset + x) * *(kp + x_offset);
				}
			}
			*(dst + y * W + x) = uByte(float(sum) / divisor);
		}
	}
	return timer.TimeSinceLastCall();
}
double ConvolveGPU(uByte* src, uByte * dst, int W, int H, uByte* kernel, HighPrecisionTime timer) {
	
	unsigned char* imageTemp = nullptr;
	unsigned char* imageBlurred = nullptr;
	cudaError_t cudaTest;
	int ConvolveSize = W*H;
	cudaTest = cudaSetDevice(0);
	if (cudaTest != cudaSuccess) {
		cout << "Error with device" << endl;
	}

	cudaTest = cudaMalloc(&imageTemp, ConvolveSize);
	if (cudaTest != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
	}
	else {
		//cout << "suscsess" << endl;
	}
	cudaTest = cudaMalloc(&imageBlurred, ConvolveSize);
	if (cudaTest != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
	}
	else {
		//cout << "suscsess" << endl;
	}

	cudaTest = cudaDeviceSynchronize();
	if (cudaTest != cudaSuccess) {
		cout << "cudaSync failed!" << endl;
	}

	cudaTest = cudaMemcpy(imageTemp, src, ConvolveSize, cudaMemcpyHostToDevice);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy failed!" << endl;
	}

	int blocksNeeded = (ConvolveSize + 1023) / 1024;
	cudaDeviceSynchronize();

	boxKernel<< < blocksNeeded, 1024 >> > (imageTemp, imageBlurred, ConvolveSize);
	
	cudaTest = cudaMemcpy(dst, imageBlurred, ConvolveSize, cudaMemcpyDeviceToHost);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy2 failed!" << endl;
	}
	
}

