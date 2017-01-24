#include <iostream>
#include <iomanip>
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "HPT1.h"
#include <stdio.h>


using namespace std;
using namespace cv;

Mat host_image;
Mat orig_image;
unsigned char * device_src = nullptr;
unsigned char * device_dst = nullptr;
HighPrecisionTime t;
size_t image_bytes;
string window_name("Ouput");
const int THRESHOLD_SLIDER_MAX = 256;
int threshold_slider = 0;
int inner_slider = 0;
int outer_slider = 100;
double gpu_accumulator = 0;
int gpu_counter = 0;
double cpu_accumulator = 0;
int cpu_counter = 0;
bool gpu_mode = true;
typedef unsigned char uByte;



__global__ void boxKernel(uByte* data, uByte* data2,int w, int h) {

	int PlaceX = ((blockIdx.x *blockDim.x) + threadIdx.x);
	int PlaceY = (blockIdx.y *blockDim.y) + threadIdx.y;

	if (PlaceX < w && PlaceY < h) {
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
void ConvolveGPU(uByte* src, uByte * dst, int W, int H, uByte* kernel, HighPrecisionTime timer) {

	uByte* imageTemp = nullptr;
	uByte* imageBlurred = nullptr;
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
	
	cudaTest = cudaMalloc(&imageBlurred, ConvolveSize);
	if (cudaTest != cudaSuccess) {
		cout << "cudaMalloc failed!" << endl;
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

	boxKernel << < blocksNeeded, 1024 >> > (imageTemp, imageBlurred, host_image.cols, host_image.rows);

	cudaTest = cudaMemcpy(dst, imageBlurred, ConvolveSize, cudaMemcpyDeviceToHost);
	if (cudaTest != cudaSuccess) {
		cout << "cudacpy2 failed!" << endl;
	}

}
double ConvolveCPU(uByte* src, uByte * dst, int W, int H, uByte* kernel, HighPrecisionTime timer){
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


void Threshold(Mat & image, int t)
{
	assert(image.channels() == 1);
	unsigned char lt = static_cast<unsigned char>(t);
	const long long e = reinterpret_cast<long long>(image.data + image.cols * image.rows);

	if (t == 256)
	{
		memset(image.data, 0, image.rows * image.cols);
	}
	else
	{
		for (long long p = reinterpret_cast<long long>(image.data); p < e; p++)
		{
			*((unsigned char *)p) = (*((unsigned char *)p) >= lt) ? 255 : 0;
		}
	}
}

__global__ void vignette(const unsigned char * src, unsigned char * dst, float inner, float outer, const size_t width, const size_t height)
{
	// the xIndex and yIndex will be used cordinates pixels of the image
	// NOTE
	// NOTE This assumes that we are treating this as a two dimensional data structure and the blocks will be used in the same way
	// NOTE
	size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	// Checking to see if the indexs are within the bounds of the image
	if (xIndex < width && yIndex < height)
	{
		// offset represents postion of the current pixel in the one dimensional array
		size_t offset = yIndex * width + xIndex;
		// Shift the pixel oriented coordinates into image resolution independent coordinates
		// where 0, 0 is the center of the image.
		float x = xIndex / float(height) - float(width) / float(height) / 2.0f;
		float y = yIndex / float(height) - 0.5f;
		//Calculates current pixels distance from the center where the cordinates are 0, 0
		float d = sqrtf(x * x + y * y);
		if (d < inner)
		{
			// if d is less than inner boundary, we don't change that specific image pixel
			*(dst + offset) = *(src + offset);
		}
		else if (d > outer)
		{
			// if d is greater than outer boundary, we set it to 0 so it becomes black
			*(dst + offset) = 0;
		}
		else
		{
			// If in between the inner and outer boundaries, it will be a shade of gray.
			// NOTE
			// NOTE  This assumes... by the time we get here, we have checked that outer does not equal inner
			// NOTE  This also assumes ... by the time we get here, we have made inner less than outer
			// NOTE
			float v = 1 - (d - inner) / (outer - inner);
			*(dst + offset) = (unsigned char)(*(src + offset) * v);
		}
	}
}

__global__ void kernel(const unsigned char * src, unsigned char * dst, int level, const size_t width, const size_t height)
{
	const size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if (xIndex < width && yIndex < height)
	{
		size_t o = yIndex * width + xIndex;
		if (level == 256)
		{
			*(dst + o) = 0;
		}
		else
		{
			*(dst + o) = (*(src + o) >= level) ? 255 : 0;
		}
		// Notice how the below version avoids having an 'if' statement.
		// I wonder if this is truly correct - I'll have to test this
		// carefully someday but it works correctly. I figured the
		// subtraction should cause an underflow which the shift might
		// propagate through the rest of the byte so as to cause 255.
		// *(dst + o) = ~((*(src + o) - level - 1) >> 7);
	}
}

void on_trackbar(int, void *)
{
	dim3 grid((host_image.cols + 1023) / 1024, host_image.rows);
	double d;

	int i = inner_slider;
	int o = outer_slider;
	if (i > o)
	{
		swap(i, o);
	}
	if (i == o)
	{
		o++;
	}
	float inner = i / 100.0f;
	float outer = o / 100.0f;

	if (gpu_mode)
	{
		if (cudaMemcpy(device_src, orig_image.ptr(), image_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
		{
			cerr << "cudaMemcpy failed at " << __LINE__ << endl;
			cudaDeviceReset();
			exit(1);
		}

		t.TimeSinceLastCall();
		kernel << <grid, 1024 >> >(device_src, device_dst, threshold_slider, host_image.cols, host_image.rows);
		cudaDeviceSynchronize();
		gpu_accumulator += t.TimeSinceLastCall();
		gpu_counter++;
		cout << "GPU AVG " << setw(12) << fixed << setprecision(8) << gpu_accumulator / ((double)gpu_counter) << " seconds";
		vignette << <grid, 1024 >> >(device_dst, device_src, inner, outer, host_image.cols, host_image.rows);
		cudaDeviceSynchronize();

		t.TimeSinceLastCall();
		if (cudaMemcpy(host_image.ptr(), device_src, image_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			cerr << "cudaMemcpy failed at " << __LINE__ << endl;
		}
		d = t.TimeSinceLastCall();
		cout << " XFER: " << setw(8) << setprecision(4) << ((double)image_bytes) / (d * 1024.0 * 1024.0 * 1024.0) << " GB/s" << endl;
	}
	else
	{
		t.TimeSinceLastCall();
		host_image = orig_image;
		Threshold(host_image, threshold_slider);
		cpu_accumulator += t.TimeSinceLastCall();
		cpu_counter++;
		cout << "CPU AVG " << setw(12) << fixed << setprecision(8) << cpu_accumulator / ((double)cpu_counter) << " seconds";
		cout << endl;
	}
	imshow(window_name, host_image);
}

int main(int argc, char * argv[])
{
	if (argc != 2)
	{
		cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}
	host_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!host_image.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cout << "Image has: " << host_image.channels() << " channels." << endl;
	cout << "Image has size: " << host_image.cols << " x " << host_image.rows << " pixels." << endl;

	cvtColor(host_image, host_image, cv::COLOR_RGB2GRAY);
	host_image.copyTo(orig_image);
	cout << "Converted to gray." << endl;

	if (cudaSetDevice(0) != cudaSuccess)
	{
		cerr << "cudaSetDevice(0) failed." << endl;
		cudaDeviceReset();
		exit(1);
	}
	image_bytes = host_image.rows * host_image.cols;
	cudaMalloc(&device_src, image_bytes);
	cudaMalloc(&device_dst, image_bytes);
	if (device_dst == nullptr || device_src == nullptr)
	{
		cerr << "cudaMalloc failed on either device_src or device_dst at " << __LINE__ << endl;
		cudaDeviceReset();
		exit(1);
	}

	// Copy the source image to the device. Note - although the kernel will be
	// called repeatedly, this is the only time we'll copy TO the device as the
	// image processing operation does not harm the original image.
	if (cudaMemcpy(device_src, host_image.data, image_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cerr << "cudaMemcpy failed at " << __LINE__ << endl;
		cudaDeviceReset();
		exit(1);
	}

	namedWindow(window_name, WINDOW_KEEPRATIO);
	resizeWindow(window_name, host_image.cols / 10, host_image.rows / 10);
	createTrackbar("Threshold", window_name, &threshold_slider, THRESHOLD_SLIDER_MAX, on_trackbar);
	createTrackbar("Inner", window_name, &inner_slider, 100, on_trackbar);
	createTrackbar("Outer", window_name, &outer_slider, 100, on_trackbar);
	on_trackbar(threshold_slider, 0);

	int k;

	while ((k = waitKey(10)) != 'q')
	{
		if (k == 'm')
		{
			gpu_mode = !gpu_mode;
			on_trackbar(0, nullptr);
		}
	}
	destroyAllWindows();

	if (device_src != nullptr)
		cudaFree(device_src);
	if (device_dst != nullptr)
		cudaFree(device_dst);
	cudaDeviceReset();
	return 0;
}