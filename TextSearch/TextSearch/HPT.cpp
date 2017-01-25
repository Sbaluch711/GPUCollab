#include "HPT.h"

HighPrecisionTime::HighPrecisionTime()
{
#if	defined(_WIN32) || defined(_WIN64)
	QueryPerformanceFrequency(&this->ticksPerSecond);
	QueryPerformanceCounter(&this->initializationTicks);
	previousTicks = initializationTicks;
#endif
}

double HighPrecisionTime::TimeSinceLastCall()
{
	double result = 0.0;
#if	defined(_WIN32) || defined(_WIN64)
	LARGE_INTEGER now;
	LARGE_INTEGER t;

	QueryPerformanceCounter(&now);
	t.QuadPart = now.QuadPart - previousTicks.QuadPart;
	result = ((double)t.QuadPart) / ((double)ticksPerSecond.QuadPart);
	previousTicks = now;
#endif
	return result;
}

double HighPrecisionTime::TotalTime()
{
	double result = 0.0;
#if	defined(_WIN32) || defined(_WIN64)
	LARGE_INTEGER now;
	LARGE_INTEGER t;

	QueryPerformanceCounter(&now);
	t.QuadPart = now.QuadPart - initializationTicks.QuadPart;
	result = ((double)t.QuadPart) / ((double)ticksPerSecond.QuadPart);
#endif
	return result;
}