
#pragma once
#ifndef	HIGHPRECISIONTIME_H
#define	HIGHPRECISIONTIME_H

#if	defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#endif

class HighPrecisionTime
{
public:
	HighPrecisionTime();
	double TimeSinceLastCall();
	double TotalTime();

private:

#if	defined(WIN32) || defined(_WIN64)
	LARGE_INTEGER initializationTicks;
	LARGE_INTEGER ticksPerSecond;
	LARGE_INTEGER previousTicks;
#endif

};

#endif