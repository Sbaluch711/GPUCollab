
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "HPT.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

#define GIGA (1 <<30)

int main()
{
	HighPrecisionTime readingTimer;
	char* bitmap = new char[GIGA];
	
	ifstream bigFile("C:\\Users\\educ\\Documents\\enwiki-latest-abstract.xml");

	readingTimer.TimeSinceLastCall();
	bigFile.read(bitmap, GIGA);
	cout << readingTimer.TimeSinceLastCall() << endl;

	bigFile.close();
	system("pause");
	return 0;
	// process data in buffer

	//open file
	//time loading giga
	//print file
	//close
	//exit
}

