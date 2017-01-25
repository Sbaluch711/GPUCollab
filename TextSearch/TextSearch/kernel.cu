
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "HPT.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

#define GIGA (1 <<30)
#define BSIZE (GIGA/8)
int main()
{
	HighPrecisionTime readingTimer;
	char* buffer = new char[GIGA]();
	char* bitmap = new char[BSIZE]();

	ifstream bigFile("C:\\Users\\educ\\Documents\\enwiki-latest-abstract.xml");

	readingTimer.TimeSinceLastCall();
	bigFile.read(buffer, GIGA);
	cout << readingTimer.TimeSinceLastCall() << endl;

	bigFile.close();

	system("pause");
	return 0;
}

