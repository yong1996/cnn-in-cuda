#include <stdio.h>

#include <stdlib.h>
#include <fcntl.h>
#include <iostream>
#include <cstdlib>
#include <cassert>

#include "string.h"


void write_ppm(char* filename, int xsize, int ysize, int maxval, float **pic)
{
	FILE* fp;
	//int x, y;

	fp = fopen(filename, "wb");
	if (!fp)
	{
		fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
		exit(-1);
	}

	fprintf(fp, "P6\n");
	fprintf(fp, "%d %d\n%d\n", xsize, ysize, maxval);

	int numpix = xsize * ysize;
	for (int i = 0; i < xsize; i++) {
        for (int j = 0; j < ysize; j++){
            unsigned char uc = (unsigned char)pic[i][j];
		    fprintf(fp, "%c%c%c", uc, uc, uc);
        }	
	}

	fclose(fp);
}