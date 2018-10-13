#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "filter.h"
#include "omp.h"

//each process has its own checksum
long double prev_checksum = 0, new_checksum = 0;

//Normalized filter for blurring
const float blurFilter[3][3] = {{1/16.0, 2/16.0, 1/16.0}, {2/16.0, 4/16.0, 2/16.0}, {1/16.0, 2/16.0, 1/16.0}};

//Convolute every pixel fro source and write the results to destination. (s=1)
void blur(unsigned char* source, unsigned char* dest, int startRow, int endRow, int startCol, int endCol, int width, int height, int isRGB) {
	int i, j;
	/*note: the "if" is outside of the loops to increace efficency
	(don't check the same condition for every loop)*/
  if (isRGB) {
		#pragma omp parallel for collapse(2)
		for (i = startRow; i <= endRow; i++) {
			for (j = startCol; j <= endCol; j++) {
				blurRGB(source, dest, i, j*3, width*3+6, height);
    	}
  	}
	} else {
		#pragma omp parallel for collapse(2)
		for (i = startRow; i <= endRow; i++) {
			for (j = startCol; j <= endCol; j++) {
				blurGrey(source, dest, i, j, width+2, height);
      }
    }
  }
}

/*Blur a 3x3 block around the selected pixel*/
void blurGrey(unsigned char* source, unsigned char* dest, int x, int y, int width, int height) {
	int i, j, p=0;
	float cell = 0;
	for (i = x-1; i <= x+1; i++) {	//x axis
    int q = 0;
		for (j = y-1; j <= y+1; j++) {	//y axis
			cell += source[width * i + j] * blurFilter[p][q];	//apply fliter
		  q++;
    }
    p++;
  }
	dest[width * x + y] = cell;	//commit the result
	new_checksum += cell;
}

/*Blur a 3x3 block around the selected pixel for every color separately.*/
void blurRGB(unsigned char* source, unsigned char* dest, int x, int y, int width, int height) {
	int i, j, p=0;
	float red = 0, green = 0, blue = 0;
	for (i = x-1; i <= x+1 ; i++) {	//x axis
    int q = 0;
		for (j = y-3; j <= y+3 ; j+=3) {	//y axis
			red += source[width * i + j]* blurFilter[p][q];
			green += source[width * i + j+1] * blurFilter[p][q];
			blue += source[width * i + j+2] * blurFilter[p][q];
      q++;
		}
    p++;
  }
	//commit results
	dest[width * x + y] = red;
	dest[width * x + y+1] = green;
	dest[width * x + y+2] = blue;
	new_checksum += red+green+blue;
}
