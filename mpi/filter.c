#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "mpi.h"
#include "filter.h"

const float blurFilter[3][3] = {{1/16.0, 2/16.0, 1/16.0}, {2/16.0, 4/16.0, 2/16.0}, {1/16.0, 2/16.0, 1/16.0}};

void blur(unsigned char* source, unsigned char* dest, int startRow, int endRow, int startCol, int endCol, int width, int height, int isRGB) {
	int i, j;
    if (isRGB) {
		for (i = startRow ; i <= endRow ; i++) {
			for (j = startCol ; j <= endCol ; j++) {
				blurRGB(source, dest, i, j*3, width*3+6, height);
    	}
  	}
	} else {
		for (i = startRow ; i <= endRow ; i++) {
			for (j = startCol ; j <= endCol ; j++) {
				blurGrey(source, dest, i, j, width+2, height);
      }
    }
  }
}

void blurGrey(unsigned char* source, unsigned char* dest, int x, int y, int width, int height) {
	int i, j, k=0;
	float cell = 0;
	for (i = x-1; i <= x+1 ; i++) {
        int m = 0;
		for (j = y-1; j <= y+1 ; j++) {
			cell += source[width * i + j] * blurFilter[k][m];
            m++;
        }
        k++;
    }
	dest[width * x + y] = cell;
}

void blurRGB(unsigned char* source, unsigned char* dest, int x, int y, int width, int height) {
	int i, j, k=0;
	float red = 0, green = 0, blue = 0;
	for (i = x-1; i <= x+1 ; i++) {
        int m = 0;
		for (j = y-3; j <= y+3 ; j+=3) {
			red += source[width * i + j]* blurFilter[k][m];
			green += source[width * i + j+1] * blurFilter[k][m];
			blue += source[width * i + j+2] * blurFilter[k][m];
            m++;
		}
        k++;
    }
	dest[width * x + y] = red;
	dest[width * x + y+1] = green;
	dest[width * x + y+2] = blue;
}
