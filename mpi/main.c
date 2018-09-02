#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include "mpi.h"
#include "filter.h"

MPI_Status status;

MPI_Datatype rowGrey;
MPI_Datatype rowRGB;

MPI_Request topSend;
MPI_Request bottomSend;
MPI_Request topRcv;
MPI_Request bottomRcv;

int getFileSize(int width, int height, int isRGB) {
    if (isRGB) {
        return 3 * width * height;
    } else {
        return width * height;
    }
}

int fileMemAllocSize(int rows, int columns, int isRGB) {
    if (isRGB) {
        return (rows + 2) * (columns*3 + 6);
    } else {
        return (rows + 2) * (columns + 2);
    }
}

void broadcastInfo(int* isRGB, int* width, int* height, int* iterations, int* splitRowNum) {
    MPI_Bcast(isRGB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(splitRowNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
	int width, height, loops, t, splitRowNum, rows, cols;
	char *imageName;
    int isRGB;
    int comm_rank, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);	

    if (comm_rank == 0) {
        if (argc != 6) {
            fprintf(stderr, "Bad input provided\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            exit(EXIT_FAILURE);
        }
        isRGB = atoi(argv[1]);
        imageName = malloc((strlen(argv[2]) + 1) * sizeof(char));
        strcpy(imageName, argv[2]);
        width = atoi(argv[3]);
        height = atoi(argv[4]);
        loops = atoi(argv[5]);
        if (height % comm_size == 0) {
            splitRowNum = comm_size;
        } else {
            fprintf(stderr, "Bad config provided\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            exit(EXIT_FAILURE);
        }
    } else {
        imageName = malloc((strlen(argv[2])+1) * sizeof(char));
        strcpy(imageName, argv[2]);
    }
    broadcastInfo(&isRGB, &width, &height, &loops, &splitRowNum);
	
    rows = height / splitRowNum;
    cols = width;

    MPI_Type_contiguous(cols, MPI_BYTE, &rowGrey);
	MPI_Type_commit(&rowGrey);
	MPI_Type_contiguous(3*cols, MPI_BYTE, &rowRGB);
	MPI_Type_commit(&rowRGB);

	 /* Compute starting row and column */
    int start_row = comm_rank * rows;
    int start_col = 0;

	/* Init arrays */
	unsigned char* source = NULL;
    unsigned char* dest = NULL;
	MPI_File fh;
	int filesize = getFileSize(width, height, isRGB);
	source = calloc(fileMemAllocSize(rows, cols, isRGB), sizeof(unsigned char));
	dest = calloc(fileMemAllocSize(rows, cols, isRGB), sizeof(unsigned char));
	if (source == NULL || dest == NULL) {
        fprintf(stderr, "%s: Not enough memory\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
	}

	/* Parallel read */
	MPI_File_open(MPI_COMM_WORLD, imageName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
	if (isRGB) {
        int i;
		for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(fh, 3*(start_row + i-1) * width + 3*start_col, MPI_SEEK_SET);
			MPI_File_read(fh, &source[i*(cols*3+6)+3], cols*3, MPI_BYTE, &status);
		}
	} else {
        int i;
        for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(fh, (start_row + i-1) * width + start_col, MPI_SEEK_SET);
			MPI_File_read(fh, &source[i*(cols+2)+1], cols, MPI_BYTE, &status);
		}
	}
	MPI_File_close(&fh);

    int top, bottom;
    if (start_row != 0) {
        top = comm_rank - 1;
    } else {
        top = -1;
    }
    if (start_row + rows != height) {
        bottom = comm_rank + 1;
    } else {
        bottom = -1;
    }
	
	MPI_Barrier(MPI_COMM_WORLD);

    double startTime = MPI_Wtime();
	/* Convolute "loops" times */
	for (t = 0 ; t < loops ; t++) {
        /* Send and request borders */
		if (isRGB) {
			if (top != -1) {
				MPI_Isend(&source[3*cols+6 + 1], 1, rowRGB, top, 0, MPI_COMM_WORLD, &topSend);
				MPI_Irecv(&source[3], 1, rowRGB, top, 0, MPI_COMM_WORLD, &topRcv);
			}
			if (bottom != -1) {
				MPI_Isend(&source[rows*(3*cols+6)], 1, rowRGB, bottom, 0, MPI_COMM_WORLD, &bottomSend);
				MPI_Irecv(&source[(rows+1)*(3*cols+6)], 1, rowRGB, bottom, 0, MPI_COMM_WORLD, &bottomRcv);
			}
		} else {
            if (top != -1) {
				MPI_Isend(&source[cols+3], 1, rowGrey, top, 0, MPI_COMM_WORLD, &topSend);
				MPI_Irecv(&source[1], 1, rowGrey, top, 0, MPI_COMM_WORLD, &topRcv);
			}
			if (bottom != -1) {
				MPI_Isend(&source[rows*(cols+2) + 1], 1, rowGrey, bottom, 0, MPI_COMM_WORLD, &bottomSend);
				MPI_Irecv(&source[(rows+1)*(cols+2)+1], 1, rowGrey, bottom, 0, MPI_COMM_WORLD, &bottomRcv);
			}
		}

		blur(source, dest, 1, rows, 1, cols, cols, rows, isRGB);

		if (top != -1) {
			MPI_Wait(&topRcv, &status);
			blur(source, dest, 1, 1, 2, cols-1, cols, rows, isRGB);
		}
		if (bottom != -1) {
			MPI_Wait(&bottomRcv, &status);
			blur(source, dest, rows, rows, 2, cols-1, cols, rows, isRGB);
		}

		if (top != -1)
			MPI_Wait(&topSend, &status);
		if (bottom != -1)
			MPI_Wait(&bottomSend, &status);

        unsigned char* temp;
		temp = source;
        source = dest;
        dest = temp;
	}
    double totalTime = MPI_Wtime() - startTime;

	char *outimageName = malloc(12 * sizeof(char));
	strcpy(outimageName, "blurred.raw");
	MPI_File outFile;
	MPI_File_open(MPI_COMM_WORLD, outimageName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outFile);
	if (isRGB) {
        int i;
		for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(outFile, 3*(start_row + i-1) * width + 3*start_col, MPI_SEEK_SET);
			MPI_File_write(outFile, &source[(cols*3+6)*i + 3], cols*3, MPI_BYTE, MPI_STATUS_IGNORE);
		}
	} else {
        int i;
        for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(outFile, (start_row + i-1) * width + start_col, MPI_SEEK_SET);
			MPI_File_write(outFile, &source[(width+2)*i + 1], cols, MPI_BYTE, MPI_STATUS_IGNORE);
		}
	}
	MPI_File_close(&outFile);

    if (comm_rank != 0)
        MPI_Send(&totalTime, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    else {
        double commPassedTime;
        int i;
        for (i = 1 ; i != comm_size ; i++) {
            MPI_Recv(&commPassedTime, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            if (commPassedTime > totalTime)
                totalTime = commPassedTime;
        }
        printf("Max process passed time: %f\n", totalTime);
    }

    MPI_Finalize();
	return EXIT_SUCCESS;
}