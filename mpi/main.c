#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <math.h>
#include "mpi.h"
#include "filter.h"

MPI_Status status;

MPI_Datatype rowGrey;
MPI_Datatype rowRGB;
MPI_Datatype colGrey;
MPI_Datatype colRGB;

MPI_Request topSend;
MPI_Request bottomSend;
MPI_Request leftSend;
MPI_Request rightSend;
MPI_Request topRcv;
MPI_Request bottomRcv;
MPI_Request leftRcv;
MPI_Request rightRcv;

int fileMemAllocSize(int rows, int columns, int isRGB) {
    if (isRGB) {
        return (rows + 2) * (columns + 2) * 3;
    } else {
        return (rows + 2) * (columns + 2);
    }
}

void broadcastInfo(int* splitRowNum, int* splitColNum, int* comm_size_sqrt_int) {
    MPI_Bcast(splitRowNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(splitColNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(comm_size_sqrt_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
	  int width, height, loops, t, splitRowNum, splitColNum, rows, cols, comm_size_sqrt_int;
	  char *imageName;
    int isRGB,comm_rank, comm_size;
    double comm_size_sqrt;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);  //size of the group associated with a communicator
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);  //rank of the calling process in the communicator

    /* check input arguments */
    if (comm_rank == 0) {
        if (argc != 6) {
            fprintf(stderr, "Bad input provided\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);  //terminate all tasks in comm
            exit(EXIT_FAILURE);
        }
    }
    /*get arguments*/
    isRGB = atoi(argv[1]);
    imageName = argv[2];
    width = atoi(argv[3]);
    height = atoi(argv[4]);
    loops = atoi(argv[5]);
    /*check arguments for compliance*/
    if(comm_rank == 0) {
        comm_size_sqrt = sqrt(comm_size);
        if ((comm_size_sqrt_int = (int) comm_size_sqrt) != comm_size_sqrt ||
            width % comm_size_sqrt_int != 0 ||
            height % comm_size_sqrt_int != 0) {
            fprintf(stderr, "Bad config provided\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            exit(EXIT_FAILURE);
        } else {
            splitRowNum = comm_size_sqrt_int;
            splitColNum = comm_size_sqrt_int;
        }
    }
    printf("arg %s %s %s %s %s\n", argv[1], argv[2], argv[3], argv[4], argv[5]);
    printf("int %d %s %d %d %d\n", isRGB, imageName, width, height, loops );

    //broadcast info from the rank=0 process to all other
    broadcastInfo(&splitRowNum, &splitColNum, &comm_size_sqrt_int);

    rows = height / splitRowNum;
    cols = width / splitColNum;

    MPI_Type_contiguous(cols, MPI_BYTE, &rowGrey);
  	MPI_Type_commit(&rowGrey);
  	MPI_Type_contiguous(3*cols, MPI_BYTE, &rowRGB);
  	MPI_Type_commit(&rowRGB);
  	MPI_Type_vector(rows, 1, cols+2, MPI_BYTE, &colGrey);
  	MPI_Type_commit(&colGrey);
  	MPI_Type_vector(rows, 3, 3*cols+6, MPI_BYTE, &colRGB);
  	MPI_Type_commit(&colRGB);

    //calculate the starting row/col
    int start_row = (comm_rank / comm_size_sqrt_int) * rows;
    int start_col = (comm_rank % comm_size_sqrt_int) * cols;

  //allocate space for the file
	unsigned char* source = NULL;
  unsigned char* dest = NULL;
	MPI_File mpi_file;
	int filesize = fileMemAllocSize(rows, cols, isRGB);
	source = calloc(filesize, sizeof(unsigned char));
	dest = calloc(filesize, sizeof(unsigned char));
	if (source == NULL || dest == NULL) {
        fprintf(stderr, "%s: Failed to allocate memory\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
	}

	// Parallel read the file to the source array
	MPI_File_open(MPI_COMM_WORLD, imageName, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);
	if (isRGB) {
        int i;
		for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(mpi_file, 3*(start_row + i-1) * width + 3*start_col, MPI_SEEK_SET);
			MPI_File_read(mpi_file, &source[i*(cols*3+6)+3], cols*3, MPI_BYTE, &status);
		}
	} else {
        int i;
        for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(mpi_file, (start_row + i-1) * width + start_col, MPI_SEEK_SET);
			MPI_File_read(mpi_file, &source[i*(cols+2)+1], cols, MPI_BYTE, &status);
		}
	}
	MPI_File_close(&mpi_file);

  //find neighbor blocks
    int top, bottom, left, right;
    if (start_row != 0) {
        top = comm_rank - splitColNum;
    } else {  //first line blocks have no top
        top = -1;
    }
    if (start_row + rows != height) {
        bottom = comm_rank + splitColNum;
    } else {  //last line blocks have no bottom
        bottom = -1;
    }
    if (start_col != 0) {
        left = comm_rank - 1;
    } else { //left column blocks have no left
        left = -1;
    }
    if (start_col + cols != width) {
        right = comm_rank + 1;
    } else {  //right column blocks have no right
        right = -1;
    }

	MPI_Barrier(MPI_COMM_WORLD); //block until all processes reach this routine

  double startTime = MPI_Wtime();
	// Convolute "loops" times
	for (t = 0 ; t < loops ; t++) {
        /* Send and request borders */
		if (isRGB) {
			if (top != -1) { //not topmost block
				MPI_Isend(&source[3*cols+6 + 3], 1, rowRGB, top, 0, MPI_COMM_WORLD, &topSend);
				MPI_Irecv(&source[3], 1, rowRGB, top, 0, MPI_COMM_WORLD, &topRcv);
			}
			if (bottom != -1) {  //not bottommost block
				MPI_Isend(&source[rows*(3*cols+6) + 3], 1, rowRGB, bottom, 0, MPI_COMM_WORLD, &bottomSend);
				MPI_Irecv(&source[(rows+1)*(3*cols+6) + 3], 1, rowRGB, bottom, 0, MPI_COMM_WORLD, &bottomRcv);
			}
			if (left != -1) { //not leftmost block
                MPI_Isend(&source[3*cols+6 + 3], 1, colRGB, left, 0, MPI_COMM_WORLD, &leftSend);
                MPI_Irecv(&source[3*cols+6], 1, colRGB, left, 0, MPI_COMM_WORLD, &leftRcv);
			}
			if (right != -1) { //not rightmost block
                MPI_Isend(&source[3*cols+6 + 3*cols], 1, colRGB, right, 0, MPI_COMM_WORLD, &rightSend);
                MPI_Irecv(&source[3*cols+6 + 3*cols+3], 1, colRGB, right, 0, MPI_COMM_WORLD, &rightRcv);
			}
		} else {
            if (top != -1) {  //not first block
				MPI_Isend(&source[cols+3], 1, rowGrey, top, 0, MPI_COMM_WORLD, &topSend);
				MPI_Irecv(&source[1], 1, rowGrey, top, 0, MPI_COMM_WORLD, &topRcv);
			}
			if (bottom != -1) { //not last block
				MPI_Isend(&source[rows*(cols+2) + 1], 1, rowGrey, bottom, 0, MPI_COMM_WORLD, &bottomSend);
				MPI_Irecv(&source[(rows+1)*(cols+2)+1], 1, rowGrey, bottom, 0, MPI_COMM_WORLD, &bottomRcv);
			}
            if (left != -1) { //not leftmost block
                MPI_Isend(&source[cols+2 + 1], 1, colRGB, left, 0, MPI_COMM_WORLD, &leftSend);
                MPI_Irecv(&source[cols+2], 1, colRGB, left, 0, MPI_COMM_WORLD, &leftRcv);
            }
            if (right != -1) { //not rightmost block
                MPI_Isend(&source[cols+2 + cols], 1, colRGB, right, 0, MPI_COMM_WORLD, &rightSend);
                MPI_Irecv(&source[cols+2 + cols+1], 1, colRGB, right, 0, MPI_COMM_WORLD, &rightRcv);
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
		if (left != -1) {
		    MPI_Wait(&leftRcv, &status);
            blur(source, dest, 2, rows-1, 1, 1, cols, rows, isRGB);
		}
		if (right != -1) {
		    MPI_Wait(&rightRcv, &status);
            blur(source, dest, 2, rows-1, cols, cols, cols, rows, isRGB);
		}

		if (top != -1 && left != -1) {
		    blur(source, dest, 1, 1, 1, 1, cols, rows, isRGB);
		}
		if (top != -1 && right != -1) {
		    blur(source, dest, 1, 1, cols, cols, cols, rows, isRGB);
		}
		if (bottom != -1 && left != -1) {
		    blur(source, dest, rows, rows, 1, 1, cols, rows, isRGB);
		}
		if (bottom != -1 && right != -1) {
		    blur(source, dest, rows, rows, cols, cols, cols, rows, isRGB);
		}

		if (top != -1) {
            MPI_Wait(&topSend, &status);
        }
		if (bottom != -1) {
            MPI_Wait(&bottomSend, &status);
        }
		if (left != -1) {
            MPI_Wait(&leftSend, &status);
        }
        if (right != -1) {
            MPI_Wait(&rightSend, &status);
        }

        unsigned char* temp;
		temp = source;
        source = dest;
        dest = temp;
	}
    double totalTime = MPI_Wtime() - startTime;

	char *outimageName = malloc(12 * sizeof(char));
	strcpy(outimageName, "blurred.raw");
	MPI_File outFile;
	//#nem why not:
	//MPI_File_open(MPI_COMM_WORLD, "blurred.raw", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outFile);
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
			MPI_File_write(outFile, &source[(cols+2)*i + 1], cols, MPI_BYTE, MPI_STATUS_IGNORE);
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
