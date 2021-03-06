#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <mpi.h>
#include <omp.h>
#include "filter.h"

MPI_Status status;

//contiguous datatypes for communicating border rows & cols
MPI_Datatype RGB_PIXEL; //3*MPI_BYTE
MPI_Datatype rowGrey;
MPI_Datatype rowRGB;
MPI_Datatype colGrey;
MPI_Datatype colRGB;

//mpi requests for sharing border pixels
MPI_Request topSend;
MPI_Request bottomSend;
MPI_Request leftSend;
MPI_Request rightSend;
MPI_Request top_leftSend;
MPI_Request top_rightSend;
MPI_Request bot_leftSend;
MPI_Request bot_rightSend;
MPI_Request topRcv;
MPI_Request bottomRcv;
MPI_Request leftRcv;
MPI_Request rightRcv;
MPI_Request top_leftRcv;
MPI_Request top_rightRcv;
MPI_Request bot_leftRcv;
MPI_Request bot_rightRcv;

int fileMemAllocSize(int rows, int columns, int isRGB) {
    if (isRGB) {
        return (rows + 2) * (columns + 2) * 3;
    } else {
        return (rows + 2) * (columns + 2);
    }
}

void broadcastInfo(int* splitRowNum, int* splitColNum) {
    MPI_Bcast(splitRowNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(splitColNum, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void OptimizeCommPerimeter(int num_proc, int width, int height,int* splitColNum,
  int* splitRowNum) {
  unsigned int  min_perimeter = width*height*4, rows, block_width, block_height,
                comm_perimeter,total_grid_perimeter;
  for(int col=1; col<=num_proc; col++){
    if( num_proc % col == 0){
      rows = num_proc / col;
      block_width = width / col;
      block_height = height/rows;
      total_grid_perimeter = num_proc*(2*block_height+2*block_width);
      comm_perimeter = total_grid_perimeter - (2*height+2*width);
      if( comm_perimeter < min_perimeter){
        *splitColNum = col;
        *splitRowNum = rows;
        min_perimeter = comm_perimeter;
      }
    }
  }
  printf("Optimum Perimeter (Communication overhead): %d\n", min_perimeter);
}

int main(int argc, char** argv) {
  int width, height, loops, t, splitRowNum, splitColNum, rows, cols;
	char *imageName;
  int isRGB,comm_rank, comm_size;
  //Init MPI enviroment and share arguments between processes
  MPI_Init(&argc, &argv);
  //size of the group associated with a communicator
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  //rank of the calling process in the communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  /* check input arguments */
  if (comm_rank == 0) {
    if (argc != 6) {
      fprintf(stderr, "Bad input provided\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);  //end all tasks in comm
      exit(EXIT_FAILURE);
    }
  }
  /*get arguments*/
  isRGB = atoi(argv[1]);
  imageName = argv[2];
  width = atoi(argv[3]);
  height = atoi(argv[4]);
  loops = atoi(argv[5]);
  /*Find the best way to partition the image*/
  if(comm_rank == 0) {
    OptimizeCommPerimeter(comm_size,width,height,&splitColNum,&splitRowNum);
    if ( width % splitColNum || height % splitRowNum) {
      fprintf(stderr, "Can't divide. Try different process_num.\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      exit(EXIT_FAILURE);
    }
    printf("Splliting %dx rows, %dx cols\n", splitRowNum, splitColNum);
  }
  //broadcast info from the rank=0 process to others
  broadcastInfo(&splitRowNum, &splitColNum);
  //rows and cols per process
  rows = height / splitRowNum;
  cols = width / splitColNum;
  if( comm_rank == 0)
    printf("Block dimensions: %dx%d\n", cols, rows);

  //Create contiguous types for rows & cols
  /*note: MPI_Type_contiguous is for making a new datatype which is count copies
          of the existing one. This is useful to simplify the processes of
          sending a number of datatypes together as you don't need to keep track
          of their combined size.*/
  MPI_Type_contiguous(3, MPI_BYTE, &RGB_PIXEL);
  MPI_Type_commit(&RGB_PIXEL);
  MPI_Type_contiguous(cols, MPI_BYTE, &rowGrey);
  MPI_Type_commit(&rowGrey);
  MPI_Type_contiguous(3*cols, MPI_BYTE, &rowRGB);
  MPI_Type_commit(&rowRGB);
  /*Columns need a vector type with a stride between pixles
  equal to the width of the array (=cols+2)*/
  MPI_Type_vector(rows, 1, cols+2, MPI_BYTE, &colGrey);
  MPI_Type_commit(&colGrey);
  MPI_Type_vector(rows, 3, 3*(cols+2), MPI_BYTE, &colRGB);
  MPI_Type_commit(&colRGB);

  //calculate the starting row/col
  int start_row = (comm_rank / splitColNum) * rows;
  int start_col = (comm_rank % splitColNum) * cols;

  /*Allocate space for the arrays but with a 1pixel border around them:
    source: the array which we read from
    destination: here we store the convolution results after every loop*/
	unsigned char* source = NULL;
  unsigned char* dest = NULL;
	int filesize = fileMemAllocSize(rows, cols, isRGB);
	source = calloc(filesize, sizeof(unsigned char));
	dest = calloc(filesize, sizeof(unsigned char));
	if (source == NULL || dest == NULL) {
        fprintf(stderr, "%s: Failed to allocate memory\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
	}

	/*Read the process block from file to the source array row by row,
  filling only the inner pixels, not the border ones.*/
  MPI_File mpi_file;
	MPI_File_open(MPI_COMM_WORLD, imageName, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_file);
	if (isRGB) {
    int i;
		for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(mpi_file, ((start_row + i-1) * width + start_col)*3, MPI_SEEK_SET);
			MPI_File_read(mpi_file, &source[(i*(cols+2)+1)*3], cols*3, MPI_BYTE, &status);
		}
	} else {
    int i;
    for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(mpi_file, (start_row + i-1) * width + start_col, MPI_SEEK_SET);
			MPI_File_read(mpi_file, &source[i*(cols+2)+1], cols, MPI_BYTE, &status);
		}
	}
	MPI_File_close(&mpi_file);

  //Find neighbor processes/blocks so we can request border rows & cols
  int top=-1, bottom=-1, left=-1, right=-1,
      top_left=-1,top_right=-1,bot_left=-1, bot_right=-1;
  if (start_row != 0)
      top = comm_rank - splitColNum;
  if (start_row + rows != height)
    bottom = comm_rank + splitColNum;
  if (start_col != 0)
    left = comm_rank - 1;
  if (start_col + cols != width)
    right = comm_rank + 1;
  //Find corner processes/blocks
  if(top != -1 && left != -1)
    top_left = top - 1;
  if(top != -1 && right != -1)
    top_right = top + 1;
  if(bottom != -1 && left !=- 1)
    bot_left = bottom - 1;
  if(bottom != -1 && right != -1)
    bot_right = bottom + 1;

	MPI_Barrier(MPI_COMM_WORLD); //block until all processes know their neighbor

  double startTime = MPI_Wtime();
	// Convolute "loops" times
	for (t = 0 ; t < loops ; t++) {
    /* Send and request borders (non-blocking)*/
		if (isRGB) {
			if (top != -1) { //top border
				MPI_Isend(&source[3*(cols+2 + 1)], 1, rowRGB, top, 0, MPI_COMM_WORLD, &topSend);
				MPI_Irecv(&source[3], 1, rowRGB, top, 0, MPI_COMM_WORLD, &topRcv);
        if(top_left != -1) {
          MPI_Isend(&source[3*(cols+2+1)], 1, RGB_PIXEL, top_left, 0, MPI_COMM_WORLD, &top_leftSend);
          MPI_Irecv(&source[0], 1, RGB_PIXEL, top_left, 0, MPI_COMM_WORLD, &top_leftRcv);
        }
        if(top_right != -1){
          MPI_Isend(&source[3*(cols+2+cols)], 1, RGB_PIXEL, top_right, 0, MPI_COMM_WORLD, &top_rightSend);
          MPI_Irecv(&source[3*(cols+1)], 1, RGB_PIXEL, top_right, 0, MPI_COMM_WORLD, &top_rightRcv);
        }
			}
			if (bottom != -1) {  //bottom border
				MPI_Isend(&source[3*(rows*(cols+2) + 1)], 1, rowRGB, bottom, 0, MPI_COMM_WORLD, &bottomSend);
				MPI_Irecv(&source[(rows+1)*(3*cols+6) + 3], 1, rowRGB, bottom, 0, MPI_COMM_WORLD, &bottomRcv);
        if(bot_left != -1){ //bottom left corner
          MPI_Isend(&source[3*(rows*(cols+2) + 1)], 1, RGB_PIXEL, bot_left, 0, MPI_COMM_WORLD, &bot_leftSend);
          MPI_Irecv(&source[3*((rows+1)*(cols+2))], 1, RGB_PIXEL, bot_left, 0, MPI_COMM_WORLD, &bot_leftRcv);
        }
        if(bot_right != -1){ //bottom right corner
          MPI_Isend(&source[3*(rows*(cols+2)+cols)], 1, RGB_PIXEL, bot_right, 0, MPI_COMM_WORLD, &bot_rightSend);
          MPI_Irecv(&source[3*((rows+1)*(cols+2)+cols+1)], 1, RGB_PIXEL, bot_right, 0, MPI_COMM_WORLD, &bot_rightRcv);
        }

			}
			if (left != -1) { //left border
        MPI_Isend(&source[3*cols+6 + 3], 1, colRGB, left, 0, MPI_COMM_WORLD, &leftSend);
        MPI_Irecv(&source[3*cols+6], 1, colRGB, left, 0, MPI_COMM_WORLD, &leftRcv);
			}
			if (right != -1) { //right border
        MPI_Isend(&source[3*cols+6 + 3*cols], 1, colRGB, right, 0, MPI_COMM_WORLD, &rightSend);
        MPI_Irecv(&source[3*cols+6 + 3*cols+3], 1, colRGB, right, 0, MPI_COMM_WORLD, &rightRcv);
			}
		} else {
      if (top != -1) {  //top border
				MPI_Isend(&source[cols+2 + 1], 1, rowGrey, top, 0, MPI_COMM_WORLD, &topSend);
				MPI_Irecv(&source[1], 1, rowGrey, top, 0, MPI_COMM_WORLD, &topRcv);
        if(top_left != -1) {  //top left corner
          MPI_Isend(&source[cols+2+1], 1, MPI_BYTE, top_left, 0, MPI_COMM_WORLD, &top_leftSend);
          MPI_Irecv(&source[0], 1, MPI_BYTE, top_left, 0, MPI_COMM_WORLD, &top_leftRcv);
        }
        if(top_right != -1){  //top right corner
          MPI_Isend(&source[cols+2+cols], 1, MPI_BYTE, top_right, 0, MPI_COMM_WORLD, &top_rightSend);
          MPI_Irecv(&source[cols+1], 1, MPI_BYTE, top_right, 0, MPI_COMM_WORLD, &top_rightRcv);
        }
			}
			if (bottom != -1) { //bottom border
				MPI_Isend(&source[rows*(cols+2) + 1], 1, rowGrey, bottom, 0, MPI_COMM_WORLD, &bottomSend);
				MPI_Irecv(&source[(rows+1)*(cols+2)+1], 1, rowGrey, bottom, 0, MPI_COMM_WORLD, &bottomRcv);
        if(bot_left != -1){ //bottom left corner
          MPI_Isend(&source[rows*(cols+2) + 1], 1, MPI_BYTE, bot_left, 0, MPI_COMM_WORLD, &bot_leftSend);
          MPI_Irecv(&source[(rows+1)*(cols+2)], 1, MPI_BYTE, bot_left, 0, MPI_COMM_WORLD, &bot_leftRcv);
        }
        if(bot_right != -1){ //bottom right corner
          MPI_Isend(&source[rows*(cols+2)+cols], 1, MPI_BYTE, bot_right, 0, MPI_COMM_WORLD, &bot_rightSend);
          MPI_Irecv(&source[(rows+1)*(cols+2)+cols+1], 1, MPI_BYTE, bot_right, 0, MPI_COMM_WORLD, &bot_rightRcv);
        }
			}
      if (left != -1) { //left border
        MPI_Isend(&source[cols+2 + 1], 1, colGrey, left, 0, MPI_COMM_WORLD, &leftSend);
        MPI_Irecv(&source[cols+2], 1, colGrey, left, 0, MPI_COMM_WORLD, &leftRcv);
      }
      if (right != -1) { //right border
        MPI_Isend(&source[cols+2 + cols], 1, colGrey, right, 0, MPI_COMM_WORLD, &rightSend);
        MPI_Irecv(&source[cols+2 + cols+1], 1, colGrey, right, 0, MPI_COMM_WORLD, &rightRcv);
      }
		}
    /*Covolute internal pixels (the ones that dont depend on the border)*/
		blur(source, dest, 2, rows-1, 2, cols-1, cols, rows, isRGB);
    /*Wait to receive and convolute border-dependent pixels*/
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
    if(top_left != -1) {
      MPI_Wait(&top_leftRcv, &status);
      blur(source, dest, 1, 1, 1, 1, cols, rows, isRGB);
    }
    if(top_right != -1) {
      MPI_Wait(&top_rightRcv, &status);
      blur(source, dest, 1, 1, cols, cols, cols, rows, isRGB);
    }
    if(bot_left != -1) {
      MPI_Wait(&bot_leftRcv, &status);
      blur(source, dest, rows, rows, 1, 1, cols, rows, isRGB);
    }
    if(bot_right != -1) {
      MPI_Wait(&bot_rightRcv, &status);
      blur(source, dest, rows, rows, cols, cols, cols, rows, isRGB);
    }

    unsigned char* temp;
    temp = source;
    source = dest;
    dest = temp;

    //checksum
    if( prev_checksum == new_checksum )
      fprintf(stderr, "%d Filter failed. Image checksum did not change.\n"
              "%Le vs %Le\n", comm_rank, prev_checksum, new_checksum);
    prev_checksum = new_checksum;
    new_checksum = 0;

    //wait until you have sent all borders
		if (top != -1)
      MPI_Wait(&topSend, &status);
		if (bottom != -1)
      MPI_Wait(&bottomSend, &status);
		if (left != -1)
      MPI_Wait(&leftSend, &status);
    if (right != -1)
      MPI_Wait(&rightSend, &status);
    if(top_left != -1)
      MPI_Wait(&top_leftSend, &status);
    if(top_right != -1)
      MPI_Wait(&top_rightSend, &status);
    if(bot_left != -1)
      MPI_Wait(&bot_leftSend, &status);
    if(bot_right != -1)
      MPI_Wait(&bot_rightSend, &status);

	}/*end of loop*/
  double totalTime = MPI_Wtime() - startTime;

  //Write the convoluted block to a new file
	MPI_File outFile;
	MPI_File_open(MPI_COMM_WORLD, "blurred.raw",
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outFile);
	if (isRGB) {
    int i;
		for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(outFile, 3*(start_row + i-1) * width + 3*start_col,
                    MPI_SEEK_SET);
			MPI_File_write( outFile, &source[(cols*3+6)*i + 3], cols*3, MPI_BYTE,
                      MPI_STATUS_IGNORE);
		}
	} else {
    int i;
    for (i = 1 ; i <= rows ; i++) {
			MPI_File_seek(outFile, (start_row + i-1) * width + start_col, MPI_SEEK_SET);
			MPI_File_write(outFile, &source[(cols+2)*i + 1], cols, MPI_BYTE, MPI_STATUS_IGNORE);
		}
	}
	MPI_File_close(&outFile);

  //Processes send their times to the master process and the max time is printed
  if (comm_rank != 0)
      MPI_Send(&totalTime, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  else {
      double* commPassedTimes = malloc(comm_size*sizeof(double));
      commPassedTimes[0] = totalTime;
      int i;
      //find max time
      for (i = 1 ; i != comm_size ; i++) {
        MPI_Recv(&commPassedTimes[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
        if (commPassedTimes[i] > totalTime)
          totalTime = commPassedTimes[i];
      }
      //calculate idle times
      for(i=0; i<comm_size; i++){
        printf("\nproc%d Idle time: %fs\n", i,totalTime-commPassedTimes[i]);
      }
      printf("\nMax process passed time: %f\n", totalTime);
  }

  //Terminate the enviroment and exit
  MPI_Finalize();
  free(source);
  free(dest);
	return EXIT_SUCCESS;
}
