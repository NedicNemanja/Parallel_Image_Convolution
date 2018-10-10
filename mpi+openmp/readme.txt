compile using mpicc -o main main.c filter.h filter.c -lm

run using mpirun -np ./main [processors_num] [isRGB] [image_relative_path] [width] [height] [iterations]

processors_num, width, height, iterations: int values
isRGB: int value. 1 for RGB, 0 for greyscale
image_relative_path: char*


DESIGN
----------------------------------------------------------------------------------
Why non-blocking communication?
NON-BLOCKING COMMUNICATION : For Non-Blocking Communication, the application creates a request for communication for send and / or receive and gets back a handle and then terminates. That's all that is needed to guarantee that the process is executed. I.e the MPI library is notified that the operation has to be executed.

For the sender side, this allows overlapping computation with communication.

For the receiver side, this allows overlapping a part of the communication overhead , i.e copying the message directly into the address space of the receiving side in the application.
---------------------------------------------------------------------------------
