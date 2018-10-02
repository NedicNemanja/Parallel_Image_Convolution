compile using mpicc -o main main.c filter.h filter.c

<<<<<<< HEAD
run using mpirun -np ./main [processors_num] [isRGB] [image_relative_path] [width] [height] [iterations]
=======
run using mpirun -np [processors_num] ./main [isRGB] [image_relative_path] [width] [height] [iterations]
>>>>>>> 15d271acc0ec103728db717907cfc7403b9d49a6

processors_num, width, height, iterations: int values
isRGB: int value. 1 for RGB, 0 for greyscale
image_relative_path: char*
