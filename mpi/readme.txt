compile using mpicc -o main main.c filter.h filter.c

run using mpirun -np [processors_num] [isRGB] [image_relative_path] [width] [height] [iterations]

processors_num, width, height, iterations: int values
isRGB: int value. 1 for RGB, 0 for greyscale
image_relative_path: char*