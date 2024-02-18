#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex{
  double real;
  double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;

    double z_real2, z_imag2, lengthsq;

    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;

        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq =  z_real2 + z_imag2;
        iter++;
    }
    while ((iter < MAX_ITER) && (lengthsq < 4.0));

    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg; 
    int temp;
    pgmimg = fopen(filename, "wb"); 
    fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File   
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);  // Writing Width and Height
    fprintf(pgmimg, "255\n");  // Writing the maximum gray value 
    int count = 0; 
    
    for (int i = 0; i < HEIGHT; i++) { 
        for (int j = 0; j < WIDTH; j++) { 
            temp = image[i][j]; 
            fprintf(pgmimg, "%d ", temp); // Writing the gray values in the 2D array to the file 
        } 
        fprintf(pgmimg, "\n"); 
    } 
    fclose(pgmimg); 
} 

void worker(int rank, int size, int row_size) {
    int image[HEIGHT][WIDTH] = {0};
    struct complex c;
    int start_row = (rank - 1) * row_size;
    int end_row = start_row + row_size < HEIGHT ? start_row + row_size : HEIGHT;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < WIDTH; j++) {
            c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
            image[i][j] = cal_pixel(c);
        }
    }

    // Send calculated segment back to master
    for (int i = start_row; i < end_row; i++) {
        MPI_Send(image[i], WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

void master(int size, int row_size) {
    int image[HEIGHT][WIDTH] = {0};
    MPI_Status status;

    for (int rank = 1; rank < size; rank++) {
        int start_row = (rank - 1) * row_size;
        int end_row = start_row + row_size < HEIGHT ? start_row + row_size : HEIGHT;

        for (int i = start_row; i < end_row; i++) {
            MPI_Recv(image[i], WIDTH, MPI_INT, rank, 0, MPI_COMM_WORLD, &status);
        }
    }

    save_pgm("mandelbrot-static.pgm", image);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int row_size = HEIGHT / (size - 1);

    if (rank == 0 && size < 2) {
        fprintf(stderr, "Usage: mpirun -np <n> %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double AVG = 0;
    int N = 10; // number of trials
    double total_time[N];

    for (int k=0; k<N; k++){
        clock_t start_time = clock();  // Start measuring time

        if (rank == 0) {
            master(size, row_size);
        } else {
            worker(rank, size, row_size);
        }

        clock_t end_time = clock(); // End measuring time

        total_time[k] = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        if (rank == 0) {
            printf("Execution time of trial [%d]: %f seconds\n", k , total_time[k]);
            AVG += total_time[k];
        }
    }

    if (rank == 0) {
        printf("The average execution time of 10 trials is: %f ms\n", AVG/N*1000);
    }

    MPI_Finalize();
    return 0;
}
