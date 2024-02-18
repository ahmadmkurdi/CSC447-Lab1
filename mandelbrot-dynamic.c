#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "mpi.h"

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

#define DATA_TAG 66
#define RESULT_TAG 77
#define TERMINATE_TAG 88

/* complex number plane */
#define REAL_MIN -2.0
#define REAL_MAX 2.0
#define IMAG_MIN -2.0
#define IMAG_MAX 2.0

struct Complex {
    double real, imag;
};

int cal_pixel(struct Complex c)
{
    double z_real, z_imag;
    double z_real_2, z_imag_2;

    int count;
    
    z_real = 0.0;
    z_imag = 0.0;
    z_real_2 = z_real * z_real;
    z_imag_2 = z_imag * z_imag;

    for (count = 0; count < MAX_ITER && ((z_real_2 + z_imag_2) < 4.0); count++)
    {
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real_2 - z_imag_2 + c.real;

        z_real_2 = z_real * z_real;
        z_imag_2 = z_imag * z_imag;
    };
    return count;
}

void master(int size)
{
    int i, k, count, row;

    int *colors = (int *)malloc(sizeof(int) * (WIDTH + 1));

    static unsigned char image_data[HEIGHT][WIDTH];

    count = 0;
    row = 0;
    for (k = 1; k < size; k++)
    {
        MPI_Send(&row, 1, MPI_INT, k, DATA_TAG, MPI_COMM_WORLD);
        count++;
        row++;
    }
    MPI_Status status;

    do
    {
        MPI_Recv(colors, WIDTH, MPI_INT, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &status);
        count--;

        if (row < HEIGHT)
        {
            MPI_Send(&row, 1, MPI_INT, status.MPI_SOURCE, DATA_TAG, MPI_COMM_WORLD);
            row++;
            count++;
        }
        else
        {
            MPI_Send(&row, 1, MPI_INT, status.MPI_SOURCE, TERMINATE_TAG, MPI_COMM_WORLD);
        }
        for (i = 0; i < WIDTH; i++)
        {
            image_data[colors[0]][i] = colors[i + 1];
        }

    } while (count > 0);

    FILE *fp;
    char *filename = "dynamic.pgm";

    fp = fopen(filename, "w"); /* open file for writing */
    fprintf(fp, "P2\n%d %d\n255\n", WIDTH, HEIGHT); /* write ASCII header to the file */

    /* write image data bytes to the file */
    for(int i = 0; i < HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
            fprintf(fp, "%d ", image_data[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp); /* close file */
}

void worker()
{
    int *colors = malloc(sizeof(int) * (WIDTH + 1));
    MPI_Status status;
    int row, x, rank;

    const double SCALE_REAL = (REAL_MAX - REAL_MIN) / WIDTH;
    const double SCALE_IMAG = (IMAG_MAX - IMAG_MIN) / HEIGHT;

    MPI_Recv(&row, 1, MPI_INT, 0, DATA_TAG, MPI_COMM_WORLD, &status);
    struct Complex c;
    while (status.MPI_TAG == DATA_TAG)
    {
        colors[0] = row;
        c.imag = IMAG_MIN + row * SCALE_IMAG;
        for (x = 0; x < WIDTH; x++)
        {
            c.real = REAL_MIN + x * SCALE_REAL;
            colors[x + 1] = cal_pixel(c);
        }
        MPI_Send(colors, WIDTH, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);
        MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
}

int main(int argc, char *argv[])
{
    MPI_Status status;
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (size < 2)
    {
        printf("Need at least 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Time benchmarking setup
    double AVG = 0.0;
    const int N = 10; // Number of trials
    double total_time[N];

    for (int k = 0; k < N; k++){
        clock_t start_time = clock();  // Start measuring time

        if (rank == 0) {
            master(size);
        }
        else {
            worker();
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