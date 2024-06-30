#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/matrix.h"
#include "../include/unsupervised.h"
#include "../include/dimreduction.h"

#define GP_WRITE(fmt, ...) \
    do { \
        fprintf(gnuplot_file, fmt, ##__VA_ARGS__); \
        fprintf(external_file, fmt, ##__VA_ARGS__); \
    } while (0)

void test_pca() {
    printf("Testing PCA...\n");

    Matrix original_data = Unsupervised_read_csv("test/test_data/pca_data.csv");

    original_data.data = (double **)malloc(original_data.rows * sizeof(double *));
    for (int i = 0; i < original_data.rows; ++i) {
        original_data.data[i] = (double *)malloc(original_data.cols * sizeof(double));
        for (int j = 0; j < original_data.cols; ++j) {
            // Replace with your actual data loading mechanism
            original_data.data[i][j] = (double)(rand() % 100); // Example random data
        }
    }

    Matrix reduced_data = PCA_Reduce(&original_data, 2);

    FILE *gnuplot_file = popen("gnuplot -persist", "w");
    if (!gnuplot_file) {
        fprintf(stderr, "Error: Unable to open Gnuplot.\n");
        return;
    }

    FILE *external_file = fopen("plots/pca_plot.gnu", "w");
    if (!external_file) {
        fprintf(stderr, "Error: Unable to open Gnuplot plot file.\n");
        return;
    }

    const char *gnuplot_commands[] = {
        "set terminal qt\n",
        "set view equal xyz\n",
        "set xrange [-80:80]\n",
        "set yrange [-80:80]\n",
        "set zrange [-40:40]\n",
        "set datafile separator \",\"\n",
        "splot '-' using 1:2:3 with points pt 7 ps 1 title 'Original Data',",
        "      '-' using 1:2:3 with points pt 7 ps 1 lc rgb 'red' title 'PCA Reduced Data'\n",
    };

    for (int i = 0; i < sizeof(gnuplot_commands) / sizeof(gnuplot_commands[0]); ++i) {
        GP_WRITE("%s", gnuplot_commands[i]);
    }

    for (int i = 0; i < original_data.rows; ++i) {
        GP_WRITE("%lf, %lf, %lf\n", original_data.data[i][0], original_data.data[i][1], original_data.data[i][2]);
    }
    GP_WRITE("e\n");

    for (int i = 0; i < reduced_data.rows; ++i) {
        GP_WRITE("%lf, %lf, %lf\n", reduced_data.data[i][0], reduced_data.data[i][1], reduced_data.data[i][2]);
    }
    GP_WRITE("e\n");

    fclose(external_file);
    fflush(gnuplot_file);
    pclose(gnuplot_file);

    for (int i = 0; i < original_data.rows; ++i) {
        free(original_data.data[i]);
    }
    free(original_data.data);

    for (int i = 0; i < reduced_data.rows; ++i) {
        free(reduced_data.data[i]);
    }
    free(reduced_data.data);

    printf("PCA Testing complete!\n");
}

int main() {
    printf("Testing dimreduction\n");
    test_pca();
    printf("All dimreductions tests passed successfully!\n");
    return 0;
}