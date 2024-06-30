#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/matrix.h"
#include "../include/unsupervised.h"

#define GP_WRITE(fmt, ...) \
        do { \
            fprintf(gnuplot_file, fmt, ##__VA_ARGS__); \
        } while (0)

void plot_clusters(const Matrix *X, const Matrix *labels) {
    FILE *gnuplot_file = popen("gnuplot -persist", "w");
    if (!gnuplot_file) {
        fprintf(stderr, "Error opening gnuplot\n");
        return;
    }

    GP_WRITE("set title 'K-Means Clustering'\n");
    GP_WRITE("set xlabel 'Feature 1'\n");
    GP_WRITE("set ylabel 'Feature 2'\n");
    GP_WRITE("set style data points\n");
    GP_WRITE("set pointsize 1.5\n");
    
    GP_WRITE("set palette maxcolors 10\n");
    GP_WRITE("set palette defined ( 0 'red', 1 'green', 2 'blue', 3 'yellow', 4 'cyan', 5 'magenta', 6 'orange', 7 'brown', 8 'violet', 9 'gray' )\n");

    int max_cluster = (int)Vector_max(labels);

    GP_WRITE("plot ");
    for (int c = 0; c <= max_cluster; ++c) {
        if (c > 0) GP_WRITE(", ");
        GP_WRITE("'-' using 1:2:3 with points palette title 'Cluster %d'", c + 1);
    }
    GP_WRITE("\n");

    for (int c = 0; c <= max_cluster; ++c) {
        for (int i = 0; i < X->rows; ++i) {
            if ((int)labels->data[i][0] == c) {
                GP_WRITE("%f %f %d\n", X->data[i][0], X->data[i][1], c);
            }
        }
        GP_WRITE("e\n");
    }

    fflush(gnuplot_file);
    pclose(gnuplot_file);
}


void test_kmeans_clustering() {
    printf("Testing K-Means Clustering\n");

    srand(time(NULL));

    Matrix X = Unsupervised_read_csv("test/test_data/3_means_clusters.csv");
    int k = 3;

    KMeansModel model = KMeans(k, &X);
    KMeans_train(&model);

    plot_clusters(&X, &model.labels);

    Matrix_free(X);
    KMeans_free(&model);
    printf("K-Means test passed successfuly\n");
}

int main() {
    test_kmeans_clustering();

    return 0;
}