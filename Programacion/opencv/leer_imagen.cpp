#include <opencv2/opencv.hpp>
#include <cstdio>

int main(void)
{
    cv::Mat Image;

    Image = cv::imread("./pinzas.png");

    int ncols = Image.cols;
    int nrows = Image.rows;

    printf("\nLeyendo imagen de %d x %d pixeles...\n", nrows, ncols);

    return 0;
}
