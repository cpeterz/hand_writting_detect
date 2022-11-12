//目的是把opencv自带的图像 digits.png 中的每个手写数字拆开并放到对应的文件夹下方便训练

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    char ad[128] = {0};
    int filename = 0, filenum = 0;
    Mat img = imread("/home/peter/handwritting_detect/cutting/digits.png");
    if (img.empty())
    {
        cout << " no img" << endl;
    }
    imshow("img", img);
    waitKey(1000);
    Mat gray;

    cvtColor(img, gray, COLOR_BGR2GRAY);
    int b = 20;
    int m = gray.rows / b; //原图为1000*2000
    int n = gray.cols / b; //裁剪为5000个20*20的小图块，m行n列

    for (int i = 0; i < m; i++)
    {
        int offsetRow = i * b; //行上的偏移量
        if (i % 5 == 0 && i != 0)
        {
            filename++;
            filenum = 0;
        }
        for (int j = 0; j < n; j++)
        {
            int offsetCol = j * b; //列上的偏移量
            // sprintf(ad, "/home/peter/handwritting_detect/data/data%d/%d.jpg", filename, filenum++);
            if ( i - filename*5 != 4)
                sprintf(ad, "/home/peter/handwritting_detect/data/data%d/train/%d.jpg", filename, filenum++);
            else
                sprintf(ad, "/home/peter/handwritting_detect/data/data%d/test/%d.jpg", filename, filenum++ - 400);

            //截取20*20的小块
            Mat tmp;
            gray(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
            imwrite(ad, tmp);
        }
    }
    return 0;
}