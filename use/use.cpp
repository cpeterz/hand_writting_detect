#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace ml;

int main()
{
    string Imgpath = "/home/peter/handwritting_detect/my_img";
    Mat m_img;
    m_img = imread("/home/peter/handwritting_detect/8.png");
    if (m_img.empty())
    {
        cout << "no image!" << endl;
        return 0;
    }
    //处理得到二值图
    cvtColor(m_img, m_img, CV_RGB2BGR);
    Mat gray_img;
    cvtColor(m_img, gray_img, CV_BGR2GRAY);
    threshold(gray_img, m_img, 100, 255, THRESH_BINARY_INV);
    imshow("m_img", m_img);
    int key = waitKey(0);
    //导入svm模型
    string modelpath = "/home/peter/handwritting_detect/svm.xml";
    Ptr<SVM> svm = StatModel::load<SVM>(modelpath);
    //处理数据格式
    resize(m_img, m_img, Size(8, 16), (0, 0), (0, 0), INTER_AREA);
    Mat p = m_img.reshape(1, 1);
    p.convertTo(p, CV_32FC1);
    int response = (int)svm->predict(p);

    cout<<"这个数字是:"<< response <<endl;

    return 0;
}