#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <dirent.h>

using namespace std;
using namespace cv;
using namespace ml;

void getFiles(string path, vector<string> &files);
int get_number(Mat &trainingImages, vector<int> &trainingLabels, int num);

int main()
{
    //获取训练数据
    Mat classes;
    Mat trainData;
    Mat trainImages;
    vector<int> trainLabels;
    //从data文件夹获取数据
    for (int i = 0; i <= 9; i++)
    {
        if (get_number(trainImages, trainLabels, i) != 1)
        {
            cout << "Path" << i << "have some problem !" << endl;
            return 0;
        }
    }
    //用本地变量保存数据，方便训练
    Mat(trainImages).copyTo(trainData);
    trainData.convertTo(trainData, CV_32FC1);
    Mat(trainLabels).copyTo(classes);

    //配置SVM训练器参数
    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);    //设置模型类型
    svm->setKernel(SVM::LINEAR); //设置内核类型
    svm->setDegree(0);    // 针对多项式核函数degree的设置
    svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 10000, 0.001)); //设置终止条件
    // svm->setGamma(1);
    // svm->setCoef0(0);   //针对多项式/sigmoid核函数的设置
    svm->setC(1); // C越大，误分错误减少，但是余量较小
    // svm->setNu(1);//设置v-SVC、一类SVM和v-SVR参数
    // svm->setP(0);  //  为设置e-SVR中损失函数的值
    cout << "开始训练！" << endl;
    //训练
    svm->train(trainData, cv::ml::ROW_SAMPLE, classes);
    cout << "终于训练好了，累得够呛..." << endl;
    //保存模型
    // cout << "test" << endl;
    svm->save("../../svm.xml");
    cout << "已将训练结果放到svm.xml文件中" << endl;
    return 0;
}
void getFiles(string _path, std::vector<std::string> &_files)
{
    DIR *dir;
    dir = opendir(_path.c_str());
    struct dirent *ptr;
    std::vector<std::string> file;
    while ((ptr = readdir(dir)) != NULL)
    {
        if (ptr->d_name[0] == '.')
        {
            continue;
        }
        file.push_back(ptr->d_name);
    }
    closedir(dir);
    // struct dirent 结构体对象读出来的文件名是按照dir->d_off来排序的，
    //所以直接得到的文件名称是乱序的，如果我们希望按照文件名称排序，则必须要在文件名称读出来之后自己进行一次排序。
    sort(file.begin(), file.end());
    _files = file;
}

int get_number(Mat &trainingImages, vector<int> &trainingLabels, int num)
{
    string filePath = "/home/peter/handwritting_detect/data/data" + to_string(num) + "/train";
    vector<string> files;
    getFiles(filePath, files);
    // cout << "filename" << files[1].c_str() << endl;
    int count = files.size();
    // makecout << count << endl;
    for (int i = 0; i < count; i++)
    {
        string Imagepath = filePath + "/" + files[i].c_str();
        Mat SrcImage = imread(Imagepath, 0);
        if (SrcImage.empty())
        {
            cout << i << "filename_error:" << Imagepath << endl;
            cout << "no image" << endl;
            return -1;
        }
        resize(SrcImage, SrcImage, Size(8, 16), (0, 0), (0, 0), INTER_AREA);
        SrcImage = SrcImage.reshape(1, 1);
        trainingImages.push_back(SrcImage);
        trainingLabels.push_back(num);
    }
    cout << "数字" << num << "数据获取成功" << endl;
    return 1;
}
