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

int main()
{
    //声明+部分初始化
    //定义测试的数字数量
    int count = 9;
    //文件夹路径
    string filePath;
    //每张图片的路径
    string ImagePath;
    //存储每张图片的名字
    vector<string> files;
    //每个文件夹下图片的数量
    int number;
    //记录训练情况
    int good_result[9] = {0};
    int bad_result[9] = {0};
    int bad_num = 0;
    int good_num = 0;
    //加载模型数据
    string modelpath = "/home/peter/handwritting_detect/svm.xml";
    Ptr<SVM> svm = StatModel::load<SVM>(modelpath);
    //循环判断模型各个数字的识别情况
    for (int j = 0; j <= count; j++)
    {
        int result = 0;
        filePath = "/home/peter/handwritting_detect/data/data" + to_string(j) + "/test";
        // files为vector类型，每次循环要将他清空
        files.clear();
        getFiles(filePath, files);

        number = files.size();
        // cout << number << endl;
        for (int i = 0; i < number; i++)
        {
            ImagePath = filePath + "/" + files[i].c_str();
            // out << "testpath" << i << ImagePath << endl;
            Mat inMat = imread(ImagePath, 0);
            // cout << files[i].c_str()<<endl;
            resize(inMat, inMat, Size(8, 16), (0, 0), (0, 0), INTER_AREA);
            Mat p = inMat.reshape(1, 1);
            p.convertTo(p, CV_32FC1);
            int response = (int)svm->predict(p);
            // cout << i << ":" << response << endl;
            if (response == j)
            {
                result++;
            }
        }
        double right = 100 * ((double)result) / number;
        printf("数字%d正确率为%.2f%\n", j, right);
        if (right >= 90)
            good_result[good_num++] = j;
        if (right < 90)
            bad_result[bad_num++] = j;
    }
    cout << "模型对数字 ";
    for (int i = 0; i < good_num; i++)
    {
        cout << to_string(good_result[i]) << " ";
    }
    cout << "效果很好" << endl;

    if (bad_num > 0)
    {
        cout << "模型对数字 ";
        for (int i = 0; i < bad_num; i++)
            cout << to_string(bad_result[i]) << " ";
        cout << "效果不行，建议更改参数或加入更多高质量训练集" << endl;
    }
    else
    {
        cout << "所有数字识别准确率均在百分之九十以上，好耶！" << endl;
    }
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
    sort(file.begin(), file.end());
    _files = file;
}