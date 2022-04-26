#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <regex>
#include <string>

using namespace cv;
using namespace std;

string preprocess_string(string line){
    line = std::regex_replace(line, std::regex("\n"), "");
    line = line.erase(0,1);
    line = std::regex_replace(line, std::regex(","), "");
    line = std::regex_replace(line, std::regex(";"), "");
    
    return line;
}

vector<float> parseStringToFloat(string toparse){
    stringstream input_stringstream(toparse);
    string parsed;
    vector<float> values;
    while (getline(input_stringstream,parsed,' '))
    {
        cout << parsed << "\n";
        values.push_back((_Float64)std::stof(parsed));
    }
    return values;
}


int main(int argc, char **argv) {

    std::string const HOME = std::getenv("HOME") ? std::getenv("HOME") : ".";
    ifstream myfile("res_task2.txt");
    string line;
    int i = 0;
    vector<float> R_elem;
    vector<float> T_elem;
    string parsed;
    while(getline(myfile,line,']')){
        cout<<line<<"\n";
        line = preprocess_string(line);
        if(i==0){
            R_elem = parseStringToFloat(line);
            i++;
        }else if (i==1){
            T_elem = parseStringToFloat(line);
        }
    }
    myfile.close();

    //float data[] = {0.9998002499593387, -0.001816728490463582, 0.01990376041947669, 0.001657230921219589, 0.9999664098270534, 0.008027004626090049, -0.01991767473671921, -0.007992416104373901, 0.9997696772346604};
    Mat R = Mat(3,3,CV_32F,R_elem.data());
    cout << R;
    Mat T = Mat(3,1,CV_32F,T_elem.data());


    ifstream myfile_left("res_task1_left.txt");
    i = 0;
    vector<float> K_left_elem;
    vector<float> d_left_elem;
    while(getline(myfile_left,line,']')){
        cout<<line<<"\n";
        line = preprocess_string(line);
        if(i==0){
            K_left_elem = parseStringToFloat(line);
            i++;
        }else if (i==1){
            d_left_elem = parseStringToFloat(line);
        }
    }
    myfile_left.close();
    Mat K_left = Mat(3,3,CV_32F,K_left_elem.data());
    Mat d_left = Mat(1,5,CV_32F,d_left_elem.data());

    ifstream myfile_right("res_task1_right.txt");
    i = 0;
    vector<float> K_right_elem;
    vector<float> d_right_elem;
    while(getline(myfile_right,line,']')){
        cout<<line<<"\n";
        line = preprocess_string(line);
        if(i==0){
            K_right_elem = parseStringToFloat(line);
            i++;
        }else if (i==1){
            d_right_elem = parseStringToFloat(line);
        }
    }
    myfile_right.close();
    Mat K_right = Mat(3,3,CV_32F,K_right_elem.data());
    Mat d_right = Mat(5,1,CV_32F,d_right_elem.data());

    string image_left =  HOME + "/Robot_Vision/Robot_Vision/stereo_data/left/picture_000001.png";
    string image_right = HOME + "/Robot_Vision/Robot_Vision/stereo_data/right/picture_000001.png";


    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);

    int color_mode = -1;
    Mat img1 = imread(image_left, color_mode);
    Mat img2 = imread(image_right, color_mode);

    Size img_size = img1.size();

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];
    cv::Size frameSize(2208, 1242);
    Mat conv;
    R.convertTo(conv, CV_64F);
    R = conv;
    T.convertTo(conv, CV_64F);
    T = conv;
    K_left.convertTo(conv, CV_64F);
    K_left = conv;
    d_left.convertTo(conv, CV_64F);
    d_left = conv;
    K_right.convertTo(conv, CV_64F);
    K_right = conv;
    d_right.convertTo(conv, CV_64F);
    d_right = conv;

    stereoRectify(K_left,  d_left,
                   K_right,  d_right,
                  frameSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, frameSize, &validRoi[0], &validRoi[1]);

    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(K_left, d_left, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(K_right, d_right, R2, P2, img_size, CV_16SC2, map21, map22);

    cout << K_left << "\n" << d_left;
    Mat img1r, img2r;
    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);
    img1 = img1r;
    img2 = img2r;

    sgbm->setPreFilterCap(63);
    int sgbmWinSize = 0;
    int numberOfDisparities = 16;
    cout << numberOfDisparities;
    sgbm->setBlockSize(sgbmWinSize);

    int cn = img1.channels();

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(15);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    //MODE_HH MODE_HH4 MODE_SGBM_3WAY MODE_SGBM
    sgbm->setMode(StereoSGBM::MODE_SGBM);

    Mat disp, disp8;
    sgbm->compute(img1, img2, disp);
    disp.convertTo(disp8, CV_8U);
    imshow("disp_name", disp8);
    waitKey(0);
    return 0;
}