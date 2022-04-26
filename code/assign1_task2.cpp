#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <regex>
#include <fstream>

using namespace cv;
using namespace std;

#define HE 6
#define WI 9

vector<vector<Point3f>> detectFeaturePoints(vector<String> &fileNames,vector<vector<Point2f>> &imagePoints,vector<Point3f> &objp){
    vector<vector<Point3f>> objectPoints;
    std::size_t i = 0;
    for (auto const &f : fileNames) {
        std::cout << std::string(f) << std::endl;

        // 2. Read in the image an call cv::findChessboardCorners()
        cv::Mat img = cv::imread(fileNames[i]);
        cv::Mat gray;
        cv::Size patternSize(WI - 1, HE - 1);
        cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

        bool patternFound = cv::findChessboardCorners(gray, patternSize, imagePoints[i], cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

        // 2. Use cv::cornerSubPix() to refine the found corner detections
        if(patternFound){
            cv::cornerSubPix(gray, imagePoints[i],cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            objectPoints.push_back(objp);
        }

        i++;
    }
    return objectPoints;
}

int main(int argc, char** argv)
{
    //height and width of the chessboard

    std::vector<cv::String> fileNamesRight;
    std::vector<cv::String> fileNamesLeft;
    
    //define home variable 
    std::string const HOME = std::getenv("HOME") ? std::getenv("HOME") : ".";
    cv::glob(HOME + "/Robot_Vision/Robot_Vision/CALIB_DATA/right/picture*.png", fileNamesRight, false);
    cv::glob(HOME + "/Robot_Vision/Robot_Vision/CALIB_DATA/left/picture*.png", fileNamesLeft, false);

    string file;
    auto it = fileNamesLeft.begin();
    while(it!=fileNamesLeft.end()){
        string element = *it;
        file = element.substr(element.find_last_of("/")+1,element.back());
        bool found = false;
        for(const auto &rvalue: fileNamesRight){
            if (rvalue.find(file) != std::string::npos) {
                std::cout << "found: " << file << '\n';
                found = true;
                break;
            }
        }
        if(!found){
            std::cout << "not found deleting: " << file << "\n";
            it = fileNamesLeft.erase(it);
        }else{
            it++;
        }
    }

    fileNamesRight.clear();
    for(auto &file : fileNamesLeft){
        fileNamesRight.push_back(std::regex_replace(file,std::regex("left"),"right"));
    }

    
    cv::Size patternSize(WI - 1, HE - 1);
    vector<vector<Point2f> > imagePointsLeft(fileNamesLeft.size());
    vector<vector<Point2f> > imagePointsRight(fileNamesLeft.size());
    vector<vector<Point3f> > objectPoints;

    int checkerBoard[2] = {WI,HE};
    // Defining the world coordinates for 3D points
    std::vector<cv::Point3f> objp;
    for(int i = 1; i<checkerBoard[1]; i++){
      for(int j = 1; j<checkerBoard[0]; j++){
        objp.push_back(cv::Point3f(j,i,0));
      }
    }

      // Detect feature points
    objectPoints = detectFeaturePoints(fileNamesLeft,imagePointsLeft,objp);
    detectFeaturePoints(fileNamesRight,imagePointsRight,objp);
    
    // Stereo Calibration
    cv::Size frameSize(2208, 1242);

    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = initCameraMatrix2D(objectPoints,imagePointsLeft,frameSize,0);
    cameraMatrix[1] = initCameraMatrix2D(objectPoints,imagePointsRight,frameSize,0);

    cout << cameraMatrix[0] << "\n" << cameraMatrix[1] << "\n" << distCoeffs[0];
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePointsLeft, imagePointsRight,
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    frameSize, R, T, E, F,
                    CALIB_FIX_ASPECT_RATIO +
                    CALIB_ZERO_TANGENT_DIST +
                    CALIB_USE_INTRINSIC_GUESS +
                    CALIB_SAME_FOCAL_LENGTH +
                    CALIB_RATIONAL_MODEL +
                    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    
    cout << "done with RMS error=" << rms << endl;
    cout << "\nR = " << R <<"\nT= " << T << "\n";

    ofstream myfile;
    myfile.open("res_task2.txt");
    if(myfile.is_open()){
        myfile << R << "\n" << T;
    }
    myfile.close();
    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];


    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  frameSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, frameSize, &validRoi[0], &validRoi[1]);

    Mat map11, map12, map21, map22;
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, frameSize, CV_16SC2, map11, map12);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, frameSize, CV_16SC2, map21, map22);

    Mat img1r, img2r;

    cv::Mat img1 = cv::imread(fileNamesLeft[0]);
    cv::Mat img2 = cv::imread(fileNamesLeft[0]);
    cout << cameraMatrix[0] << "\n" << distCoeffs[0];
    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);

    imshow("rectified",img1r);
    cv::waitKey(0);

    imshow("rectified",img2r);
    cv::waitKey(0);

    return 0;
}