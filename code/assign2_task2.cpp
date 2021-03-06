#include <iostream>
#include <opencv2/calib3d.hpp>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <iterator>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
const char* keys =
    "{ help h |                  | Print help message. }"
    "{ input1 | /home/michael/RV/robot_vision/IMG_CAL_DATA/left08.png | Path to input image 1. }"
    "{ input2 | /home/michael/RV/robot_vision/IMG_CAL_DATA/left10.png | Path to input image 2. }"
    "{ input3 | /home/michael/RV/robot_vision/IMG_CAL_DATA/right08.png | Path to input image 3. }";

template <typename T>
static float distancePointLine(const cv::Point_<T> point, const cv::Vec<T,3>& line)
{
  //Line is given as a*x + b*y + c = 0
  return std::fabs(line(0)*point.x + line(1)*point.y + line(2))
      / std::sqrt(line(0)*line(0)+line(1)*line(1));
}

template <typename T1, typename T2>
static void drawEpipolarLines(const std::string title, const cv::Matx<T1,3,3> F,
                const cv::Mat& img1, const cv::Mat& img2,
                const std::vector<cv::Point_<T2>> points1,
                const std::vector<cv::Point_<T2>> points2,
                const float inlierDistance = -1)
{
  CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
  cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
  cv::Rect rect1(0,0, img1.cols, img1.rows);
  cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
  /*
   * Allow color drawing
   */
  if (img1.type() == CV_8U)
  {
    cv::cvtColor(img1, outImg(rect1), COLOR_GRAY2BGR);
    cv::cvtColor(img2, outImg(rect2), COLOR_GRAY2BGR);
  }
  else
  {
    img1.copyTo(outImg(rect1));
    img2.copyTo(outImg(rect2));
  }
  std::vector<cv::Vec<T2,3>> epilines1, epilines2;
  cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
  cv::computeCorrespondEpilines(points2, 2, F, epilines2);
 
  CV_Assert(points1.size() == points2.size() &&
        points2.size() == epilines1.size() &&
        epilines1.size() == epilines2.size());
 
  cv::RNG rng(0);
  for(size_t i=0; i<points1.size(); i++)
  {
    if(inlierDistance > 0)
    {
      if(distancePointLine(points1[i], epilines2[i]) > inlierDistance ||
        distancePointLine(points2[i], epilines1[i]) > inlierDistance)
      {
        //The point match is no inlier
        continue;
      }
    }
    /*
     * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
     */
    cv::Scalar color(rng(256),rng(256),rng(256));
 
    cv::line(outImg(rect2),
      cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
      cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
      color);
    cv::circle(outImg(rect1), points1[i], 3, color, -1, LINE_AA);
 
    cv::line(outImg(rect1),
      cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
      cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
      color);
    cv::circle(outImg(rect2), points2[i], 3, color, -1, LINE_AA);
  }
  cv::imshow(title, outImg);
  cv::waitKey();
  cv::imwrite(title,outImg);
}

void compute_detection(Mat img1, Mat img2, Ptr<Feature2D> detector,string name, string is_type,bool isrightleft){

    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    if(is_type == "fnb"){
        detector->detect(img1, keypoints1);
        detector->detect(img2, keypoints2);

        Ptr<DescriptorExtractor> featureExtractor = BriefDescriptorExtractor::create();
        featureExtractor->compute(img1, keypoints1, descriptors1);
        featureExtractor->compute(img2, keypoints2, descriptors2);
    } else {
        detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
        detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );
    }
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    if(is_type == "orb" || "fnb") {
        matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    }
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //imwrite(name,img_matches);

    Mat img_keypoints1;
    drawKeypoints(img1,keypoints1,img_keypoints1);


    Mat img_keypoints2;
    drawKeypoints(img2,keypoints2,img_keypoints2);

    std::shuffle(good_matches.begin(), good_matches.end(), std::mt19937(std::random_device()()));

    int i = 0;
    std::vector<cv::Point2f> points1, points2;
    for (std::vector<cv::DMatch>::const_iterator it= good_matches.begin();
       it!= good_matches.end(); ++it)
    {
        i++;
        //left keypoints
        float x= keypoints1[it->queryIdx].pt.x;
        float y= keypoints1[it->queryIdx].pt.y;
        points1.push_back(cv::Point2f(x,y));
        //right keypoints
        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        points2.push_back(cv::Point2f(x,y));
        if (i == 20)
            break;
    }


    float data[] = {9.842439e+02, 0.000000e+00, 6.900000e+02, 0.000000e+00, 9.808141e+02, 2.331966e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00};
    Mat K_left = Mat(3,3,CV_32F,data);

    float data2[] = {9.895267e+02, 0.000000e+00, 7.020000e+02, 0.000000e+00, 9.878386e+02, 2.455590e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00};
    Mat K_right = Mat(3,3,CV_32F,data2);

    vector<Point2f> points1_norm, points2_norm;
    undistortPoints(points1, points1_norm, K_left,noArray());
    if(isrightleft){
        undistortPoints(points2, points2_norm, K_right,noArray());
    }else{
        undistortPoints(points2, points2_norm, K_left,noArray());
    }

    Matx<float, 3, 3> E = findEssentialMat(points1_norm,points2_norm,Matx33f::eye());
    Matx<float, 3,3> K_inv_t_second,K_inv_left;

    Matx<float, 3,3> f_fromExentialMat;
    if(isrightleft){
        invert(K_right, K_inv_t_second);
        transpose(K_inv_t_second, K_inv_t_second);
        invert(K_left,K_inv_left);
        f_fromExentialMat = K_inv_t_second * E * K_inv_left;
    }else{
        invert(K_left, K_inv_t_second);
        transpose(K_inv_t_second, K_inv_t_second);
        invert(K_left,K_inv_left);
        f_fromExentialMat = K_inv_t_second * E * K_inv_left;
    }
    Matx<float, 3, 3> f_8point = findFundamentalMat(points1, points2, FM_8POINT);
    Matx<float, 3, 3> f_8pointRansac = findFundamentalMat(points1, points2, FM_RANSAC);


    Mat lineimage;
    cvtColor(img1,lineimage,COLOR_GRAY2BGR);
    drawEpipolarLines(name + "fm-8point.png", f_8point, img1, img2, points1, points2);
    drawEpipolarLines(name + "fm-ransac.png", f_8pointRansac, img1, img2, points1, points2);
    drawEpipolarLines(name + "em-ransac.png", f_fromExentialMat, img1, img2,  points1, points2);


}

int main( int argc, char* argv[] )
{
    CommandLineParser parser( argc, argv, keys );
    Mat img1 = imread( samples::findFile( parser.get<String>("input1") ), IMREAD_GRAYSCALE );
    Mat img2 = imread( samples::findFile( parser.get<String>("input2") ), IMREAD_GRAYSCALE );
    Mat img3 = imread( samples::findFile( parser.get<String>("input3") ), IMREAD_GRAYSCALE );
    
    if ( img1.empty() || img2.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }

    //--------------------------------------------------------------------------------------------
    // SURF
    //--------------------------------------------------------------------------------------------
    int minHessian = 10000;

    Ptr<SURF> detector = SURF::create( minHessian );
    compute_detection(img1, img2,detector,"SURF-left-left-", "", false);
    compute_detection(img1,img3,detector,"SURF-left-right-", "", true);

    //--------------------------------------------------------------------------------------------
    // SIFT
    //--------------------------------------------------------------------------------------------

    Ptr<SIFT> detector_sift = SIFT::create(200);
    compute_detection(img1, img2,detector_sift,"SIFT-left-left-", "", false);
    compute_detection(img1,img3,detector_sift,"SIFT-left-right-", "", true);

    //--------------------------------------------------------------------------------------------
    // ORB
    //--------------------------------------------------------------------------------------------

    Ptr<ORB> detector_orb = ORB::create();
    compute_detection(img1, img2,detector_orb,"orb-left-left-", "orb", false);
    compute_detection(img1,img3,detector_orb,"orb-left-right-","orb", true);

    //--------------------------------------------------------------------------------------------
    // Fast & Brief
    //--------------------------------------------------------------------------------------------

    Ptr<FastFeatureDetector> detector_fast = FastFeatureDetector::create(100);
    compute_detection(img1, img2,detector_fast,"f&b-left-left-", "fnb", false);
    compute_detection(img1,img3,detector_fast,"f&b-left-right-", "fnb", true);

    return 0;
}