#include <iostream>
#include <opencv2/calib3d.hpp>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>
#include <fstream>
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




void compute_detection(Mat img1, Mat img2, Ptr<Feature2D> detector,string name,string name1,string name2,string is_type){

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
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    if(is_type == "orb" || "fnb"){
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
    imwrite(name,img_matches);
    
    //-- Show detected matches
    imshow(name, img_matches );
    waitKey();

    Mat img_keypoints1;
    drawKeypoints(img1,keypoints1,img_keypoints1);
    imshow(name1,img_keypoints1);
    waitKey();
    imwrite(name1,img_keypoints1);

    Mat img_keypoints2;
    drawKeypoints(img2,keypoints2,img_keypoints2);
    imshow(name2,img_keypoints2);
    waitKey();
    imwrite(name2,img_keypoints2);


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
    compute_detection(img1, img2,detector,"SURF-left-left.png","SURF-left08.png","SURF-left10.png", "");
    compute_detection(img1,img3,detector,"SURF-left-right.png","SURF-left08.png","SURF-right08.png", "");


    //--------------------------------------------------------------------------------------------
    // SIFT
    //--------------------------------------------------------------------------------------------

    Ptr<SIFT> detector_sift = SIFT::create(200);
    compute_detection(img1, img2,detector_sift,"SIFT-left-left.png","SIFT-left08.png","SIFT-left10.png", "");
    compute_detection(img1,img3,detector_sift,"SIFT-left-right.png","SIFT-left08.png","SIFT-right08.png", "");

    //--------------------------------------------------------------------------------------------
    // ORB
    //--------------------------------------------------------------------------------------------

    Ptr<ORB> detector_orb = ORB::create();
    compute_detection(img1, img2,detector_orb,"orb-left-left.png","orb-left08.png","orb-left10.png", "orb");
    compute_detection(img1,img3,detector_orb,"orb-left-right.png","orb-left08.png","orb-right08.png", "orb");

    //--------------------------------------------------------------------------------------------
    // Fast & Brief
    //--------------------------------------------------------------------------------------------

    Ptr<FastFeatureDetector> detector_fast = FastFeatureDetector::create(100);
    compute_detection(img1, img2,detector_fast,"f&b-left-left.png","f&b-left08.png","f&b-left10.png", "fnb");
    compute_detection(img1,img3,detector_fast,"f&b-left-right.png","f&b-left08.png","f&b-right08.png", "fnb");

    return 0;
}