#pragma once

#ifndef TRACKER_H
#define TRACKER_H
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>      //for imshow
#include "utils.h"// Drawing and printing functions
#include <fstream>  


using namespace std;
using namespace cv;

const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box
const int stats_update_period = 10; // On-screen statistics are updated every 10 frames

class Tracker
{
	double ransac_confidence_ = 0.99; // The confidence required
	double maximum_threshold_ = 50.0;// The maximum threshold value
// The name of the current test scene
	bool use_magsac_plus_plus_ = true; // A flag to decide if MAGSAC++ or MAGSAC should be used
	bool draw_results_ = true; // A flag determining if the results should be visualized
	double drawing_threshold_ = 2.5;
	vector<KeyPoint> firstKeyPoint;

public:
	Tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher) :
		detector(_detector),
		matcher(_matcher)
	{}
	void setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats);
	Mat process(const Mat frame, Stats& stats);
	void setFirstKp(vector<KeyPoint> kp) {
		firstKeyPoint = kp;
	}
	vector<KeyPoint> getFirstKp() {
		return firstKeyPoint;
	}
	Ptr<Feature2D> getDetector() {
		return detector;
	}
protected:
	Ptr<Feature2D> detector;
	Ptr<DescriptorMatcher> matcher;
	Mat first_frame, first_desc;
	vector<KeyPoint> first_kp;
	vector<Point2f> object_bb;
};
void Tracker::setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats)
{
	Point* ptMask = new Point[bb.size()];
	const Point* ptContain = { &ptMask[0] };
	int iSize = static_cast<int>(bb.size());
	for (size_t i = 0; i < bb.size(); i++) {
		ptMask[i].x = static_cast<int>(bb[i].x);
		ptMask[i].y = static_cast<int>(bb[i].y);
	}
	first_frame = frame.clone();
	Mat matMask = Mat::zeros(frame.size(), CV_8UC1);
	//CV_8UC1 unsigned 8-bit single-channel data; can be used for grayscale image or binary image
	cv::fillPoly(matMask, &ptContain, &iSize, 1, cv::Scalar::all(255));
	detector->detectAndCompute(first_frame, matMask, first_kp, first_desc);
	setFirstKp(first_kp);
	stats.keypoints = (int)first_kp.size();
	drawBoundingBox(first_frame, bb);
	putText(first_frame, title, Point(0, 60), FONT_HERSHEY_PLAIN, 5, Scalar::all(0), 4);

	object_bb = bb;
	delete[] ptMask;

}
Mat Tracker::process(const Mat frame, Stats& stats)
{

	TickMeter tm;
	vector<KeyPoint> kp;
	Mat desc;
	tm.start();
	detector->detectAndCompute(frame, noArray(), kp, desc);
	stats.keypoints = (int)kp.size();
	vector< vector<DMatch> > matches;
	vector<KeyPoint> matched1, matched2;
	matcher->knnMatch(first_desc, desc, matches, 2);
	for (unsigned i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
			matched1.push_back(first_kp[matches[i][0].queryIdx]);
			matched2.push_back(kp[matches[i][0].trainIdx]);
		}
	}
	stats.matches = (int)matched1.size();
	Mat inlier_mask, homography;
	vector<KeyPoint> inliers1, inliers2;
	vector<DMatch> inlier_matches;


	if (matched1.size() >= 4) {
		homography = findHomography(Points(matched1), Points(matched2),
			USAC_MAGSAC, ransac_thresh, inlier_mask);
	}
	tm.stop();
	stats.fps = 1. / tm.getTimeSec();
	if (matched1.size() < 4 || homography.empty()) {
		Mat res;
		hconcat(first_frame, frame, res);
		stats.inliers = 0;
		stats.ratio = 0;
		return res;
	}

	for (unsigned i = 0; i < matched1.size(); i++) {
		if (inlier_mask.at<uchar>(i)) {
			int new_i = static_cast<int>(inliers1.size());
			inliers1.push_back(matched1[i]);
			inliers2.push_back(matched2[i]);
			inlier_matches.push_back(DMatch(new_i, new_i, 0));
		}
	}
	stats.inliers = (int)inliers1.size();
	stats.ratio = stats.inliers * 1.0 / stats.matches;
	vector<Point2f> new_bb;
	perspectiveTransform(object_bb, new_bb, homography);
	Mat frame_with_bb = frame.clone();
	if (stats.inliers >= bb_min_inliers) {
		drawBoundingBox(frame_with_bb, new_bb);
	}
	Mat res;
	drawMatches(first_frame, inliers1, frame_with_bb, inliers2,
		inlier_matches, res,
		Scalar(255, 0, 0), Scalar(255, 0, 0));
	return res;
}

#endif
