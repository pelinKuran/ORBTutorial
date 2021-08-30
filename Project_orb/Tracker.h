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
/* 
Mat Tracker::process2(const Mat frame, Stats& stats) {

	TickMeter tm;
	Mat points;
	vector<int> ground_truth;
	vector<KeyPoint> kp;
	vector<KeyPoint> firstKp = getFirstKp();
	Mat desc;
	tm.start();
	detector->detectAndCompute(frame, noArray(), kp, desc);

	readAnnotatedPoint(firstKp, kp, points, ground_truth);
	const size_t point_number = points.rows;
	magsac::utils::DefaultHomographyEstimator estimator; // The robust homography estimator class containing the function for the fitting and residual calculation
	gcransac::Homography model; // The estimated model
	std::vector<int> refined_labels = ground_truth;
	refineManualLabeling<gcransac::Homography, magsac::utils::DefaultHomographyEstimator>(
		points, // The data points
		refined_labels, // The refined labeling
		estimator, // The model estimator
		2.0); // The used threshold in pixels

	std::vector<int> ground_truth_inliers = getSubsetFromLabeling(ground_truth, 1),
		refined_inliers = getSubsetFromLabeling(refined_labels, 1);
	if (ground_truth_inliers.size() < refined_inliers.size())
		ground_truth_inliers.swap(refined_inliers);

	const size_t reference_inlier_number = ground_truth_inliers.size();
	gcransac::sampler::ProgressiveNapsacSampler<4> main_sampler(&points,
		{ 16, 8, 4, 2 },	// The layer of grids. The cells of the finest grid are of dimension 
							// (source_image_width / 16) * (source_image_height / 16)  * (destination_image_width / 16)  (destination_image_height / 16), etc.
		estimator.sampleSize(), // The size of a minimal sample
		{ static_cast<double>(first_frame.cols), // The width of the source image
			static_cast<double>(first_frame.rows), // The height of the source image
			static_cast<double>(frame.cols), // The width of the destination image
			static_cast<double>(frame.rows) },  // The height of the destination image
		0.5); // The length (i.e., 0.5 * <point number> iterations) of fully blending to global sampling 

	MAGSAC<cv::Mat, magsac::utils::DefaultHomographyEstimator> magsac
	(use_magsac_plus_plus_ ?
		MAGSAC<cv::Mat, magsac::utils::DefaultHomographyEstimator>::MAGSAC_PLUS_PLUS :
		MAGSAC<cv::Mat, magsac::utils::DefaultHomographyEstimator>::MAGSAC_ORIGINAL);
	magsac.setMaximumThreshold(maximum_threshold_); // The maximum noise scale sigma allowed
	magsac.setIterationLimit(1e4); // Iteration limit to interrupt the cases when the algorithm run too long.
	magsac.setReferenceThreshold(2.0);
	int iteration_number = 0; // Number of iterations required
	ModelScore score; // The model score

	std::chrono::time_point<std::chrono::system_clock> end,
		start = std::chrono::system_clock::now();
	magsac.run(points, // The data points
		ransac_confidence_, // The required confidence in the results
		estimator, // The used estimator
		main_sampler, // The sampler used for selecting minimal samples in each iteration
		model, // The estimated model
		iteration_number, // The number of iterations
		score); // The score of the estimated model
	end = std::chrono::system_clock::now();
	/* if (model.descriptor.size() == 0)
	{
		// Clean up the memory occupied by the images
		first_frame.release();
		//frame.release();
		return;
	}*/
/* 
	if (draw_results_)
	{
		MostSimilarInlierSelector< magsac::utils::DefaultHomographyEstimator>
			inlierSelector(estimator.sampleSize() + 1,
				maximum_threshold_);

		std::vector<size_t> selectedInliers;
		double bestThreshold;
		inlierSelector.selectInliers(points,
			estimator,
			model,
			selectedInliers,
			bestThreshold);

		// The labeling implied by the estimated model and the drawing threshold
		std::vector<int> obtained_labeling(points.rows, 0);

		for (const auto& inlierIdx : selectedInliers)
			obtained_labeling[inlierIdx] = 1;

		cv::Mat out_image;

		// Draw the matches to the images
		drawMatches<double, int>(points, // All points 
			obtained_labeling, // The labeling obtained by OpenCV
			first_frame, // The source image
			frame, // The destination image
			out_image); // The image with the matches drawn
		return out_image;
		// Show the matches
		/*std::string window_name = "Visualization with threshold = " + std::to_string(drawing_threshold_) + " px; Maximum threshold is = " + std::to_string(maximum_threshold_);
		showImage(out_image, // The image with the matches drawn
			window_name, // The name of the window
			1600, // The width of the window
			900); // The height of the window
		out_image.release(); // Clean up the memory
	}
	 */
	/*

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
		/* homography = findHomography(Points(matched1), Points(matched2),
			RANSAC, ransac_thresh, inlier_mask);*/

			/* }
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
	Mat out_image;}
	*/

#endif