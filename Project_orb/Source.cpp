#include "Tracker.h";

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, "{@input_path |0|input path can be a camera id, like 0,1,2 or a video filename}");
	parser.printMessage();
	string input_path = parser.get<string>(0);
	string video_name = input_path;
	VideoCapture video_in;
	if ((isdigit(input_path[0]) && input_path.size() == 1))
	{
		int camera_no = input_path[0] - '0';
		video_in.open(camera_no);
	}
	else {
		video_in.open(video_name);
	}
	if (!video_in.isOpened()) {
		cerr << "Couldn't open " << video_name << endl;
		return 1;
	}
	Stats stats, orb_stats;

	Ptr<ORB> orb = ORB::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	Tracker orb_tracker(orb, matcher);
	Mat frame;
	namedWindow(video_name, WINDOW_NORMAL);
	cout << "\nPress any key to stop the video and select a bounding box" << endl;
	while (waitKey(1) < 1)
	{
		video_in >> frame;
		resizeWindow(video_name, frame.size());
		imshow(video_name, frame);
	}

	vector<Point2f> bb;
	Rect uBox = selectROI(video_name, frame);
	bb.push_back(Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y)));
	bb.push_back(Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y)));
	bb.push_back(Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y + uBox.height)));
	bb.push_back(Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y + uBox.height)));

	orb_tracker.setFirstFrame(frame, bb, "ORB", stats);
	Stats orb_draw_stats;
	Mat orb_res, res_frame;
	int i = 0;
	for (;;) {
		i++;
		bool update_stats = (i % stats_update_period == 0);
		video_in >> frame;
		// stop the program if no more images
		if (frame.empty()) break;


		orb->setMaxFeatures(stats.keypoints);
		orb_res = orb_tracker.process(frame, stats);
		orb_stats += stats;
		if (update_stats) {
			orb_draw_stats = stats;
		}

		drawStatistics(orb_res, orb_draw_stats);
		vconcat(orb_res, res_frame);
		imshow(video_name, res_frame);


		if (waitKey(1) == 27) break; //quit on ESC button
	}

	orb_stats /= i - 1;
	printStatistics("ORB", orb_stats);
	return 0;
}