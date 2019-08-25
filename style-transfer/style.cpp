#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/shape_utils.hpp"
#include <iostream>
#include <stdio.h>
#include <cstdio>

using namespace std;
using namespace cv;
using namespace dnn;


string get_output_filename(int frame_number)
{
	string outfile = "style_test.";
	string frame_string = to_string(frame_number);
	frame_string = string(4 - frame_string.length(), '0') + frame_string;
	outfile = outfile + frame_string + ".jpg";
	return outfile;
}

int main(void) {

	Net deep;
	String _windowName = "Live Style Transfer";
	VideoCapture _capture;

	// Load the network
	deep = readNet("franc.bin", "franc.xml");
	// Try to make OpenVino work
	deep.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
	deep.setPreferableTarget(DNN_TARGET_OPENCL);

	// Open the stream.
	_capture.open(0);
	if (!_capture.isOpened())
		return -2;

	// Get input res
	int outCameraWidth = _capture.get(CAP_PROP_FRAME_WIDTH);
	int outCameraHeight = _capture.get(CAP_PROP_FRAME_HEIGHT);

	// Frame count
	int frame_number = 0;
	// Storage for input key
	int key = 1;
	bool is_capturing = false;

	while (true) {
		Mat frame;
		_capture >> frame;
		if (frame.empty())
			return 0;


		// Hold space to see original image
		if (key!= 32){
			// Make blob and feed to Neural Net
			Mat blob, out;
			blob = blobFromImage(frame, 1. / 256, Size(512, 512), (127, 127, 127), true, false);
			deep.setInput(blob);
			out = deep.forward();

			// Reverse offset
			out += 1.;
			out *= .5;

			// Re-sort channels for Open CV
			Size siz(out.size[2], out.size[3]);
			Mat r = Mat(siz, CV_32F, out.ptr(0, 0));
			Mat g = Mat(siz, CV_32F, out.ptr(0, 1));
			Mat b = Mat(siz, CV_32F, out.ptr(0, 2));
			Mat final;
			Mat chn[] = { b, g, r };
			merge(chn, 3, final);

			// Conver to 8bit and resize back to original res
			final.convertTo(final, CV_8U, 255);
			resize(final, frame, frame.size());
		}

		//Press "c" to toggle capturing images
		if (key == 99) {
			is_capturing = !is_capturing;
		}
		if (is_capturing)
		{
			imwrite(get_output_filename(frame_number), frame);
			frame_number += 1;
		}

		imshow(_windowName, frame);
		key = waitKey(1);

	}
	return 0;
}
