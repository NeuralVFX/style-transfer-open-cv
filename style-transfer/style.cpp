#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/shape_utils.hpp"
#include <iostream>
#include <stdio.h>


using namespace std;
using namespace cv;
using namespace dnn;



int main(void) {

	Net deep;
	String _windowName = "Live Style Transfer";
	VideoCapture _capture;

	// Load the network
	deep = readNet("pic_dec.onnx");

	// Open the stream.
	_capture.open(1);
	if (!_capture.isOpened())
		return -2;

	// Get input res
	int outCameraWidth = _capture.get(CAP_PROP_FRAME_WIDTH);
	int outCameraHeight = _capture.get(CAP_PROP_FRAME_HEIGHT);

	while (true) {
		Mat frame;
		_capture >> frame;
		if (frame.empty())
			return 0;


		// Hold space to see original image
		if (waitKey(1000) != 32){
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

		// Press "c" to capture image
		if (waitKey(1000) == 99) {
			imwrite("style_test.jpg", frame);
		}

		imshow(_windowName, frame);
		waitKey(1);

	}
	return 0;
}
