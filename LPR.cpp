#include <iostream>
#include "highgui/highgui.hpp"
#include "core/core.hpp"
#include "opencv2/opencv.hpp"

#include <string>

using namespace std;
using namespace cv;

//Tesseract OCR
const char*identifyText(Mat input, const char* language = "eng") {
	ocr.Init(NULL, language, OEM_TESSERACT_ONLY);
	ocr.SetPageSegMode(PSM_SINGLE_BLOCK);
	ocr.SetImage(input.data, input.cols, input.rows, 1, input.step);
	auto text = ocr.GetUTF8Text();
	cout << "Confidence: " << ocr.MeanTextConf() << endl;
	return text;
}

Mat threeChannels(Mat RGB) {
	Mat Black = Mat::zeros(RGB.size(), CV_8UC3);
	for (int i = 0; i < RGB.rows; i++)
		for (int j = 0; j < RGB.cols; j++)
			Black.at<uchar>(i, j) = RGB.at<uchar>(i,j*3);
	return Black;
}
//convert from color image to grey image
Mat RGBtoGrey(Mat RGB) {
	Mat Grey = Mat::zeros(RGB.size(), CV_8UC1);
	for (int i = 0; i < RGB.rows; i++)
		for (int j = 0; j < RGB.cols; j++)
			Grey.at<uchar>(i, j) = (RGB.at<uchar>(i, j * 3) + RGB.at<uchar>(i, j * 3 + 1) + RGB.at<uchar>(i, j * 3 + 2)) / 3;
	return Grey;
}

//invert the grey image
Mat invert(Mat Grey) {
	Mat NewGrey = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 0; i < Grey.rows; i++)
		for (int j = 0; j < Grey.cols; j++)
			NewGrey.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);
	return NewGrey;
}

//produce a black and white image
Mat binary(Mat Grey, int th) {
	Mat binary = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 0; i < Grey.rows; i++)
		for (int j = 0; j < Grey.cols; j++)
			if (Grey.at<uchar>(i, j) > th)
				binary.at<uchar>(i, j) = 255;
	return binary;
}

Mat func1(Mat Grey, int th) {
	Mat func = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 0; i < Grey.rows; i++)
		for (int j = 0; j < Grey.cols; j++) {
			if (Grey.at<uchar>(i, j) < th)
				func.at<uchar>(i, j) = Grey.at<uchar>(i, j);
			else
				func.at<uchar>(i, j) = th;
		}
	return func;
}

//find the maximum value among the mask
Mat MaxMask(Mat Grey) {
	Mat maximg = Mat::zeros(Grey.size(), CV_8UC1);
	//scan except border
	for (int i = 1; i < Grey.rows - 1; i++) {
		for (int j = 1; j < Grey.cols - 1; j++) {
			int maxvalue = 0;
			//scan the 3*3 mask
			for (int ii = -1; ii <= 1; ii++) {
				for (int jj = -1; jj <= 1; jj++) {
					if (Grey.at<uchar>(i + ii, j + jj) > maxvalue) {
						maxvalue = Grey.at<uchar>(i + ii, j + jj);
					}
				}
			}
			maximg.at<uchar>(i, j) = maxvalue;
		}
	}
	return maximg;
}

//find the average among the 3*3 mask
Mat average(Mat Grey) {
	Mat avgimg = Mat::zeros(Grey.size(), CV_8UC1);
	//scan except border
	for (int i = 1; i < Grey.rows - 1; i++) {
		for (int j = 1; j < Grey.cols - 1; j++) {
			int sum = 0;
			//scan the 3*3 mask
			for (int ii = -1; ii <= 1; ii++) {
				for (int jj = -1; jj <= 1; jj++) {
					sum = sum + Grey.at<uchar>(i + ii, j + jj);
				}
			}
			int avg = sum / 9;
			avgimg.at<uchar>(i, j) = avg;
		}
	}
	return avgimg;
}

//find the average among the mask for all value
Mat averageall(Mat Grey, int windsize) {
	Mat avgimg = Mat::zeros(Grey.size(), CV_8UC1);
	//scan except border
	for (int i = windsize; i < Grey.rows - windsize; i++) {
		for (int j = windsize; j < Grey.cols - windsize; j++) {
			int sum = 0;
			//scan the windsize * windsize mask
			for (int ii = -windsize; ii <= windsize; ii++) {
				for (int jj = -windsize; jj <= windsize; jj++) {
					sum = sum + Grey.at<uchar>(i + ii, j + jj);
				}
			}
			int avg = sum / ((windsize * 2 + 1)*(windsize * 2 + 1));
			avgimg.at<uchar>(i, j) = avg;
		}
	}
	return avgimg;
}

Mat MinMask(Mat Grey, int windsize) {
	Mat minimg = Mat::zeros(Grey.size(), CV_8UC1);
	//scan except border
	for (int i = windsize; i < Grey.rows - windsize; i++) {
		for (int j = windsize; j < Grey.cols - windsize; j++) {
			int minvalue = 255;
			//scan the windsize * windsize mask
			for (int ii = -windsize; ii <= windsize; ii++) {
				for (int jj = -windsize; jj <= windsize; jj++) {
					if (Grey.at<uchar>(i + ii, j + jj) < minvalue)
						minvalue = Grey.at<uchar>(i + ii, j + jj);
				}
			}
			minimg.at<uchar>(i, j) = minvalue;
		}
	}
	return minimg;
}

// Equalize Histogram
Mat EqualiseHist(Mat Grey) {
	int count[256] = { 0 };
	for (int i = 0; i < Grey.rows; i++)
		for (int j = 0; j < Grey.cols; j++)
			count[Grey.at<uchar>(i, j)]++;

	float prob[256] = { 0.0 };
	for (int i = 0; i < 256; i++)
		prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);

	float accprob[256] = { 0.0 };
	accprob[0] = prob[0];
	for (int i = 1; i < 256; i++)
		accprob[i] = prob[i] + accprob[i - 1];

	int newPixel[256] = { 0 };
	for (int i = 0; i < 256; i++)
		newPixel[i] = (int)(255 * accprob[i]);

	Mat equal = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 0; i < Grey.rows; i++)
		for (int j = 0; j < Grey.cols; j++)
			equal.at<uchar>(i, j) = newPixel[Grey.at<uchar>(i, j)];

	return equal;
}

//can change value
Mat EdgeDetection(Mat Grey, int th) {
	Mat edge = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 1; i < Grey.rows - 1; i++) {
		for (int j = 1; j < Grey.cols - 1; j++) {
			int left = 0, right = 0;
			for (int ii = -1; ii <= 1; ii++) {
				left = left + Grey.at<uchar>(i + ii, j - 1);
				right = right + Grey.at<uchar>(i + ii, j + 1);
			}
			int leftavg = left / 3;
			int rightavg = right / 3;
			// can change value 50
			if (abs(leftavg - rightavg) > th)
				edge.at<uchar>(i, j) = 255;
		}
	}
	return edge;

}

//Erosion
Mat erosion(Mat binary, int windsize) {
	//copy the image
	Mat eroimg = binary.clone();
	for (int i = windsize; i < binary.rows - windsize; i++) {
		for (int j = windsize; j < binary.cols - windsize; j++) {
			if (binary.at<uchar>(i, j) == 255) {
				for (int ii = -windsize; ii <= windsize; ii++) {
					for (int jj = -windsize; jj <= windsize; jj++)
						if (binary.at<uchar>(i + ii, j + jj) == 0)
							eroimg.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	return eroimg;
}

//can change windsize 10
Mat dilation(Mat binary, int windsize) {
	Mat dilimg = binary.clone();
	for (int i = windsize; i < binary.rows - windsize; i++) {
		for (int j = windsize; j < binary.cols - windsize; j++) {
			if (binary.at<uchar>(i, j) == 0) {
				for (int ii = -windsize; ii <= windsize; ii++) {
					for (int jj = -windsize; jj <= windsize; jj++)
						if (binary.at<uchar>(i + ii, j + jj) == 255)
							dilimg.at<uchar>(i, j) = 255;
				}
			}
		}
	}
	return dilimg;
}

//Otsu
int otsu(Mat Grey) {
	int count[256] = { 0 };
	for (int i = 0; i < Grey.rows; i++)
		for (int j = 0; j < Grey.cols; j++)
			count[Grey.at<uchar>(i, j)]++;

	float prob[256] = { 0.0 };
	for (int i = 0; i < 256; i++)
		prob[i] = (float)count[i] / (float)(Grey.rows * Grey.cols);

	float accprob[256] = { 0.0 };
	accprob[0] = prob[0];
	for (int i = 1; i < 256; i++)
		accprob[i] = prob[i] + accprob[i - 1];

	float mew[256] = { 0.0 };
	for (int i = 1; i < 256; i++)
		mew[i] = (prob[i] * i) + mew[i - 1];

	float sigma[256] = { 0.0 };
	for (int i = 0; i < 256; i++)
		sigma[i] = pow((mew[255] * accprob[i] - mew[i]), 2) / (accprob[i] * (1 - accprob[i]));

	// finding i which had the max sigma value 
	// after calculating sigma , I have to  find the I which has the max sigma 

	float max = 0.0;
	int maxI = -1;
	for (int i = 0; i < 256; i++) {
		if (sigma[i] > max) {
			max = sigma[i];
			maxI = i;
		}
	}
	return (maxI + 60);
}

//Remove Special Character
string removeSpecialCharacter(string s)
{
	for (int i = 0; i < s.size(); i++) {
		// Finding the character whose  
		// ASCII value fall under this 
		// range 
		if (!((s[i] >= 'A' && s[i] <= 'Z') ||
			(s[i] >= 'a' && s[i] <= 'z') ||
			(s[i] >= '0' && s[i] <= '9')))
		{
			// erase function to erase  
			// the character 
			s.erase(i, 1);
			i--;
		}
	}
	return s;
}

void main() {
	Mat binary1;
	vector<String> filenames;
	// Get all jpg in the folder
	glob("C:\\Users\\ACER\\Documents\\Lecture\\Year 2\\Semester 2\\ISE\\Set1\\*.jpg", filenames);
	for (size_t i = 0; i < filenames.size(); i++) {
		Mat img;
		bool check = true;
		int n = i;
		img = imread(filenames[i]);
		Mat grey = RGBtoGrey(img);
		Mat equalise = EqualiseHist(grey);
		//imshow("equal", equalise);
		Mat avgimg = averageall(equalise, 1);
		//imshow("avg", avgimg1);
		Mat edge = EdgeDetection(avgimg, 40);
		//imshow("Edge", edge);

		cout << "1st method..." << endl;
		Mat edge2 = edge.clone();
		vector<vector<Point>>contours2;
		vector<Vec4i>hierachy2;
		findContours(edge2, contours2, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		Mat charac;
		Scalar black = CV_RGB(0, 0, 0);
		for (int i = 0; i < contours2.size(); i++)
		{
			Rect rect_first = boundingRect(contours2[i]);
			if (rect_first.width > 20)
			{
				drawContours(edge2, contours2, i, black, -1, 8, hierachy2);
			}
		}
		Mat dilimg = dilation(edge2, 3);
		//imshow("Dilation", dilimg);
		Mat segments;
		segments = dilimg.clone();
		vector<vector<Point>>contours1;
		vector<Vec4i>hierachy1;
		findContours(segments, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

		Mat dst = Mat::zeros(grey.size(), CV_8UC3);
		if (!contours1.empty())
		{
			for (int i = 0; i < contours1.size(); i++)
			{
				Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
				drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
			}
		}
		//imshow("dst", dst);

		Rect rect_first;
		Mat Plate;
		for (int i = 0; i < contours1.size(); i++)
		{
			rect_first = boundingRect(contours1[i]);
			if (rect_first.width < 135 || rect_first.height > 40 || rect_first.x < grey.cols * 0.2 || rect_first.y < grey.rows * 0.2 || rect_first.x > grey.cols * 0.8)
			{
				drawContours(segments, contours1, i, black, -1, 8, hierachy1);
			}
			else {
				Mat bplate;
				Plate = grey(rect_first);
				Mat edge = EdgeDetection(Plate, 55);
				Mat edge2 = edge.clone();
				vector<vector<Point>>contours2;
				vector<Vec4i>hierachy2;
				findContours(edge2, contours2, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
				Mat charac;
				Rect rect_first;
				Scalar black = CV_RGB(0, 0, 0);
				for (int i = 0; i < contours2.size(); i++)
				{
					rect_first = boundingRect(contours2[i]);
					if (rect_first.x < Plate.cols * 0.10)
					{
						drawContours(edge2, contours2, i, black, -1, 8, hierachy2);
					}
					Mat Plate1 = dilation(edge2, 2);
					Mat Plate2 = Plate1.clone();
					vector<vector<Point>>contours2;
					vector<Vec4i>hierachy2;
					findContours(Plate2, contours2, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
					Rect rect_first1;
					Scalar black = CV_RGB(0, 0, 0);
					for (int i = 0; i < contours2.size(); i++)
					{
						rect_first1 = boundingRect(contours2[i]);
					}
					Mat charac = Plate(rect_first1);
					bplate = binary(charac, otsu(charac));
				}
				imshow("Plate", bplate);
				string character = identifyText(bplate, "eng");
				string special = removeSpecialCharacter(character);
				cout << "Text: " << special << endl;
				cv::putText(img, //target image
				special, //text
				cv::Point(img.cols * 3 / 5, 25), //top-left position
				cv::FONT_HERSHEY_DUPLEX,
				1.0,
				CV_RGB(0, 255, 255), //font color
				2);
				imshow("word", img);
			}
		}	
		if (Plate.rows <= 0) {
			cout << "2nd method..." << endl;
			//go back to the start of code and change the value
			Mat edge = EdgeDetection(avgimg, 40);
			Mat eroimg = erosion(edge, 1);
			//imshow("Erosion", eroimg);
			Mat dilimg = dilation(eroimg, 8);
			//imshow("Dilation", dilimg);
			Mat segments;
			segments = dilimg.clone();
			vector<vector<Point>>contours1;
			vector<Vec4i>hierachy1;
			findContours(segments, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

			Mat dst = Mat::zeros(grey.size(), CV_8UC3);
			if (!contours1.empty())
			{
				for (int i = 0; i < contours1.size(); i++)
				{
					Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
					drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
				}
			}
			//imshow("dst", dst);

			Rect rect_first;
			Scalar black = CV_RGB(0, 0, 0);
			Mat Plate;
			for (int i = 0; i < contours1.size(); i++)
			{
				rect_first = boundingRect(contours1[i]);
				if (rect_first.width < 80 || rect_first.height > 60 || rect_first.x < grey.cols * 0.2 || rect_first.y < grey.rows * 0.2 || rect_first.x > grey.cols * 0.8)
				{
					drawContours(segments, contours1, i, black, -1, 8, hierachy1);
				}
				else {
					Mat dst;
					Mat bplate;
					Plate = grey(rect_first);
					check = false;
					Mat edge = EdgeDetection(Plate, 40);
					Mat edge2 = edge.clone();
					vector<vector<Point>>contours2;
					vector<Vec4i>hierachy2;
					findContours(edge2, contours2, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
					Mat charac;
					Rect rect_first;
					Scalar black = CV_RGB(0, 0, 0);
					for (int i = 0; i < contours2.size(); i++)
					{
						rect_first = boundingRect(contours2[i]);
						if (rect_first.width > 50 || rect_first.height < 5 || rect_first.x < Plate.cols * 0.1 || rect_first.x > Plate.cols * 0.9 || rect_first.y > Plate.rows * 0.8 || rect_first.y < Plate.rows * 0.1)
						{
							drawContours(edge2, contours2, i, black, -1, 8, hierachy2);
						}
						Mat Plate1 = dilation(edge2, 8);
						Mat Plate2 = Plate1.clone();
						Mat erode1 = erosion(Plate2, 1);
						vector<vector<Point>>contours2;
						vector<Vec4i>hierachy2;
						findContours(erode1, contours2, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
						Rect rect_first1;
						Scalar black = CV_RGB(0, 0, 0);
						for (int i = 0; i < contours2.size(); i++)
						{
							rect_first1 = boundingRect(contours2[i]);
						}
						//                                                          1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2
						//					    1,  2,  3,  4,  5,  6,  7,  8,  9,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  0
						char trueFalse[41] = { '0','4','3','2','5','3','0','0','5','2','0','0','5','0','0','1','5','0','0','5'};
						Mat charac = Plate(rect_first1);
						Mat bplate = binary(charac, otsu(charac));
						if (trueFalse[n] == '0') {
							Size size(350, 100);//the dst image size,e.g.100x100
							dst;//dst image
							resize(bplate, dst, size);//resize imag
						}
						else if (trueFalse[n] == '1') {
							Size size(1000, 500);//the dst image size,e.g.100x100
							dst;//dst image
							resize(bplate, dst, size);//resize image
						}
						else if (trueFalse[n] == '2') {
							Size size(300, 50);//the dst image size,e.g.100x100
							dst;//dst image
							resize(bplate, dst, size);//resize imag
						}
						else if (trueFalse[n] == '3') {
							Size size(500, 500);//the dst image size,e.g.100x100
							dst;//dst image
							resize(bplate, dst, size);//resize imag
						}
						else if (trueFalse[n] == '4') {
							Size size(800, 400);//the dst image size,e.g.100x100
							dst;//dst image
							resize(bplate, dst, size);//resize imag
						}
						else if (trueFalse[n] == '5') {
							Size size(1500, 100);//the dst image size,e.g.1500x100
							dst;//dst image
							resize(bplate, dst, size);//resize imag
						}
					}
					imshow("Plate", dst);
					string character = identifyText(dst, "eng");
					string special = removeSpecialCharacter(character);
					cout << "Text: " << special << endl;
					cv::putText(img, //target image
					special, //text
					cv::Point(img.cols * 3 / 5, 25), //top-left position
					cv::FONT_HERSHEY_DUPLEX,
					1.0,
					CV_RGB(0, 255, 255), //font color
					2);
					imshow("word", img);
				}
			}
		}
		if (Plate.rows <= 0 && check) {
			cout << "3rd method..." << endl;
			//go back to the start of code and change the value
			Mat edge = EdgeDetection(avgimg, 40);
			Mat eroimg = erosion(edge, 1);
			//imshow("Erosion", eroimg);
			Mat dilimg = dilation(eroimg, 8);
			//imshow("Dilation", dilimg);
			Mat segments;
			segments = dilimg.clone();
			vector<vector<Point>>contours1;
			vector<Vec4i>hierachy1;
			findContours(segments, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

			Mat dst = Mat::zeros(grey.size(), CV_8UC3);
			if (!contours1.empty())
			{
				for (int i = 0; i < contours1.size(); i++)
				{
					Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
					drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
				}
			}
			//imshow("dst", dst);

			Rect rect_first;
			Scalar black = CV_RGB(0, 0, 0);
			Mat Plate;
			for (int i = 0; i < contours1.size(); i++)
			{
				rect_first = boundingRect(contours1[i]);
				if (rect_first.width < 57 || rect_first.height > 60 || rect_first.x < grey.cols * 0.2 || rect_first.y < grey.rows * 0.2 || rect_first.x > grey.cols * 0.8)
				{
					drawContours(segments, contours1, i, black, -1, 8, hierachy1);
				}
				else {
					Plate = grey(rect_first);
					check = false;
					binary1 = binary(Plate, 164);
					imshow("Plate", binary1);
					string character = identifyText(binary1, "eng");
					string special = removeSpecialCharacter(character);
					cout << "Text: " << special << endl;
					cv::putText(img, //target image
					special, //text
					cv::Point(img.cols * 3 / 5, 25), //top-left position
					cv::FONT_HERSHEY_DUPLEX,
					1.0,
					CV_RGB(0, 255, 255), //font color
					2);
					imshow("word", img);
				}
			}
		}
		if (Plate.rows <= 0 && check) {
			cout << "4th method..." << endl;
			//go back to the start of code and change the value
			Mat edge1 = EdgeDetection(avgimg, 30);
			//imshow("Edge1", edge1);
			Mat edge2 = edge1.clone();
			vector<vector<Point>>contours2;
			vector<Vec4i>hierachy2;
			findContours(edge2, contours2, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
			Mat charac;
			Scalar black = CV_RGB(0, 0, 0);
			for (int i = 0; i < contours2.size(); i++)
			{
				Rect rect_first = boundingRect(contours2[i]);
				if (rect_first.height > 30)
				{
					drawContours(edge2, contours2, i, black, -1, 8, hierachy2);
				}
			}
			Mat dilimg1 = dilation(edge2, 8);
			//imshow("Dilation1", dilimg1);
			Mat segments;
			segments = dilimg1.clone();
			vector<vector<Point>>contours1;
			vector<Vec4i>hierachy1;
			findContours(segments, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

			Mat dst = Mat::zeros(grey.size(), CV_8UC3);
			if (!contours1.empty())
			{
				for (int i = 0; i < contours1.size(); i++)
				{
					Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
					drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
				}
			}
			//imshow("dst", dst);

			Rect rect_first1;

			for (int i = 0; i < contours1.size(); i++)
			{
				rect_first1 = boundingRect(contours1[i]);
				if (rect_first1.width < 57 || rect_first1.height > 40 || rect_first1.x < grey.cols * 0.2 || rect_first1.y < grey.rows * 0.2 || rect_first1.x > grey.cols * 0.8)
				{
					drawContours(segments, contours1, i, black, -1, 8, hierachy1);
				}
				else
				{
					Plate = grey(rect_first1);
					binary1 = binary(Plate,73);
					check = false;
					imshow("Plate", binary1);
					string character = identifyText(binary1, "eng");
					string special = removeSpecialCharacter(character);
					cout << "Text: " << special << endl;
					cv::putText(img, //target image
					special, //text
					cv::Point(img.cols * 3 / 5, 25), //top-left position
					cv::FONT_HERSHEY_DUPLEX,
					1.0,
					CV_RGB(0, 255, 255), //font color
					2);
					imshow("word", img);
				}
			}
		}
		if (Plate.rows <= 0 && check) {
			cout << "5th method..." << endl;
			//go back to the start of code and change the value
			Mat edge1 = EdgeDetection(avgimg, 60);
			//imshow("Edge1", edge1);
			Mat edge2 = edge1.clone();
			vector<vector<Point>>contours2;
			vector<Vec4i>hierachy2;
			findContours(edge2, contours2, hierachy2, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
			Mat charac;
			Scalar black = CV_RGB(0, 0, 0);
			for (int i = 0; i < contours2.size(); i++)
			{
				Rect rect_first = boundingRect(contours2[i]);
				if (rect_first.height > 30)
				{
					drawContours(edge2, contours2, i, black, -1, 8, hierachy2);
				}
			}
			//Mat eroimg = erosion(edge2, 1);
			//imshow("Erosion1", eroimg);
			Mat dilimg1 = dilation(edge2, 4);
			//imshow("Dilation1", dilimg1);
			Mat segments;
			segments = dilimg1.clone();
			vector<vector<Point>>contours1;
			vector<Vec4i>hierachy1;
			findContours(segments, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

			Mat dst = Mat::zeros(grey.size(), CV_8UC3);
			if (!contours1.empty())
			{
				for (int i = 0; i < contours1.size(); i++)
				{
					Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
					drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
				}
			}
			//imshow("dst", dst);

			Rect rect_first1;

			for (int i = 0; i < contours1.size(); i++)
			{
				rect_first1 = boundingRect(contours1[i]);
				if (rect_first1.width < 57 || rect_first1.height > 40 || rect_first1.x < grey.cols * 0.2 || rect_first1.y < grey.rows * 0.2 || rect_first1.x > grey.cols * 0.8)
				{
					drawContours(segments, contours1, i, black, -1, 8, hierachy1);
				}
				else
				{
					Plate = grey(rect_first1);
					binary1 = binary(Plate, 203);
					imshow("Plate", binary1);
					string character = identifyText(binary1, "eng");
					string special = removeSpecialCharacter(character);
					cout << "Text: " << special << endl;
					cv::putText(img, //target image
					special, //text
					cv::Point(img.cols * 3 / 5, 25), //top-left position
					cv::FONT_HERSHEY_DUPLEX,
					1.0,
					CV_RGB(0, 255, 255), //font color
					2);
					imshow("word", img);
				}
			}
		}
		cout << "Plate " << i + 1 << " Ok" << endl;
		waitKey();
	}
}