#include "stdafx.h"

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctype.h>
#include <time.h>
#include <windows.h>
#include <process.h>
#include <direct.h>

#include "cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include "aruco.h"
#include "cvdrawingutils.h"

using namespace cv;
using namespace std;
using namespace aruco;

#define PI 3.1415926

// split a string by separator
vector<string> split(string str, string separator)
{
	vector<string> result;
	int cutAt;
	while ((cutAt = str.find_first_of(separator)) != str.npos) {
		if (cutAt > 0) {
			result.push_back(str.substr(0, cutAt));
		}
		str = str.substr(cutAt + 1);
	}
	if (str.length() > 0) {
		result.push_back(str);
	}
	return result;
}

double PeriodCnstr(double _val_in, double _val_low, double _val_up) {	
	double prd;
	prd = _val_up - _val_low;
	double _val_out;
	int k = floor((_val_in-_val_low)/prd);
	_val_out = _val_in - k*prd;	
	return _val_out;
}

double Interpolate(double _time1, double _val1, double _time2, double _val2, double _time_out) {
	double dt_1_2 = _time2 - _time1;
	double dt_2_out = _time_out - _time2;

	dt_1_2 = PeriodCnstr(dt_1_2, -30, 30);
	dt_2_out = PeriodCnstr(dt_2_out, -30, 30);

	double val_out;

	if (abs(dt_1_2) <= 0.005) {
		val_out = _val2;
	}
	else {
		val_out = _val2 + (_val2-_val1)*dt_2_out/dt_1_2;		
	}
	return val_out;
}

int main(int argc, char** argv)
{
	// usage: datasetPath markSize logType
	// markSize: mm
	// logType: 0 pzf, 1 cfc

	if( argc != 4) {
		cout << "Usage error!!!" << endl;
		cout << "Usage: dataPath markSize logType" << endl;
		waitKey();
		return 1;
	}

	// init dataset path
/*	string strDataPath = "../../data/2016031114-pioneer-forward-clock/";*/
	string strDataPath = argv[1];
	float MarkerSize = atof(argv[2]);
	int logType = atoi(argv[3]);

	Mat im;	

	string strImgPath;
	switch (logType)
	{
	case 0:
		strImgPath = strDataPath+"image/";
		break;
	case 1:
		strImgPath = strDataPath+"image/picture";
		break;
	case 2:
		strImgPath = strDataPath+"image/";
		break;
	}
	
	string strLogPath = strDataPath+"log.txt";
	string strConfigPath = strDataPath+"config/CamConfig.yml";

	// init aruco
	aruco::CameraParameters CamParam;
	MarkerDetector MDetector;
	vector<Marker> Markers;
	
	CamParam.readFromXMLFile(strConfigPath);

	int ThePyrDownLevel = 0;
// 	int ThresParam1 = 19;
// 	int ThresParam2 = 35;
	int ThresParam1 = 19;
	int ThresParam2 = 15;

	MDetector.pyrDown(ThePyrDownLevel);
	MDetector.setCornerRefinementMethod(MarkerDetector::LINES);
	MDetector.setThresholdParams( ThresParam1,ThresParam2);	

	// read log file
	stringstream logFilePath_stream;
	logFilePath_stream << strLogPath;
	string logFilePath;
	logFilePath_stream >> logFilePath;
	ifstream logFile_stream(logFilePath);
	string s_tmp;
	vector<string> vecStrTmp;
	vector<double> vec_timeOdo;
	vector<double> vec_timeCam;
	vector<double> vec_xOdo, vec_yOdo, vec_thetaOdo;
	while( logFile_stream >> s_tmp ){
		// read time info
		vecStrTmp = split(s_tmp, ":");
		if(vecStrTmp.size() == 3) {
			vec_timeOdo.push_back(atof(vecStrTmp[2].c_str()));
			logFile_stream >> s_tmp;
			vec_timeCam.push_back(atof(s_tmp.c_str()));
		}

		switch(logType){
		case 0:
			// read odometry info: log file version PZF
			if (s_tmp == "O:"){
				logFile_stream >> s_tmp;
				vecStrTmp = split(s_tmp, ",");
				double xOdo = 1000*atof(vecStrTmp[0].c_str());
				double yOdo = 1000*atof(vecStrTmp[1].c_str());
				double thetaOdo = atof(vecStrTmp[2].c_str());
				vec_xOdo.push_back(xOdo);
				vec_yOdo.push_back(yOdo);
				vec_thetaOdo.push_back(thetaOdo);
			}
			break;
		case 1:
			// read odometry info: log file version CFC
			if (vecStrTmp[0] == "O"){
				vecStrTmp = split(vecStrTmp[1], ",");
				double xOdo = atof(vecStrTmp[0].c_str());
				double yOdo = atof(vecStrTmp[1].c_str());
				double thetaOdo = atof(vecStrTmp[2].c_str())*PI/180;
				vec_xOdo.push_back(xOdo);
				vec_yOdo.push_back(yOdo);
				vec_thetaOdo.push_back(thetaOdo);
			}
			break;
		case 2:
			// read odometry info: log file version CFC
			if (vecStrTmp[0] == "O"){
				vecStrTmp = split(vecStrTmp[1], ",");
				double xOdo = atof(vecStrTmp[0].c_str());
				double yOdo = atof(vecStrTmp[1].c_str());
				double thetaOdo = atof(vecStrTmp[2].c_str())*PI/180;
				vec_xOdo.push_back(xOdo);
				vec_yOdo.push_back(yOdo);
				vec_thetaOdo.push_back(thetaOdo);
			}
			break;
		default:
			break;
		}
	}
	int numFrame = vec_timeOdo.size();

	// init record
	ofstream recFileMk;
	ofstream recFileOdo;
	mkdir("../rec");
	string strRecFileMkPath = "../rec/Mk.rec";
	string strRecFileOdoPath = "../rec/Odo.rec";
	recFileMk.clear();
	recFileMk.open(strRecFileMkPath.c_str(),ios::out);
	recFileMk << "# aruco mark observation info" << endl;
	recFileMk << "# format: lp id rvec(x y z) tvec(x y z) ptimg(x1 y1 x2 y2 x3 y3 x4 y4)" << endl;

	recFileOdo.clear();
	recFileOdo.open(strRecFileOdoPath.c_str(),ios::out);
	recFileOdo << "# odometry info" << endl;
	recFileOdo << "# format: lp timeOdo timeCam x y theta" << endl;

	// init video record

	VideoWriter writer("../rec/Rec.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(640, 480)); 

	// main loops
	for (int lp = 0; lp < numFrame-1; lp ++) {

		// synchronization
		double xOdoSyn, yOdoSyn, thetaOdoSyn;
		xOdoSyn = Interpolate(vec_timeOdo[lp], vec_xOdo[lp], vec_timeOdo[lp+1], vec_xOdo[lp+1], vec_timeCam[lp]);
		yOdoSyn = Interpolate(vec_timeOdo[lp], vec_yOdo[lp], vec_timeOdo[lp+1], vec_yOdo[lp+1], vec_timeCam[lp]);
		double theta1, theta2;
		theta1 = vec_thetaOdo[lp];
		theta2 = vec_thetaOdo[lp+1];
		theta2 = PeriodCnstr(theta2, theta1-PI, theta1+PI);
		thetaOdoSyn = Interpolate(vec_timeOdo[lp], theta1, vec_timeOdo[lp+1], theta2, vec_timeCam[lp]);
		thetaOdoSyn = PeriodCnstr(thetaOdoSyn, -PI, PI);

		// write into recOdo
		recFileOdo << lp << " ";
		recFileOdo << vec_timeOdo[lp] << " ";
		recFileOdo << vec_timeCam[lp] << " ";
// 		recFileOdo << vec_xOdo[lp] << " ";
// 		recFileOdo << vec_yOdo[lp] << " ";
// 		recFileOdo << vec_thetaOdo[lp] << " ";
		recFileOdo << xOdoSyn << " " << yOdoSyn << " " << thetaOdoSyn << " ";
		recFileOdo << endl;

		// load image
		char localImgFileName[10];
		sprintf(localImgFileName, "%d.bmp", lp);
		string strImgFileName = localImgFileName;
		string strImgFullPath = strImgPath+strImgFileName;
		im = imread(strImgFullPath);

		// detect aruco
		MDetector.detect(im,Markers,CamParam,MarkerSize);
		//for each marker, draw info and its boundaries in the image
		for (unsigned int i=0;i<Markers.size();i++) {
// 			cout << Markers[i].id << ' ';
// 			cout << Markers[i].Rvec.at<float>(0,0) << ' ' << Markers[i].Rvec.at<float>(1,0) << ' ' << Markers[i].Rvec.at<float>(2,0) << ' ';
// 			cout << Markers[i].Tvec.at<float>(0,0) << ' ' << Markers[i].Tvec.at<float>(1,0) << ' ' << Markers[i].Tvec.at<float>(2,0) << ' ';
// 			cout << Markers[i][0].x << ' ' << Markers[i][0].y << ' ';
// 			cout << Markers[i][1].x << ' ' << Markers[i][1].y << ' ';
// 			cout << Markers[i][2].x << ' ' << Markers[i][2].y << ' ';
// 			cout << Markers[i][3].x << ' ' << Markers[i][3].y << ' ';
// 			cout << endl;

			// write into recMk
			recFileMk << lp << " ";
			recFileMk << Markers[i].id << ' ';
			recFileMk << Markers[i].Rvec.at<float>(0,0) << ' ' << Markers[i].Rvec.at<float>(1,0) << ' ' << Markers[i].Rvec.at<float>(2,0) << ' ';
			recFileMk << Markers[i].Tvec.at<float>(0,0) << ' ' << Markers[i].Tvec.at<float>(1,0) << ' ' << Markers[i].Tvec.at<float>(2,0) << ' ';
			recFileMk << Markers[i][0].x << ' ' << Markers[i][0].y << ' ';
			recFileMk << Markers[i][1].x << ' ' << Markers[i][1].y << ' ';
			recFileMk << Markers[i][2].x << ' ' << Markers[i][2].y << ' ';
			recFileMk << Markers[i][3].x << ' ' << Markers[i][3].y << ' ';
			recFileMk << endl;

			// draw marker detection
			Markers[i].draw(im,Scalar(0,0,255),2);
		}

		cout << "lp:" << lp << endl;
		imshow("im", im);
//		writer << im;
//		imshow("thresh", MDetector.getThresholdedImage());
		waitKey(3);
	}	

	// end
	cout << "end ..." << endl;
	recFileMk.close();
	recFileOdo.close();
	writer.release();
	waitKey();
	return 0;
}
