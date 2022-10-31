#include <iostream>
#include <vector>
#include <string>
#include "opencv.hpp"
#include "inference_engine.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <fstream> 
#include <io.h>
#include <stdio.h>
#include <direct.h>
#include <windows.h>
#include <algorithm>
 

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;


int isExist(string Path);
Mat manage(Mat Imgbw);


//read model 
int readModel(std::string binPath, std::string xmlPath, InferenceEngine::CNNNetwork& model)
{
	InferenceEngine::Core ie;
	if (binPath.empty() || xmlPath.empty())
	{
		return -1;
	}
	try
	{
		model = ie.ReadNetwork(xmlPath, binPath);
	}
	catch (...)
	{
		return -1;
	}

	return 0;
}

//模型识别
Mat predict(InferenceEngine::CNNNetwork& model,
	InferenceEngine::InferRequest& infer_request,
	cv::Mat& src_image)
{
	std::string input_name = model.getInputsInfo().begin()->first;
	std::string output_name = model.getOutputsInfo().begin()->first;

	InferenceEngine::Blob::Ptr input = infer_request.GetBlob(input_name);
	InferenceEngine::Blob::Ptr output = infer_request.GetBlob(output_name);

	Mat image, Mask;
	int image_width = src_image.cols;
	int image_height = src_image.rows;

	cv::resize(src_image, image, cv::Size(input->getTensorDesc().getDims()[2],
		input->getTensorDesc().getDims()[3]));

	cv::resize(src_image, Mask, cv::Size(input->getTensorDesc().getDims()[2],
		input->getTensorDesc().getDims()[3]));

	size_t channels_number = input->getTensorDesc().getDims()[1];
	size_t image_size = input->getTensorDesc().getDims()[3] * input->getTensorDesc().getDims()[2];
	size_t h = input->getTensorDesc().getDims()[2];
	size_t w = input->getTensorDesc().getDims()[3];

	auto input_data = input->buffer()
		.as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
	float mean[3] = { 113.32432f,121.01931f,134.7733f };
	float std[3] = { 64.109604f,64.60054f,67.17994f };
	for (size_t row = 0; row < h; row++)
	{
		for (size_t col = 0; col < w; col++) {
			for (size_t ch = 0; ch < channels_number; ++ch)
			{
				input_data[image_size * ch + row * w + col] = ((image.at<cv::Vec3b>(row, col)[ch]-mean[ch])/std[ch])/ 255.0f;
			}
		}
	}

	infer_request.Infer();

	auto output_data = output->buffer().
		as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

	//cv::resize(src_image, src_image, cv::Size(input->getTensorDesc().getDims()[2],
	//	input->getTensorDesc().getDims()[3]));

	for (int row = 0; row < h; row++)
	{
		for (int col = 0; col < w; col++) {
			if (output_data[row * w + col] > 0) {
				Mask.at<cv::Vec3b>(row, col) = Vec3b(255, 255, 255);
			}
			else
			{
				Mask.at<cv::Vec3b>(row, col) = Vec3b(0, 0, 0);
			}
		}
	}

	Mask = manage(Mask);

	cv::resize(Mask, Mask, cv::Size(image_width,image_height));


	for (int row = 0; row < image_height; row++)
	{
		for (int col = 0; col < image_width; col++) {
			if (Mask.at<cv::Vec3b>(row, col)[0] == 0) {
				src_image.at<cv::Vec3b>(row, col) = Vec3b(255,255,255);
			}
		}
	}

	//cv::resize(src_image, src_image, cv::Size(image_width, image_height));

	cv::cvtColor(src_image, src_image, cv::COLOR_BGR2RGB);

	return Mask;
}

Mat manage(Mat Imgbw) {
	 
	Mat Imglabels, Imgstats, Imgcentriods,Temp;
	Temp = Imgbw;
	cvtColor(Imgbw, Imgbw, COLOR_BGR2GRAY);

	
	int Imglabelnum = cv::connectedComponentsWithStats(Imgbw, Imglabels, Imgstats, Imgcentriods);//返回连通域的数量
												//输入图片，输出图片，输出图片信息  输出图片重心
	
	//Imgstats包含了一些信息，如果想知道标签为i的连通域的一些信息，可以如下访问
	
	//double left = Imgstats.at<int>(i, CC_STAT_LEFT); //连通域的boundingbox的最左边
	//double top = Imgstats.at<int>(i, CC_STAT_TOP);//连通域的boundingbox的最上边
	//double width = Imgstats.at<int>(i, CC_STAT_WIDTH);//连通域的宽
	//double height Imgstats.at<int>(i, CC_STAT_HEIGHT);//连通域的高

	//Point2f pt;  //pt就是重心
	int area=0,area2=0;
	int area_big_tag=0, area_big_tag2=0;
	bool flag=true;
	cv::Mat output = cv::Mat(Imgbw.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	if (Imglabelnum>1)
	{	
		for (size_t i = 0; i < Imglabelnum; i++)
		{
			if (area < Imgstats.at<int>(i, CC_STAT_AREA)) {
				area = Imgstats.at<int>(i, CC_STAT_AREA);//连通域的面积
				area_big_tag2 = area_big_tag;
				area_big_tag = i;
				if (i != 0) {
					area2 = Imgstats.at<int>(area_big_tag2, CC_STAT_AREA);
				}	
			}
			else if(area2 < Imgstats.at<int>(i, CC_STAT_AREA))
			{
				area2 = Imgstats.at<int>(i, CC_STAT_AREA);
				area_big_tag2 = i;
			}
		}
		

		for (int y = 0; y < Imgbw.rows; y++)
		{
			for (int x = 0; x < Imgbw.cols; x++)
			{

				if (flag){
					if (Temp.at<cv::Vec3b>(y, x) == Vec3b(0, 0, 0)) {
						if (Imglabels.at<int>(y, x) == area_big_tag)
						{
							area_big_tag = area_big_tag2;
							flag = false;
						}
						else if(Imglabels.at<int>(y, x) == area_big_tag2)
						{
							flag = false;
						}
					}
				}

				if (Imglabels.at<int>(y, x) == area_big_tag) {
					output.at<cv::Vec3b>(y, x) = Vec3b(255, 255, 255);
				}
			}
			
		}
	}


	return output;
}

int initiateModel(std::string binPath, std::string xmlPath,
	InferenceEngine::CNNNetwork& model,
	InferenceEngine::InferRequest& infer_request)
{
	InferenceEngine::Core ie;


	int readFlag = readModel(binPath, xmlPath, model);
	if (readFlag == -1)
	{
		//read model failed
		return -1;
	}

	//prepare input blobs
	InferenceEngine::InputInfo::Ptr input_info = model.getInputsInfo().begin()->second;;
	std::string input_name = model.getInputsInfo().begin()->first;
	input_info->setLayout(InferenceEngine::Layout::NCHW);
	input_info->setPrecision(InferenceEngine::Precision::FP32);

	//prepare output blobs
	InferenceEngine::DataPtr output_info = model.getOutputsInfo().begin()->second;
	std::string output_name = model.getOutputsInfo().begin()->first;
	output_info->setPrecision(InferenceEngine::Precision::FP32);

	//load model to device
	InferenceEngine::ExecutableNetwork 	executable_network = ie.LoadNetwork(model, "CPU");

	//create infer request
	infer_request = executable_network.CreateInferRequest();

	cout << "初始化模型成功" << endl;
	return 0;
}

int getpic(InferenceEngine::CNNNetwork model, InferenceEngine::InferRequest infer_request,std::string imagePath,std::string outPath[])//cv::Mat image
{

	cv::Mat image = cv::imread(imagePath);
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	if (image.empty()) {
		std::cout << "图片为空！" << std::endl;
		return -1;
	}
	imwrite(outPath[1], predict(model, infer_request, image));
	imwrite(outPath[0], image);
	return 0;
}

int getAllImagePath(std::string path, std::vector<cv::String>& imagePathList)
{
	cv::glob(path, imagePathList);
	return 0;
}


int speedTest(InferenceEngine::CNNNetwork model, InferenceEngine::InferRequest infer_request,std::string imagePath)//cv::Mat image
{
	std::vector<cv::String> imagePathList;

	cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	if (image.empty()) {
		return -1;
	}

	int num = 1000;
	clock_t startTime = clock();
	for (int i = 0; i < num; i++) 	predict(model, infer_request, image);
	clock_t endTime = clock();

	std::cout << "inference one image " << num << "times run time: " << double(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms\n";
	return 0;
}

int getVideo(InferenceEngine::CNNNetwork model,InferenceEngine::InferRequest infer_request ,string videoPath,string outputPath[]) {

	//打开视频文件：其实就是建立一个VideoCapture结构
	VideoCapture capture(videoPath);
	//检测是否正常打开:成功打开时，isOpened返回ture
	if (!capture.isOpened()) {
		cout << "fail to open!" << endl;
		return -1;
	}
		

	//获取整个帧数
	long totalFrameNumber = capture.get(CAP_PROP_FRAME_COUNT);
	
	cout << "整个视频共" << totalFrameNumber << "帧" << endl;


	//设置开始帧()
	long frameToStart = 1;
	capture.set(CAP_PROP_POS_FRAMES, frameToStart);
	


	if (totalFrameNumber < frameToStart)
	{
		cout << "结束帧小于开始帧，程序错误，即将退出！" << endl;
		return -1;
	}

	//获取帧率
	double rate = capture.get(CAP_PROP_FPS);


	//定义一个用来控制读取视频循环结束的变量
	bool stop = false;

	//承载每一帧的图像
	Mat frame,frame_mask;
	
	Size size;
	size.width = capture.get(CAP_PROP_FRAME_WIDTH);
	size.height = capture.get(CAP_PROP_FRAME_HEIGHT);
	//显示每一帧的窗口
	//namedWindow( "Extractedframe" );

	//两帧间的间隔时间:
	//int delay = 1000/rate;
	double delay = 1000 / rate;


	//利用while循环读取帧
	//currentFrame是在循环体中控制读取到指定的帧后循环结束的变量
	long currentFrame = frameToStart;
	
	VideoWriter video(outputPath[0], CAP_OPENCV_MJPEG, rate, size, true);//mp4可行，avi不
	VideoWriter video_mask(outputPath[1], CAP_OPENCV_MJPEG, rate, size, true);
	DWORD start_time = GetTickCount();
	
	while (currentFrame	<= totalFrameNumber)
	{
		capture >> frame;
		if (frame.empty()) {
			cout << "视频结束" << endl;
			break;
		}

		cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
		frame_mask = predict(model, infer_request, frame);

		video_mask.write(frame_mask);
		// 流操作符，把图片传入视频
		video.write(frame);
		
		currentFrame++;
		cv::imshow("video", frame);
		cv::waitKey(1);

	}

	DWORD end_time = GetTickCount();

	cout << "识别FPS: " << 1000/((end_time - start_time)/totalFrameNumber) << " FPS" << endl;

	//关闭视频文件
	capture.release();
	video.release();
	//waitKey(0);
	return 0;
}

//文件不存在返回-1，文件夹返回1，文件返回2
int isExist(string Path) {
	int mode = 0; // 判定目录或文件是否存在的标识符
	try
	{
		if (Path.find_last_of(".")==-1) {
			_mkdir(Path.c_str());
		}
	}
	catch (const std::exception&)
	{
		cout << "创建文件夹失败！" << endl;
	}
	if (_access(Path.c_str(), mode))
	{
		//system("mkdir head");
		cout << "文件或文件夹不存在！" << endl;
		return -1;
	}

	WIN32_FIND_DATAA FindFileData;
	FindFirstFileA(Path.c_str(), &FindFileData);
	//FindFirstFileA(tempPath,&FindFileData);
	if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
	{
		//Folder
		cout << "Folder" << endl;
		return 1;
	}
	else
	{
		//File
		cout << "File" << endl;
		return 2;
	}
}


void picToVideo() {

	// 从一个文件夹下读取多张jpg图片
	String pattern = "F:\\test_pic\\*.png";

	vector<String> fn;
	//读取文件下所有文件
	glob(pattern, fn, false);

	size_t count = fn.size();

	Mat src0 = imread(fn[0]);
	Size size = src0.size();
	// 构造一个VideoWriter
	VideoWriter video("test.mp4", CAP_OPENCV_MJPEG, 25.0, size);//mp4可行，avi不行
	
	for (size_t i = 0; i < count; i++)
	{
		Mat image = imread(fn[i]);
		// 这个大小与VideoWriter构造函数中的大小一致。
		resize(image, image, size);
		// 流操作符，把图片传入视频
		video << image;
	}
	cout << "处理完毕！" << endl;
	// 处理完之后会在得到一个名为test.avi的视频文件。


}

//返回1时文件是图片,返回-1时文件时MP4
int getFileName(string videoPath,string output[]) {

	int pos2 = videoPath.find_last_of('.');
	string FileType(videoPath.substr(pos2+1,videoPath.size()));
	string FileName(videoPath.substr(0,pos2));

	if (FileType == "jpg" || FileType == "png" || FileType == "bmp") {
		output[0] = FileName + "_result.jpg";
		output[1] = FileName + "_mask.jpg";
		return 1;
	}
	if (FileType == "avi" || FileType == "mp4")
	{
		output[0] = FileName + "_result.mp4";
		output[1] = FileName + "_mask.mp4";
		return -1;
	}
}

void getFiles(string path, vector<string>& files)
{
	//文件句柄，此处intptr_t是用于跨平台运行，
	//例如原本在32位电脑所用文件地址编码long即可，
	//但在64位电脑中运行则需要long long而我们使用intptr_t即可解决这一问题
	intptr_t  hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;

	if ((hFile = _findfirst(p.assign(path).append("*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表s
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


int main(int args,char* argv[])
{

	std::string binPath = "model.bin";
	std::string xmlPath = "model.xml";
	vector<string> FileList;
	string FileName[2];
	string Path;
	
	namedWindow("video", 0);
	resizeWindow("video", 1280, 720);
	
	if (args > 1) {
		Path = argv[1];
		cout << Path << endl;
		getFiles(Path, FileList);
	}
	
	InferenceEngine::CNNNetwork model;
	InferenceEngine::InferRequest infer_request;

	if (initiateModel(binPath, xmlPath, model, infer_request)) {
		std::cout << "读取模型失败，模型初始化失败！" << std::endl;
		return -1;
	}

	for (size_t i = 0; i < FileList.size(); i++)
	{
		if (getFileName(FileList[i], FileName)==-1) {
			cout << FileList[i] << endl;
			getVideo(model, infer_request, FileList[i], FileName);
		}
		else
		{
			getpic(model, infer_request, FileList[i], FileName);
		}
		
	}

	waitKey(0);
	system("pause");
}

