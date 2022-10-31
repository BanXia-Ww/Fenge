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

//ģ��ʶ��
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

	
	int Imglabelnum = cv::connectedComponentsWithStats(Imgbw, Imglabels, Imgstats, Imgcentriods);//������ͨ�������
												//����ͼƬ�����ͼƬ�����ͼƬ��Ϣ  ���ͼƬ����
	
	//Imgstats������һЩ��Ϣ�������֪����ǩΪi����ͨ���һЩ��Ϣ���������·���
	
	//double left = Imgstats.at<int>(i, CC_STAT_LEFT); //��ͨ���boundingbox�������
	//double top = Imgstats.at<int>(i, CC_STAT_TOP);//��ͨ���boundingbox�����ϱ�
	//double width = Imgstats.at<int>(i, CC_STAT_WIDTH);//��ͨ��Ŀ�
	//double height Imgstats.at<int>(i, CC_STAT_HEIGHT);//��ͨ��ĸ�

	//Point2f pt;  //pt��������
	int area=0,area2=0;
	int area_big_tag=0, area_big_tag2=0;
	bool flag=true;
	cv::Mat output = cv::Mat(Imgbw.size(), CV_8UC3, cv::Scalar(0, 0, 0));
	if (Imglabelnum>1)
	{	
		for (size_t i = 0; i < Imglabelnum; i++)
		{
			if (area < Imgstats.at<int>(i, CC_STAT_AREA)) {
				area = Imgstats.at<int>(i, CC_STAT_AREA);//��ͨ������
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

	cout << "��ʼ��ģ�ͳɹ�" << endl;
	return 0;
}

int getpic(InferenceEngine::CNNNetwork model, InferenceEngine::InferRequest infer_request,std::string imagePath,std::string outPath[])//cv::Mat image
{

	cv::Mat image = cv::imread(imagePath);
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	if (image.empty()) {
		std::cout << "ͼƬΪ�գ�" << std::endl;
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

	//����Ƶ�ļ�����ʵ���ǽ���һ��VideoCapture�ṹ
	VideoCapture capture(videoPath);
	//����Ƿ�������:�ɹ���ʱ��isOpened����ture
	if (!capture.isOpened()) {
		cout << "fail to open!" << endl;
		return -1;
	}
		

	//��ȡ����֡��
	long totalFrameNumber = capture.get(CAP_PROP_FRAME_COUNT);
	
	cout << "������Ƶ��" << totalFrameNumber << "֡" << endl;


	//���ÿ�ʼ֡()
	long frameToStart = 1;
	capture.set(CAP_PROP_POS_FRAMES, frameToStart);
	


	if (totalFrameNumber < frameToStart)
	{
		cout << "����֡С�ڿ�ʼ֡��������󣬼����˳���" << endl;
		return -1;
	}

	//��ȡ֡��
	double rate = capture.get(CAP_PROP_FPS);


	//����һ���������ƶ�ȡ��Ƶѭ�������ı���
	bool stop = false;

	//����ÿһ֡��ͼ��
	Mat frame,frame_mask;
	
	Size size;
	size.width = capture.get(CAP_PROP_FRAME_WIDTH);
	size.height = capture.get(CAP_PROP_FRAME_HEIGHT);
	//��ʾÿһ֡�Ĵ���
	//namedWindow( "Extractedframe" );

	//��֡��ļ��ʱ��:
	//int delay = 1000/rate;
	double delay = 1000 / rate;


	//����whileѭ����ȡ֡
	//currentFrame����ѭ�����п��ƶ�ȡ��ָ����֡��ѭ�������ı���
	long currentFrame = frameToStart;
	
	VideoWriter video(outputPath[0], CAP_OPENCV_MJPEG, rate, size, true);//mp4���У�avi��
	VideoWriter video_mask(outputPath[1], CAP_OPENCV_MJPEG, rate, size, true);
	DWORD start_time = GetTickCount();
	
	while (currentFrame	<= totalFrameNumber)
	{
		capture >> frame;
		if (frame.empty()) {
			cout << "��Ƶ����" << endl;
			break;
		}

		cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
		frame_mask = predict(model, infer_request, frame);

		video_mask.write(frame_mask);
		// ������������ͼƬ������Ƶ
		video.write(frame);
		
		currentFrame++;
		cv::imshow("video", frame);
		cv::waitKey(1);

	}

	DWORD end_time = GetTickCount();

	cout << "ʶ��FPS: " << 1000/((end_time - start_time)/totalFrameNumber) << " FPS" << endl;

	//�ر���Ƶ�ļ�
	capture.release();
	video.release();
	//waitKey(0);
	return 0;
}

//�ļ������ڷ���-1���ļ��з���1���ļ�����2
int isExist(string Path) {
	int mode = 0; // �ж�Ŀ¼���ļ��Ƿ���ڵı�ʶ��
	try
	{
		if (Path.find_last_of(".")==-1) {
			_mkdir(Path.c_str());
		}
	}
	catch (const std::exception&)
	{
		cout << "�����ļ���ʧ�ܣ�" << endl;
	}
	if (_access(Path.c_str(), mode))
	{
		//system("mkdir head");
		cout << "�ļ����ļ��в����ڣ�" << endl;
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

	// ��һ���ļ����¶�ȡ����jpgͼƬ
	String pattern = "F:\\test_pic\\*.png";

	vector<String> fn;
	//��ȡ�ļ��������ļ�
	glob(pattern, fn, false);

	size_t count = fn.size();

	Mat src0 = imread(fn[0]);
	Size size = src0.size();
	// ����һ��VideoWriter
	VideoWriter video("test.mp4", CAP_OPENCV_MJPEG, 25.0, size);//mp4���У�avi����
	
	for (size_t i = 0; i < count; i++)
	{
		Mat image = imread(fn[i]);
		// �����С��VideoWriter���캯���еĴ�Сһ�¡�
		resize(image, image, size);
		// ������������ͼƬ������Ƶ
		video << image;
	}
	cout << "������ϣ�" << endl;
	// ������֮����ڵõ�һ����Ϊtest.avi����Ƶ�ļ���


}

//����1ʱ�ļ���ͼƬ,����-1ʱ�ļ�ʱMP4
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
	//�ļ�������˴�intptr_t�����ڿ�ƽ̨���У�
	//����ԭ����32λ���������ļ���ַ����long���ɣ�
	//����64λ��������������Ҫlong long������ʹ��intptr_t���ɽ����һ����
	intptr_t  hFile = 0;
	//�ļ���Ϣ
	struct _finddata_t fileinfo;
	string p;

	if ((hFile = _findfirst(p.assign(path).append("*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,����֮
			//�������,�����б�s
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
		std::cout << "��ȡģ��ʧ�ܣ�ģ�ͳ�ʼ��ʧ�ܣ�" << std::endl;
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

