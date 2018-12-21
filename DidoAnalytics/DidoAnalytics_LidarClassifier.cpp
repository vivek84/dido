/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_LidarClassifier.cpp
* @author  	SL
* @version 	1
* @date    	2017-10-25
* @brief    classifies lidar blobs using a SVM
*****************************************************************************
**/

#include "GlobalIncludes.h"

#include "DidoAnalytics_LidarClassifier.h"
#include "DidoMain_lidRecording.h"


namespace overview
{
	DidoAnalytics_LidarClassifier::lidarFeatures DidoAnalytics_LidarClassifier::generateFeatures(const std::vector<DidoLidar_rangeData>& blob)
	{
		std::vector<OV_WorldPoint> wpblob;
		for (auto & p : blob)
		{
			wpblob.push_back(convLidar(p));
		}
		return generateFeatures(wpblob);
	}
	constexpr float LidarClassifierSizeNormalisation = 7.0f;

	DidoAnalytics_LidarClassifier::lidarFeatures DidoAnalytics_LidarClassifier::generateFeatures(const std::vector<OV_WorldPoint>& points)
	{
		lidarFeatures rval;
		if (points.empty()) return rval;
		//calculate the size
		float minx = points[0].x;
		float maxx = points[0].x;
		float miny = points[0].y;
		float maxy = points[0].y;
		float maxz = points[0].z;
		float minz = points[0].z;
		OV_WorldPoint mean(0, 0, 0);
		for (auto & p : points)
		{
			maxx = std::max(p.x, maxx);
			minx = std::min(p.x, minx);
			maxy = std::max(p.y, maxy);
			miny = std::min(p.y, miny);
			maxz = std::max(p.z, maxz);
			minz = std::min(p.z, minz);
			mean += p;
		}
		mean = mean *(1.0f/ points.size());
		float width = sqrt((maxx - minx)*(maxx - minx) + (maxy - miny)*(maxy - miny));
		float height = maxz - minz;

		//calculate covariance
		cv::Mat covariance, pointsMat = cv::Mat(points).reshape(1,(int)points.size()), meanMat = cv::Mat(mean).t(), eigenvalues;
		cv::calcCovarMatrix(pointsMat, covariance, meanMat, CV_COVAR_NORMAL | CV_COVAR_USE_AVG | CV_COVAR_SCALE | CV_COVAR_ROWS, CV_32FC1);
		//calculate the eigenvalues of the covariance
		cv::eigen(covariance, eigenvalues);

		//append them all together
		for(int i = 0; i < 14; i++)	rval[i].index = i +1;
		rval[14].index = -1;
		rval[0].value = std::min(width / LidarClassifierSizeNormalisation, 1.0f);
		rval[1].value = std::min(height / LidarClassifierSizeNormalisation, 1.0f);
		for (int i = 0; i < 9; i++)
		{
			rval[i + 2].value = covariance.at<float>(i) / (width*height + 1);
		}
		for (int i = 0; i < 3; i++)
		{
			rval[i + 11].value = eigenvalues.at<float>(i) / (width*height + 1);
		}
		return rval;
	}


	OVTrack_FLClassification DidoAnalytics_LidarClassifier::classifyBlob(const std::vector<DidoLidar_rangeData>& blob)
	{
		std::vector<OV_WorldPoint> wpblob;
		for (auto & p : blob)
		{
			wpblob.push_back(convLidar(p));
		}
		return classifyBlob(wpblob);
	}

	OVTrack_FLClassification DidoAnalytics_LidarClassifier::classifyBlob(const std::vector<OV_WorldPoint>& blob)
	{
		if (!_svm) throw std::runtime_error("SVM was not defined");
		lidarFeatures features = generateFeatures(blob);
		double probabilities[OVTrack_NFClass];
		int val = (int)svm_predict_probability(_svm, features.data(), probabilities);

		OVTrack_FLClassification rval;
		float sortedprobs[OVTrack_NFClass];
		for (int i = 0; i < _svm->nr_class; i++)
		{
			sortedprobs[_svm->label[i]] = (float)probabilities[i];
		}

		//how do I get a confidence? perhaps from the distance from the support vector?
		rval.classvals[static_cast<int>(FirstLevelClassification::Unknown)] = (float)sortedprobs[0];
		rval.classvals[static_cast<int>(FirstLevelClassification::Human)] = (float)sortedprobs[1];
		rval.classvals[static_cast<int>(FirstLevelClassification::Vehicle)] = (float)sortedprobs[2];
		rval.classvals[static_cast<int>(FirstLevelClassification::Animal)] = 0;
		rval.classvals[static_cast<int>(FirstLevelClassification::Object)] = 0;
		rval.confidence = confidence;	//???

		return rval;
	}
	DidoAnalytics_LidarClassifier::~DidoAnalytics_LidarClassifier()
	{
		if (_svm) svm_free_and_destroy_model(&_svm);
	}
	void DidoAnalytics_LidarClassifier::save(std::string saveFile)
	{
		svm_save_model(saveFile.c_str(), _svm);
	}
	void DidoAnalytics_LidarClassifier::load(std::string loadfile)
	{
		try
		{
			if (_svm) svm_free_and_destroy_model(&_svm);
			_svm = svm_load_model(loadfile.c_str());
		}
		catch (std::exception e)
		{
			LOG_ERROR << "failed during loading of SVM save file: " << loadfile;
			throw std::runtime_error("couldn't load given SVM file");
		}
	}
	void DidoAnalytics_LidarClassifier::setConfidence(float cf)
	{
		confidence = cf;
	}

	void DidoAnalytics_LidarClassifier::train(std::vector<std::string> labelsFiles, std::vector<std::string> blobPrefixs)
	{
		std::vector<double> svmlabels;
		//parse the inputs into one training set
		std::vector<DidoAnalytics_LidarClassifier::lidarFeatures> datastore;
		//using repetition to compensate for imbalanced data (augmentation will be done to the files instead)
		int npositive = 0;
		int nnegative = 0;
		lidarFeatures lastpositive;
		int lastlabel = 0;

		for (int i = 0; i < labelsFiles.size() && i < blobPrefixs.size(); i++)
		{
			std::fstream lfile(labelsFiles[i], std::ios::in);
			if (lfile.is_open())
			{
				std::string line;
				int j = 0;
				while (getline(lfile, line))
				{
				//each line is a new frame of the input
					std::stringstream linestream(line);
					std::fstream bfile(blobPrefixs[i] + std::to_string(j++) + ".lcluster", std::ios::in | std::ios::binary);
					if (bfile.is_open())
					{
						try
						{
							auto blobs = recording::readBlobClusters(bfile);
							for (auto & b : blobs)
							{
								int label;
								char delim;
								linestream >> label; 
								if(linestream.bad()) break;
								if (label < 0) continue;	//features labelled -1 were unable to be clearly labelled

								linestream >> delim;
								lidarFeatures features = generateFeatures(b);
								if (label == 0)
								{
									if (nnegative - 10 > npositive && !lastpositive.empty() )
									{
										datastore.push_back(lastpositive);
										svmlabels.push_back(lastlabel);

									}
									nnegative++;
								}
								else
								{
									lastpositive = features;
									lastlabel = label;
									npositive++;
								}
								datastore.push_back(features);
								svmlabels.push_back(label);
							}
						}
						catch (std::exception e)
						{
							LOG_ERROR << "error whilst parsing blob " << blobPrefixs[i] << std::to_string(j - 1) << ".lcluster : \n" << e.what();				
						}
						bfile.close();
					}
					else LOG_WARN << "failed to open cluster file " << blobPrefixs[i] << std::to_string(j - 1) << ".lcluster";
				}
				lfile.close();
			}
			else LOG_WARN << "failed to open label file " << labelsFiles[i];
		}
		if (datastore.empty() || svmlabels.empty()) throw std::runtime_error("there was no data in the training set");
		if (_svm) svm_free_and_destroy_model(&_svm);
		svm_problem trainingData;
		trainingData.l = (int)svmlabels.size();
		trainingData.y = svmlabels.data();
		try
		{
			trainingData.x = (svm_node **)malloc(datastore.size() * sizeof(svm_node *));
			for (size_t i = 0; i < datastore.size(); i++) trainingData.x[i] = datastore[i].data();
			//train on it
			_svm = svm_train(&trainingData, &svparameters);
			//save it and load it again to release the pointers to the other data...
			svm_save_model("__modeltemp.xml", _svm);
			svm_free_and_destroy_model(&_svm);
			_svm = svm_load_model("__modeltemp.xml");
			free(trainingData.x);
		}
		catch (std::exception e)
		{
			if (trainingData.x) free(trainingData.x);
			throw e;
		}
	}
	std::vector<float> DidoAnalytics_LidarClassifier::test(std::vector<std::string> labelsFiles, std::vector<std::string> blobPrefixs)
	{
		int npsuccess = 0, nvsuccess = 0, nnsuccess = 0;
		int nptrials = 0, nvtrials = 0, nntrials = 0;
		for (int i = 0; i < labelsFiles.size() && i < blobPrefixs.size(); i++)
		{
			std::fstream lfile(labelsFiles[i], std::ios::in);
			if (lfile.is_open())
			{
				std::string line;
				int j = 0;
				while (getline(lfile, line))
				{
					//each line is a new frame of the input
					std::stringstream linestream(line);
					std::fstream bfile(blobPrefixs[i] + std::to_string(j++) + ".lcluster", std::ios::in | std::ios::binary);
					if (bfile.is_open())
					{
						try
						{
							auto blobs = recording::readBlobClusters(bfile);
							for (auto & b : blobs)
							{
								int label;
								char delim;
								linestream >> label;
								if (linestream.bad()) break;
								if (label < 0) continue;	//features labelled -1 were unable to be clearly labelled
								linestream >> delim;

								//test it here
								auto cf = classifyBlob(b);
								switch (label)
								{
								case 0:
									if (cf.classvals[static_cast<int>(FirstLevelClassification::Unknown)] > 0.5) nnsuccess++;
									nntrials++;
									break;
								case 1:
									if (cf.classvals[static_cast<int>(FirstLevelClassification::Human)] > 0.5) npsuccess++;
									nptrials++;
									break;
								case 2:

									if (cf.classvals[static_cast<int>(FirstLevelClassification::Vehicle)] > 0.5) nvsuccess++;
									nvtrials++;
									break;
								default:
									LOG_INFO << "unknown label";
									break;
								}
							}
						}
						catch (std::exception e)
						{
							LOG_ERROR << "error whilst parsing blob " << blobPrefixs[i] << std::to_string(j - 1) << ".lcluster : \n" << e.what();
						}
						bfile.close();
					}
					else LOG_WARN << "failed to open cluster file " << blobPrefixs[i] << std::to_string(j - 1) << ".lcluster";
				}
				lfile.close();
			}
			else LOG_WARN << "failed to open label file " << labelsFiles[i];
		}
		LOG_INFO << "Testing Complete";
		LOG_INFO << "there were " << npsuccess << " successful Human classifications out of " << nptrials << " clear examples";
		LOG_INFO << "there were " << nvsuccess << " successful Vehicle classifications out of " << nvtrials << " clear examples";
		LOG_INFO << "there were " << nnsuccess << " successful Other classifications out of " << nntrials << " clear examples";
		int ntotsucces = npsuccess + nnsuccess + nvsuccess;
		int ntotrials = nvtrials + nptrials + nntrials;
		LOG_INFO << "there were " << ntotsucces << " successful classifications out of " << ntotrials << " clear examples";

		std::vector<float> rval;
		rval.push_back((float)npsuccess / nptrials);
		rval.push_back((float)nvsuccess / nvtrials);
		rval.push_back((float)nnsuccess / nntrials);
		rval.push_back ((float)ntotsucces / ntotrials);
		return rval;
	}
	DidoAnalytics_LidarClassifier::DidoAnalytics_LidarClassifier(svm_parameter svpars) : svparameters(svpars), _svm(nullptr)
	{

	}

	DidoAnalytics_LidarClassifier::DidoAnalytics_LidarClassifier(std::string loadfile, svm_parameter svpars) : svparameters(svpars), _svm(nullptr)
	{
		try
		{
			_svm = svm_load_model(loadfile.c_str());

			if (!_svm) throw std::runtime_error("couldn't load given SVM file");
		}
		catch (std::exception e)
		{
			LOG_ERROR << "failed during loading of lidar SVM save file: " << loadfile;
			throw std::runtime_error("couldn't load given SVM file");
		}
	}
	svm_parameter lidarSVMDefaultParameters()
	{
		svm_parameter rval;
		rval.kernel_type = RBF;
		rval.svm_type = C_SVC;
		rval.C = 1.8;
		rval.cache_size = 800;
		rval.coef0 = 0;
		rval.degree = 3;
		rval.gamma = 3.0;
		rval.nu = 0.5;
		rval.eps = 1e-3;
		rval.p = 0.1;
		rval.shrinking = 0;
		rval.probability = 1;
		rval.nr_weight = 0;
		rval.weight_label = nullptr;
		rval.weight = nullptr;
		return rval;
	}
}