/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_ThermalClassifier.cpp
* @author  	SL
* @version 	1
* @date    	2017-11-09
* @brief    classifies thermal blobs using a SVM
*****************************************************************************
**/

/* LBP code liscence
Copyright(c) 2011, philipp <bytefish[at]gmx.de>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met :
*Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the organization nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED.IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "GlobalIncludes.h"

#include "DidoAnalytics_ThermalClassifier.h"
#include "DidoMain_recording.h"


namespace overview
{
	

	//LBP implimentation using bilinear interpolation
	static void ELBP_(const cv::Mat& src, cv::Mat& dst, int radius) {

		int neighbors = 8; // set bounds...

		dst = cv::Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1);
		for (int n = 0; n< neighbors; n++) {
			// sample points
			float x = static_cast<float>(radius) * cosf(2.0f*pi*n / static_cast<float>(neighbors));
			float y = static_cast<float>(radius) * -sinf(2.0f*pi*n / static_cast<float>(neighbors));
			// relative indices
			int fx = static_cast<int>(floor(x));
			int fy = static_cast<int>(floor(y));
			int cx = static_cast<int>(ceil(x));
			int cy = static_cast<int>(ceil(y));
			// fractional part
			float ty = y - fy;
			float tx = x - fx;
			// set interpolation weights
			float w1 = (1 - tx) * (1 - ty);
			float w2 = tx  * (1 - ty);
			float w3 = (1 - tx) *      ty;
			float w4 = tx  *      ty;
			// iterate through your data
			for (int i = radius; i < src.rows - radius; i++) {
				for (int j = radius; j < src.cols - radius; j++) {
					float t = w1*src.at<thermalType>(i + fy, j + fx) + w2*src.at<thermalType>(i + fy, j + cx) + w3*src.at<thermalType>(i + cy, j + fx) + w4*src.at<thermalType>(i + cy, j + cx);
					// we are dealing with floating point precision, so add some little tolerance
					dst.at<unsigned char>(i - radius, j - radius) += ((t > src.at<thermalType>(i, j)) && (abs(t - src.at<thermalType>(i, j)) > std::numeric_limits<float>::epsilon())) << n;
				}
			}
		}
	}


	//to normalise our results closer to the 0-1 range
	constexpr float descriptorscaling = 1.3f;
	DidoAnalytics_ThermalClassifier::thermalFeatures DidoAnalytics_ThermalClassifier::generateFeatures(const cv::Mat & img)
	{
		//precalculated uniformity check - could be replaced with a rotation invariant feature if we prefer
		static unsigned char lbpmap[256] = { 0,2,3,4,5,1,6,7,8,1,1,1,9,1,10,11,12,1,1,1,1,1,1,1,13,1,1,1,14,1,15,16,17,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,18,1,1,1,1,1,1,1,19,1,1,1,20,1,21,22,23,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,24,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,25,1,1,1,1,1,1,1,26,1,1,1,27,1,28,29,30,31,1,32,1,1,1,33,1,1,1,1,1,1,1,34,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,35,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,36,37,38,1,39,1,1,1,40,1,1,1,1,1,1,1,41,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,42,43,44,1,45,1,1,1,46,1,1,1,1,1,1,1,47,48,49,1,50,1,1,1,51,52,53,1,54,55,56,57,58 };

		DidoAnalytics_ThermalClassifier::thermalFeatures rval;
		if (img.empty()) return rval;
		//scale then pad the image to the correct size
		cv::Mat padded = img;
		float scalefactor = std::min((float)patchRows / padded.rows, (float)patchCols / padded.cols);
		cv::resize(padded, padded, cv::Size((int)ceil(padded.cols*scalefactor), (int)ceil(padded.rows*scalefactor)));

		int hpad = patchCols - padded.cols;
		int vpad = patchRows - padded.rows;
		//we have edge and texture features, so we want the border to be as similar to the edge of the image but also constant
		cv::copyMakeBorder(padded, padded, vpad / 2, vpad - vpad / 2, hpad / 2, hpad - hpad / 2, cv::BORDER_CONSTANT, img.at<thermalType>(0,0));

		//smooth the patch to clean up some noise
		cv::GaussianBlur(padded, padded, cv::Size(3, 3), 1);
		//generate the HOG features for this size
		std::vector<float> descriptors;
		hog.compute(padded, descriptors);
		int rvind = 0;	//allowing us to use libsvm's sparse representation better

		for (int i = 0; i < descriptors.size(); i++)
		{
			if (i >= rval.size())
			{
				LOG_WARN << "size mismatch between generated features and expected size";
				break;
			}
			if (descriptors[i] > 0.00001)
			{
				rval[rvind].index = i;
				rval[rvind++].value = descriptors[i] * descriptorscaling;
			}
		}

		if (useLBP)
		{
			cv::Mat features;
			ELBP_(padded, features, lbpwidth);
			//map the LBP features to the simpler set
			cv::LUT(features, cv::Mat(cv::Size(256, 1), CV_8UC1, &lbpmap[0]), features);

			//pad the features back to size
			cv::copyMakeBorder(features, features, lbpwidth, lbpwidth, lbpwidth, lbpwidth, cv::BORDER_CONSTANT, 0);

			std::vector<cv::Mat> windows;
			std::vector<int> channels, histsize;
			std::vector<const float *> histranges;
			const float histrange[] = { 0,59 };
			for (int r = 0; r < patchRows; r += blockSize)
			{
				for (int c = 0; c < patchCols; c += blockSize)
				{
					windows.push_back(features(cv::Rect(c, r, blockSize, blockSize)));
					channels.push_back(0);
					histsize.push_back(59);
					histranges.push_back(histrange);
				}
			}
			cv::Mat hist;
			//we can't do more than 32 images in a go, so we'll batch them
			for (int i = 0; i < windows.size(); i += 1)
			{
				cv::Mat tmphist;
				cv::calcHist(windows.data() + i, 1, channels.data() + i, cv::Mat(), tmphist, 1, histsize.data() + i, histranges.data() + i);
				if (hist.empty()) hist = tmphist;
				else cv::hconcat(hist, tmphist, hist);
			}
			//remove the zero bins
			hist = hist(cv::Rect(0, 1, hist.cols, hist.rows - 1));

			//normalise the histograms
			hist = hist / (blockSize*blockSize);
			hist.reshape(1, 1);
			for (int i = 0; i < hist.cols; i++)
			{
				if (i + descriptors.size() >= rval.size())
				{
					LOG_WARN << "size mismatch between generated features and expected size";
					break;
				}
				if (hist.at<float>(i) > 0.00001)
				{
					rval[rvind].value = hist.at<float>(i) * descriptorscaling;
					rval[rvind++].index = i + (int)descriptors.size();
				}
			}
		}
		return rval;
	}

	OVTrack_FLClassification DidoAnalytics_ThermalClassifier::classifyBlob(const cv::Mat & img)
	{
		if (!_svm)
		{
			LOG_ERROR << "no SVM defined";
			return OVTrack_FLClassification();
		}
		auto features = generateFeatures(img);
		double probabilities[OVTrack_NFClass];
		int val = (int)svm_predict_probability(_svm, features.data(), probabilities);

		OVTrack_FLClassification rval;
		rval.classvals[static_cast<int>(FirstLevelClassification::Unknown)] = 0;
		rval.classvals[static_cast<int>(FirstLevelClassification::Human)] = 0;
		rval.classvals[static_cast<int>(FirstLevelClassification::Vehicle)] = 0;
		rval.classvals[static_cast<int>(FirstLevelClassification::Animal)] = 0;
		rval.classvals[static_cast<int>(FirstLevelClassification::Object)] = 0;
		for (int i = 0; i < _svm->nr_class; i++)
		{
			rval.classvals[_svm->label[i]] = (float)probabilities[i];
		}


		rval.confidence = confidence;	//???

		return rval;
	}
	DidoAnalytics_ThermalClassifier::~DidoAnalytics_ThermalClassifier()
	{
		if (_svm) svm_free_and_destroy_model(&_svm);
	}
	void DidoAnalytics_ThermalClassifier::save(std::string saveFile)
	{
		svm_save_model(saveFile.c_str(), _svm);
	}
	void DidoAnalytics_ThermalClassifier::load(std::string loadfile)
	{
		try
		{
			if (_svm) svm_free_and_destroy_model(&_svm);
			_svm = svm_load_model(loadfile.c_str());
		}
		catch (std::exception e)
		{
			LOG_ERROR << "failed during loading of SVM save file: " << loadfile;
		//	throw std::runtime_error("couldn't load given SVM file");
		}
	}
	void DidoAnalytics_ThermalClassifier::setConfidence(float cf)
	{
		confidence = cf;
	}

	DidoAnalytics_ThermalClassifier::safe_svm_problem DidoAnalytics_ThermalClassifier::loadTrainData(std::array<std::vector<std::string>, OVTrack_NFClass> labelFolders)
	{
		const int mean = 6;
		const int variance = 2;
		//parse the inputs into one training set
		std::array<std::vector<thermalFeatures>, OVTrack_NFClass> datastore;

		for (int i = 0; i < labelFolders.size(); i++)
		{
			for (auto & folder : labelFolders[i])
			{
				try
				{
					//go through the folder and open every image in it
					for (auto & f : std::experimental::filesystem::directory_iterator(folder))
					{
						try
						{
							cv::Mat img = cv::imread((f.path().generic_string()), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
							datastore[i].push_back(generateFeatures(img));
							//data augmentation
							//flip it
							cv::Mat flipped;
							cv::flip(img, flipped, 1);
							datastore[i].push_back(generateFeatures(flipped));
							//add some noise
							cv::Mat noise = cv::Mat(img.size(), CV_8UC1);
							cv::randn(noise, mean, variance);
							cv::Mat offset = (cv::Mat::ones(img.size(), CV_8UC1) * mean);
							cv::Mat noisy = img - offset + noise;
							datastore[i].push_back(generateFeatures(noisy));
							flipped = flipped - offset + noise;
							datastore[i].push_back(generateFeatures(flipped));
						}
						catch (cv::Exception & e)
						{
							LOG_WARN << "error handling file " << (f.path().generic_string()) << e.what();
						}
					}
				}
				catch (std::exception e)
				{
					LOG_WARN << "error handling folder " << folder << " : " << e.what();
				}
			}
		}

		//we want similar numbers of examples from each class, but not too many duplicates
		int maxExamples = std::max(std::max((int)datastore[static_cast<int>(FirstLevelClassification::Human)].size(), (int)datastore[static_cast<int>(FirstLevelClassification::Vehicle)].size()),std::max((int)datastore[static_cast<int>(FirstLevelClassification::Animal)].size(), (int)datastore[static_cast<int>(FirstLevelClassification::Unknown)].size()));

		for (int cs = 0; cs < OVTrack_NFClass; cs++)
		{
			if (maxExamples > (int)datastore[cs].size())
			{
				datastore[cs].insert(datastore[cs].end(), datastore[cs].begin(),
					datastore[cs].begin() + std::min((maxExamples - (int)datastore[cs].size()), (int)datastore[cs].size()));
			}
		}

		std::vector<double> svmlabels;
		std::vector<thermalFeatures> svmdata;
		for (int i = 0; i < OVTrack_NFClass; i++)
		{
			svmlabels.insert(svmlabels.end(), datastore[i].size(), (double)i);
			svmdata.insert(svmdata.end(), datastore[i].begin(), datastore[i].end());
		}
		safe_svm_problem rval;
		rval.allocateData(svmdata.data(), svmlabels.data(), (int)svmlabels.size());
		return rval;
	}

	DidoAnalytics_ThermalClassifier::safe_svm_problem DidoAnalytics_ThermalClassifier::loadTestData(std::array<std::vector<std::string>, OVTrack_NFClass> labelFolders)
	{
		//parse the inputs into one training set
		std::array<std::vector<thermalFeatures>, OVTrack_NFClass> datastore;

		for (int i = 0; i < labelFolders.size(); i++)
		{
			for (auto & folder : labelFolders[i])
			{
				try
				{
					//go through the folder and open every image in it
					for (auto & f : std::experimental::filesystem::directory_iterator(folder))
					{
						try
						{
							cv::Mat img = cv::imread((f.path().generic_string()), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
							datastore[i].push_back(generateFeatures(img));
						}
						catch (cv::Exception & e)
						{
							LOG_WARN << "error handling file " << (f.path().generic_string()) << e.what();
						}
					}
				}
				catch (std::exception e)
				{
					LOG_WARN << "error handling folder " << folder << " : " << e.what();
				}
			}
		}

		std::vector<double> svmlabels;
		std::vector<thermalFeatures> svmdata;
		for (int i = 0; i < OVTrack_NFClass; i++)
		{
			svmlabels.insert(svmlabels.end(), datastore[i].size(), (double)i);
			svmdata.insert(svmdata.end(), datastore[i].begin(), datastore[i].end());
		}
		safe_svm_problem rval;
		rval.allocateData(svmdata.data(), svmlabels.data(), (int)svmlabels.size());
		return rval;
	}

	void DidoAnalytics_ThermalClassifier::train(const svm_problem & problem)
	{
		if (_svm) svm_free_and_destroy_model(&_svm);

		//train on it
		_svm = svm_train(&problem, &svparameters);
		//save it and load it again to release the pointers to the other data...
		svm_save_model("__modeltemp.xml", _svm);
		svm_free_and_destroy_model(&_svm);
		_svm = svm_load_model("__modeltemp.xml");
	}

	std::vector<float> DidoAnalytics_ThermalClassifier::test(const svm_problem & problem)
	{
		if (!_svm) throw std::runtime_error("svm wasn't defined");
		int npsuccess = 0, nvsuccess = 0, nnsuccess = 0, nasuccess = 0;
		//confusion matrix variables
		int pv = 0, pn = 0, pa = 0;
		int vp = 0, vn = 0, va = 0;
		int ap = 0, av = 0, an = 0;

		int nptrials = 0, nvtrials = 0, nntrials = 0, natrials = 0;
		float pscore = 0, vscore = 0, nscore = 0, ascore = 0;

		auto p = problem;
		for (int i = 0; i < p.l; i++)
		{
			double probabilities[OVTrack_NFClass], sortedps[OVTrack_NFClass];
			int val = (int)svm_predict_probability(_svm, p.x[i], probabilities);
			for (int i = 0; i < _svm->nr_class; i++)
			{
				sortedps[_svm->label[i]] = probabilities[i];
			}

			switch ((int)p.y[i])
			{
			case static_cast<int>(FirstLevelClassification::Unknown) :
				if (sortedps[static_cast<int>(FirstLevelClassification::Unknown)] > 0.6) nnsuccess++;
				nntrials++;
				nscore += (float)sortedps[static_cast<int>(FirstLevelClassification::Unknown)];
				break;
			case static_cast<int>(FirstLevelClassification::Human) :
				if (sortedps[static_cast<int>(FirstLevelClassification::Human)] > 0.6) npsuccess++;
				nptrials++;
				pscore += (float)sortedps[static_cast<int>(FirstLevelClassification::Human)];
				break;
			case static_cast<int>(FirstLevelClassification::Vehicle) :
				if (sortedps[static_cast<int>(FirstLevelClassification::Vehicle)] > 0.6) nvsuccess++;
				nvtrials++;
				vscore += (float)sortedps[static_cast<int>(FirstLevelClassification::Vehicle)];
				break;
			case static_cast<int>(FirstLevelClassification::Animal) :
				if (sortedps[static_cast<int>(FirstLevelClassification::Animal)] > 0.6) nasuccess++;
				natrials++;
				ascore += (float)sortedps[static_cast<int>(FirstLevelClassification::Animal)];
				break;
			default:
				LOG_INFO << "unknown label";
				break;
			}
		}
		LOG_INFO << "Testing Complete";
		LOG_INFO << "there were " << npsuccess << " successful Human classifications out of " << nptrials << " clear examples";
		LOG_INFO << "there were " << nvsuccess << " successful Vehicle classifications out of " << nvtrials << " clear examples";
		LOG_INFO << "there were " << nnsuccess << " successful Other classifications out of " << nntrials << " clear examples";
		LOG_INFO << "there were " << nasuccess << " successful Animal classifications out of " << natrials << " clear examples";
		int ntotsucces = npsuccess + nnsuccess + nvsuccess + nasuccess;
		int ntotrials = nvtrials + nptrials + nntrials + natrials;
		LOG_INFO << "there were " << ntotsucces << " successful classifications out of " << ntotrials << " clear examples";



		std::vector<float> rval;
		/* alternative score criteria
		rval.push_back((float)pscore / nptrials);
		rval.push_back((float)vscore / nvtrials);
		rval.push_back((float)nscore / nntrials);
		rval.push_back((float)ascore / natrials);
		rval.push_back((float)(pscore + vscore + nscore + ascore) / ntotrials);
		/*/

		//*
		rval.push_back((float)npsuccess / nptrials);
		rval.push_back((float)nvsuccess / nvtrials);
		rval.push_back((float)nnsuccess / nntrials);
		rval.push_back((float)nasuccess / natrials);
		rval.push_back((float)ntotsucces / ntotrials);
		//*/
		return rval;
	}
	
	void DidoAnalytics_ThermalClassifier::train(std::array<std::vector<std::string>, OVTrack_NFClass> labelFolders)
	{
		safe_svm_problem problem = loadTrainData(labelFolders);
		train(problem.getProblem());
	}

	std::vector<float> DidoAnalytics_ThermalClassifier::test(std::array<std::vector<std::string>, OVTrack_NFClass> labelFolders)
	{
		std::vector<cv::Mat> datastore;
		std::vector<int> labels;

		for (int i = 0; i < labelFolders.size(); i++)
		{
			for (auto & folder : labelFolders[i])
			{
				try
				{
					//go through the folder and open every image in it
					for (auto & f : std::experimental::filesystem::directory_iterator(folder))
					{
						try
						{
							cv::Mat img = cv::imread((f.path().generic_string()), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
							datastore.push_back((img));
							labels.push_back(i);
						}
						catch (cv::Exception & e)
						{
							LOG_WARN << "error handling file " << (f.path().generic_string()) << e.what();
						}
					}
				}
				catch (std::exception e)
				{
					LOG_WARN << "error handling folder " << folder << " : " << e.what();
				}
			}
		}
		return test(labels, datastore);
	}

	std::vector<float> DidoAnalytics_ThermalClassifier::test(std::vector<int> labels, std::vector<cv::Mat> blobs)
	{
		int npsuccess = 0, nvsuccess = 0, nnsuccess = 0, nasuccess = 0;
		int nptrials = 0, nvtrials = 0, nntrials = 0, natrials = 0;
		for (int i = 0; i < labels.size() && i < blobs.size(); i++)
		{
			auto b = blobs[i];
			int label = labels[i];

			//test it here
			auto cf = classifyBlob(b);
			switch (label)
			{
			case static_cast<int>(FirstLevelClassification::Unknown) :
				if (cf.classvals[static_cast<int>(FirstLevelClassification::Unknown)] > 0.6) nnsuccess++;
				nntrials++;
				break;
			case static_cast<int>(FirstLevelClassification::Human) :
				if (cf.classvals[static_cast<int>(FirstLevelClassification::Human)] > 0.6) npsuccess++;
				nptrials++;
				break;
			case static_cast<int>(FirstLevelClassification::Vehicle) :
				if (cf.classvals[static_cast<int>(FirstLevelClassification::Vehicle)] > 0.6) nvsuccess++;
				nvtrials++;
				break;
			case static_cast<int>(FirstLevelClassification::Animal) :
					if (cf.classvals[static_cast<int>(FirstLevelClassification::Animal)] > 0.6) nasuccess++;
				natrials++;
				break;
			default:
				LOG_INFO << "unknown label";
				break;
			}
		}
		LOG_INFO << "Testing Complete";
		LOG_INFO << "there were " << npsuccess << " successful Human classifications out of " << nptrials << " clear examples";
		LOG_INFO << "there were " << nvsuccess << " successful Vehicle classifications out of " << nvtrials << " clear examples";
		LOG_INFO << "there were " << nnsuccess << " successful Other classifications out of " << nntrials << " clear examples";
		LOG_INFO << "there were " << nasuccess << " successful Animal classifications out of " << natrials << " clear examples";
		int ntotsucces = npsuccess + nnsuccess + nvsuccess + nasuccess;
		int ntotrials = nvtrials + nptrials + nntrials + natrials;
		LOG_INFO << "there were " << ntotsucces << " successful classifications out of " << ntotrials << " clear examples";

		std::vector<float> rval;
		rval.push_back((float)npsuccess / nptrials);
		rval.push_back((float)nvsuccess / nvtrials);
		rval.push_back((float)nnsuccess / nntrials);
		rval.push_back((float)nasuccess / natrials);
		rval.push_back ((float)ntotsucces / ntotrials);
		return rval;
	}

	DidoAnalytics_ThermalClassifier::DidoAnalytics_ThermalClassifier(std::string loadfile, svm_parameter svpars) : svparameters(svpars), _svm(nullptr),
		hog(cv::Size(patchCols,patchRows),cv::Size(blockSize, blockSize),cv::Size(blockStride, blockStride),cv::Size(cellSize, cellSize), numBins)
	{
		try
		{
			_svm = svm_load_model(loadfile.c_str());
	//		if (!_svm) throw std::runtime_error("couldn't load given SVM file");
		}
		catch (std::exception e)
		{
			LOG_ERROR << "failed during loading of thermal SVM save file: " << loadfile;
			throw std::runtime_error("couldn't load given SVM file");
		}
	}
	DidoAnalytics_ThermalClassifier::DidoAnalytics_ThermalClassifier(svm_parameter svpars) : svparameters(svpars), _svm(nullptr),
		hog(cv::Size(patchCols, patchRows), cv::Size(blockSize, blockSize), cv::Size(blockStride, blockStride), cv::Size(cellSize, cellSize), numBins)
	{
	}

	svm_parameter thermalSVMDefaultParameters()
	{
		svm_parameter rval;
		rval.kernel_type = RBF;
		rval.svm_type = C_SVC;
		rval.C = 1.8;
		rval.cache_size = 800;
		rval.coef0 = 0;
		rval.degree = 3;
		rval.gamma = 0.164;
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

	const svm_problem DidoAnalytics_ThermalClassifier::safe_svm_problem::getProblem(int startind, int ndata) const 
	{
		startind = std::min(startind, _ndata - 1);
		if (ndata < 0) ndata = _ndata - startind;
		ndata = std::min(ndata, _ndata);
		svm_problem rval;
		rval.l = ndata;
		rval.y = _labels + startind;
		rval.x = _data + startind;
		return rval;
	}

	DidoAnalytics_ThermalClassifier::safe_svm_problem::~safe_svm_problem()
	{
		if (_data)
		{
			for (int i = 0; i < _ndata; i++)
			{
				if (_data[i]) free(_data[i]);
			}
			free(_data);
		}
		if (_labels) free(_labels);
	}
	DidoAnalytics_ThermalClassifier::safe_svm_problem::safe_svm_problem(safe_svm_problem && mv)
	{
		_data = mv._data;
		_labels = mv._labels;
		_ndata = mv._ndata;
		mv._ndata = 0;
		mv._data = nullptr;
		mv._labels = nullptr;
	}
	DidoAnalytics_ThermalClassifier::safe_svm_problem::safe_svm_problem(const safe_svm_problem & cp)
	{
		_ndata = cp._ndata;
		//we keep a double size store to allow rotations for the slices
		_labels = (double*)malloc(2 * _ndata * sizeof(double));
		memcpy(_labels, cp._labels, _ndata * sizeof(double));
		_data = (svm_node **)malloc(2*_ndata * sizeof(svm_node *));
		for (int i = 0; i < _ndata; i++)
		{
			_data[i] = (svm_node *)malloc(thermalFeatures().size() * sizeof(svm_node));
			memcpy(_data[i], cp._data[i], thermalFeatures().size() * sizeof(svm_node));
		}
		//duplicate the pointers rather than the data
		memcpy(_labels + _ndata, _labels, _ndata * sizeof(double));
		memcpy(_data + _ndata, _data, _ndata* sizeof(svm_node *));
	}
	void DidoAnalytics_ThermalClassifier::safe_svm_problem::allocateData(thermalFeatures * data, double * labels, int ndata)
	{
		if (_data)
		{
			for (int i = 0; i < _ndata; i++)
			{
				if (_data[i]) free(_data[i]);
			}
			free(_data);
		}
		if (_labels) free(_labels);
		//we keep a double size store to allow rotations for the slices
		_labels = (double*)malloc(2* ndata * sizeof(double));
		//shuffle the data
		std::vector<size_t> shuffleinds(ndata);
		_data = (svm_node **)malloc( 2* ndata * sizeof(svm_node *));
		for (int i = 0; i < ndata; i++) shuffleinds[i] = i;
		std::random_shuffle(shuffleinds.begin(), shuffleinds.end());

		for (int i = 0; i < ndata; i++)
		{
			_labels[i] = labels[shuffleinds[i]];
			_data[i] = (svm_node *)malloc(data[i].size() * sizeof(svm_node));
			memcpy(_data[i], data[shuffleinds[i]].data(), data[i].size() * sizeof(svm_node));
		}
		_ndata = ndata;
		//duplicate the pointers rather than the data
		memcpy(_labels + _ndata, _labels, _ndata * sizeof(double));
		memcpy(_data + _ndata, _data, _ndata * sizeof(svm_node *));
	}
}