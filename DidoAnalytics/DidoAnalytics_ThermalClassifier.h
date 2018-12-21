/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_ThermalClassifier.h
* @author  	SL
* @version 	1
* @date    	2017-11-09
* @brief    classifies thermal blobs using a SVM
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoAnalytics_ThermalClassifier
#define __DidoAnalytics_ThermalClassifier


#pragma once
#include "OVTrack.h"
#include "svm.h"

/*defines ---------------------------*/

namespace overview
{
	//convenience container to allow us to initialise and handle the paramters more easily
	svm_parameter thermalSVMDefaultParameters();

	class DidoAnalytics_ThermalClassifier
	{
	protected:
		svm_model * _svm = nullptr;
		svm_parameter svparameters;
		
		/*LBP parameters*/
		static const bool useLBP = true;
		int lbpwidth = 3;
		static const int lbpbins = 58;

		/*HOG descriptor and paramteres*/
		cv::HOGDescriptor hog;
		static const int patchCols = 64;
		static const int patchRows = 128;
		static const int numBins = 9;		//has to be 9
		static const int blockSize = 16;	//has to be 16	for gpu support
		static const int cellSize = 8;		//has to be 8
		static const int blockStride = 16;

		static const int nfeatures = (1 + (patchCols - blockSize)/ blockStride)*(1 + (patchRows - blockSize)/ blockStride)*(blockSize/cellSize)*(blockSize/cellSize)*numBins + 
			1 + ((useLBP)? lbpbins*(patchCols/blockSize)*(patchRows/blockSize) : 0);
		float confidence = 0.6f;

	public:
		typedef std::array<svm_node, nfeatures> thermalFeatures;

		/*procduces the feature vector for the given image patch*/
		thermalFeatures generateFeatures(const cv::Mat & blob);

		//construct/destruct/move/copyable container for the svm_problem class that reduces the risk of memory leaks
		//only usable for problems with feature vectors that are thermalFeatures
		class safe_svm_problem
		{
			double * _labels;
			svm_node ** _data;
			int _ndata;

		public:
			const svm_problem getProblem(int startind = 0, int size = -1) const;
			int getNdata() const { return _ndata; }

			safe_svm_problem() : _labels(nullptr), _data(nullptr), _ndata(0){}
			~safe_svm_problem();
			safe_svm_problem(safe_svm_problem && mv);
			safe_svm_problem(const safe_svm_problem & cp);
			//data allocation
			void allocateData(thermalFeatures * data, double * labels,  int ndata);
		};

		//parameters that define how to adjust the bounding box of a target before feeding it to this
		//estimation of bad pixels around a thermal image
		static const int borderWidth = 8;
		//extra space used to improve classification quality
		static const int inflateWidth = 30;

		//produces a first level classification for the given blob
		OVTrack_FLClassification classifyBlob(const cv::Mat & blob);
	
		~DidoAnalytics_ThermalClassifier();
	
		//saves the SVM to the given xml file
		void save(std::string saveFile);
		//loads the SVM from the given xml file
		void load(std::string loadFile);

		//defines the expected confidence in the overall classification from the detector
		void setConfidence(float cf);

		//functions for convenience and data caching
		//loads the given dataset of images, extracts their features and stores those features as a SVM problem
		safe_svm_problem loadTestData(std::array<std::vector<std::string>, OVTrack_NFClass> labelFolders);
		//as above, but with data augmentation
		safe_svm_problem loadTrainData(std::array<std::vector<std::string>, OVTrack_NFClass> labelFolders);
		//trains the SVM using the given svm_problem
		void train(const svm_problem & problem);
		//tests the SVM using the given svm_problem
		//returns the %success rate for each class and overall
		std::vector<float> test(const svm_problem & problem);

		//loads the training data then trains the SVM
		void train(std::array<std::vector<std::string>, OVTrack_NFClass> labelFolders);
		//loads the given testing data and tests the SVM on it
		//returns the %success rate for each class and overall
		std::vector<float> test(std::array<std::vector<std::string>, OVTrack_NFClass> labelFolders);
		std::vector<float> test(std::vector<int> labels, std::vector<cv::Mat> blobs);

		//constructor that loads the SVM data
		DidoAnalytics_ThermalClassifier(std::string loadfile, svm_parameter svpars);
		//constructor without loading an SVM
		DidoAnalytics_ThermalClassifier(svm_parameter svpars);
	};
}
#endif
