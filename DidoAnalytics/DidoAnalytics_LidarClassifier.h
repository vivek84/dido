/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_LidarClassifier.h
* @author  	SL
* @version 	1
* @date    	2017-10-25
* @brief    classifies lidar blobs using a SVM
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoAnalytics_LidarClassifier
#define __DidoAnalytics_LidarClassifier


#pragma once
#include "OVTrack.h"
#include "svm.h"

/*defines ---------------------------*/

namespace overview
{
	//convenience container to allow us to initialise and handle the paramters more easily
	svm_parameter lidarSVMDefaultParameters();

	class DidoAnalytics_LidarClassifier
	{
	protected:
		svm_model * _svm = nullptr;
		svm_parameter svparameters;
		typedef std::array<svm_node, 15> lidarFeatures;
		
		//creates a feature vector from the given cluster of lidar points in either polar or euclidean coordinates
		lidarFeatures generateFeatures(const std::vector<DidoLidar_rangeData> & blob);
		lidarFeatures generateFeatures(const std::vector<OV_WorldPoint> & blob);
		float confidence = 0.5f;
	public:
		//classifies the given cluster of lidar points using the SVM
		OVTrack_FLClassification classifyBlob(const std::vector<DidoLidar_rangeData> & blob);
		OVTrack_FLClassification classifyBlob(const std::vector<OV_WorldPoint> & blob);
	
		~DidoAnalytics_LidarClassifier();
	
		//saves the SVM to the given xml file
		void save(std::string saveFile);
		//loads the SVM from the given xml file
		void load(std::string loadFile);

		//sets the expected overall confidence in this classifier
		void setConfidence(float cf);
		
		//trains the SVM on the blobs loaded from the given files with the given labels
		void train(std::vector<std::string> labelsFiles, std::vector<std::string> blobPrefixs);
		//returns the %success rate for each class and overall
		std::vector<float> test(std::vector<std::string> labelsFiles, std::vector<std::string> blobPrefixs);

		DidoAnalytics_LidarClassifier(svm_parameter svpars);
		//constructor that loads the SVM data
		DidoAnalytics_LidarClassifier(std::string loadfile, svm_parameter svpars);
	};
}
#endif
