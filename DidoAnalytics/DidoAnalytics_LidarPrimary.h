/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2018 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_LidarPrimary.h
* @author  	SL
* @version 	1
* @date    	2018-01-22
* @brief   	produces tracks using lidar
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoAnalytics_LidarPrimary
#define __DidoAnalytics_LidarPrimary


#pragma once

#include "DidoAnalytics_LidarBGSUB.h"
#include "DidoMain_ParamEnv.h"
#include "DidoAnalytics_LidarClassifier.h"
#include "DidoAnalytics_Phase2.h"
#include "DidoLidar_primary.h"

/*defines ---------------------------*/

namespace overview
{

	/*inputs are pointers to managed GPU memory sections - they are already allocated with cudaMallocManaged*/
	class DidoAnalytics_LidarPrimary : public DidoAnalytics_Phase2
	{
	protected:

		DidoAnalytics_LidarBGSUB bgsub_lidar;
		DidoAnalytics_LidarClassifier lidClassifier;
		
		//clustering parameter for the lidar object detector
		float epsilon;
		//how many neighbours a point needs to count as core for the DBSCAN clustering for the lidar
		int ncore;
		//minimum lidar points to count as a detection
		int minLidpts;
		
        //the minum probability from the target Kalman filter to allow a proposed association
		float minPassoc;

		//history parameter for the lidar background subtractor
		int lidhist;

		//threshold of confidence in the probability that a target is from an Unknown classification before you reject it
		float rejectionThreshold;
        
		//produces a set of detections from the lidar frame
		virtual	std::vector<std::shared_ptr<OVTrack>> getLidarTargets(const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar, const Dido_Timestamp & now);
        
        std::shared_ptr<DidoLidar_primary> lidar;
        /*the processing loop*/
        void runFunction();
        std::thread runThread;
        bool running = false;
        
        /*que to collect the lidar return data in*/
		std::queue<std::shared_ptr<DidoLidar::DidoLidar_Pano>> lidque;
		std::mutex quemut;
		std::condition_variable quevar;
        
	public:
		virtual ~DidoAnalytics_LidarPrimary() { stop(); };

        virtual void start() override;
        virtual void stop() override;
        
		//sets the rejection threshold
		void setRejectionThreshold(float rt);
		//loads the lidar classification model from the given xml file
		void loadLidarSVMModel(std::string file);
		//saves the lidar classification model from the given xml file
		void saveLidarSVMModel(std::string file);
        
		DidoAnalytics_LidarPrimary(DidoMain_ParamEnv & pars, std::shared_ptr<DidoLidar_primary> _lid);
	};
}
#endif
