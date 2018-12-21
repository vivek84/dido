/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_LateFused.h
* @author  	SL
* @version 	1
* @date    	2017-09-27
* @brief   	produces tracks using late fusion of lidar tracks with thermal
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoAnalytics_LateFused
#define __DidoAnalytics_LateFused


#pragma once

#include "OVTrack.h"
#include "DidoAnalytics.h"
#include "DidoAnalytics_LidarBGSUB.h"
#include "DidoMain_ParamEnv.h"
#include "DidoMain_ObjEnv.h"
#include "DidoLidar_CUDA.h"
#include "DidoAnalytics_LidarClassifier.h"
#include "DidoAnalytics_ThermalClassifier.h"

/*defines ---------------------------*/

namespace overview
{

	/*inputs are pointers to managed GPU memory sections - they are already allocated with cudaMallocManaged*/
	class DidoAnalytics_LateFused : public DidoAnalytics
	{
	protected:
		//per pixel range estimation wants this
		DidoLidar_CUDA lsynth;
		float verticalDisplacement, verticalFOV, verticalOffset;

		//per thread sets of GPU allocated data for the dense range estimation
		std::map<std::thread::id, std::vector<DidoLidar_Managed_Pano<float>>> m_range_panos_max, m_range_panos_min;
		std::map<std::thread::id, std::vector<DidoLidar_Managed_Pano<unsigned char>>> m_analytics_pano;

		DidoAnalytics_LidarBGSUB bgsub_lidar;
		DidoAnalytics_LidarClassifier lidClassifier;
		DidoAnalytics_ThermalClassifier thermClassifier;
		
		//clustering parameter for the lidar object detector
		float epsilon;
		//how many neighbours a point needs to count as core for the DBSCAN clustering for the lidar
		int ncore;
		//minimum lidar points to count as a detection
		int minLidpts;
		//minimum area of the thermal bounding box to count as a target
		int minThermPts;
		//paramters for noise reduction and cluster smoothing for the thermal detections
		int erodeWidth, dilateWidth;

		//horizontal field of view per thermal frame in radians
		float width;
		//square of how close a thermal and lidar detection should be to be combined
		float mergedist;
		//the minum probability from the target Kalman filter to allow a proposed association
		float minPassoc;
		//history paramter for the thermal background subtractor
		int hist;
		//history parameter for the lidar background subtractor
		int lidhist;
		//initial variance for the thermal GMM 
		float thermVar;
		//storage for the background subtractors for each thermal angle
		std::map<MapFloat, cv::Ptr<cv::BackgroundSubtractorMOG2>> bgr_therm;
		//tangent of the angle between the top and center of the thermal frame (used for geometric range/height estimation)
		float tanAlpha;

		//threshold of confidence in the probability that a target is from an Unknown classification before you reject it
		float rejectionThreshold;

		//produces a set of detections from the lidar frame
		virtual	std::vector<std::shared_ptr<OVTrack>> getLidarTargets(const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar, const Dido_Timestamp & now);
		//produces a set of detections from the given thermal frame
		virtual std::vector<std::shared_ptr<OVTrack>> getThermalTargets(const std::vector<cv::Mat> & imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar, const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles, const Dido_Timestamp & now);

		/*produces potential tracks from the data and associates using the Kalman Filter*/
		virtual std::vector<std::shared_ptr<OVTrack>> produceTargets(const std::vector<cv::Mat> & imgs , const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar,
			const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles, const Dido_Timestamp & now) override;
		
	public:
		virtual ~DidoAnalytics_LateFused() = default;

		//sets the rejection threshold
		void setRejectionThreshold(float rt);
		//loads the lidar classification model from the given xml file
		void loadLidarSVMModel(std::string file);
		//saves the lidar classification model from the given xml file
		void saveLidarSVMModel(std::string file);

		//loads the thermal classification model from the given xml file
		void loadThermalSVMModel(std::string file);
		//saves the thermal classification model from the given xml file
		void saveThermalSVMModel(std::string file);

		DidoAnalytics_LateFused(DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj);
	};

	//subclass that only uses the lidar input
	class DidoAnalytics_OnlyLidar : public DidoAnalytics_LateFused
	{
	protected:
		//this one doesn't use the thermal at all, so is simpler
		virtual std::vector<std::shared_ptr<OVTrack>> produceTargets(const std::vector<cv::Mat> & imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar,
			const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles, const Dido_Timestamp & now) override;
	public:
		DidoAnalytics_OnlyLidar(DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj);
	};

	//only uses the lidar input and displays the background subtraction results
	class DidoAnalytics_OnlyLidar_debug : public DidoAnalytics_OnlyLidar
	{
	protected:
		float dispScaling;
		int displayWidth;
		virtual	std::vector<std::shared_ptr<OVTrack>> getLidarTargets(const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar, const Dido_Timestamp & now) override;
	public:
		DidoAnalytics_OnlyLidar_debug(DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj, float _dispScaling = 3.0f, int _displayWidth = 600);
	};

	//only uses the thermal
	class DidoAnalytics_OnlyThermal : public DidoAnalytics_LateFused
	{
	protected:
		//without lidar we have to make a hack for the ranges
	virtual std::vector<std::shared_ptr<OVTrack>> getThermalTargets(const std::vector<cv::Mat> & imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar, const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles, const Dido_Timestamp & now);

		//this one doesn't use the lidar at all, so is simpler
		virtual std::vector<std::shared_ptr<OVTrack>> produceTargets(const std::vector<cv::Mat> & imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar,
			const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles, const Dido_Timestamp & now) override;
	public:
		DidoAnalytics_OnlyThermal(DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj);
	};
}
#endif
