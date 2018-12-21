/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoFusedAnalytics_Overview.h
* @author  	SL
* @version 	1
* @date    	2017-08-08
* @brief   	Implimentation of the analytics using our own object detectors and a third party classifier
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoFusedAnalytics_Overview
#define __DidoFusedAnalytics_Overview


#pragma once
#include "DidoFusedAnalytics.h"
#include "DidoFusedAnalytics_3dBgSub_CUDA.h"
#include "DidoFusedAnalytics_3dObjDetect_CUDA.h"
#include "DidoLidar_ManagedPano.h"
#include "DidoMain_ParamEnv.h"
#include "DidoMain_ObjEnv.h"
#include "DidoAnalytics/DidoAnalytics_ThermalClassifier.h"

/*defines ---------------------------*/

namespace overview
{
//what exactly does this give us? - something to test with once vivek has made the pano? seems low priority
class DidoFusedAnalytics_Overview : public DidoFusedAnalytics
{
	DidoAnalytics_ThermalClassifier tclass;

	std::map<MapFloat, DidoFusedAnalytics_3dBgSub_CUDA> bgrs;
	DidoFusedAnalytics_3dObjDetect_CUDA obdet;
	std::shared_ptr<OVTrack_Storage> trackStore;
	//parameters
	float max_range = 69.8f;
	//tan of the angle between the center and the bottom of the frame
	float tanalpha = 0.1989f;

	float imgwidth = 30 * 2 * pi / 360;//how wide each image is in radians
	
	//we have to add some set of functionality that will handle the between frames association of these targets
	//it will probably need to maintain some local copy of metastatistics or similar
	int history = 100;
	float tVar = 32.0f, rVar = 2.0f;
	float bgThresh = 4.0f, genThresh = 3.0f;

	float rejectionThreshold = 0.7f;

	const bool _regionBased;
	int bgrScale;

	float minPAssoc = 0.8f; //the square of the maximum separation on the target and it's prediction to make an association

	//we also need a behaviour classifier

protected:
	/*generates the detections from the frame and initialises the tracks on the heap*/
	virtual std::vector<std::shared_ptr<OVTrack>> processFrame(const cv::Mat & img, const unsigned  char * pano, const float * depths_min, const float * depths_max, float startangle, Dido_Timestamp ts, size_t rows, size_t cols) override;

	virtual OVTrack_Classification classifyROI(const cv::Mat & img);

public:
	virtual ~DidoFusedAnalytics_Overview() = default;
	/*needs a pointer to the global store on construction*/
	DidoFusedAnalytics_Overview(int history = 150, bool regionBased = false);
	DidoFusedAnalytics_Overview(DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj);

	virtual void connectTrackStore(std::shared_ptr<OVTrack_Storage> store) { trackStore = store; }

	//parameter setting functions

	//object detection parameters
	void setNcore(int nc) {	obdet.setNcore(nc);}
	void setMinpoints(int mp) { obdet.setMinPoints(mp); }
	void setRangeScaling(float rs) { obdet.setRangeScaling(rs); }

	//background subtraction parameters
	void setHistory(int hist_) { history = hist_;  for (auto & bg : bgrs) bg.second.setHistory(hist_); }
	void setVariance(float initTempVar, float initRangeVar) { tVar = initTempVar; rVar = initRangeVar; for (auto & bg : bgrs) bg.second.setVariance(initTempVar, initRangeVar); }
	void setThresholds(float backgroundThresh, float generativeThresh) //it's reccomended that generative threshold is less than background Threshold
	{ 
		bgThresh = backgroundThresh; genThresh = generativeThresh;
		for (auto & bg : bgrs) bg.second.setThresholds(backgroundThresh, generativeThresh); 
	}

	//other parameters
	void setMaxRange(float mr) { max_range = mr;}
	void setTanAlpha(float ta) { tanalpha = ta; }
	void setImageWidth(float iw) { imgwidth = iw; }
};
}
#endif
