/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_EarlyLateFused.h
* @author  	SL
* @version 	1
* @date    	2017-10-18
* @brief   	produces tracks using late fusion of lidar tracks with tracks from early fusion
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoAnalytics_EarlyLateFused
#define __DidoAnalytics_EarlyLateFused


#pragma once
#include "DidoAnalytics_LateFused.h"
#include "DidoFusedAnalytics.h"
#include "DidoLidar_CUDA.h"

/*defines ---------------------------*/

namespace overview
{
	class DidoAnalytics_EarlyLateFused : public DidoAnalytics_LateFused
	{
		std::shared_ptr<DidoFusedAnalytics> fusedA;
		//do we use the edges when doing fusion
		bool dumbFusion;
	protected:
		virtual std::vector<std::shared_ptr<OVTrack>> getThermalTargets(const std::vector<cv::Mat> & imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar,
			const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles, const Dido_Timestamp & now) override;
		virtual void connectTrackStore(std::shared_ptr<OVTrack_Storage> ts) override { trackStore = ts; fusedA->connectTrackStore(ts); }
	public:
		DidoAnalytics_EarlyLateFused(DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj, std::shared_ptr<DidoFusedAnalytics> fa, bool dumbfused = false);
	};
}
#endif
