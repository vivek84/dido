/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoFusedAnalytics.cpp
* @author  	SL
* @version 	1
* @date    	2017-03-22
* @brief   	Abstract class that defines how we interact with out video analytics providers
			This implements all interactions with the global store of tracks
*****************************************************************************
**/

#include "GlobalIncludes.h"

#include "DidoFusedAnalytics.h"
#include "PTZActuator.h"

namespace overview
{	
	std::vector<std::shared_ptr<OVTrack>> DidoFusedAnalytics::analyse3DFrame(const std::vector<cv::Mat> & imgs, const std::vector<DidoLidar_Managed_Pano<unsigned char>> & panos, const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles, const std::vector<DidoLidar_Managed_Pano<float>> & depths_min, const std::vector<DidoLidar_Managed_Pano<float>> & depths_max, size_t rows, size_t cols)
	{
		auto ts = Dido_Clock::now();
		std::vector<std::shared_ptr<OVTrack>> rval;
		for (int i = 0; i < imgs.size(); i++)	//could have spare outputs defined
		{
			//skip shuttered frames
			if (startangles[i] < 0) continue;
			auto vec = processFrame(imgs[i], panos[i].data(), depths_min[i].data(), depths_max[i].data(), startangles[i], timestamps[i], rows, cols);
			/*apply the tracks to the globals*/
			rval.insert(rval.end(), vec.begin(), vec.end());
		}
		return rval;
	}
}