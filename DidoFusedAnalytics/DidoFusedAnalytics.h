/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoFusedAnalytics.h
* @author  	SL
* @version 	1
* @date    	2017-03-22
* @brief   	Abstract class that defines how we interact with out video analytics providers
			This implements all interactions with the global store of tracks
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoFusedAnalytics
#define __DidoFusedAnalytics


#pragma once

#include "OVTrack.h"

#include "DidoLidar_ManagedPano.h"
#include "db_tracks.h"
#include "ControlInterface_Area.h"

/*defines ---------------------------*/

namespace overview
{

	/*inputs are pointers to managed GPU memory sections - they are already allocated with cudaMallocManaged*/
	class DidoFusedAnalytics
	{
	protected:
		/*virtual functions that must be implemented by the children that include how we actually get the data from the analytics*/

		/*generates the detections from the frame and initialises the tracks on the heap*/
		virtual std::vector<std::shared_ptr<OVTrack>> processFrame(const cv::Mat & frame,const  unsigned  char * pano, const float * depths_min, const float * depths_max, float startangle, Dido_Timestamp ts, size_t rows, size_t cols) = 0;

	public:
		virtual ~DidoFusedAnalytics() = default;
		/*needs a pointer to the global store on construction*/
		DidoFusedAnalytics() = default;
		virtual void connectTrackStore(std::shared_ptr<OVTrack_Storage> store) {/*noop*/ }

		/*both these functions take an input and apply their elements directly to the tracks*/
		/*function called as part of the main pipeline*/
		std::vector<std::shared_ptr<OVTrack>> analyse3DFrame(const std::vector<cv::Mat> & imgs ,const std::vector<DidoLidar_Managed_Pano<unsigned char>> & panos, const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles,
			const std::vector<DidoLidar_Managed_Pano<float>> & depths_min, const std::vector<DidoLidar_Managed_Pano<float>> & depths_max, size_t rows, size_t cols);
	};
}
#endif
