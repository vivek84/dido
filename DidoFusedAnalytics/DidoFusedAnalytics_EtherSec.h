/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoFusedAnalytics_EtherSec.h
* @author  	SL
* @version 	1
* @date    	2017-04-20
* @brief   	Implementation of the analytics using Ethersec's interface
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoFusedAnalytics_EtherSec
#define __DidoFusedAnalytics_EtherSec


#pragma once
#include "DidoFusedAnalytics.h"
#include "EtherSecInterfaceProxy.h"

/*defines ---------------------------*/
//predeclarations

namespace overview
{
	//what exactly does this give us? - something to test with once vivek has made the pano? seems low priority
	class DidoFusedAnalytics_EtherSec : public DidoFusedAnalytics
	{
		cv::Size imgsize;
		//the height above the centerline of an object at the top of the panorama 1 meter away
		float tanalpha = 0.364f;
		void ES_processCamera(CESI_Camera & cam, std::vector<std::shared_ptr<OVTrack>> & trcks, Dido_Timestamp & t, float sa);
		//std::unique_ptr<esi::OvVa<unsigned char>> vidAnalytics;
		CESI_FrameOutput dresults;
		int ncams = 15;


		//CNN member elements
		//IESI_CNNAdapter * CNNAdapter;

	protected:

		/*generates the detections from the frame and initialises the tracks on the heap*/
		virtual std::vector<std::shared_ptr<OVTrack>> processFrame(const cv::Mat & img, const unsigned  char * pano, const float * depths_min, const float * depths_max, float startangle, Dido_Timestamp ts, size_t rows, size_t cols) override;

	public:
		~DidoFusedAnalytics_EtherSec() final;
		/*needs a pointer to the global store on construction*/
		DidoFusedAnalytics_EtherSec(cv::Size _imgSize );
		/*anything that has to be ran at startup by the system that can't/shouldn't be in the constructor*/
		void startUp(int ncams, size_t ncols, size_t nrows);
	};
}
#endif
