/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_EarlyFused.h
* @author  	SL
* @version 	1
* @date    	2017-09-27
* @brief   	produces tracks using early fusion of lidar data with thermal
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoAnalytics_EarlyFused
#define __DidoAnalytics_EarlyFused


#pragma once

#include "OVTrack.h"

#include "db_tracks.h"
#include "ControlInterface_Area.h"
#include "DidoMain_ParamEnv.h"
#include "DidoMain_ObjEnv.h"
#include "DidoAnalytics.h"
#include "DidoLidar_CUDA.h"
#include "DidoFusedAnalytics.h"
#include "DidoLidar_ManagedPano.h"

/*defines ---------------------------*/

namespace overview
{

	/*inputs are pointers to managed GPU memory sections - they are already allocated with cudaMallocManaged*/
	class DidoAnalytics_EarlyFused : public DidoAnalytics
	{
		//various parameters of internal elements are contained in the objects themselves

		DidoLidar_CUDA lsynth;
		float verticalDisplacement, verticalFOV, verticalOffset;

		std::shared_ptr<DidoFusedAnalytics> analytics;

		//containter for the GPU data so we allocate less
		std::map<std::thread::id, std::vector<DidoLidar_Managed_Pano<float>>> m_range_panos_max, m_range_panos_min;
		std::map<std::thread::id, std::vector<DidoLidar_Managed_Pano<unsigned char>>> m_analytics_pano;
		
		float width;
		bool dumbFusion;
		
	protected:

		/*produces potential tracks from the data - doesn't associate*/
		virtual std::vector<std::shared_ptr<OVTrack>> produceTargets(const std::vector<cv::Mat> & imgs , const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar,
			const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles, const Dido_Timestamp & now) override;

		virtual void connectTrackStore(std::shared_ptr<OVTrack_Storage> ts) override { trackStore = ts; analytics->connectTrackStore(ts); }
		
	public:
		virtual ~DidoAnalytics_EarlyFused() = default;
		/*needs a pointer to the global store on construction*/
		DidoAnalytics_EarlyFused(std::shared_ptr<DidoFusedAnalytics> fAnalytics, const DidoMain_ParamEnv & pars, const DidoMain_ObjEnv & obj, bool dumbfused = false );
	};
}
#endif
