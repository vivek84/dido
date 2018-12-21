/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_EarlyLateFused.cpp
* @author  	SL
* @version 	1
* @date    	2017-10-18
* @brief   	produces tracks using late fusion of lidar tracks with tracks from early fusion
*****************************************************************************
**/

#include "GlobalIncludes.h"

#include "DidoAnalytics_EarlyLateFused.h"
#include "PTZActuator.h"

namespace overview
{
	std::vector<std::shared_ptr<OVTrack>> DidoAnalytics_EarlyLateFused::getThermalTargets(const std::vector<cv::Mat>& imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar, const std::vector<Dido_Timestamp>& timestamps, const std::vector<float>& startangles, const Dido_Timestamp & now)
	{
		//do the synthesis locally, then go on
		int nrows = imgs[0].rows;
		int ncols = imgs[0].cols;
		if (nrows*ncols == 0) return std::vector<std::shared_ptr<OVTrack>>();	//skip the world of empty images
		auto id = std::this_thread::get_id();

		std::vector<float *> max_range_ptrs, min_range_ptrs;
		std::vector<const thermalType*> therm_ptrs;

		for (int i = 0; i < imgs.size(); i++)
		{
			if (m_range_panos_max[id].size() <= i)
			{
				m_range_panos_max[id].emplace_back(ncols, nrows);
				m_range_panos_min[id].emplace_back(ncols, nrows);
				m_analytics_pano[id].emplace_back(ncols, nrows);
			}
			//check that they are big enough (allows us to catch panos that were allocated based on small images)
			if (m_analytics_pano[id][i].ncols() < ncols || m_analytics_pano[id][i].nrows() < nrows) m_analytics_pano[id][i] = DidoLidar_Managed_Pano<thermalType>(ncols, nrows);
			if (m_range_panos_min[id][i].ncols() < ncols || m_range_panos_min[id][i].nrows() < nrows) m_range_panos_min[id][i] = DidoLidar_Managed_Pano<float>(ncols, nrows);
			if (m_range_panos_max[id][i].ncols() < ncols || m_range_panos_max[id][i].nrows() < nrows) m_range_panos_max[id][i] = DidoLidar_Managed_Pano<float>(ncols, nrows);

			if (!imgs[i].empty()) m_analytics_pano[id][i].copyFrom(imgs[i].data);
			therm_ptrs.push_back((const thermalType*)m_analytics_pano[id][i].data());
			max_range_ptrs.push_back(m_range_panos_max[id][i].r_data());
			min_range_ptrs.push_back(m_range_panos_min[id][i].r_data());
		}
		if (dumbFusion)
		{
			lsynth.dumbSynthesis(ncols, nrows, min_range_ptrs, max_range_ptrs, lidar ? lidar->size() : 0, lidar ? lidar->data() : nullptr,
				verticalDisplacement, verticalFOV, verticalOffset, startangles, width);
		}
		else
		{
			lsynth.cudaLaserSynthesis_infill(therm_ptrs, ncols, nrows, min_range_ptrs, max_range_ptrs,
				lidar ? lidar->size() : 0, lidar ? lidar->data() : nullptr, verticalDisplacement, verticalFOV, verticalOffset, startangles, width);
		}
		LOG_DEBUG << "Synthesised range and thermal data" << std::endl;

		return fusedA->analyse3DFrame(imgs, m_analytics_pano[id], timestamps, startangles, m_range_panos_min[id], m_range_panos_max[id], nrows, ncols);
	}

	DidoAnalytics_EarlyLateFused::DidoAnalytics_EarlyLateFused(DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj, std::shared_ptr<DidoFusedAnalytics> fa, bool dumb)
		: DidoAnalytics_LateFused(pars, obj), fusedA(fa), dumbFusion(dumb)
	{
	}

}