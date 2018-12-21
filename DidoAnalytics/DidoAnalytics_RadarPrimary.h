/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2018 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_RadarPrimary.h
* @author  	SL
* @version 	1
* @date    	2018-01-25
* @brief   	produces tracks using radar
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoAnalytics_RadarPrimary
#define __DidoAnalytics_RadarPrimary


#pragma once

#include "DidoMain_ParamEnv.h"
#include "DidoAnalytics_Phase2.h"
#include "SpotterRadar.h"

/*defines ---------------------------*/

namespace overview
{

	/*inputs are pointers to managed GPU memory sections - they are already allocated with cudaMallocManaged*/
	class DidoAnalytics_RadarPrimary : public DidoAnalytics_Phase2
	{
	protected:
		SpotterRadar radar;

		std::vector<std::shared_ptr<OVTrack>> parseRadarTrack(const SpotterRadar_track & tk, Dido_Timestamp now);

		//the minum probability from the target Kalman filter to allow a proposed association
		float minPassoc;

		std::thread runthread;
		bool running = false;
		void runfunction();
        
	public:
		virtual ~DidoAnalytics_RadarPrimary() { stop(); };

        virtual void start() override;
        virtual void stop() override;
        
		DidoAnalytics_RadarPrimary(DidoMain_ParamEnv & pars, std::string radarIP);
	};
}
#endif
