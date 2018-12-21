/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2018 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_RadarPrimary.cpp
* @author  	SL
* @version 	1
* @date    	2018-01-25
* @brief   	produces tracks using radar
*****************************************************************************
**/

#include "GlobalIncludes.h"

#include "DidoAnalytics_RadarPrimary.h"

namespace overview
{
	std::vector<std::shared_ptr<OVTrack>> DidoAnalytics_RadarPrimary::parseRadarTrack(const SpotterRadar_track & tks, Dido_Timestamp now)
	{
		std::vector<std::shared_ptr<OVTrack>> rval;
		for (auto & t : tks)
		{
			DidoLidar_rangeData rbear((float)t.range, (float)t.horizontalAngle* pi / 180, (float)t.verticalAngle* pi / 180);
			OV_WorldPoint loc = convLidar(rbear);
			//estimate velocity using heading and radial velocity
			float absspeed = abs((float)t.radVelocity / cosf((float)fmod((float)t.heading - (float)t.horizontalAngle, 180) *pi / 180));
			OV_WorldPoint vel = convLidar(DidoLidar_rangeData(absspeed, (float)t.heading, 0));
			//estimate the size using the radar cross section
			float width = (float)sqrt(t.radarCrossSection);
			
			rval.push_back(std::make_shared<OVTrack>(-1, loc, now, (float)t.accuracy));
			rval.back()->setVel(vel);
			//apply the tracking
			static long newid = 0;
			float maxdist = minPassoc;
			long id = -1;
			for (auto t : *trackStore)
			{
				if (t == nullptr) continue;
				auto sep = t->obsCloseEnough(rval.back()->getLoc(), now);
				if (sep > maxdist)
				{
					maxdist = sep;
					id = t->trackID;
				}
			}
			if (id < 0)
			{
				id = newid++;
			}
			rval.back()->trackID = id;
		}

		return rval;
	}
	void DidoAnalytics_RadarPrimary::runfunction()
	{
		while (running)
		{
			try
			{
				auto now = Dido_Clock::now();
				auto tks = radar.getCurrentTracks();
				auto targvec = parseRadarTrack(tks, now);
				//associate tracks with existing targets
				updateGlobalTracks(targvec, now);
			}
			catch (const std::runtime_error & e)
			{
				LOG_WARN << "failed to read radar data with message: " << e.what();
			}
		}
	}
	/*runs the analytics in a parallel thread until told to stop*/
	void DidoAnalytics_RadarPrimary::start()
    {
		if (!running)
		{
			running = true;
			runthread = std::thread(&DidoAnalytics_RadarPrimary::runthread, this);
		}
    }

	/*stops it running*/
	void DidoAnalytics_RadarPrimary::stop()
    {
		if (running)
		{
			running = false;
			if (runthread.joinable()) runthread.join();
		}
    }

	DidoAnalytics_RadarPrimary::DidoAnalytics_RadarPrimary(DidoMain_ParamEnv & pars, std::string ip)
        :radar(ip),	DidoAnalytics_Phase2(350)
	{
	}
    
}