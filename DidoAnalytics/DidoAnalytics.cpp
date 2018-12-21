/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics.cpp
* @author  	SL
* @version 	1
* @date    	2017-09-27
* @brief   	Abstract class that defines the interface of how we turn our wide area sensor data into tracks
			This implements all interactions with the global store of tracks
*****************************************************************************
**/

#include "GlobalIncludes.h"

#include "DidoAnalytics.h"
#include "PTZActuator.h"

namespace overview
{

	DidoAnalytics::DidoAnalytics()
		: timeouttime(std::chrono::duration_cast<Dido_Clock::duration>(std::chrono::milliseconds(400)))
	{
	}

	void DidoAnalytics::analyse3DFrame(const std::vector<cv::Mat> & imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar , const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles)
	{
		if (timestamps.empty()) return;
		auto ts = timestamps.front();
		for (auto & t : timestamps)
		{
			if (t > ts) ts = t;
		}

		auto vec = produceTargets(imgs, lidar,  timestamps, startangles, ts);
		/*apply the tracks to the globals*/
		for (auto & v : vec)
		{
			auto pos = v->getLoc();
			if (pos.dot(pos) < minrange*minrange) continue;//skip detections within the minimum range
			bool ignored = false;
			//check if it's inside an ignore area		
			for (auto & a : IgnoreAreas)
			{
				if (a.isInArea(pos))
				{
					ignored = true;
					break;
				}
			}
			if (ignored) continue;

			bool newtrack = true;
			for (auto & t : *trackStore)
			{
				if ((t) && t->trackID == v->trackID)
				{
					//check if it crosses a boundary and alert as appropriate
					OV_WorldPoint sloc = t->getLoc();
					for (auto & b : Boundaries)
					{
						if (b.crosses(sloc, pos))
						{
							LOG_INFO << "Boundary crossed by object " << t->trackID << std::endl;
							//TODO - alert properly
						}
					}

					newtrack = false;
					t->updateWithPartialTrack(*v);
					t->threat = calculateThreat(t);
					break;
				}
			}
			if (newtrack)
			{
				/*add it to the storage - use the first null location*/
				for (auto & t : *trackStore)
				{
					if (!t)
					{
						t = v;
						break;
					}
				}
			}
		}
		/*decay the undetected globals*/
		for (auto & t : *trackStore)
		{
			if (t != nullptr)
			{
				if ((t->getCurrObs().timestamp + timeouttime < ts) )
				{
					//get it to predict it's location and update it's current timestamp
					t->noObsPredict(ts);
					t->threat = calculateThreat(t);

					if (t->detectionConfidence < confidenceThreshold)
					{
						t.reset();
					}
				}
				/*put the tracks into the database */
				else
				{
					if (dbconnect) (*dbconnect)(*t);
				}
			}
		}
		/*alarm on any humans or vehicles inside the area of interest*/
		for (auto & t : *trackStore)
		{
			if (t)
			{
				auto classif = t->getFirstLevelClassification();
				if (classif == FirstLevelClassification::Human || classif == FirstLevelClassification::Vehicle)
				{
					//check if it's inside an ignore area		
					for (auto & a : AreasOfInterest)
					{
						auto pos = t->getLoc();
						if (a.isInArea(pos))
						{
							LOG_INFO << "target " << t->trackID << "triggered an alarm at " << t->getLoc() << std::endl; //TODO - replace this with a real alarm
							break;
						}
					}
				}
			}
		}

	}

	void DidoAnalytics::setTimeoutTime(int miliseconds)
	{
		timeouttime = std::chrono::duration_cast<Dido_Clock::duration>(std::chrono::milliseconds(miliseconds));
	}

	void DidoAnalytics::setAngleSlack(float as)
	{
		angleSlack = as;
	}
	void DidoAnalytics::setMinRange(float mr)
	{
		minrange = mr;
	}
	std::vector<std::shared_ptr<OVTrack>> DidoAnalytics_Dummy::produceTargets(const std::vector<cv::Mat>& imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano>& lidar, const std::vector<Dido_Timestamp>& timestamps, const std::vector<float>& startangles, const Dido_Timestamp & now)
	{
		std::vector<std::shared_ptr<OVTrack>>rval;
		OV_WorldPoint cpos((float)std::max(20 - frame % 40, 1), (float)std::max(frame % 40 - 20, 1), 1.5f);
		frame++;
		rval.push_back(std::make_shared<OVTrack>(1, cpos, now, 0.5f));
		return rval;
	}

	float DidoAnalytics::calculateThreat(std::shared_ptr<OVTrack> trk)
	{
		auto pos = trk->getLoc();
		//iterative updates towards 1/d *classconf*detconf
		return trk->threat *(1 - gamma) + gamma*((1 / sqrt(pos.dot(pos)))* trk->detectionConfidence* trk->getFirstLevelClassificationWeight(FirstLevelClassification::Human));
	}
}