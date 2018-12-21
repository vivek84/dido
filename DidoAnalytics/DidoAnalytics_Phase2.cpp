/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2018 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_Phase2.cpp
* @author  	SL
* @version 	1
* @date    	2018-01-25
* @brief   	Abstract class that defines the interface for whole detection pipeline.
*           This allows choice of sensor configuration
*****************************************************************************
**/

#include "GlobalIncludes.h"

#include "DidoAnalytics_Phase2.h"

namespace overview
{

	DidoAnalytics_Phase2::DidoAnalytics_Phase2(int timeout)
		: timeouttime(std::chrono::duration_cast<Dido_Clock::duration>(std::chrono::milliseconds(timeout)))
	{
	}

	void DidoAnalytics_Phase2::updateGlobalTracks(const std::vector<std::shared_ptr<OVTrack>> & vec, Dido_Timestamp ts)
	{
		if (!trackStore) throw std::runtime_error("trackstore must be connected before these functions are called");

		Dido_Timestamp prevstep;
		if (displayTimings)
		{
			auto ts = Dido_Clock::now();
			prevstep = ts;
		}
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
                else
                {
                    tsig(t);
                }    
			}
		}
		if (displayTimings)
		{
			auto ts = Dido_Clock::now();
			LOG_INFO << "Applying tracks took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevstep).count() << "ms";
			prevstep = ts;
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
		if (displayTimings)
		{
			auto ts = Dido_Clock::now();
			LOG_INFO << "alarms took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevstep).count() << "ms";
			prevstep = ts;
		}
		ssig(trackStore);
		if (displayTimings)
		{
			auto ts = Dido_Clock::now();
			LOG_INFO << "connected functions took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevstep).count() << "ms";
			prevstep = ts;
		}
	}

    void DidoAnalytics_Phase2::connectTrackFun(trackSignal::slot_function_type fun)
    {
        tsig.connect(fun);
    }
    
    void DidoAnalytics_Phase2::disconnectTrackFun()
    {
        tsig.disconnect_all_slots();
    }
    
    void DidoAnalytics_Phase2::connectStepFun(stepSignal::slot_function_type fun)
    {
        ssig.connect(fun);
    }
    
    void DidoAnalytics_Phase2::disconnectStepFun()
    {
        ssig.disconnect_all_slots();
    }
    
	void DidoAnalytics_Phase2::setTimeoutTime(int miliseconds)
	{
		timeouttime = std::chrono::duration_cast<Dido_Clock::duration>(std::chrono::milliseconds(miliseconds));
	}

	void DidoAnalytics_Phase2::setMinRange(float mr)
	{
		minrange = mr;
	}
    
    float DidoAnalytics_Phase2::calculateThreat(std::shared_ptr<OVTrack> trk)
	{
		auto pos = trk->getLoc();
		//iterative updates towards 1/d *classconf*detconf
		return trk->threat *(1 - gamma) + gamma*((1 / sqrt(pos.dot(pos)))* trk->detectionConfidence* trk->getFirstLevelClassificationWeight(FirstLevelClassification::Human));
	}
    
	void DidoAnalytics_Phase2_Dummy::runfunc()
	{
        while(running)
        {
            auto now = Dido_Clock::now();
            std::vector<std::shared_ptr<OVTrack>>rval;
            OV_WorldPoint cpos((float)std::max(20 - frame % 40, 1), (float)std::max(frame % 40 - 20, 1), 1.5f);
            frame++;
            rval.push_back(std::make_shared<OVTrack>(1, cpos, now, 0.5f));
            updateGlobalTracks(rval, now);
            my_sleep(200);
        }
    }
    
    void DidoAnalytics_Phase2_Dummy::start()
    {
        if(!running)
        {
            running = true;
            runthread = std::thread(&DidoAnalytics_Phase2_Dummy::runfunc, this);
        }
    }

    void DidoAnalytics_Phase2_Dummy::stop()
    {
        running = false;
        if (runthread.joinable()) runthread.join();
    }
}