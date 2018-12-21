/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2018 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_LidarPrimary.cpp
* @author  	SL
* @version 	1
* @date    	2018-01-22
* @brief   	produces tracks using lidar
*****************************************************************************
**/

#include "GlobalIncludes.h"

#include "DidoAnalytics_LidarPrimary.h"
#include "PTZActuator.h"
#include "RangeMappingFunctions.h"

namespace overview
{

	std::vector<std::shared_ptr<OVTrack>> DidoAnalytics_LidarPrimary::getLidarTargets(const std::shared_ptr<DidoLidar::DidoLidar_Pano>& lidar, const Dido_Timestamp & now)
	{
		//for timing
		Dido_Timestamp prevstep;
		if (displayTimings) prevstep = Dido_Clock::now();

		if (!trackStore) throw std::runtime_error("trackstore must be connected before this is called");
		std::vector<std::shared_ptr<OVTrack>> rval;

		if (!lidar) return rval;	//no data means no data
		
		if (displayInput)
		{
			cv::Mat lidimg = mapRanges(lidar, 10.0f, cv::Size(600,600));
			if (!lidimg.empty())
			{
				cv::imshow("lidar_in", lidimg);
				cv::waitKey(1);
			}
		}
		
		if (displayTimings)
		{
			auto ts = Dido_Clock::now();
			LOG_INFO << "displaying lidar input took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevstep).count() << "ms";
			prevstep = ts;
		}


		//background subtract the points
		std::vector<std::vector<DidoLidar_rangeData>> fgpoints;
		//cluster the foreground points
		
		if (displayWorking)
		{
			auto bgsubbed = bgsub_lidar.apply(lidar->data(), lidar->size());
			cv::Mat bgimg = mapRanges(bgsubbed.data(), (int)bgsubbed.size(), 10.0f, cv::Size(600, 600));
			if (!bgsubbed.empty())
			{
				cv::imshow("lidar_fg", bgimg);
				cv::waitKey(1);
			}
			fgpoints = bgsub_lidar.Cluster(bgsubbed);
			//looking for saturation
			cv::Mat bins(bgsub_lidar.rows(), bgsub_lidar.cols(), CV_8UC1);
			bgsub_lidar.dispayBgmodelNpts(bins.data, bins.rows*bins.cols);
			cv::Mat colorbins;
			cv::applyColorMap(bins*(256 / LIDARBGSUB_MAX_BIN_PTS), colorbins,cv::COLORMAP_JET);
			cv::imshow("bins", colorbins);
			cv::waitKey(1);
		}
		else
		{
			fgpoints = bgsub_lidar.applyAndCluster(lidar->data(), lidar->size());
		}

		if (displayTimings)
		{
			auto ts = Dido_Clock::now();
			LOG_INFO << "clustering took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevstep).count() << "ms";
			prevstep = ts;
		}

		for (auto & v : fgpoints)
		{
			if (v.size() < minLidpts) continue;
			//turn it into world points
			std::vector<OV_WorldPoint> wpts;
			for (auto & p : v) wpts.push_back(convLidar(p));
			//classify the blob
			auto cf = lidClassifier.classifyBlob(wpts);
			if (cf.classvals[static_cast<int>(FirstLevelClassification::Unknown)] > rejectionThreshold) continue;	//drop things we are confident are not people
			//make a blob
			rval.push_back(trackFromPoints(wpts, now));
			rval.back()->replaceFLClassification(cf);

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
		if (displayTimings)
		{
			auto ts = Dido_Clock::now();
			LOG_INFO << "classification and association took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevstep).count() << "ms";
		}
		return rval;
	}
    
	void DidoAnalytics_LidarPrimary::setRejectionThreshold(float rt)
	{
		rejectionThreshold = rt;
	}

	void DidoAnalytics_LidarPrimary::loadLidarSVMModel(std::string file)
	{
		lidClassifier.load(file);
	}

	void DidoAnalytics_LidarPrimary::saveLidarSVMModel(std::string file)
	{
		lidClassifier.save(file);
	}

    
    void DidoAnalytics_LidarPrimary::runFunction()
		{
			while (running)
			{
				std::shared_ptr<DidoLidar::DidoLidar_Pano> lidpan;
				//get the data
				{
					std::unique_lock<std::mutex> ulock(quemut);
					while (lidque.empty() && running)
					{
						quevar.wait(ulock);
					}
					if (!running) return;
					while (!lidque.empty())
					{
						lidpan = lidque.front();
						lidque.pop();
					}
				}
				if (lidpan)
				{
					auto now = Dido_Clock::now();
					//run the analytics
					updateGlobalTracks(getLidarTargets(lidpan, now), now);
				}
			}
		}


	/*runs the analytics in a parallel thread until told to stop*/
	void DidoAnalytics_LidarPrimary::start()
    {
		if (!running)
		{
			running = true;
			lidar->connect([&](std::shared_ptr<DidoLidar::DidoLidar_Pano> pano) {
				std::lock_guard<std::mutex> lck(quemut);
				lidque.push(pano);
				quevar.notify_one();
			});
			runThread = std::thread(&DidoAnalytics_LidarPrimary::runFunction, this);
			lidar->run();
		}
    }

	/*stops it running*/
	void DidoAnalytics_LidarPrimary::stop()
    {
        lidar->stop();
        if (running)
        {
            running = false;
            lidar->disconnect();
            quevar.notify_all();
            if (runThread.joinable())
            {
                runThread.join();
            }
        }
    }

	DidoAnalytics_LidarPrimary::DidoAnalytics_LidarPrimary(DidoMain_ParamEnv & pars, std::shared_ptr<DidoLidar_primary> lid)
		: bgsub_lidar(pars.binwidth, pars.binheight, pars.lidMindist, pars.lidHistory, pars.lidBGThresh,pars.lideps, pars.lidncore),
		lidhist(pars.lidHistory), epsilon(pars.lideps), ncore(pars.lidncore), minLidpts(pars.minLidpts), minPassoc(pars.minPassoc),
		lidClassifier(pars.lidSvmSaveFile, lidarSVMDefaultParameters()), rejectionThreshold(pars.rejectionThreshold), lidar(lid),
		DidoAnalytics_Phase2(250)
	{

	}
    
}