/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics.h
* @author  	SL
* @version 	1
* @date    	2017-09-27
* @brief   	Abstract class that defines the interface of how we turn our wide area sensor data into tracks
			This implements all interactions with the global store of tracks
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoAnalytics
#define __DidoAnalytics


#pragma once

#include "OVTrack.h"

#include "db_tracks.h"
#include "ControlInterface_Area.h"
#include "DidoLidar.h"

/*defines ---------------------------*/

namespace overview
{
	class DidoAnalytics
	{
	protected:
		/*pointer to the global set of tracks*/
		std::shared_ptr<OVTrack_Storage> trackStore;
		/*the SQL database that the gui uses to find out about the tracks*/
		std::shared_ptr<overview::db::store_to_database> dbconnect;

		/*how long between detections before we timeout a track*/
		Dido_Clock::duration timeouttime;

		/*how much angular deviation from the last recorded location do we allow a track that we're updating with the 2D classifier*/
		float angleSlack = 0.5;

		//the minimum range past which a target is accepted
		float minrange = 0.7f;

		/*how low our confidence needs to go before we remove a detection*/
		float confidenceThreshold = 0.2f;

		/*produces potential tracks from the data - doesn't associate*/
		virtual std::vector<std::shared_ptr<OVTrack>> produceTargets(const std::vector<cv::Mat> & imgs , const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar,
			const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles, const Dido_Timestamp & now) =0;

		//threat learning rate
		float gamma = 0.2f;
		float calculateThreat(std::shared_ptr<OVTrack> trk);

	public:
		virtual ~DidoAnalytics() = default;
		DidoAnalytics();

		void connnectDB(std::shared_ptr<overview::db::store_to_database> db) { dbconnect = db; }
		virtual void connectTrackStore(std::shared_ptr<OVTrack_Storage> ts) { trackStore = ts; }

		/*both these functions take an input and apply their elements directly to the tracks*/
		/*function called as part of the main pipeline*/
		void analyse3DFrame(const std::vector<cv::Mat> & imgs ,const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar , const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles);

		/*internal storage of the areas of interest - public so they can be easily edited and added to
		 * not thread safe so if the parameters are being edited, the analytics must not be running*/
		std::vector<ControlInterface_Area> AreasOfInterest;
		std::vector<ControlInterface_Area> IgnoreAreas;
		std::vector<ControlInterface_Boundary> Boundaries;

		/*parameter setting functions*/
		void setTimeoutTime(int miliseconds);
		void setAngleSlack(float as);
		void setMinRange(float mr);
	};

	//dummy analytics that simply produces a track that moves from 10,0 to 1,0 then to 1,10 then back again
	class DidoAnalytics_Dummy : public DidoAnalytics
	{
		int frame = 0;
	public:
		// Inherited via DidoAnalytics
		virtual std::vector<std::shared_ptr<OVTrack>> produceTargets(const std::vector<cv::Mat>& imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano>& lidar, const std::vector<Dido_Timestamp>& timestamps, const std::vector<float>& startangles, const Dido_Timestamp & now) override;
	};
}
#endif
