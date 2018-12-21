/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_LateFused.cpp
* @author  	SL
* @version 	1
* @date    	2017-09-27
* @brief   	produces tracks using late fusion of lidar tracks with thermal
*****************************************************************************
**/

#include "GlobalIncludes.h"

#include "DidoAnalytics_LateFused.h"
#include "PTZActuator.h"
#include "RangeMappingFunctions.h"

namespace overview
{

	namespace detail
	{
		//from some cylindrical geometry - alpha is the angle between the top of the image and the centerpoint
		static inline float pixel_to_height(int pixel, size_t rows, float depth, float tan_alpha)
		{
			int offset = (int)rows / 2 - pixel;
			return depth*(offset) / sqrt((offset*offset) + rows*rows / (4 * tan_alpha));
		}
	}

	std::vector<std::shared_ptr<OVTrack>> DidoAnalytics_LateFused::getLidarTargets(const std::shared_ptr<DidoLidar::DidoLidar_Pano>& lidar, const Dido_Timestamp & now)
	{
		std::vector<std::shared_ptr<OVTrack>> rval;

		if (!lidar) return rval;	//no data means no data
		//background subtract the points
		auto fgpoints =  bgsub_lidar.applyAndCluster(lidar->data(), lidar->size());
		//cluster the foreground points
		
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
		}
		return rval;
	}

	std::vector<std::shared_ptr<OVTrack>> DidoAnalytics_LateFused::getThermalTargets(const std::vector<cv::Mat>& imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar, const std::vector<Dido_Timestamp>& timestamps, const std::vector<float>& startangles, const Dido_Timestamp & now)
	{
		std::vector<std::shared_ptr<OVTrack>> rval;
		int nrows = imgs[0].rows;
		int ncols = imgs[0].cols;

		if (nrows*ncols == 0) return rval;	//skip the world of empty images

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

			if(!imgs[i].empty()) m_analytics_pano[id][i].copyFrom(imgs[i].data);
			therm_ptrs.push_back(m_analytics_pano[id][i].data());
			max_range_ptrs.push_back(m_range_panos_max[id][i].r_data());
			min_range_ptrs.push_back(m_range_panos_min[id][i].r_data());
		}

		lsynth.cudaLaserSynthesis_infill(therm_ptrs, ncols, nrows, min_range_ptrs, max_range_ptrs,
			lidar ? lidar->size() : 0, lidar ? lidar->data() : nullptr, verticalDisplacement, verticalFOV, verticalOffset, startangles, width);

		LOG_DEBUG << "Synthesised range and thermal data" << std::endl;

		for (int i = 0; i < startangles.size(); i++)
		{
			//skip shuttered frames
			if (startangles[i] < 0) continue;
			if (imgs[i].empty()) continue;
			//background subtract
			if (bgr_therm.find(startangles[i]) == bgr_therm.end())
			{
				bgr_therm.emplace(std::make_pair(startangles[i], cv::createBackgroundSubtractorMOG2(hist, thermVar, false)));
			}
			cv::Mat mask;
			bgr_therm.at(startangles[i])->apply(imgs[i], mask);
			//denoise - erode and dilate
			erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erodeWidth, erodeWidth)));
			dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilateWidth, dilateWidth))); //this is also designed to reconnect broken blobs

			std::vector<std::vector<cv::Point> > contours;

			findContours(mask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
			if (contours.empty()) continue;
			//use the lidar data to work out the range to the target

			//copy down the data so we can use it
			cv::Mat rangeImgMin(nrows, ncols, CV_32FC1);
			m_range_panos_min[id][i].copyTo((float*)rangeImgMin.data);
			cv::Mat rangeImgMax(nrows, ncols, CV_32FC1);
			m_range_panos_max[id][i].copyTo((float*)rangeImgMax.data);

			//apply the mask
			cv::multiply( rangeImgMin, mask, rangeImgMin,1.0, CV_32FC1);
			cv::multiply(rangeImgMax, mask, rangeImgMax, 1.0, CV_32FC1);

			//handle the NAN points - set them at zero
			cv::Mat nanmask = (rangeImgMin != rangeImgMin);
			cv::Mat mrMat = cv::Mat::zeros(rangeImgMin.size(), CV_32FC1);
			mrMat.copyTo(rangeImgMin, nanmask);
			nanmask = rangeImgMax != rangeImgMax;
			mrMat.copyTo(rangeImgMax, nanmask);

			//turn objects into bounding boxes
			std::vector<std::vector<cv::Point> > contours_poly(contours.size());
			for (int j = 0; j < contours.size(); j++)
			{
				cv::approxPolyDP(cv::Mat(contours[j]), contours_poly[j], 3, 1);
				auto rect = cv::boundingRect(cv::Mat(contours_poly[j]));
				if (rect.area() < minThermPts) continue;
				//work out the range from the estimates
				int  nnonzerorange = cv::countNonZero(rangeImgMin(rect));
				double range = nnonzerorange > 0 ? (cv::sum(rangeImgMin(rect))[0] + cv::sum(rangeImgMax(rect))[0])/ 2*nnonzerorange : DIDOLIDAR_MAXDIST;

				//classify the bounding box
				cv::Rect crect(std::max(rect.x - DidoAnalytics_ThermalClassifier::inflateWidth, DidoAnalytics_ThermalClassifier::borderWidth), std::max(rect.y - DidoAnalytics_ThermalClassifier::inflateWidth, DidoAnalytics_ThermalClassifier::borderWidth),
					std::min(rect.width + DidoAnalytics_ThermalClassifier::inflateWidth, imgs[i].cols - DidoAnalytics_ThermalClassifier::borderWidth - rect.x), std::min(rect.height + DidoAnalytics_ThermalClassifier::inflateWidth, imgs[i].rows - DidoAnalytics_ThermalClassifier::borderWidth- rect.y));
				if (crect.area() <= 0) crect = rect;
				auto classific = thermClassifier.classifyBlob(imgs[i](crect)); //inflate the ROI for the classifier
				if (classific.classvals[static_cast<int>(FirstLevelClassification::Unknown)] > rejectionThreshold) continue;
				float theta = startangles[i] + (width)*((float)(rect.x  + rect.width/ 2) / ncols);
				OV_WorldPoint pos(
					(float)range*cosf(theta),
					(float)range*sinf(theta),
					pixel_to_height(rect.y +  (rect.height/ 4), nrows, (float)range, tanAlpha));	//the interesting part of a person detection is the top half

				rval.push_back(std::make_shared<OVTrack>(0, pos, now, 0.3f));
				rval.back()->setBoundingBox(rect, startangles[i]);
				rval.back()->replaceFLClassification(classific);
			}
		}

		//cluster together nearby targets (as they are probably overlapping/multiply detected
		for (int i = 0; i < rval.size(); i++)
		{
			if (!rval[i]) continue;
			auto p = rval[i]->getLoc();
			for (int j = i + 1; j< rval.size(); j++)
			{
				if (!rval[j]) continue;
				//check if they are close enough to merge
				auto sep = rval[j]->getLoc() - p;
				if (sep.dot(sep) < mergedist)
				{
					rval[i]->updateWithPartialTrack(*rval[j]);
					rval[j].reset();
				}
			}
		}
		rval.erase(std::remove_if(rval.begin(), rval.end(), [](const std::shared_ptr<OVTrack> & p) {return p == nullptr; }), rval.end());
		return rval;
	}

	std::vector<std::shared_ptr<OVTrack>> DidoAnalytics_LateFused::produceTargets(const std::vector<cv::Mat> & imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar,
		const std::vector<Dido_Timestamp> &  timestamps, const std::vector<float> & startangles, const Dido_Timestamp & now)
	{
		int nrows = imgs[0].rows;
		int ncols = imgs[0].cols;
		auto id = std::this_thread::get_id();

		//get inputs per modality
		auto ltargs = getLidarTargets(lidar, now);
		auto ttargs = getThermalTargets(imgs, lidar, timestamps, startangles, now);

		//we need to combine the inputs, then match them to the outputs in space
		for (auto & l : ltargs)
		{
			auto p = l->getLoc();
			for (auto & t : ttargs)
			{
				if (!t) continue;
				//check if they are close enough to merge
				auto sep = t->getLoc() - p;

				if (sep.dot(sep) < mergedist)
				{
					t->updateWithPartialTrack(*l);
					l = t; //move the track like this so that we preserve the lidar size estimates, that are possibly better?
					t.reset();
					//break; //it's possible to have multiple detections of the same target, so we need to also collate those
				}
			}
		}
		//check for any other multiple detections
		for (int i = 0; i < ttargs.size(); i++)
		{
			if (!ttargs[i]) continue;
			auto p = ttargs[i]->getLoc();
			for (int j = i +1; j< ttargs.size(); j++)
			{
				if (!ttargs[j]) continue;
				//check if they are close enough to merge
				auto sep = ttargs[j]->getLoc() - p;
				if ( sep.dot(sep) < mergedist)
				{
					ttargs[i]->updateWithPartialTrack(*ttargs[j]);
					ttargs[j].reset();
				}
			}
		}

		//now append the removed ones
		for (auto & t : ttargs)
		{
			if (t) ltargs.push_back(t);
		}

		//compare to outputs
		static long newid = 0;
		for (auto & l : ltargs)
		{
			float maxdist = minPassoc;
			long id = -1;
			for (auto t : *trackStore)
			{
				if (t == nullptr) continue;
				auto sep = t->obsCloseEnough(l->getLoc(), now);
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
			l->trackID = id;
		}

		return ltargs;
	}

	void DidoAnalytics_LateFused::setRejectionThreshold(float rt)
	{
		rejectionThreshold = rt;
	}

	void DidoAnalytics_LateFused::loadLidarSVMModel(std::string file)
	{
		lidClassifier.load(file);
	}

	void DidoAnalytics_LateFused::saveLidarSVMModel(std::string file)
	{
		lidClassifier.save(file);
	}

	void DidoAnalytics_LateFused::loadThermalSVMModel(std::string file)
	{
		thermClassifier.load(file);
	}

	void DidoAnalytics_LateFused::saveThermalSVMModel(std::string file)
	{
		thermClassifier.save(file);
	}

	DidoAnalytics_LateFused::DidoAnalytics_LateFused(DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj)
		: bgsub_lidar(pars.binwidth, pars.binheight, pars.lidMindist, pars.lidHistory, pars.lidBGThresh,pars.lideps, pars.lidncore),
		hist(pars.history), lidhist(pars.lidHistory), epsilon(pars.lideps), ncore(pars.lidncore), minLidpts(pars.minLidpts), thermVar(pars.tVar), tanAlpha(pars.tanAlpha), minPassoc(pars.minPassoc), mergedist(pars.mergedist),
		lidClassifier(pars.lidSvmSaveFile, lidarSVMDefaultParameters()), thermClassifier(pars.thermalSvmSaveFile, thermalSVMDefaultParameters()), rejectionThreshold(pars.rejectionThreshold), width(obj.globalThermal->panoWidth()),
		erodeWidth(pars.erodeWidth), dilateWidth(pars.dilateWidth), minThermPts(pars.minpoints)
	{
		verticalDisplacement = pars.vertDisp;
		verticalFOV = pars.vertFOV;
		verticalOffset = pars.vertOff;

		lsynth.setBorderDecay(pars.borderDecay);
		lsynth.setCritDist(pars.critDist);
		lsynth.setCritTemp(pars.critTemp);
		lsynth.setNsteps(pars.nsteps);
		lsynth.setOutScale(1);
		lsynth.setScale(pars.scale);

	}
	
	std::vector<std::shared_ptr<OVTrack>> DidoAnalytics_OnlyLidar::produceTargets(const std::vector<cv::Mat>& imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano>& lidar, const std::vector<Dido_Timestamp>& timestamps, const std::vector<float>& startangles, const Dido_Timestamp & now)
	{
		//get inputs per modality
		auto ltargs = getLidarTargets(lidar, now);

		//compare to outputs
		static long newid = 0;
		for (auto & l : ltargs)
		{
			float maxdist = minPassoc;
			long id = -1;
			for (auto t : *trackStore)
			{
				if (t == nullptr) continue;
				auto sep = t->obsCloseEnough(l->getLoc(), now);
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
			l->trackID = id;
		}

		return ltargs;
	}

	DidoAnalytics_OnlyLidar::DidoAnalytics_OnlyLidar(DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj) : DidoAnalytics_LateFused(pars, obj)
	{
	}

	std::vector<std::shared_ptr<OVTrack>> DidoAnalytics_OnlyLidar_debug::getLidarTargets(const std::shared_ptr<DidoLidar::DidoLidar_Pano>& lidar, const Dido_Timestamp & now)
	{
		std::vector<std::shared_ptr<OVTrack>> rval;

		if (!lidar) return rval;	//no data means no data
									//background subtract the points
		auto fgpoints = bgsub_lidar.apply(lidar->data(), lidar->size());
		//display the foreground points
		cv::Mat fgimg = mapRanges(fgpoints.data(), (int)fgpoints.size(), dispScaling, cv::Size(displayWidth, displayWidth));
		cv::imshow("foreground points", fgimg);
		cv::waitKey(1);
		//cluster the foreground points
		auto fgtargs = bgsub_lidar.Cluster(fgpoints);

		for (auto & v : fgtargs)
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
		}
		return rval;
	}

	DidoAnalytics_OnlyLidar_debug::DidoAnalytics_OnlyLidar_debug(DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj, float _dispScaling, int _displayWidth) : DidoAnalytics_OnlyLidar(pars, obj), dispScaling(_dispScaling), displayWidth(_displayWidth)
	{

	}

	std::vector<std::shared_ptr<OVTrack>> DidoAnalytics_OnlyThermal::produceTargets(const std::vector<cv::Mat>& imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano>& lidar, const std::vector<Dido_Timestamp>& timestamps, const std::vector<float>& startangles, const Dido_Timestamp & now)
	{
		//get inputs per modality
		auto ltargs = getThermalTargets(imgs, lidar, timestamps, startangles, now);

		//compare to outputs
		static long newid = 0;
		for (auto & l : ltargs)
		{
			float maxdist = minPassoc;
			long id = -1;
			for (auto t : *trackStore)
			{
				if (t == nullptr) continue;
				auto sep = t->obsCloseEnough(l->getLoc(), now);
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
			l->trackID = id;
		}

		return ltargs;
	}

	DidoAnalytics_OnlyThermal::DidoAnalytics_OnlyThermal(DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj) : DidoAnalytics_LateFused(pars, obj)
	{
	}

	
	std::vector<std::shared_ptr<OVTrack>> DidoAnalytics_OnlyThermal::getThermalTargets(const std::vector<cv::Mat>& imgs, const std::shared_ptr<DidoLidar::DidoLidar_Pano> & lidar, const std::vector<Dido_Timestamp>& timestamps, const std::vector<float>& startangles, const Dido_Timestamp & now)
	{
		std::vector<std::shared_ptr<OVTrack>> rval;
		for (int i = 0; i < startangles.size(); i++)
		{
			//skip shuttered frames
			if (startangles[i] < 0) continue;
			if (imgs[i].empty()) continue;
			//background subtract
			if (bgr_therm.find(startangles[i]) == bgr_therm.end())
			{
				bgr_therm.emplace(std::make_pair(startangles[i], cv::createBackgroundSubtractorMOG2(hist, thermVar, false)));
			}
			cv::Mat mask;
			bgr_therm.at(startangles[i])->apply(imgs[i], mask);
			//denoise - erode and dilate
			erode(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erodeWidth, erodeWidth)));
			dilate(mask, mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilateWidth, dilateWidth))); //this is also designed to reconnect broken blobs

			std::vector<std::vector<cv::Point> > contours;

			findContours(mask, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

			//turn objects into bounding boxes
			std::vector<std::vector<cv::Point> > contours_poly(contours.size());
			for (int j = 0; j < contours.size(); j++)
			{
				if (contours[j].empty()) continue;
				cv::approxPolyDP(cv::Mat(contours[j]), contours_poly[j], 3, 1);
				auto rect = cv::boundingRect(cv::Mat(contours_poly[j]));
				if (rect.area() < minThermPts) continue;
				cv::Rect crect(std::max(rect.x - DidoAnalytics_ThermalClassifier::inflateWidth, DidoAnalytics_ThermalClassifier::borderWidth), std::max(rect.y - DidoAnalytics_ThermalClassifier::inflateWidth, DidoAnalytics_ThermalClassifier::borderWidth),
					std::min(rect.width + DidoAnalytics_ThermalClassifier::inflateWidth, imgs[i].cols - DidoAnalytics_ThermalClassifier::borderWidth - rect.x), std::min(rect.height + DidoAnalytics_ThermalClassifier::inflateWidth, imgs[i].rows - DidoAnalytics_ThermalClassifier::borderWidth - rect.y));
				if (crect.area() <= 0) crect = rect;
				//classify the bounding box
				auto classific = thermClassifier.classifyBlob(imgs[i](crect));
				if (classific.classvals[static_cast<int>(FirstLevelClassification::Unknown)] > rejectionThreshold) continue;

				//approximate range by assuming that everything is a person standing on the ground plane
				int offset = abs((int)imgs[i].rows / 2 - ((rect.y)));
				float height = 1.7f;
				//this may even be smart, as most of the time with occluded people you still get their heads
				float depth = height*sqrt((offset*offset) + imgs[i].rows*imgs[i].rows / (4 * tanAlpha)) / offset;

				//turn the rect into a track
				float theta = startangles[i] + (width)*(((float)(rect.x + rect.width / 2)) / imgs[i].cols);
				OV_WorldPoint pos(
					depth*cos(theta),
					depth*sin(theta),
					detail::pixel_to_height(rect.y + rect.height / 4, imgs[i].rows, depth, tanAlpha));

				rval.push_back(std::make_shared<OVTrack>(0, pos, now, 0.3f));
				rval.back()->setBoundingBox(rect, startangles[i]);
				rval.back()->replaceFLClassification(classific);
			}
		}
		return rval;
	}
	
}