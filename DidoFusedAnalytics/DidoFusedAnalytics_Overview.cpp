/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoFusedAnalytics_Overview.cpp
* @author  	SL
* @version 	1
* @date    	2017-08-08
* @brief   	Implimentation of the analytics using our own object detectors and a third party classifier
*****************************************************************************
**/

#include "GlobalIncludes.h"
#include "DidoFusedAnalytics_Overview.h"

namespace overview
{
	namespace
	{
		//defining a conversion from the dido analytics boundin box to a cvRect
		cv::Rect rectFromBB(const DidoFusedAnalytics_BoundingBox & bb)
		{
			return cv::Rect(cv::Point(bb.x, bb.y), cv::Point(bb.max_x, bb.max_y));
		}
	}

std::vector<std::shared_ptr<OVTrack>> DidoFusedAnalytics_Overview::processFrame(const cv::Mat & img, const unsigned  char * pano, const float * depths_min, const float * depths_max, float startangle, Dido_Timestamp ts, size_t rows, size_t cols)
{
	DidoLidar_Managed_Pano<float> fgranges_min((int)cols, (int)rows), fgranges_max((int)cols, (int)rows);
	auto timePoint = Dido_Clock::now();
	//bgr
	//construct a new one if there isnt' one
	if (bgrs.find(startangle) == bgrs.end())
	{
		bgrs.emplace(std::make_pair(startangle, DidoFusedAnalytics_3dBgSub_CUDA((int)rows, (int)cols, _regionBased, bgrScale)));
		bgrs.at(startangle).setHistory(history);
		bgrs.at(startangle).setThresholds(bgThresh, genThresh);
		bgrs.at(startangle).setVariance(tVar,rVar);
	}
	bgrs.at(startangle).apply(pano, depths_min, depths_max, fgranges_min.r_data(), fgranges_max.r_data());
	//object detection
	auto bbs = obdet.detectBlobs(fgranges_min.data(), fgranges_max.data(), (int)rows, (int)cols);
	
	//run the classifier on the images of the target
	//and turn the bounding boxes into tracks
	std::vector<std::shared_ptr<OVTrack>> rval;
	
	for (auto & b : bbs)
	{
		//check that the bounding box is sane
		if (b.x < 0 || b.y < 0 || b.x > cols || b.y > rows || b.max_x < 0 || b.max_y < 0 || b.max_x > cols || b.max_y > rows) continue;
		cv::Rect objbb = rectFromBB(b);
		cv::Rect crect (std::max(objbb.x - DidoAnalytics_ThermalClassifier::inflateWidth, DidoAnalytics_ThermalClassifier::borderWidth), std::max(objbb.y - DidoAnalytics_ThermalClassifier::inflateWidth, DidoAnalytics_ThermalClassifier::borderWidth),
			std::min(objbb.width + DidoAnalytics_ThermalClassifier::inflateWidth, img.cols - DidoAnalytics_ThermalClassifier::borderWidth - objbb.x), std::min(objbb.height + DidoAnalytics_ThermalClassifier::inflateWidth, img.rows - DidoAnalytics_ThermalClassifier::borderWidth - objbb.y));
		if (crect.area() <= 0) crect = objbb;
		auto classific = classifyROI(img(crect));

		//work out it's position in a cylindrical projection
		float theta =  startangle + (imgwidth)*(((float)(b.x + b.max_x) / 2) / cols);
		OV_WorldPoint pos(
			b.avdepth*cos(theta),
			b.avdepth*sin(theta),
			//aim at three quarters up because we want faces
			pixel_to_height((b.y *3 + b.max_y)/4, rows, b.avdepth,tanalpha));

		static long newid = 0;

		float minp = minPAssoc;
		long id = -1;
		for (auto t : *trackStore)
		{
			if (t == nullptr) continue;
			auto sep = t->obsCloseEnough(pos,timePoint);
			if (sep > minp)
			{
				minp = sep;
				id = t->trackID;
			}
		}
		if (id < 0)
		{
			id = newid++;
		}

		rval.push_back(std::make_shared<OVTrack>(id, pos, timePoint, 1 / (b.uncertainty + 1)));

		/*size calculation*/
		rval.back()->width = b.avdepth * 2 * pi*(b.max_x - b.x)/ cols;
		rval.back()->height = pixel_to_height( b.max_y , rows, b.avdepth, tanalpha) - pixel_to_height(b.y , rows, b.avdepth, tanalpha);
		/*if it's close enough that we can get a range to it, we're more confident*/
		if (b.avdepth < max_range) rval.back()->detectionConfidence = rval.back()->detectionConfidence + 0.1f;
		rval.back()->replaceClassification(classific);
//		rval.back()->replaceBehaviour(behav);
		rval.back()->setBoundingBox(objbb,startangle);
		//work out the target's "threat level"		
	}

	return rval;
}

OVTrack_Classification DidoFusedAnalytics_Overview::classifyROI(const cv::Mat & img)
{
	OVTrack_Classification rval;
	rval.FirstLevel = tclass.classifyBlob(img);
	return rval;
}


DidoFusedAnalytics_Overview::DidoFusedAnalytics_Overview(int hist, bool rbased)
	: history(hist), _regionBased(rbased), bgrScale(2), tclass("thermalSVM.xvl", thermalSVMDefaultParameters())
{ 
}

DidoFusedAnalytics_Overview::DidoFusedAnalytics_Overview(DidoMain_ParamEnv & pars,	DidoMain_ObjEnv & obj)
	:max_range(pars.maxLidarRange), tanalpha(pars.tanAlpha), imgwidth(obj.globalThermal->panoWidth()), history(pars.history), tVar(pars.tVar), rVar(pars.rVar),
	bgThresh(pars.bgThresh), genThresh(pars.genThresh), _regionBased(pars.bgrRegionBased), minPAssoc(pars.minPassoc), bgrScale(pars.fusedBGRScale), rejectionThreshold(pars.rejectionThreshold),
	tclass(pars.thermalSvmSaveFile, thermalSVMDefaultParameters()), obdet(pars.minpoints,pars.rangescaling,pars.ncore,pars.epsilon)
{
}
}