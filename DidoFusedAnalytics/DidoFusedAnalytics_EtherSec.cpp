/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoFusedAnalytics_EtherSec.cpp
* @author  	SL
* @version 	1
* @date    	2017-04-20
* @brief   	Implementation of the analytics using Ethersec's interface
*****************************************************************************
**/

#include "GlobalIncludes.h"
#include "EtherSecInterfaceProxy.h"
#include "DidoFusedAnalytics_EtherSec.h"
#include "DidoLidar_CUDA.h"


namespace overview
{
	//need to populate this correctly
	enum CESI_ObjProb_Types
	{
		human = 0,
		car,
		van,
		bus,
		goat,
		horse,
		unknown
	};




	constexpr float max_range = 70;

	void DidoFusedAnalytics_EtherSec::startUp(int _ncams, size_t ncols, size_t nrows)
	{
	/*	LOG_INFO << "starting up the Ethersec Analytics" << std::endl;

		vidAnalytics = std::make_unique<esi::OvVa<unsigned char>>(_ncams, nrows, ncols, 3, ncols/_ncams);
		ncams = _ncams;
		if (!vidAnalytics->IsInit()) throw std::exception("video analytics wouldn't load");


		LOG_INFO << "Starting up the CNN" << std::endl;

		HRESULT hr = CoInitialize(NULL);
		hr = CoCreateInstance(CLSID_ESI_CNNAdapter, NULL, CLSCTX_INPROC,
			IID_IESI_CNNAdapter, (void **)&CNNAdapter);
		if (hr < 0 )
		{
			LOG_ERROR << "ERROR - Could not create the CNN." << std::endl;
			if (CNNAdapter)
			{
				CNNAdapter->Release();
			}
			return;
		}
		//TODO - invoke it 

		long isinit;
		hr = CNNAdapter->IsCNNProcessorInited(&isinit);
		if (hr < 0 || isinit < 1)
		{
			LOG_ERROR << "ERROR - Could not initalise the CNN." << std::endl;
			if (CNNAdapter)
			{
				CNNAdapter->Release();
			}
			return;
		}
		*/
	}

	static OVTrack_Classification convertESClass(CESI_TrackingObjectOpticalCameraData & ESClassific)
	{
		OVTrack_Classification rval;
		for (int i = 0; i < 5; i++)
		{
			switch (ESClassific.m_ObjProb_Type[i])
			{
				//this may not be correct
			case CESI_ObjProb_Types::human:
				rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Human)] = std::max(rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Human)], ESClassific.m_ObjProb_Type_Confidence[i]);
				rval.FirstLevel.confidence = std::max(ESClassific.m_ObjProb_Type_Confidence[i], rval.FirstLevel.confidence);
				break;
			case CESI_ObjProb_Types::car:
				rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Vehicle)] = std::max(rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Vehicle)], ESClassific.m_ObjProb_Type_Confidence[i]);
				rval.FirstLevel.confidence = std::max(ESClassific.m_ObjProb_Type_Confidence[i], rval.FirstLevel.confidence);
				rval.vehicle.classvals[static_cast<int>(VehicleSubClassification::Car)] = ESClassific.m_ObjProb_Type_Confidence[i];
				rval.vehicle.confidence = std::max(ESClassific.m_ObjProb_Type_Confidence[i], rval.vehicle.confidence);
				break;
			case CESI_ObjProb_Types::van:
				rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Vehicle)] = std::max(rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Vehicle)], ESClassific.m_ObjProb_Type_Confidence[i]);
				rval.FirstLevel.confidence = std::max(ESClassific.m_ObjProb_Type_Confidence[i], rval.FirstLevel.confidence);
				rval.vehicle.classvals[static_cast<int>(VehicleSubClassification::Van)] = ESClassific.m_ObjProb_Type_Confidence[i];
				rval.vehicle.confidence = std::max(ESClassific.m_ObjProb_Type_Confidence[i], rval.vehicle.confidence);
				break;
			case CESI_ObjProb_Types::bus:
				rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Vehicle)] = std::max(rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Vehicle)], ESClassific.m_ObjProb_Type_Confidence[i]);
				rval.FirstLevel.confidence = std::max(ESClassific.m_ObjProb_Type_Confidence[i], rval.FirstLevel.confidence);
				rval.vehicle.classvals[static_cast<int>(VehicleSubClassification::Bus)] = ESClassific.m_ObjProb_Type_Confidence[i];
				rval.vehicle.confidence = std::max(ESClassific.m_ObjProb_Type_Confidence[i], rval.vehicle.confidence);
				break;
			case CESI_ObjProb_Types::goat:
				rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Animal)] = std::max(rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Animal)], ESClassific.m_ObjProb_Type_Confidence[i]);
				rval.FirstLevel.confidence = std::max(ESClassific.m_ObjProb_Type_Confidence[i], rval.FirstLevel.confidence);
				rval.animal.classvals[static_cast<int>(AnimalSubClassification::Goat)] = ESClassific.m_ObjProb_Type_Confidence[i];
				rval.animal.confidence = std::max(ESClassific.m_ObjProb_Type_Confidence[i], rval.animal.confidence);
				break;
			case CESI_ObjProb_Types::horse:
				rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Animal)] = std::max(rval.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Animal)], ESClassific.m_ObjProb_Type_Confidence[i]);
				rval.FirstLevel.confidence = std::max(ESClassific.m_ObjProb_Type_Confidence[i], rval.FirstLevel.confidence);
				rval.animal.classvals[static_cast<int>(AnimalSubClassification::Horse)] = ESClassific.m_ObjProb_Type_Confidence[i];
				rval.animal.confidence = std::max(ESClassific.m_ObjProb_Type_Confidence[i], rval.animal.confidence);
				break;
			default:
				break;
			}
		}

		//normalise them
		float total = 0;
		for (auto & c : rval.FirstLevel.classvals)
		{
			total += c;
		}
		if (total > 0)
		{
			for (auto & c : rval.FirstLevel.classvals)
			{
				c = c / total;
			}
		}

		total = 0;
		for (auto & c : rval.human.classvals)
		{
			total += c;
		}
		if (total > 0)
		{
			for (auto & c : rval.human.classvals)
			{
				c = c / total;
			}
		}

		total = 0;
		for (auto & c : rval.vehicle.classvals)
		{
			total += c;
		}
		if (total > 0)
		{
			for (auto & c : rval.vehicle.classvals)
			{
				c = c / total;
			}
		}

		total = 0;
		for (auto & c : rval.animal.classvals)
		{
			total += c;
		}
		if (total > 0)
		{
			for (auto & c : rval.animal.classvals)
			{
				c = c / total;
			}
		}

		total = 0;
		for (auto & c : rval.object.classvals)
		{
			total += c;
		}
		if (total > 0)
		{
			for (auto & c : rval.object.classvals)
			{
				c = c / total;
			}
		}
		return rval;
	}

	/*function to convert ethersec's number of instances into a confidence*/
	float instancesToConfidence(int instances)
	{
		return (instances > 0) ? std::min(1.0f, (float)instances / 20) : 0;
	}

	static OVTrack_Behaviour convertESBehav(CESI_TrackingObjectNeualNetworkDetections & dets)
	{
		OVTrack_Behaviour rval;
		//zero initialise
		rval.classvals.fill(0);
		rval.classvals[static_cast<int>(BehaviourClassification::BodyDrag)] = (dets.m_BodyDragDetectionInstances > 0) ? (float)dets.m_BodyDrag / dets.m_BodyDragDetectionInstances : 0;
		rval.classvals[static_cast<int>(BehaviourClassification::Crawl)] = (dets.m_CrawlDetectionInstances > 0) ? (float)dets.m_Crawl / dets.m_CrawlDetectionInstances : 0;
		rval.classvals[static_cast<int>(BehaviourClassification::Crouch)] = (dets.m_CrouchingDetectionInstances > 0) ? (float)dets.m_Crouching / dets.m_CrouchingDetectionInstances : 0;
		rval.classvals[static_cast<int>(BehaviourClassification::LogRoll)] = (dets.m_LogRollDetectionInstances > 0) ? (float)dets.m_LogRoll / dets.m_LogRollDetectionInstances : 0;
		rval.classvals[static_cast<int>(BehaviourClassification::Walking)] = (dets.m_RunningWalkingDetectionInstances > 0) ? (float)dets.m_RunningWalking / dets.m_RunningWalkingDetectionInstances : 0;

		//confidence is based on the number of detections the best classification has
		rval.confidence = instancesToConfidence(std::max(std::max(dets.m_BodyDrag, std::max(dets.m_Crawl, dets.m_Crouching)), std::max(dets.m_LogRoll, dets.m_RunningWalking)));
		rval.classvals[static_cast<int>(BehaviourClassification::Unknown)] = 1 - rval.confidence;

		//normalise it
		float total = 0;
		for (int i = 0; i < OVTrack_NBehav; i++)
		{
			total += rval.classvals[i];
		}
		if (total > 0)
		{
			for (int i = 0; i < OVTrack_NBehav; i++)
			{
				rval.classvals[i] = rval.classvals[i] / total;
			}
		}
		return rval;
	}

	//they give us the bounding box as per their image, not ours
	constexpr int ES_image_rows = 287;
	constexpr int ES_image_cols = 384;

	/*function called on each ethersec camera to get the detections*/
	void DidoFusedAnalytics_EtherSec::ES_processCamera(CESI_Camera & cam, std::vector<std::shared_ptr<OVTrack>> & trcks, Dido_Timestamp & ts, float sa)
	{
		for (int i = 0; i < cam.m_nTrackedObjects; i++)
		{
			cv::Rect bb = cam.m_ESI_TrackingObject[i].m_BoundingBox;
			//correct the rotation
			bb.x += cam.m_CameraNum*ES_image_cols;
			//scale it
			bb.x = (bb.x*imgsize.width) / (ES_image_cols*ncams);
			bb.width = (bb.width*imgsize.width) / (ES_image_cols*ncams);
			bb.y = (bb.y*imgsize.height) / ES_image_rows;
			bb.height = (bb.height*imgsize.height) / ES_image_rows;
			//work out it's position in a cylindrical projection
			float theta = 2 * pi*(bb.x + bb.width / 2) / imgsize.width;
			OV_WorldPoint pos(
				cam.m_ESI_TrackingObject[i].m_Depth*sin(theta),
				cam.m_ESI_TrackingObject[i].m_Depth*cos(theta),
				(bb.y + bb.height)*cam.m_ESI_TrackingObject[i].m_Depth*tanalpha / imgsize.height);
			/*create the track (for now we generate our own GUID based on the IDInFrame*/
			std::shared_ptr<OVTrack> nTrack = std::make_shared<OVTrack>((cam.m_ESI_TrackingObject[i].m_GUID == -1) ? (cam.m_ESI_TrackingObject[i].m_IDInFrame + (cam.m_CameraNum << 8)) : cam.m_ESI_TrackingObject[i].m_GUID, pos, ts, 0.4f);
			/*size calculation*/
			nTrack->width = cam.m_ESI_TrackingObject[i].m_Depth * 2 * pi*bb.width / imgsize.width;
			nTrack->height = bb.height*cam.m_ESI_TrackingObject[i].m_Depth*tanalpha / imgsize.height;
			/*if it's close enough that we can get a range to it, we're more confident*/
			if (cam.m_ESI_TrackingObject[i].m_Depth < max_range) nTrack->detectionConfidence = 0.6f;
			nTrack->replaceClassification(convertESClass(cam.m_ESI_TrackingObject[i].m_OpticalCassification));
			nTrack->replaceBehaviour(convertESBehav(cam.m_ESI_TrackingObject[i].m_ESI_TrackingObjectNeualNetworkDetections));
			nTrack->setBoundingBox(bb,sa);
			trcks.push_back(nTrack);
		}
	}


	std::vector<std::shared_ptr<OVTrack>> DidoFusedAnalytics_EtherSec::processFrame(const cv::Mat & img, const unsigned  char * pano, const float * depths_min, const float * depths_max, float startangle, Dido_Timestamp ts, size_t rows, size_t cols)
	{
		std::vector<std::shared_ptr<OVTrack>> rval;
		//for the test, convert it to 3 channel unsigned char
		//DidoLidar_Managed_Pano<unsigned char> ucharpano((int)rows*3, (int)cols);
		//DidoLidar_Convert16UC1To8UC3(pano, ucharpano.data(), (int)(rows*cols));
		//give it to Ethersec
		dresults =  ThermalPanoramaDetect(pano, depths_min,  rows,  cols);

		//vidAnalytics->ProcessFrame(ucharpano.data(), depths, cols*3, cols*sizeof(float));
		//dresults = vidAnalytics->GetResults();

		//process it
		for (int i = 0; i < dresults.nCams; i++)
		{
			ES_processCamera(dresults.m_Camera[i], rval, ts, startangle);
		}
		return rval;
	}
	DidoFusedAnalytics_EtherSec::~DidoFusedAnalytics_EtherSec()
	{
	//	if (CNNAdapter) CNNAdapter->Release();
	}

	DidoFusedAnalytics_EtherSec::DidoFusedAnalytics_EtherSec(cv::Size _imgSize)
		: imgsize(_imgSize)
	{
	}
}