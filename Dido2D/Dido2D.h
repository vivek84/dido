/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	Dido2D.h
* @author  	SL
* @version 	1
* @date    	2017-03-29
* @brief    Class that handles the reception of the data from the PTZ, including interfacing for face recognition and visual analytics
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __Dido2D
#define __Dido2D


#pragma once

#include "OVTrack.h"
#include "db_tracks.h"
#include "ShareableTrackList.h"

//temporary proxy until we get an actual face rec system
#include "FaceDetector/FaceDetector.h"


/*defines ---------------------------*/


namespace overview
{
	/*abstraction of the interface to the Dido2D so that alternative inheritance is efficient*/

	class Dido2D_iface
	{
	protected:
		/*pointer to the global set of tracks*/
		std::shared_ptr<OVTrack_Storage> trackStore;

		//tools to allow us to give up on bad targets
		int nobsCTarg = 0;
		std::weak_ptr<const OVTrack> ctarg;

		//expected error in angle estimate
		float angleSlack = 0.3f;

		//condition variable to allow the DMM to tell it when the camera is still
		std::condition_variable camStillCondVar;
		std::mutex camStateMut;
		bool cameraStill = false;
		OV_PTZBear cameraBearing;
		//pointer to the target iterator so that we can progress our target selection when we have a target
		std::shared_ptr<ShareableTrackList> targetit;

		//the PTZ camera
		std::shared_ptr<DidoVCap> vcap;

		//main loop thread elements
		std::thread loopthread;
		//the core processing loop of the object
		virtual void loopfunction() = 0;
		bool running = false;
	public:

		/*needs a pointer to the global store on construction*/
		Dido2D_iface(std::shared_ptr<OVTrack_Storage> _trackstore, std::string _videoAddress);
		Dido2D_iface(std::shared_ptr<OVTrack_Storage> _trackstore,  std::shared_ptr<DidoVCap> _vcap);


		/*runs continuously - calling this function starts it's thread*/
		void start();
		/*stops it running*/
		void stop();

		/*parameter setting functions*/
		void setVideoAddress(std::string vidaddress);


		//functions called by the DMM to communicate the motion status of the camera
		void declareCameraStopped(OV_PTZBear const & cambear);
		void declareCameraMoving();
		void setTargetIterator(std::shared_ptr < ShareableTrackList > targetIt);

	};


	class Dido2D: public Dido2D_iface
	{
	protected:
		/*the SQL database that the gui uses to find out about the tracks*/
		std::shared_ptr<overview::db::store_to_database> dbconnect;
		int currentFaceID;	//this object defines the unique ID for each face shot

		//some connection to the Identification stuff
		FaceDetector face;

		//2d person classifier
		cv::Ptr<cv::cuda::HOG> hog;

		virtual void loopfunction() override;

	public:
		~Dido2D();
		/*needs a pointer to the global store on construction*/
		Dido2D(std::shared_ptr<OVTrack_Storage> _trackstore, std::shared_ptr<overview::db::store_to_database> _dbconnect, std::string _videoAddress,
			std::string cascadefile = "haarcascade_frontalface_alt.xml", std::string DeepFolder = "../vgg_face_caffe", std::string faceFolder = "facefiles", bool pretend = false);
		Dido2D(std::shared_ptr<OVTrack_Storage> _trackstore, std::shared_ptr<overview::db::store_to_database> _dbconnect, std::shared_ptr<DidoVCap> _vcap,
			std::string cascadefile = "haarcascade_frontalface_alt.xml", std::string DeepFolder = "../vgg_face_caffe", std::string faceFolder = "facefiles", bool pretend = false);

		/*generates classifications from an RGB image - so we don't require two instances of heavy engines
		returns a vector of the classifications in the image and the ROI associated with each classification in the case of mulitple classifications
		*/
		virtual std::vector<std::pair<OVTrack_Classification, cv::Rect> > classifyImage(const cv::Mat & img);

		//whether to show the faces detected per frame (for display purposes)
		bool displayFaces = true;

		//other face recognition parameters
		void setDeepNetFolder(std::string folder);
		void setFaceFeatFolder(std::string folder);
		void setHaarParameters(int minNeighbours, int minSize);
	};
}
#endif
