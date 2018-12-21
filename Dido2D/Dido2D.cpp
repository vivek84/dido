/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	Dido2D.cpp
* @author  	SL
* @version 	1
* @date    	2017-03-29
* @brief    Class that handles the reception of the data from the PTZ, including interfacing for face recognition and visual analytics
*****************************************************************************
**/

#include "GlobalIncludes.h"
#include "Dido2D.h"
#include "PTZActuator.h"

namespace overview
{
	void Dido2D::loopfunction()
	{
		while (running && keep_running())
		{
			OV_PTZBear bearing;
			//wait to be told that you're still
			{
				std::unique_lock<std::mutex> lock(camStateMut);
				while (!cameraStill && keep_running())
				{
					camStillCondVar.wait(lock);
				}
				bearing = cameraBearing;
			}
			{
				//get the current image
				cv::Mat img;
				vcap->read(img);
				if (!img.empty())
				{
					auto now = Dido_Clock::now();
					//run it through face rec
					LOG_INFO << "detecting faces" << std::endl;
					auto faces = face.detectFaces(img);
					//classify objects in the imnage
					auto classifics = classifyImage(img);

					auto fit = faces.begin();
					auto cit = classifics.begin();


					if (displayFaces & !faces.empty())
					{
						cv::Mat displayface;
						for (auto & f : faces)
						{
							if (std::get<0>(f).area() > 10)
							{
								cv::Mat nf = img(std::get<0>(f));
								cv::resize(nf, nf, cv::Size(255, 255));
								int step = 0;
								for(auto & id : std::get<1>(f))
									cv::putText(nf, std::to_string(id), cv::Point(30*(++step), 30), 1, 1.2, cv::Scalar(0, 255, 255));
								if (displayface.empty())
								{
									displayface = nf;
								}
								else
								{
									cv::hconcat(displayface, nf, displayface);
								}
							}
						}
						if (!displayface.empty()) cv::imshow("faces", displayface);
						cv::waitKey(1);
					}

					//put the faces on the targets
					while (fit!=faces.end() || cit != classifics.end())
					{		
						float idq, fxadjust, fyadjust;
						if (fit != faces.end() && std::get<0>(*fit).area() > 10)
						{
							idq = std::get<2>(*fit);
							/*the adjustment to the center of the face shot*/
							fxadjust = bearing.zoom*((float)(std::get<0>(*fit).x + std::get<0>(*fit).width / 2) / img.cols - 0.5f);
							fyadjust = bearing.zoom*((float)(std::get<0>(*fit).y + std::get<0>(*fit).height / 2) / img.rows - 0.5f);

							//save it -  how do we record to whom it belongs? use the ID!!
							if (dbconnect) (*dbconnect)(img(std::get<0>(*fit)), std::get<1>(*fit));
						}
						else
						{
							idq = fxadjust = fyadjust = 0;
						}
						//find the person it belongs to and attach it to them
						float fmindist = (bearing.zoom + angleSlack) * 2;
						std::weak_ptr<OVTrack> ftarget;

						float cxadjust, cyadjust;
						if (cit != classifics.end())
						{
							cxadjust = bearing.zoom*((float)(cit->second.x + cit->second.width / 2) / img.cols - 0.5f);
							cyadjust = bearing.zoom*((float)(cit->second.y + cit->second.height / 2) / img.rows - 0.5f);
						}
						else
						{
							cxadjust = cyadjust = 0;
						}
						//find the person it belongs to and attach it to them
						float cmindist = fmindist;
						std::weak_ptr<OVTrack> ctarget;
						for (std::weak_ptr<OVTrack> wt : *trackStore)	//want copies so that we keep them alive
						{
							//explicitly create a weak then lock it
							if (auto t = wt.lock())
							{
								//measure to the predicted location
								auto bear = PTZActuator::getBearing(t->predictLoc(now));
								if (fit != faces.end() && std::get<0>(*fit).area() > 10)
								{
									float fdist = abs(bear.pan - bearing.pan - fxadjust) + abs(bear.tilt - bearing.tilt - fyadjust);
									if (fdist < fmindist)
									{
										fmindist = fdist;
										ftarget = t;
									}
								}
								if (cit != classifics.end())
								{
									float cdist = abs(bear.pan - bearing.pan - cxadjust) + abs(bear.tilt - bearing.tilt - cyadjust);
									if (cdist < cmindist)
									{
										cmindist = cdist;
										ctarget = t;
									}
								}
							}
						}
						if (auto tmp = ftarget.lock())
						{
							tmp->pushBackFaceShot(std::get<1>(*fit).front(), now, idq);
							LOG_DEBUG << "gave face " << std::get<1>(*fit).front() << " to target " << tmp->trackID << std::endl;

							if (tmp == targetit->lock())
							{
								//move down the targetlist if we've updated it.
								if (targetit->notAtEnd())(*targetit)++;
							}
						}

						if (auto tmp =  ctarget.lock())
						{
							tmp->updateClassification(cit->first);
							LOG_DEBUG << "updated classification for target " << tmp->trackID << std::endl;

							if (tmp == targetit->lock())
							{
								//move down the targetlist if we've updated it.
								if (targetit->notAtEnd())(*targetit)++;
							}
						}

						if (fit != faces.end()) fit++;
						if (cit != classifics.end()) cit++;
					}
					LOG_INFO << "finished detecting faces";
				}
			}
			if (targetit->lock() == ctarg.lock())
			{
				nobsCTarg++;
				if (nobsCTarg > 3) 
				{
					nobsCTarg = 0;
					if (targetit->notAtEnd())(*targetit)++;
				}
			}
			else
			{
				nobsCTarg = 0;
				ctarg = targetit->getTarg();
			}
		}
	}

	Dido2D::~Dido2D()
	{
		stop();
	}

	Dido2D::Dido2D(std::shared_ptr<OVTrack_Storage> _trackstore, std::shared_ptr<overview::db::store_to_database> _dbconnect, 
		std::string _videoAddress, std::string cascadefile, std::string deepfolder, std::string facefolder, bool pretend)
		: Dido2D_iface(_trackstore, _videoAddress), dbconnect(_dbconnect), face(cascadefile, deepfolder, facefolder, pretend)
	{
		hog = cv::cuda::HOG::create();
		hog->setGroupThreshold(0);
	}

	Dido2D::Dido2D(std::shared_ptr<OVTrack_Storage> _trackstore, std::shared_ptr<overview::db::store_to_database> _dbconnect, 
		std::shared_ptr<DidoVCap> _vcap, std::string cascadefile, std::string deepfolder, std::string facefolder, bool pretend)
		: Dido2D_iface(_trackstore, _vcap),dbconnect(_dbconnect), face(cascadefile, deepfolder, facefolder, pretend)
	{
		hog = cv::cuda::HOG::create();
		hog->setGroupThreshold(0);
	}

	std::vector<std::pair<OVTrack_Classification, cv::Rect>> Dido2D::classifyImage(const cv::Mat & img)
	{
		//use the standard OpenCV pedestrian detector perhaps?
		cv::cuda::GpuMat gpimg(img);
		cv::cuda::cvtColor(gpimg, gpimg, CV_BGR2GRAY);
		std::vector<cv::Rect> dets;
		std::vector<double> confidences(0);
		hog->detectMultiScale(gpimg, dets, &confidences);
		std::vector<std::pair<OVTrack_Classification, cv::Rect>> rval;
		for (size_t i = 0; i < dets.size() && i < confidences.size(); i++)
		{
			OVTrack_Classification cs;
			cs.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Human)] = 0.8f;
			cs.FirstLevel.classvals[static_cast<int>(FirstLevelClassification::Object)] = 0.2f;
			cs.FirstLevel.confidence = (float)confidences[i];
			rval.push_back(std::pair<OVTrack_Classification, cv::Rect>(cs, dets[i]));
		}
		return rval;
	}

	Dido2D_iface::Dido2D_iface(std::shared_ptr<OVTrack_Storage> _trackstore, std::string _videoAddress)
		:trackStore(_trackstore), vcap(new DidoVCap(_videoAddress))
	{
	}

	Dido2D_iface::Dido2D_iface(std::shared_ptr<OVTrack_Storage> _trackstore, std::shared_ptr<DidoVCap> _vcap)
		: trackStore(_trackstore), vcap(_vcap)
	{
	}

	void Dido2D_iface::start()
	{
		if (!running)
		{
			running = true;
			loopthread = std::thread(&Dido2D_iface::loopfunction, this);
		}
	}

	void Dido2D_iface::stop()
	{
		if (running)
		{
			running = false;
			{
				std::lock_guard<std::mutex> lock(camStateMut);
				cameraStill = true;
			}
			camStillCondVar.notify_all();
			if (loopthread.joinable())loopthread.join();
		}
	}

	void Dido2D_iface::setVideoAddress(std::string vidaddress)
	{
		vcap->open(vidaddress);
	}
	void Dido2D::setDeepNetFolder(std::string folder)
	{
		face.setDeepNetFolder(folder);
	}
	void Dido2D::setFaceFeatFolder(std::string folder)
	{
		face.setFaceFeatFolder(folder);
	}
	void Dido2D::setHaarParameters(int minNeighbours, int minSize)
	{
		face.mindetect = minNeighbours;
		face.minsize = cv::Size(minSize, minSize);
	}
	void Dido2D_iface::declareCameraStopped(OV_PTZBear const & cambear)
	{
		std::lock_guard<std::mutex> lock(camStateMut);
		cameraBearing = cambear;
		if (!cameraStill)
		{
			cameraStill = true;
			camStillCondVar.notify_one();
		}
	}
	void Dido2D_iface::declareCameraMoving()
	{
		std::lock_guard<std::mutex> lock(camStateMut);
		cameraStill = false;
	}
	void Dido2D_iface::setTargetIterator(std::shared_ptr<ShareableTrackList> tl)
	{
		std::lock_guard<std::mutex> lock(camStateMut);
		targetit = tl;
	}
}