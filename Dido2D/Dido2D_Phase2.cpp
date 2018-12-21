/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	Dido2D_Phase2.cpp
* @author  	SL
* @version 	1
* @date    	2018-01-24
* @brief    Class that handles the reception of the data from the PTZ for phase 2
*****************************************************************************
**/

#include "GlobalIncludes.h"
#include "Dido2D_Phase2.h"
#include "PTZActuator.h"

namespace overview
{
	void Dido2D_Phase2::loopfunction()
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
				//get the current image and give it to a target
				cv::Mat img;
				vcap->read(img);
				if (!img.empty())
				{
					auto now = Dido_Clock::now();
					
						//find the person it belongs to and attach it to them
						float fmindist = (bearing.zoom + angleSlack) * 2;
						std::weak_ptr<OVTrack> ftarget;
						for (std::weak_ptr<OVTrack> wt : *trackStore)	//want copies so that we keep them alive
						{
							//explicitly create a weak then lock it
							if (auto t = wt.lock())
							{
								//measure to the predicted location
								auto bear = PTZActuator::getBearing(t->predictLoc(now));

									float fdist = abs(bear.pan - bearing.pan) + abs(bear.tilt - bearing.tilt);
									if (fdist < fmindist)
									{
										fmindist = fdist;
										ftarget = t;
									}
							}
						}
						if (auto tmp = ftarget.lock())
						{
							cv::imwrite(saveloc + "/target_" + std::to_string(imgID) + ".png", img);
							float idq = 1.0f / (1 + tmp->IDQuality);
							tmp->pushBackFaceShot(imgID, now, idq);
							imgID++;

							if (tmp == targetit->lock())
							{
								//move down the targetlist if we've updated it.
								if (targetit->notAtEnd())(*targetit)++;
							}
						}
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

	Dido2D_Phase2::~Dido2D_Phase2()
	{
	}

	Dido2D_Phase2::Dido2D_Phase2(std::shared_ptr<OVTrack_Storage> _trackstore, std::string address)
		: Dido2D_iface(_trackstore, address)
	{
	}

	Dido2D_Phase2::Dido2D_Phase2(std::shared_ptr<OVTrack_Storage> _trackstore, std::shared_ptr<DidoVCap> _vcap)
		: Dido2D_iface(_trackstore,  _vcap)
	{
	}
}