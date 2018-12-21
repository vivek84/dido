/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoDMM.h
* @author  	SL
* @version 	1
* @date    	2017-03-07
* @brief   	Overall state machine and decision making element for the active elements of dido
*			Chooses the next target for the PTZ to look at in order to maximise the identification over all targets and time
*****************************************************************************
**/

#include "GlobalIncludes.h"
#include "DidoDMM.h"
#include "DidoDMM_TargetSelector.h"
#include "PTZException.h"

/*private functions*/
namespace overview
{
	void DidoDMM::controlThreadFunction()
	{
		while (running && keep_running())
		{
			if (!overridden)
			{
				/*are there any targets?*/
				while (sortedTargets)
				{	
					std::shared_ptr<OVTrack> tmp;
					{
						//scope for the lock
						std::lock_guard<std::mutex> tlck(targetmut);
						if (!sortedTargets->notAtEnd()) break;
						tmp = (*(*sortedTargets)).lock();
					}
					/*check that our target still exists*/
					if (tmp)
					{
						
						OV_WorldPoint vel = tmp->getVel();
						/*work out where we want to point - lead the target*/
						auto now = Dido_Clock::now();
						auto obs = tmp->getCurrObs();		//the longer it's been since we saw it, the greater our uncertainty about position so the more we zoom out
						//float timeErr =((float)std::chrono::duration_cast<std::chrono::milliseconds>(now - obs.timestamp).count()) / 1000;
						//offsetting target location from the center of the target to improve our chances of getting faces
						OV_PTZBear targLoc = ptza->standardiseBearing(ptza->getBearing( tmp->predictLoc(now) - PTZOffset + cv::Point3f(0,0,0.3f), 3.2f +  tmp->width ));
						//height varys depedning on sensor, so isn't as reliable
						/*see if the camera is close enough*/
						std::lock_guard<std::mutex> plck(ptzmut);
						try
						{
							curpos = ptza->convertPTZ(ptza->getAbsPosition());
						}
						catch (PTZException e)
						{
							LOG_WARN << e.what();
							//make sure that the 2d analytics runs, then wait a while - for DEBUG
							curpos = targLoc;
							analytics->declareCameraStopped(curpos);
							my_sleep(100);
						}
						cv::Point3f cpos = (cv::Point3f)curpos.load();
						cv::Point3f sep = cpos - (cv::Point3f)targLoc;
						if (sep.dot(sep) > fardist)
						{
							//prevent the zoom-hunt bug
							ptza->stop();
							ptza->goAbsPosition(ptza->convertBear(targLoc));
							analytics->declareCameraMoving();
							curpos = targLoc;
							my_sleep(40);
						}
						else
						{
							//notify the 2D analytics that we're pointing at a target
							analytics->declareCameraStopped(curpos);
						}
						break;
					}
					//move on down the list
					else (*sortedTargets)++;
				}
			}
			std::this_thread::yield();
			//why is this polling rather than using a condition variable to tell it when a change occurs? because we're waiting for the camera to move physically, which doesn't give us a signal
		}
	}

	void DidoDMM::selectTarget(std::shared_ptr<OVTrack_Storage> targetstore)
	{
		/*check if we have any targets*/
		bool empty = true;
		for (auto & t : *targetstore)
		{
			/*check if there are any valid targets*/
			if (t != nullptr)
			{
				empty = false;
				idle = false;
				OV_PTZBear cp;
				cp = curpos;
				/*uses MCTS to decide which target to look at next*/
				std::shared_ptr<OVTrack> tmp;
				{
					//scope for the lock
					std::lock_guard<std::mutex> tlck(targetmut);
					tmp = (sortedTargets->notAtEnd()) ? (*(*sortedTargets)).lock() : nullptr;
				}

				DidoDMM_TargetSelector ts(targetstore, cp, tmp, FaceReqAlpha, ClassifAlpha, maxDepth, maxRange, framePeriod/3, preferClassify);
				auto lst = ts.MCTS(maxSteps);
				sortedTargets->replace_list(std::move(lst));
				return;
			}
		}
		if (empty)
		{
			idle = true;
			return;
		}
	}

	DidoDMM::DidoDMM(PTZActuator * ptz, std::shared_ptr<Dido2D_iface> analytics_, OV_WorldPoint defaulttarg, float fTime, float alphaF, float alphaC, int ns, int maxd, float maxr)
		: ptza(ptz), FaceReqAlpha(alphaF), ClassifAlpha(alphaC), framePeriod(fTime), maxSteps(ns), maxDepth(maxd), maxRange(maxr*maxr), 
			defaultPosition(ptz->convertPoint(defaulttarg)), analytics(analytics_), PTZOffset(1,1,0)
	{
		sortedTargets = std::make_shared < ShareableTrackList>();
		analytics->setTargetIterator(sortedTargets);
	}

	DidoDMM::~DidoDMM()
	{
		if (running)
		{
			running = false;
			if (controlThread.joinable()) controlThread.join();
		}
	}

	void DidoDMM::nextTarget()
	{
		if ((*sortedTargets) != sortedTargets->end()) (*sortedTargets)++;
	}

	void DidoDMM::startPTZcontrol()
	{
		if (!running)
		{
			running = true;
			controlThread = std::thread(&DidoDMM::controlThreadFunction, this);
		}
	}

	void DidoDMM::stopPTZcontrol()
	{
		if (running)
		{
			running = false;
			if(controlThread.joinable()) controlThread.join();
		}
	}

	void DidoDMM::pointAtWorldLocation(const OV_WorldPoint & pt)
	{
		bool auton = !overridden;
		overridden = true;
		std::lock_guard<std::mutex>lck(ptzmut);
		if (auton)
		{
			ptza->stop();
		}
		/*our default zoom expectation will be 3 meters across*/
		ptza->goAbsPosition(ptza->convertPoint(pt - PTZOffset, 3));
	}

	void DidoDMM::pointAtTarget(std::shared_ptr<OVTrack> trg)
	{
		bool auton = !overridden;
		overridden = true;
		std::lock_guard<std::mutex>lck(ptzmut);
		if (auton)
		{
			ptza->stop();
		}
		/*a couple of meters around the target*/
		ptza->goAbsPosition(ptza->convertPoint(trg->getLoc() - PTZOffset, std::max(trg->height, trg->width) + 2));
	}

	void DidoDMM::pointAtBearing(const cv::Point2f & bearing)
	{
		bool auton = !overridden;
		overridden = true;
		std::lock_guard<std::mutex>lck(ptzmut);
		if (auton)
		{
			ptza->stop();
		}
		ptza->goAbsPosition(ptza->convertBear(bearing));
	}

	void DidoDMM::setVelocity(const cv::Vec4b & ptzf)
	{
		bool auton = !overridden;
		overridden = true;
		std::lock_guard<std::mutex>lck(ptzmut);
		if (auton)
		{
			ptza->stop();
		}
		ptza->PTZFControl(ptzf);
	}

	void DidoDMM::panTilt(uchar pan, uchar tilt)
	{
		bool auton = !overridden;
		overridden = true;
		std::lock_guard<std::mutex>lck(ptzmut);
		if (auton)
		{
			ptza->stop();
		}
		ptza->VelControl(pan, tilt);
	}

	void DidoDMM::zoomFocus(uchar zoom, uchar focus)
	{
		bool auton = !overridden;
		overridden = true;
		std::lock_guard<std::mutex>lck(ptzmut);
		if (auton)
		{
			ptza->stop();
		}
		ptza->ZoomFocusControl(zoom, focus);
	}

	OV_PTZBear DidoDMM::getPTZFacing()
	{
		std::lock_guard<std::mutex>lck(ptzmut);
		PTZPoint pt = ptza->getAbsPosition();
		curpos = ptza->convertPTZ(pt);
		return ptza->convertPTZ(pt);
	}

	bool DidoDMM::isPTZMoving()
	{
		std::lock_guard<std::mutex>lck(ptzmut);
		return ptza->isMoving();
	}

	const std::shared_ptr<const ShareableTrackList> DidoDMM::observeTargets()
	{
		return sortedTargets;
	}

	void DidoDMM::endOverride()
	{
		overridden = false;
	}

	void DidoDMM::setRange(float range)
	{
		maxRange = range*range;
	}

	float DidoDMM::getSquaredRange()
	{
		return maxRange;
	}

	void DidoDMM::setNsteps(size_t nstep)
	{
		maxSteps = (int)nstep;
	}

	int DidoDMM::getNsteps()
	{
		return maxSteps;
	}

	void DidoDMM::setDepth(int d)
	{
		maxDepth = d;
	}
	void DidoDMM::setPreferClassify(float pc)
	{
		preferClassify = pc;
	}
	void DidoDMM::setPTZOffset(float x, float y, float z)
	{
		PTZOffset = OV_WorldPoint(x, y, z);
	}
	void DidoDMM::setPTZOffset(const OV_WorldPoint & off)
	{
		PTZOffset = off;
	}
	void DidoDMM::setPTZRotation(float pan_off)
	{
		ptza->setRotOffset(pan_off);
	}
	void dumbDMM::selectTarget(std::shared_ptr<OVTrack_Storage> targetstore)
	{
		std::list<std::weak_ptr<OVTrack>> lst;
		for (auto & t : *targetstore)
		{
			if (t == nullptr) continue;
			//if we've already looked at it, drop its priority to last
			if (t->IDQuality > 0.5)
			{
				lst.push_back(t);
				continue;
			}

			//sort the targets from the closest one outwards
			auto npos = t->getLoc();
			float range = npos.dot(npos);

			auto it = lst.begin();
			while (true)
			{
				if (it == lst.end())
				{
					lst.push_back(t);
					break;
				}
				auto ipos = it->lock()->getLoc();
				if (ipos.dot(ipos) > range)
				{
					lst.insert(it, t);
					break;
				}
			}
		}

		if (lst.empty())
		{
			idle = true;
			return;
		}
		else
		{
			idle = false;
			sortedTargets->replace_list(std::move(lst));
		}

	}

}