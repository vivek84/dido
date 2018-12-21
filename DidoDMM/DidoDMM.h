/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
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

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoDMM
#define __DidoDMM


#pragma once

//#include "GlobalIncludes.h" //should be in the cpp
#include "OVTrack.h"
#include "my_sleep.h"
#include "PTZActuator.h"
#include "Dido2D.h"
#include "ShareableTrackList.h"

/*defines ---------------------------*/


/* this agent does different things depending on the state
* the core functionality is a thread that makes sure the PTZ is pointed at the location where it's target is believed to be
* states are:
* Autonomous - point the PTZ at target->getLoc()
* Idle - point the PTZ at default position
* Manual override - only point the PTZ when told
- with move commands
- with specific locations
*/

namespace overview
{
	class DidoDMM
	{
	protected:
		//this recieves commands from the CI
		/*the PTZ controlling object*/
		std::unique_ptr<PTZActuator> ptza;
		/*the video analytics piece*/
		std::shared_ptr<Dido2D_iface> analytics;
		//mutex to allow other threads to ask the PTZ questions without screwing up the serial bus
		std::mutex ptzmut;

		/*internal parameters of the model for target selection*/
		float FaceReqAlpha;
		float ClassifAlpha;
		/*the maxium range of our system squared*/
		float maxRange;
		/*the maximum number of simulations to run - effectively our computational budget*/
		int maxSteps;
		/*how many frames into the future to predict*/
		int maxDepth;
		/*time between frames in seconds*/
		float framePeriod;

		/*the current optimal order to look at the tracks in*/
		std::shared_ptr<ShareableTrackList > sortedTargets;
		/*mutex to protect this iterator*/
		std::mutex targetmut;

		/*elements for the PTZ control*/
		/*to what proportion of the zoom can we afford to be out by before we need to move the PTZ*/
		float fardist = 0.8f;

		/*the internal variables marking what state it's in*/
		bool idle = false;
		std::atomic_bool overridden = false;
		/*weight towards classifying over identifying - 0 is all identify, 1 all classify*/
		std::atomic<float> preferClassify = 0.0f;	//currently we don't have a useful classifier on the Dido2Dss

		std::atomic_int currentrequest = 0;

		std::thread controlThread;
		bool running = false;
		void controlThreadFunction();

		PTZPoint defaultPosition;
		//where the camera is facing right now
		std::atomic<OV_PTZBear> curpos;

		OV_WorldPoint PTZOffset; //the offset in world coordinates from the origin to the PTZ 

	public:
		/*function called as part of the main detect - classify - decide pipeline*/
		virtual void selectTarget(std::shared_ptr<OVTrack_Storage>targetstore);

		DidoDMM(PTZActuator * ptz, std::shared_ptr<Dido2D_iface> analytics_, OV_WorldPoint defaulttarg, float fTime = 1.0f, float alphaF = 0.5f, float alphaC = 0.5f, int ns = 300, int maxd = 12, float maxr = 400);
		~DidoDMM();

		/*tells it that an id has been acquired and to progress along the list*/
		void nextTarget();

		/*starts the PTZ control thread*/
		void startPTZcontrol();
		void stopPTZcontrol();

		/*manual PTZ override functions*/
		/*point and click position based targeting*/
		void pointAtWorldLocation(const OV_WorldPoint & pt);
		void pointAtTarget(std::shared_ptr<OVTrack> trg);
		void pointAtBearing(const cv::Point2f & bearing);

		/*direct PTZ commands*/
		//set velocity 0 is a stop everything command
		void setVelocity(const cv::Vec4b & ptzf);
		void panTilt(uchar pan, uchar tilt);
		void zoomFocus(uchar zoom, uchar focus);

		/*access to PTZ location for the face rec thread*/
		OV_PTZBear getPTZFacing();
		bool isPTZMoving();

		const std::shared_ptr<const ShareableTrackList> observeTargets();

		/*release back to autonomy*/
		void endOverride();

		/*get/set functions*/
		void setRange(float range);
		float getSquaredRange();
		void setNsteps(size_t nstep);
		int getNsteps();

		float getMaxErr() { return fardist; }
		void setMaxErr(float me) { fardist = me; }

		void setDepth(int d);

		void setPreferClassify(float pc);

		void setPTZOffset(float x, float y, float z);
		void setPTZOffset(const OV_WorldPoint & off);
		void setPTZRotation(float pan_off);
	};

	//class that instead selects the targets in a silly heuristic manner so we can validate the detector
	class dumbDMM : public DidoDMM
	{
	public: 
		dumbDMM(PTZActuator * ptz, std::shared_ptr<Dido2D> analytics_, OV_WorldPoint defaulttarg)
			: DidoDMM(ptz, analytics_, defaulttarg, 1.0f, 1.0f, 1.0f, 1,  1 )
		{
		}

		virtual void selectTarget(std::shared_ptr<OVTrack_Storage>targetstore);
	};

}
#endif
