/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	ControlInterface.h
* @author  	SL
* @version 	1
* @date    	2017-04-24
* @brief    Abstract class that defines the interface to the functionality of the system
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __ControlInterface
#define __ControlInterface

#include "ControlInterface_Area.h"
#include "OVTrack.h"
#pragma once

/*defines ---------------------------*/

namespace overview
{
	//this object is also the control interface and is a singleton with only static members so that we can use the C callbacks
	class ControlInterface
	{
	public:
		//sets the weight towards preferring classsification or identification
		virtual void setModalityClassify(float classifying) =0;

		//displacement is the separation in meters
		virtual  void setVerticalDisplacement(float vd) = 0;
		virtual float getVerticalDisplacement() = 0;
		//field of view is the total vertical angle subtended by the panorama
		virtual void setVerticalFieldOfView(float vFOV) = 0;
		virtual float getVerticalFieldOfView() = 0;
		//vertical offset is the angle from vertical to the top of the panorama
		virtual void setVerticalOffset(float voff) = 0;
		virtual float getVerticalOffset() = 0;
		//horizontal field of view is the total horizontal angle of the panorama (should be 2*pi)
		virtual void setHorizontalFieldOfView(float hFOV) = 0;
		virtual float getHorizontalFieldOfView() = 0;
		//horizontal offset is the angle of rotation between the lidar and the panorama
		virtual void setHorizontalOffset(float hoff) = 0;
		virtual float getHorizontalOffset() = 0;

		//sets whether the lidar is upside down or not
		virtual void setLidarUpFlip(bool up) = 0;
		//sets whether the lidar and pano are spinning in the same or opposite directions
		virtual void setLidarRotFlip(bool same) = 0;

		//some functions to adjust the analytics parameters
		virtual void setIgnoreAreas(const std::vector<ControlInterface_Area> & ignoreAreas) = 0;
		virtual std::vector<ControlInterface_Area> getIgnoreAreas() = 0;
		virtual void setAreasOfInterest(const std::vector<ControlInterface_Area> & AreasOfInterest) = 0;
		virtual std::vector<ControlInterface_Area> getAreasOfInterest() = 0;
		virtual void setBoundaries(const std::vector<ControlInterface_Boundary> & Boundaries) = 0;
		virtual std::vector<ControlInterface_Boundary> getBoundaries() = 0;

		//some functions to adjust the 2dAnalytics parameters
		virtual void setVideoAddress(std::string vidaddr) = 0;

		//some functions to adjust the DMM parameters
		virtual void setRange(float range) = 0;
		virtual void setDMMSteps(size_t nsteps) = 0;
		virtual void setDMMPredictionDepth(size_t depth) = 0;

		//some functions to adjust the lidar parameters
		virtual void setLidarIP(std::string dev) = 0;
		virtual void setPanoFrequency(float f) = 0;

		//some functions to adjust the laser synthesiser parameters
		virtual void setCriticalDistance(float cd) = 0;
		virtual void setCriticalTemperature(float ct) = 0;

		//functions to give motion commands to the DMM
		/*point and click position based targeting*/
		virtual void pointAtWorldLocation(const OV_WorldPoint & pt) = 0;
		virtual void pointAtTarget(std::shared_ptr<OVTrack> trg) = 0;
		virtual void pointAtBearing(const cv::Point2f & bearing) = 0;

		/*direct PTZ commands*/
		//set velocity 0 is a stop everything command
		virtual void setVelocity(const cv::Vec4b & ptzf) = 0;
		virtual void panTilt(uchar pan, uchar tilt) = 0;
		virtual void zoomFocus(uchar zoom, uchar focus) = 0;

		/*access to PTZ location*/
		virtual OV_PTZBear getPTZFacing() = 0;
		virtual bool isPTZMoving() = 0;

		/*release back to autonomy*/
		virtual void endOverride() = 0;

	};

	enum class ControlInterfaceExceptionTypes
	{
		memberNotInitialised
	};
	//special exception type that the control interface returns if the called for objects don't exist
	class ControlInterface_exception : public std::exception
	{
		const char * description;
		ControlInterfaceExceptionTypes err;
	public:
		ControlInterface_exception(const char * desc, ControlInterfaceExceptionTypes errtype) :
			err(errtype), description(desc) {
		}
		virtual  const char* what() const throw() override
		{
			return description;
		}
		ControlInterfaceExceptionTypes getErrType()
		{
			return err;
		}
	};
}
#endif
