/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAutonomous.h
* @author  	SL
* @version 	1
* @date    	2017-04-24
* @brief    Class that owns all the elements of the autonomous operation and runs the two core loops of the autonomous part of the dido system
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoAutonomous
#define __DidoAutonomous


#pragma once
#include "DidoLidar.h"
#include "PTZActuator.h"
#include "ControlInterface.h"
#include "db_tracks.h"
#include "DidoThermal.h"
#include "DidoAnalytics\DidoAnalytics.h"
#include "DidoMain_ParamEnv.h"
#include "DidoMain_ObjEnv.h"
#include "ShareableTrackList.h"

/*defines ---------------------------*/

namespace overview
{

	//this is largely static as there can only be once instance of the core driver running with the program
	namespace DidoAutonomous
	{
		/*constructs all the member variables and starts the system running*/
		void initialise(std::shared_ptr<overview::db::store_to_database> _dbs, PTZActuator * _ptza, DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj);

		/*kills everything and starts again - not yet implimented*/
		void restart();

		/*shuts it all down*/
		void shutdown();

		/*returns the DMM targets for display*/
		const std::shared_ptr<const ShareableTrackList> getDMMTargets();


		/*various parameters setting functions that make up the control interface
		*/
		class DidoControlInterface : public ControlInterface
		{
			/*is a singleton*/
			DidoControlInterface() {};
			DidoControlInterface(DidoControlInterface & cp);
		public:
			//access the singleton instance
			static std::shared_ptr<DidoControlInterface> & instance()
			{
				static std::shared_ptr<DidoControlInterface> inst( new DidoControlInterface() );
				return inst;
			}
			//sets the weight towards classification or identification
			virtual void setModalityClassify(float pc) override;
			//displacement is the separation in meters
			virtual void setVerticalDisplacement(float vd) override;
			virtual float getVerticalDisplacement()override;
			//field of view is the total vertical angle subtended by the panorama
			virtual void setVerticalFieldOfView(float vFOV);
			virtual float getVerticalFieldOfView() override;
			//vertical offset is the angle from vertical to the top of the panorama
			virtual void setVerticalOffset(float voff)override;
			virtual float getVerticalOffset()override;
			//horizontal field of view is the total horizontal angle of the panorama (should be 2*pi)
			virtual void setHorizontalFieldOfView(float hFOV)override;
			virtual float getHorizontalFieldOfView()override;
			//horizontal offset is the angle of rotation between the lidar and the panorama
			virtual void setHorizontalOffset(float hoff)override;
			virtual float getHorizontalOffset()override;

			//sets whether the lidar is upside down or not
			virtual void setLidarUpFlip(bool up)override;
			//sets whether the lidar and pano are spinning in the same or opposite directions
			virtual void setLidarRotFlip(bool same)override;

			//some functions to adjust the analytics parameters
			virtual void setIgnoreAreas(const std::vector<ControlInterface_Area> & ignoreAreas)override;
			virtual std::vector<ControlInterface_Area> getIgnoreAreas()override;
			virtual void setAreasOfInterest(const std::vector<ControlInterface_Area> & AreasOfInterest)override;
			virtual std::vector<ControlInterface_Area> getAreasOfInterest()override;
			virtual void setBoundaries(const std::vector<ControlInterface_Boundary>& Boundaries) override;
			virtual std::vector<ControlInterface_Boundary> getBoundaries() override;

			//TODO - define useful parameter setting stuff for the analytics itself - stuff like increase FAR, decrease detection probability etc


			//some functions to adjust the 2dAnalytics parameters
			virtual void setVideoAddress(std::string vidaddr)override;

			//some functions to adjust the DMM parameters
			virtual void setRange(float range)override;
			virtual void setDMMSteps(size_t nsteps)override;
			virtual void setDMMPredictionDepth(size_t depth)override;

			//some functions to adjust the lidar parameters
			virtual void setLidarIP(std::string dev)override;
			virtual void setPanoFrequency(float f)override;

			//some functions to adjust the laser synthesiser parameters
			virtual void setCriticalDistance(float cd)override;
			virtual void setCriticalTemperature(float ct)override;

			//functions to give motion commands to the DMM
			/*point and click position based targeting*/
			virtual void pointAtWorldLocation(const OV_WorldPoint & pt)override;
			virtual void pointAtTarget(std::shared_ptr<OVTrack> trg)override;
			virtual void pointAtBearing(const cv::Point2f & bearing)override;

			/*direct PTZ commands*/
			//set velocity 0 is a stop everything command
			virtual void setVelocity(const cv::Vec4b & ptzf)override;
			virtual void panTilt(uchar pan, uchar tilt)override;
			virtual void zoomFocus(uchar zoom, uchar focus)override;

			/*access to PTZ location for the face rec thread*/
			virtual OV_PTZBear getPTZFacing()override;
			virtual bool isPTZMoving()override;

			/*release back to autonomy*/
			virtual void endOverride()override;

		};

		/*debug display function*/
		std::shared_ptr< const OVTrack_Storage> getTrackStore();

	};
}
#endif
