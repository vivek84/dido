/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAutonomous.cpp
* @author  	SL
* @version 	1
* @date    	2017-04-24
* @brief    Class that owns all the elements of the autonomous operation and runs the two core loops of the autonomous part of the dido system
*****************************************************************************
**/

#include "GlobalIncludes.h"
#include "DidoAutonomous.h"
#include "Dido2D.h"
#include "DidoDMM.h"
#include "DidoLidar_CUDA.h"
#include "DidoLidar_ManagedPano.h"

#include "CUDA_Exception.h"

/*the callback functions to give to the thermal driver*/

namespace overview
{
namespace input
{
	std::mutex lid_mut;
	static overview::DidoLidar * lidptr;
	std::shared_ptr<overview::DidoLidar::DidoLidar_Pano> global_lidarPano;

	DidoThermal::frameCBFunc frame_start_callback = []()
	{
		std::lock_guard<std::mutex> lck(lid_mut);
		global_lidarPano.reset(new overview::DidoLidar::DidoLidar_Pano(lidptr->grabPano()));
		(lidptr)->startnewframe();
	};
}

#define TRYCATCH 1
/*variable initialisation*/

	namespace DidoAutonomous
	{
		//visual studio doesn't like anonymous namespaces
		namespace detail
		{
			/*pointers to object that need to be shared*/
			 /*pointer to the global set of tracks*/
			std::shared_ptr<OVTrack_Storage> trackStore;
			/*the SQL database that the gui uses to find out about the tracks*/
			std::shared_ptr<overview::db::store_to_database> dbconnect;
			/*the DMM - has to be given to some other object that also want the camera*/
			std::shared_ptr<DidoDMM> dmm;
			/*analytics are also given to the 2d system*/
			std::shared_ptr<DidoAnalytics> analytics;

			std::shared_ptr<Dido2D> visualAnalytics;
			std::shared_ptr<DidoLidar> lidar;
			std::shared_ptr<DidoLidar_CUDA> laserSynthesiser;
			std::shared_ptr<DidoThermal> thermal;

			//how big is the panorama
			int ncols, nrows;
			float width = 30 * 2 * pi / 360;

			//alignment parameters
			float verticalDisplacement = 0;
			float horizontalOffset = 0;
			float horizontalFOV = 2 * pi;
			float verticalOffset = pi - 0.22f;
			float verticalFOV = 0.22f;
			int lsynthScale = 12;

			bool thermalRunning = true;

			void thermalLoop(const std::vector<cv::Mat> & panos, const std::vector<Dido_Timestamp> & ts, const std::vector<float> & sa)
			{							
				//check that we have initialised things and wait till we do
				while ((!dmm || !analytics || !laserSynthesiser )&& keep_running()) std::this_thread::yield();
				//don't keep reallocating it - it gets overwritten anyway
#if TRYCATCH
				try
#endif
				{
					LOG_DEBUG << "Collected thermal panorama";
					std::shared_ptr<overview::DidoLidar::DidoLidar_Pano> lidPano;
					{
						std::lock_guard<std::mutex> lid_lck(input::lid_mut);
						lidPano = input::global_lidarPano;
					}
					LOG_DEBUG << "Collected Lidar Data";
					analytics->analyse3DFrame(panos, lidPano, ts, sa);
					LOG_DEBUG << "Analysed frame and produced new/updated tracks" << std::endl;
					
					//only update target choices after a full pano?
					bool fullrot = false;
					for (auto & a : sa) if (a < 0.1f) fullrot = true;
					if (fullrot)	dmm->selectTarget(trackStore);
					LOG_DEBUG << "new target for PTZ selected" << std::endl;
#if TRYCATCH
				}
				catch (CUDA_Exception e)
				{
					LOG_ERROR << "CUDA ERROR : - " << e.what() << std::endl;
					throw e;
#endif
				}
			}
			DidoThermal::analyticsCBFunc analytics_callback = thermalLoop;
		} // /detail

		

		/*public functions*/

	 /*the constructor has to make everything in the right order
		the PTZActuator should be dynamically allocated - it gets given to a std::unique_ptr<>
	 */
		void initialise(std::shared_ptr<overview::db::store_to_database> _dbs, PTZActuator * _ptza, DidoMain_ParamEnv & pars, DidoMain_ObjEnv & obj)
		{
			/*create all the objects*/
			detail::trackStore = std::make_shared<OVTrack_Storage>();
			detail::dbconnect = _dbs;
			detail::thermal = obj.globalThermal;
			detail::lidar = obj.globalLidar;
			input::lidptr = &*detail::lidar;
			
			detail::ncols = detail::thermal->ncols();
			ControlInterface_Area::ncols = detail::ncols;
			detail::nrows = detail::thermal->nrows();
			detail::analytics = obj.globalAnalytics;
			detail::analytics->connectTrackStore(detail::trackStore);
			detail::analytics->connnectDB(detail::dbconnect);
			detail::analytics->IgnoreAreas = pars.ignoreAreas;
			
			detail::visualAnalytics = std::make_shared<Dido2D>(detail::trackStore, detail::dbconnect, obj.globalVcapPTZ,
				pars.faceDetectorFile, pars.deepModelFolder, pars.faceFeatureFolder, pars.pretendToFindFaces);
			detail::visualAnalytics->setHaarParameters(pars.minHaarNeighbours, pars.minHaarSize);
			if (pars.useHeuristicDMM)
			{
				detail::dmm = std::make_shared<dumbDMM>(_ptza, detail::visualAnalytics, OV_WorldPoint(0, 0, 4));	//point vertical when have no targets for clarity
			}
			else
			{
				detail::dmm = std::make_shared<DidoDMM>(_ptza, detail::visualAnalytics, OV_WorldPoint(0, 0, 4), 1.0f/pars.frameRate,
					pars.expectedIDGain, pars.expectedClassifGain, pars.MCMaxSteps, pars.MCMaxDepth, pars.systemMaxRange);
			}
			detail::dmm->setPTZOffset(pars.PTZOffset);
			detail::dmm->setPTZRotation(pars.PTZrotOff);
			detail::dmm->setMaxErr(pars.DMMMaxErr);

			detail::laserSynthesiser = std::make_shared<DidoLidar_CUDA>();
			detail::laserSynthesiser->setScale(detail::lsynthScale);
			LOG_INFO << "autonomous objects created" << std::endl;

			/*launch the thermal panorama*/
			detail::thermal->setup(detail::analytics_callback, input::frame_start_callback);
			LOG_INFO << "thermal launched" << std::endl;
			detail::dmm->startPTZcontrol();
			LOG_INFO << "PTZ control enabled" << std::endl;

			//set the 2d analytics running
			detail::visualAnalytics->start();
			LOG_INFO << "2D analytics started" << std::endl;

			detail::thermal->setNThreads(3);

			LOG_INFO << "main thread launched" << std::endl;
		}



		void restart()
		{
			detail::thermal->stop();
			detail::dmm->stopPTZcontrol();
			detail::thermalRunning = false;
			detail::visualAnalytics->stop();
			LOG_INFO << "Processing stopped" << std::endl;
			for (auto & t : *detail::trackStore)
			{
				t.reset();
			}
			LOG_INFO << "Tracks cleared" << std::endl;
			detail::thermal->setup(detail::analytics_callback, input::frame_start_callback);
			LOG_INFO << "thermal launched" << std::endl;
			detail::dmm->startPTZcontrol();
			detail::visualAnalytics->start();
			LOG_INFO << "2D analytics started" << std::endl;
			LOG_INFO << "main thread relaunched" << std::endl;
		}


		/*stops it all and frees it all from memory*/
		void shutdown()
		{
			//stop operations
			detail::thermalRunning = false;
			if(detail::thermal) detail::thermal->stop();
			if(detail::dmm) detail::dmm->stopPTZcontrol();
			if(detail::visualAnalytics)detail::visualAnalytics->stop();
			//free the data
			detail::lidar.reset();
			detail::thermal.reset();
			detail::trackStore.reset();
			detail::dbconnect.reset();
			detail::dmm.reset();
			detail::analytics.reset();
			detail::visualAnalytics.reset();
			detail::lidar.reset();
			detail::laserSynthesiser.reset();
		}
		const std::shared_ptr<const ShareableTrackList> getDMMTargets()
		{
			if (detail::dmm) return detail::dmm->observeTargets();
			return nullptr;
		}
		/*definitions of class members*/

		void DidoControlInterface::setModalityClassify(float pc)
		{
			if (detail::dmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			detail::dmm->setPreferClassify(pc);
		}

		void DidoControlInterface::setVerticalDisplacement(float vd)
		{
			detail::verticalDisplacement = vd;
			LOG_INFO << "vertical Displacement set to " << vd << std::endl;
		}

		float DidoControlInterface::getVerticalDisplacement()
		{
			return detail::verticalDisplacement;
		}

		void DidoControlInterface::setVerticalFieldOfView(float vFOV)
		{
			detail::verticalFOV = vFOV;
			LOG_INFO << "vertical Field of View set to " << vFOV << std::endl;
		}

		float DidoControlInterface::getVerticalFieldOfView()
		{
			return detail::verticalFOV;
		}

		void DidoControlInterface::setVerticalOffset(float voff)
		{
			detail::verticalOffset = voff;
			LOG_INFO << "vertical Offset set to " << voff << std::endl;
		}

		float DidoControlInterface::getVerticalOffset()
		{
			return detail::verticalOffset;
		}

		void DidoControlInterface::setHorizontalFieldOfView(float hFOV)
		{
			detail::horizontalFOV = hFOV;
			LOG_INFO << "horizontal Field of View set to " << hFOV << std::endl;
		}

		float DidoControlInterface::getHorizontalFieldOfView()
		{
			return detail::horizontalFOV;
		}

		void DidoControlInterface::setHorizontalOffset(float hoff)
		{
			detail::horizontalOffset = hoff;
			LOG_INFO << "Horizontal offset set to " << hoff << std::endl;
		}

		float DidoControlInterface::getHorizontalOffset()
		{
			return detail::horizontalOffset;
		}

		/*for these we need to hold a local copy to prevent concurrent destruction screwing us*/
		void DidoControlInterface::setLidarUpFlip(bool up)
		{
			auto tmplid = detail::lidar;
			if (tmplid == nullptr) throw ControlInterface_exception("Lidar not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmplid->setUpsideDown(up);
			LOG_INFO << "Lidar set to be " << ((up) ? "right way up" : "upside down") << std::endl;
		}

		void DidoControlInterface::setLidarRotFlip(bool same)
		{
			auto tmplid = detail::lidar;
			if (tmplid == nullptr) throw ControlInterface_exception("Lidar not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmplid->setFlipSpin(same);
			LOG_INFO << "Lidar set to spin the " << ((same) ? "same way as the thermal" : "opposite way to the thermal") << std::endl;
		}

		void DidoControlInterface::setVideoAddress(std::string vidaddr)
		{
			auto tmpana = detail::visualAnalytics;
			if (tmpana == nullptr) throw ControlInterface_exception("Visual Analytics not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpana->setVideoAddress(vidaddr);
			LOG_INFO << "Video address set to " << vidaddr << std::endl;
		}

		void DidoControlInterface::setRange(float range)
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpdmm->setRange(range);
			LOG_INFO << "Maximum range set to " << range << std::endl;
		}

		void DidoControlInterface::setDMMSteps(size_t nsteps)
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpdmm->setNsteps((int)nsteps);
			LOG_INFO << "DMM max computation steps set to " << nsteps << std::endl;
		}

		void DidoControlInterface::setDMMPredictionDepth(size_t depth)
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpdmm->setDepth((int)depth);
			LOG_INFO << "DMM predictive depth set to " << depth << std::endl;
		}

		void DidoControlInterface::setCriticalTemperature(float ct)
		{
			auto tmplsynth = detail::laserSynthesiser;
			if (tmplsynth == nullptr) throw ControlInterface_exception("Laser Synthesiser not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmplsynth->setCritTemp(ct);
			LOG_INFO << "Critical temperature for laser synthesis to " << ct << std::endl;
		}

		void DidoControlInterface::setCriticalDistance(float cd)
		{
			auto tmplsynth = detail::laserSynthesiser;
			if (tmplsynth == nullptr) throw ControlInterface_exception("Laser Synthesiser not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmplsynth->setCritDist(cd);
			LOG_INFO << "Critical distance for laser synthesis to " << cd << std::endl;
		}

		void DidoControlInterface::setLidarIP(std::string dev)
		{
			auto tmplid = detail::lidar;
			if (tmplid == nullptr) throw ControlInterface_exception("Lidar not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmplid->setIP(dev);
			LOG_INFO << "IP address of Lidar set to " << dev << std::endl;
		}

		void DidoControlInterface::setPanoFrequency(float f)
		{
			auto tmplid = detail::lidar;
			if (tmplid == nullptr) throw ControlInterface_exception("Lidar not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmplid->setFreq(f);
			LOG_INFO << "frequency of pano rotation for the Lidar set to " << f << std::endl;
		}


		/*straight passthroughs to the DMM

		/*point and click position based targeting*/
		void DidoControlInterface::pointAtWorldLocation(const OV_WorldPoint & pt)
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpdmm->pointAtWorldLocation(pt);
			LOG_INFO << "PTZ told to point to " << pt << std::endl;
		}
		void DidoControlInterface::pointAtTarget(std::shared_ptr<OVTrack> trg)
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpdmm->pointAtTarget(trg);
			LOG_INFO << "PTZ told to point at a target at " << trg->getLoc() << std::endl;
		}

		void DidoControlInterface::pointAtBearing(const cv::Point2f & bearing)
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpdmm->pointAtBearing(bearing);
			LOG_INFO << "PTZ told to point to bearing " << bearing << std::endl;
		}

		/*direct PTZ commands*/
		//set velocity 0 is a stop everything command
		void DidoControlInterface::setVelocity(const cv::Vec4b & ptzf)
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpdmm->setVelocity(ptzf);
			LOG_INFO << "PTZ velocity set to " << ptzf << std::endl;
		}
		void DidoControlInterface::panTilt(uchar pan, uchar tilt)
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpdmm->panTilt(pan, tilt);
			LOG_INFO << "PTZ set to pan " << (int)pan << " and tilt " << (int)tilt << std::endl;
		}
		void DidoControlInterface::zoomFocus(uchar zoom, uchar focus)
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpdmm->zoomFocus(zoom, focus);
			LOG_INFO << "PTZ set to zoom " << (int)zoom << " and focus " << (int)focus << std::endl;
		}

		/*access to PTZ location for the face rec thread*/
		OV_PTZBear DidoControlInterface::getPTZFacing()
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			return tmpdmm->getPTZFacing();
		}
		bool DidoControlInterface::isPTZMoving()
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			return tmpdmm->isPTZMoving();
		}

		/*release back to autonomy*/
		void DidoControlInterface::endOverride()
		{
			auto tmpdmm = detail::dmm;
			if (tmpdmm == nullptr) throw ControlInterface_exception("DMM not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpdmm->endOverride();
			LOG_INFO << "Override ended " << std::endl;
		}

		void DidoControlInterface::setBoundaries(const std::vector<ControlInterface_Boundary>& boundaries)
		{
			auto tmpana = detail::analytics;
			if (tmpana == nullptr) throw ControlInterface_exception("Analytics not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpana->Boundaries = boundaries;
			LOG_INFO << "Boundaries updated" << std::endl;
		}

		std::vector<ControlInterface_Boundary> DidoControlInterface::getBoundaries()
		{
			auto tmpana = detail::analytics;
			if (tmpana == nullptr) throw ControlInterface_exception("Analytics not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			return tmpana->Boundaries;
		}

		//some functions to adjust the analytics parameters
		void DidoControlInterface::setIgnoreAreas(const std::vector<ControlInterface_Area> & ignoreAreas)
		{
			auto tmpana = detail::analytics;
			if (tmpana == nullptr) throw ControlInterface_exception("Analytics not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			tmpana->IgnoreAreas = ignoreAreas;
			LOG_INFO << "Ignore Areas updated" << std::endl;
		}
		std::vector<ControlInterface_Area> DidoControlInterface::getIgnoreAreas()
		{
			auto tmpana = detail::analytics;
			if (tmpana == nullptr) throw ControlInterface_exception("Analytics not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			return tmpana->IgnoreAreas;
		}
		void DidoControlInterface::setAreasOfInterest(const std::vector<ControlInterface_Area> & aoi)
		{
			auto tmpana = detail::analytics;
			if (tmpana == nullptr) throw ControlInterface_exception("Analytics not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);

			tmpana->AreasOfInterest = aoi;
			LOG_INFO << "Areas of Interest updated" << std::endl;
		}
		std::vector<ControlInterface_Area> DidoControlInterface::getAreasOfInterest()
		{
			auto tmpana = detail::analytics;
			if (tmpana == nullptr) throw ControlInterface_exception("Analytics not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			return tmpana->AreasOfInterest;
		}

		/*for debug display purposes*/
		std::shared_ptr<const OVTrack_Storage> getTrackStore()
		{
			auto tmptks = detail::trackStore;
			if (tmptks == nullptr) throw ControlInterface_exception("Trackstore not initialised", ControlInterfaceExceptionTypes::memberNotInitialised);
			return tmptks;
		}

	}
}
