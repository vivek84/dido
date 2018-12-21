/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2018 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_Phase2.h
* @author  	SL
* @version 	1
* @date    	2018-01-25
* @brief   	Abstract class that defines the interface for whole detection pipeline.
*           This allows choice of sensor configuration
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoAnalytics_Phase2
#define __DidoAnalytics_Phase2


#pragma once

#include "OVTrack.h"

#include "ControlInterface_Area.h"

/*defines ---------------------------*/

namespace overview
{
	class DidoAnalytics_Phase2
	{
	protected:
		/*pointer to the global set of tracks*/
		std::shared_ptr<OVTrack_Storage> trackStore;

		/*how long between detections before we timeout a track*/
		Dido_Clock::duration timeouttime;

		//the minimum range past which a target is accepted
		float minrange = 0.7f;

		/*how low our confidence needs to go before we remove a detection*/
		float confidenceThreshold = 0.2f;

		//threat learning rate
		float gamma = 0.2f;
		float calculateThreat(std::shared_ptr<OVTrack> trk);
        //sets the timeout in ms
		DidoAnalytics_Phase2(int timeout = 400);
        
        /*function used to actually update the global store given a set of proposals*/
        void updateGlobalTracks(const std::vector<std::shared_ptr<OVTrack>> & tracks, Dido_Timestamp ts);

		/*parameters to allow easier tuning of sensor specific parameters by displaying the inputs
		  the acutal use of these parameters is up to the inheriting class
		*/
		bool displayInput = false;
		bool displayWorking = false;
		bool displayTimings = false;

	public:
		virtual ~DidoAnalytics_Phase2() = default;

		virtual void connectTrackStore(std::shared_ptr<OVTrack_Storage> ts) { trackStore = ts; }

        /*starts and stops the system running - whilst running the trackstore is continually updated in a thread*/
        virtual void start() = 0;
        virtual void stop() = 0;
        

		/*sets the display paramters*/
		void setDisplayInput(bool di) { displayInput = di; }
		void setDisplayWorking(bool dw) { displayWorking = dw; }
		void setDisplayTimings(bool dt) { displayTimings = dt; }


		/*internal storage of the areas of interest - public so they can be easily edited and added to
		 * not thread safe so if the parameters are being edited, the analytics must not be running*/
		std::vector<ControlInterface_Area> AreasOfInterest;
		std::vector<ControlInterface_Area> IgnoreAreas;
		std::vector<ControlInterface_Boundary> Boundaries;

		/*parameter setting functions*/
		void setTimeoutTime(int miliseconds);
		void setMinRange(float mr);

        /*external functionality to apply to tracks - for example sending to sapient*/
        typedef boost::signals2::signal<void(std::shared_ptr<OVTrack>) > trackSignal;
                
        void connectTrackFun(trackSignal::slot_function_type fun);
        void disconnectTrackFun();
        
        /*signal to apply after any update step, for example running the DMM*/
        typedef boost::signals2::signal<void(std::shared_ptr<OVTrack_Storage>) > stepSignal;
        void connectStepFun(stepSignal::slot_function_type fun);
        void disconnectStepFun();
        
    private:
        trackSignal tsig;
        stepSignal ssig;
    };

	//dummy analytics that simply produces a track that moves from 10,0 to 1,0 then to 1,10 then back again
	class DidoAnalytics_Phase2_Dummy : public DidoAnalytics_Phase2
	{
		int frame = 0;
        bool running;
        std::thread runthread;
        void runfunc();
        
	public:
		// Inherited via DidoAnalytics_Phase2
        virtual void start() override;
        virtual void stop() override;
        ~DidoAnalytics_Phase2_Dummy() {stop();}
	};
}
#endif
