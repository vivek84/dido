/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoDMM_TargetSelector.h
* @author  	SL
* @version 	1
* @date    	2017-03-06
* @brief   	Special datatypes used by the model predictive control
*			Based on Monte-Carlo Tree Search as described in http://www.cameronius.com/cv/mcts-survey-master.pdf
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoDMM_TargetSelector
#define __DidoDMM_TargetSelector


#pragma once

//#include "GlobalIncludes.h" //should be in the cpp
#include "OVTrack.h"
#include "PTZActuator.h"

/*defines ---------------------------*/

namespace overview
{
	class DidoDMM_TargetSelector
	{
		/*the model state datatype - for each target, reran for each simulation*/
		struct ModelState
		{
			std::shared_ptr<OVTrack> target;
			OV_WorldPoint location;
			float IDQuality;
			float ClassificationConfidence;

			ModelState(std::shared_ptr<OVTrack> tar, OV_WorldPoint loc, float IDQ, float CC)
				: target(tar), location(loc), IDQuality(IDQ), ClassificationConfidence(CC) {}
		};

		/*the node datatype*/
		struct MCTSNode
		{
			/*the incoming action's index from the model, not the storage*/
			size_t action;
			/*total simulation reward over all it's children*/
			double Q = 0;
			/*the node depth*/
			int depth;
			/*visit count*/
			int nVisit = 0;
			/*parent node*/
			MCTSNode * parent;
			/*children*/
			std::vector<std::unique_ptr<MCTSNode >> children;

			/*constructor*/
			MCTSNode(size_t act, MCTSNode * par, size_t targetsize);
			/*destructor*/
			~MCTSNode() = default;
		};
		/*the base node that has everything else as it's children*/
		MCTSNode basenode;
		/*the initial camera position*/
		OV_PTZBear cpos;
		/*the initial target of the camera*/
		std::shared_ptr<OVTrack> initTarg;

		/*the list of targets that we have*/
		std::shared_ptr<OVTrack_Storage> targets;

		/*the depth to which we search*/
		int maxdepth;
		/*the square of how far away a target can be before it's not longer a target*/
		float maxrange;
		/*time between model steps in seconds*/
		float framePeriod;

		float FaceReqAlpha;
		float ClassifAlpha;

		/*function that works out what our expected change in IDq is when looking at a target - this is what will need to be changed depending on the identification piece*/
		float IDqGain(std::shared_ptr<OVTrack> & mtarg);


		/*updates the model using the declared action*/
		void updateModel(size_t action);

		/*Tree policy - always called from the root*/
		MCTSNode * treePolicy();
		/*default policy and backup function*/
		void defaultThenBackup(MCTSNode * expnode);
		/*bestChild function*/
		MCTSNode * bestChild(MCTSNode * parent, float cp);
		/*internal model*/
		std::vector<ModelState> model;
		float modelreward;
		OV_PTZBear modelcam;
		float prefClassify;
	public:
		/*run the algorithm for n iterations and return the current optimal path*/
		std::list<std::weak_ptr<OVTrack>> MCTS(int nsteps);
		/*constructor*/
		DidoDMM_TargetSelector(std::shared_ptr<OVTrack_Storage> t, const OV_PTZBear & campos, std::shared_ptr<OVTrack> initialTarget, float fra, float cca, int mdpth, float maxr, float ftime, float preferCkassify);
		/*the destructor here will be important*/
		~DidoDMM_TargetSelector();
	};
}
#endif
