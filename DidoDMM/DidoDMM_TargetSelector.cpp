/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoDMM_TargetSelector.h
* @author  	SL
* @version 	1
* @date    	2017-03-06
* @brief   	Special datatypes used by the model predictive control
*****************************************************************************
**/

#include "GlobalIncludes.h"
#include "DidoDMM_TargetSelector.h"

/*private functions*/

namespace overview
{
	constexpr float DIDODMM_CP(1.412f);

	DidoDMM_TargetSelector::MCTSNode::MCTSNode(size_t act, MCTSNode * par, size_t targetsize)
		: action(act), parent(par)
	{
		if (parent == nullptr) depth = 0;
		else depth = parent->depth + 1;
		children.reserve(targetsize);
		for (int i = 0; i < targetsize; i++)
		{
			children.emplace_back(nullptr);
		}
	}


	std::list<std::weak_ptr<OVTrack>> DidoDMM_TargetSelector::MCTS(int nsteps)
	{
		for (int i = 0; i < nsteps; i++)
		{
			/*reset the model*/
			modelreward = 0;
			modelcam = cpos;
			for (auto t : *targets)
			{
				if (t == nullptr) continue;
				bool extra = true;
				for (auto & m : model)
				{
					//reset each model variable - this keeps it robust to changes from the enviroment about which targets there are, as they keep the same order in the model
					if (m.target == t)
					{
						extra = false;
						m = ModelState(t, t->getLoc(), (float)t->IDQuality, t->getFirstLevelClassificationConfidence());
					}
				}
				if (extra) model.push_back(ModelState(t, t->getLoc(), (float)t->IDQuality, t->getFirstLevelClassificationConfidence()));
			}
			defaultThenBackup(treePolicy());
		}
		std::list<std::weak_ptr<OVTrack>> rval;
		auto rptr = bestChild(&basenode, 0);
		while (rptr != nullptr)
		{
			rval.push_back(model[rptr->action].target);
			//iterate down the tree
			rptr = bestChild(rptr, 0);
		}
		return rval;
	}

	DidoDMM_TargetSelector::DidoDMM_TargetSelector(std::shared_ptr<OVTrack_Storage> t, const OV_PTZBear & campos, std::shared_ptr<OVTrack> initT, float fra, float cca, int mdpth, float mr, float fp, float prefclass)
		:targets(t), cpos(campos), maxdepth(mdpth), basenode(0, nullptr, t->size()), maxrange(mr), framePeriod(fp), FaceReqAlpha(fra), ClassifAlpha(cca), prefClassify(prefclass), initTarg(initT)
	{
		basenode.depth = 0;
		if (prefClassify > 1) prefClassify = 1;
		if (prefClassify < 0) prefClassify = 0;
	}

	DidoDMM_TargetSelector::~DidoDMM_TargetSelector()
	{
	}



	float DidoDMM_TargetSelector::IDqGain(std::shared_ptr<OVTrack>& mtar)
	{
		/*we multiply by the confidence we have in the detection*/
		return mtar->detectionConfidence*FaceReqAlpha;
	}

	float zoomtime(float zoomstart, float zoomend)
	{

		return 1.0f;
	}

	constexpr float zoompar1 = -0.5878f;
	constexpr float zoompar2 = 0.0667f;
	constexpr float zoompar3 = 3.746f;

	void DidoDMM_TargetSelector::updateModel(size_t action)
	{
		for (auto & m : model)
		{
			/*update positions*/
			m.location += m.target->getVel()*framePeriod;
			m.IDQuality *= OVT_IDQDecay;
			/*check if is our target*/
			if (m.target == model[action].target)
			{
				/*work out the travel time */
				cv::Point3f nloc = PTZActuator::getBearing(m.location);
				cv::Point3f sep = nloc - (cv::Point3f)modelcam;
				//the hydra was measured to take 0.7 seconds to do a 180 move
				//each mode of motion is independant
				//zoom modelled from the sony camera spec sheet - this should really be a function from the camera instead
				float traveltime = std::max(std::max(.7f*(abs(sep.x)/ pi),zoompar1*abs(sep.z) + abs(sep.z)*zoompar2 + zoompar3 ),0.7f*(abs(sep.y)/pi));
				float f = std::max((framePeriod - traveltime), 0.0f);

				//update the idQ
				m.IDQuality += f*IDqGain(m.target)*(1 - m.IDQuality);

				//update the classification confidence
				m.ClassificationConfidence += f*ClassifAlpha*(1 - m.ClassificationConfidence);

				//update the camera position
				modelcam = (traveltime > framePeriod) ? (sep*(framePeriod/traveltime) + (cv::Point3f)modelcam) :nloc;

			}
			//only score for targets who are in range
			if (m.location.x*m.location.x + m.location.y*m.location.y < maxrange) modelreward += (m.ClassificationConfidence*prefClassify + (1-prefClassify)*m.IDQuality) / model.size();	
			//normalised score is needed for the MCTS functions to work

		}
	}

	DidoDMM_TargetSelector::MCTSNode * DidoDMM_TargetSelector::treePolicy()
	{
		MCTSNode * rval = &basenode;
		//first attempt to follow the initial target as input
		if (initTarg != nullptr)
		{
			for (int i = 0; i < rval->children.size() && i < model.size(); i++)
			{
				if (model[i].target == initTarg)
				{
					if (rval->children[i] == nullptr)
					{
						rval->children[i] = std::move(std::unique_ptr<MCTSNode>(new MCTSNode(i, rval, model.size())));
						//update the model
						updateModel(i);
						return rval->children[i].get();
					}
					rval = rval->children[i].get();
				}
			}
		}

		while (rval->depth < maxdepth)
		{
			int i = 0;
			//there are model.size() live targets
			for (; i < model.size() && i < rval->children.size(); i++)
			{
				if (rval->children[i] == nullptr)
				{
					rval->children[i] = std::move(std::unique_ptr<MCTSNode>(new MCTSNode(i, rval, model.size())));
					//update the model
					updateModel(i);
					return rval->children[i].get();
				}
			}
			if (i >= model.size())
			{
				rval = bestChild(rval, DIDODMM_CP);
				//update the model appropriately
				updateModel(rval->action);
			}
		}
		return rval;
	}

	void DidoDMM_TargetSelector::defaultThenBackup(MCTSNode * expnode)
	{
		/*default policy down the tree*/
		/*default policy is to change node every other step to a node selected at random*/
		bool changedpos = true;
		MCTSNode * currnode = expnode;
		while (currnode->depth < maxdepth)
		{
			if (changedpos)
			{
				//repeat the target
				currnode->children[currnode->action] = std::move(std::unique_ptr<MCTSNode>(new MCTSNode(currnode->action, currnode, model.size())));
				//update the model
				updateModel(currnode->action);
				currnode = currnode->children[currnode->action].get();
				changedpos = false;
			}
			else
			{
				//at random
				size_t act = abs(rand()) % model.size();
				currnode->children[act] = std::make_unique<MCTSNode>(act, currnode, model.size());
				updateModel(act);
				currnode = currnode->children[act].get();
				changedpos = true;
			}
		}

		/*backup back up it*/
		while (currnode != nullptr)
		{
			currnode->nVisit += 1;
			currnode->Q += modelreward / maxdepth;	//normalise it
			currnode = currnode->parent;
		}

	}

	DidoDMM_TargetSelector::MCTSNode * DidoDMM_TargetSelector::bestChild(MCTSNode * parent, float cp)
	{
		if (parent == nullptr || parent->children.empty()) return nullptr;
		else
		{
			double UCB = 0;
			MCTSNode * rval = parent->children.front().get();
			for (auto & c : parent->children)
			{
				if (c == nullptr) continue;
				double newCB = (c->Q / c->nVisit) + cp*sqrt(2 * log(parent->nVisit) / c->nVisit);
				if (newCB > UCB)
				{
					UCB = newCB;
					rval = c.get();
				}
			}
			return rval;
		}
	}
}