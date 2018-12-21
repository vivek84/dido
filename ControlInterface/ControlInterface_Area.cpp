/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	ControlInterface_Area.cpp
* @author  	SL
* @version 	1
* @date    	2017-05-03
* @brief   	Class that defines the container for passing areas around
*****************************************************************************
**/

#include "GlobalIncludes.h"

#include "ControlInterface_Area.h"


namespace overview
{
	int ControlInterface_Area::ncols = 7200;

	bool ControlInterface_Area::isInArea(const OV_WorldPoint & pt) const
	{
		double dist = pointPolygonTest(corners, cv::Point2f(pt.x,pt.y), false);
		return (dist >= 0);
	}

	bool ControlInterface_Boundary::crosses(const OV_WorldPoint & pt1, const OV_WorldPoint & pt2) const
	{
		cv::Point2f s1(pt2.x - pt1.x, pt2.y - pt1.y); 
		for (int i = 0; i < (int)segments.size() - 1; i++)
		{
			//for each segment on the boundary check for intersection using Cramer's rule
			cv::Point2f s2(segments[i + 1] - segments[i]);
			float s = (-s1.y*(pt1.x - segments[i].x) + s1.x*(pt1.y - segments[i].y)) / (-s2.x * s1.y + s1.x * s2.y);
			float t = (s2.x*(pt1.y - segments[i].y) - s2.y*(pt1.x - segments[i].x)) / (-s2.x * s1.y + s1.x * s2.y);
			if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
			{
				return true;
			}
		}
		return false;
	}
}