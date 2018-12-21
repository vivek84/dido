/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	ControlInterface_Area.h
* @author  	SL
* @version 	1
* @date    	2017-05-03
* @brief   	Class that defines the container for passing areas around
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __ControlInterface_Area
#define __ControlInterface_Area


#pragma once

/*defines ---------------------------*/

namespace overview
{
	//area marked in the real world on the ground plane. whether you are in the area or not is marked using your ground plane projection
	class ControlInterface_Area
	{
	public:
		std::vector<cv::Point2f> corners;
		/*helpful function to check if something is inside the area*/
		bool isInArea(const OV_WorldPoint & pt) const;
		/*system parameter corresponding to the image width - needed for wrapping*/
		static int ncols;
	};

	//similar, but a boundary not an area
	class ControlInterface_Boundary
	{
	public:
		std::vector<cv::Point2f> segments;
		/*helpful function to check if something is inside the area*/
		bool crosses(const OV_WorldPoint & pt1, const OV_WorldPoint & pt2) const;
		/*system parameter corresponding to the image width - needed for wrapping*/
		static int ncols;
	};
}

#endif
