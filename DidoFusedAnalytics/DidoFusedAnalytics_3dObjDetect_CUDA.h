/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview    
* Limited. Any unauthorised use, reproduction or transfer of this         
* program is strictly prohibited.              
* Copyright 2017 Overview Limited. (Subject to limited                    
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoFusedAnalytics_3dObjDetect_CUDA.h
* @author  	SL
* @version 	1
* @date    	2017-07-05
* @brief   	class that does the blob detection on the GPU
 *****************************************************************************
**/
#ifndef __DidoFusedAnalytics_3dObjDetect_CUDA_H
#define __DidoFusedAnalytics_3dObjDetect_CUDA_H

#pragma once

/*the function - inputs are a pointer to a row major array of floats, the number of collumns it has, the number of rows it has, pointers to an array of pan angles, of tilt angles and of ranges, the size of those arrays. Return value is a pointer to a heap allocated array of floats that are the ranges that correspond with the input temperatures.*/

namespace overview
{
	//helper bounding box because you can't include openCV directly into CUDA code
	struct DidoFusedAnalytics_BoundingBox
	{
		//top left
		int x;
		int y;
		//bottom right
		int max_x;
		int max_y;
		float avdepth;
		float uncertainty;
	};
	
	class DidoFusedAnalytics_3dObjDetect_CUDA
	{
	protected:
		int minpoints; //minimum number of points to be a blob
		float rangeScaling;
		int ncore; //minimum number of points in the neighborhood to be a core point
		int epsilon; //how far away a point can be and still be core

		//the sacale we internally downsample to
		int workingscale = 2;
	public:
		DidoFusedAnalytics_3dObjDetect_CUDA(int _minpoints = 40, float _rangeScaling = 1.40f, int _ncore = 5, int _eps  = 3) : epsilon(_eps), minpoints(_minpoints), rangeScaling(_rangeScaling), ncore(_ncore){}
		~DidoFusedAnalytics_3dObjDetect_CUDA() = default;

		//clusters the given foreground map into a series of blobs
		//returns the labelled image and an array of bounding boxes
		std::vector<DidoFusedAnalytics_BoundingBox> detectBlobs(const float * foreground_ranges_min, const float * foreground_ranges_max, int rows, int cols) const;
		
		//parameter setting functions
		void setNcore(int nc) { ncore = nc; }
		void setRangeScaling(float rs) { rangeScaling = rs; }
		void setMinPoints(int mp) { minpoints = mp; }
	};
}

#endif 