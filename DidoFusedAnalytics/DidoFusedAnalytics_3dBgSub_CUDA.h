/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview    
* Limited. Any unauthorised use, reproduction or transfer of this         
* program is strictly prohibited.              
* Copyright 2013 Overview Limited. (Subject to limited                    
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoFusedAnalytics_3dBgSub_CUDA.h
* @author  	SL
* @version 	1
* @date    	2017-04-11
* @brief   	class that does the background subtraction on the GPU
 *****************************************************************************
**/
#ifndef __DidoFusedAnalytics_3dBgSub_CUDA_H
#define __DidoFusedAnalytics_3dBgSub_CUDA_H

#pragma once

/*the function - inputs are a pointer to a row major array of floats, the number of collumns it has, the number of rows it has,
pointers to an array of pan angles, of tilt angles and of ranges, the size of those arrays. Return value is a pointer to a heap
allocated array of floats that are the ranges that correspond with the input temperatures.*/

namespace overview
{
	class DidoFusedAnalytics_3dBgSub_CUDA
	{
		int nsteps = 0;
		int scale;
		const bool useRegion;
	protected:
		float * rangeModel, *rangevars, *tempvars, *modelweights;
		float * tempmodel;
		int rows, cols;
	public:
		DidoFusedAnalytics_3dBgSub_CUDA(int _rows, int _cols, bool useregion = false, int _scale = 2);
		~DidoFusedAnalytics_3dBgSub_CUDA();

		//copy construction
		DidoFusedAnalytics_3dBgSub_CUDA(DidoFusedAnalytics_3dBgSub_CUDA & cp);
		//move construction
		DidoFusedAnalytics_3dBgSub_CUDA(DidoFusedAnalytics_3dBgSub_CUDA && mv);

		//marks foreground pixels with the range
		//inputs and outputs should already be allocated on the GPU
		void apply(const thermalType * input_t, const float * input_d_min, const float * input_d_max, float * out_min, float * out_max, float learningRate = -1);

		//parameter setting
		void setHistory(int hist_);
		void setVariance(float initTempVar, float initRangeVar);
		void setThresholds(float backgroundThresh, float generativeThresh);//it's reccomended that generative threshold is less than background Threshold
	
		//sets the total weight of all possible background models for any pixel
		void setBackgroundWeight(float TB);

		void setErrorCap(float ec) { pars.c_errrorCap = ec; }

		//structure for easier passing around of parameters to the functions
		struct bgrPars
		{
			//threshold parameters
			int           c_nmixtures = 5; // maximal number of Gaussians in mixture
			float         c_TB = 0.92f; // threshold sum of weights for background test
			float         c_Tb = 4.0f;
			float         c_Tg = 3.0f;

			//variance parameters
			float         c_varInit_r = 5.2f; // initial variance for new components
			float         c_varMin_r = 1.60f;
			float         c_varMax_r = 5.0f * c_varInit_r;

			float         c_varInit_t = 32.0f; // initial variance for new components
			float         c_varMin_t = 3.3f;
			float         c_varMax_t = 5.0f *c_varInit_t;

			//parameters to do with not having lidar data
			float			c_Tb_u;
			float			c_Tg_u;
			//paramterdefining the boundary for an outlying pixel for range mode
			float			c_errrorCap = 4.98f;
			float			c_t_varInflate = 8.0f;
			float			c_r_varInflate = 3.0f;

		};

	protected:
		//parameters
		int history = 120; // Learning rate; alpha = 1/history
		float ct = 0.017f;
		bgrPars pars;
	};

	//only runs the background subtraction on the thermal input, but still allows for use of the region based methods and similar
	class DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly
	{
		int nsteps = 0;
		int scale;
	protected:
		float *tempvars, *modelweights;
		float * tempmodel;
		int rows, cols;
	public:
		DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly(int _rows, int _cols, int _scale = 2);
		~DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly();

		//copy construction
		DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly(DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly & cp);
		//move construction
		DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly(DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly && mv);

		//marks foreground pixels with the range
		//inputs and outputs should already be allocated on the GPU
		void apply(const thermalType * input_t, const float * input_d_min, const float * input_d_max, float * out_min, float * out_max, float learningRate = -1);
		void setHistory(int hist_);
		void setVariance(float initTempVar, float initRangeVar);
		void setThresholds(float backgroundThresh, float generativeThresh);//it's reccomended that generative threshold is less than background Threshold

																		   //sets the total weight of all possible background models for any pixel
		void setBackgroundWeight(float TB);

		void setErrorCap(float ec) { pars.c_errrorCap = ec; }

	protected:
		//parameters
		int history = 120; // Learning rate; alpha = 1/history
		float ct = 0.017f;
		DidoFusedAnalytics_3dBgSub_CUDA::bgrPars pars;
	};
}

#endif 