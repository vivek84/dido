/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview    
* Limited. Any unauthorised use, reproduction or transfer of this         
* program is strictly prohibited.              
* Copyright 2017 Overview Limited. (Subject to limited                    
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoFusedAnalytics_3dBgSub_CUDA.h
* @author  	SL
* @version 	1
* @date    	2017-06-29
* @brief   	class that does the background subtraction on the GPU
 *****************************************************************************
**/

/**
*	description of the algorithm
*	this uses a standard MOG2 based background subtractor with a few key differences -

*	it checks if the background point would be in range of the lidar, and if not, adjusts the thresholds appropriately and only uses thermal instead
*	if it is in range of the lidar,  it maintains separate variances for each component, and because the lidar values are returned as a range, it models them
*	as a gaussian with mean at the center of the range and standard deviation = half the width of the range. Then the values are based simply on the joint integrals of the two distributions 
*/



#include "global_defines.h"
#include "DidoFusedAnalytics_3dBgSub_CUDA.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include "math.h"
#include "math_constants.h"
#include "cuda_profiler_api.h"

//for our logging we will throw an error that can then be caught by the surrounding code that is allowed to include boost
#include "CUDA_Exception.h"


/*
   if the computer being used doesn't have a GPU, define DIDOLIDAR_NOGPU as 1 in the preprocessor, and this wil produce some noops instead. It still requires the nvidia sdk to compile at all, however
*/



#if DIDOLIDAR_NOGPU

#else
//error handling function
static void HandleError( cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
	//	cudaDeviceReset();
		throw overview::CUDA_Exception(cudaGetErrorString( err ) , err, line, file);
    }
}
#define HANDLE_ERROR(err) {HandleError((err), __FILE__, __LINE__);} 
#endif

namespace overview
{

namespace bgrcuda
{
#define NOPOINTVALUE -1.0f
	//convenience function for swapping with
	__device__ __forceinline__ void swap(float * array, int ind1, int ind2)
	{
		float tmp = array[ind1];
		array[ind1] = array[ind2];
		array[ind2] = tmp;
	}

	template<typename T>
	__global__ void downsample(const T * in, T* out, int nrows, int ncols, int scale)
	{
		const int totind_x = (threadIdx.x + blockIdx.x * blockDim.x);
		const int totind_y = (threadIdx.y + blockIdx.y * blockDim.y);
		if (totind_x >= ncols / scale || totind_y > nrows / scale) return;
		double total = 0;
		int npts = 0;
		for (int i = 0; i < scale; i++)
		{
			int index_x = totind_x*scale + i;
			if (index_x >= ncols) continue;
			for (int j = 0; j < scale; j++)
			{
				int index_y = totind_y*scale + j;
				if (index_y >= nrows) continue;
				total += in[index_x + ncols*index_y];
				npts++;
			}
		}
		out[totind_x + totind_y*(ncols / scale)] = (T)(total / npts);
	}

	__global__ void upsample(const float * in_pano, float * out_pano,int nrows,  int ncols, int scale)
	{
		int index_x = (threadIdx.x + blockIdx.x * blockDim.x);
		int index_y = (threadIdx.y + blockIdx.y * blockDim.y);
		if (index_x / scale < ncols / scale && index_y / scale < nrows / scale)
			out_pano[index_x + index_y*ncols] = in_pano[(index_x / scale) + (index_y / scale)*(ncols / scale)];
	}

	//the actual bgr
	__global__ void mixturegaussians(const float * ranges_min, const float * ranges_max, float * rangemdl, float * rangevar, const thermalType * temps,
		float * tempmdl, float * tempvar, float * modelweights, float* out_min, float* out_max, int rows, int cols, float alphaT, float alpha1, float prune, DidoFusedAnalytics_3dBgSub_CUDA::bgrPars pars)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= cols || y >= rows)
			return;

		float r_min = ranges_min[x + y*cols];
		float r_max = ranges_max[x + y*cols];
		float r_width = r_max - r_min;
		float r_center =  (r_max + r_min)/2;
		float temp = temps[x + y*cols];

		//check if the observation has a range
		bool hasRange = r_center > 0;


		//calculate distances to the modes (+ sort)
		//here we need to go in descending order!!!

		bool background = false; // true - the pixel classified as background

		//internal:

		bool fitsPDF = false; //if it remains zero a new GMM mode will be added

		float totalWeight = 0.0f;

		//go through all modes
		int lastmode = 0;
		for (int mode = 0; mode < pars.c_nmixtures; ++mode)
		{
			int modeind = (x + (y*cols))*pars.c_nmixtures + mode;
			//skip modes with no weight
			//need only weight if fit is found
			if (modelweights[modeind] <= 0) continue;
			float weight = alpha1 * modelweights[modeind] + prune;
			lastmode++;
			//fit not found yet
			if (!fitsPDF)
			{
				bool hasModelRange = rangemdl[modeind] > 0;
				//check if it belongs to some of the remaining modes
				float t_var = tempvar[modeind];
				//our observations of range are also gaussian distibuted, so we look at the distribution of the convolution
				float r_var = rangevar[modeind] + (r_width*r_width);
            
				//calculate difference and distance
				float t_diff = tempmdl[modeind] - temp;
				float r_diff = rangemdl[modeind] - r_center;
				//weighted distance in both directions
				float dist2 = hasRange && hasModelRange ? t_diff*t_diff*r_var + r_diff*r_diff*t_var : t_diff*t_diff;
				float bgthresh = hasRange && hasModelRange ? pars.c_Tb * t_var * r_var : pars.c_Tb*t_var;
				float genthresh = hasRange && hasModelRange ? pars.c_Tg * t_var * r_var : pars.c_Tg*t_var;

				//background? - Tb - usually larger than Tg
				if (totalWeight < pars.c_TB && dist2 < bgthresh)
					background = true;

				//check fit
				if (dist2 < genthresh)
				{
					//belongs to the mode
					fitsPDF = true;

					//update distribution

					//update weight
					weight += alphaT;
					float k = alphaT / weight;

					//update variance
					float t_varnew = t_var + k * (t_diff*t_diff - t_var);
					//integrating the weighting against the probability of the observation
					float r_varnew = rangevar[modeind] + hasRange && hasModelRange ? k * ((r_width*r_width + 1)*(r_diff*r_diff) + pars.c_r_varInflate - rangevar[modeind]) : 0;

					//update means
					tempmdl[modeind] = tempmdl[modeind] - k * t_diff;
					rangemdl[modeind] = hasModelRange ? (rangemdl[modeind] - hasRange ? k *( r_diff ) : 0) : r_center;


					//limit the variance
					t_varnew = (t_varnew < pars.c_varMin_t) ? pars.c_varMin_t : (t_varnew > pars.c_varMax_t)? pars.c_varMax_t : t_varnew;
					r_varnew = (r_varnew < pars.c_varMin_r) ? pars.c_varMin_r : (r_varnew > pars.c_varMax_r)? pars.c_varMax_r : r_varnew;

					rangevar[modeind] = r_varnew;
					tempvar[modeind] = t_varnew;

					//sort
					//all other weights are at the same place and
					//only the matched (iModes) is higher -> just find the new place for it

					for (int i = mode; i > 0; --i)
					{
						//check one up
						if (weight < modelweights[(i - 1) + pars.c_nmixtures*(x + y*cols)])
							break;

						//swap one up
						swap(modelweights, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
						swap(rangevar, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
						swap(tempvar, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
						swap(rangemdl, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
						swap(tempmdl, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
					}

					//belongs to the mode - bFitsPDF becomes 1
				}
			} // !fitsPDF

			//check prune
			if (weight < -prune)
			{
				weight = 0.0f;
				lastmode--;
			}

			modelweights[modeind] = weight; //update weight by the calculated value
			totalWeight += weight;
		}

		//renormalize weights

		totalWeight = totalWeight == 0 ? 1.f : 1.f / totalWeight;
		for (int mode = 0; mode < pars.c_nmixtures; ++mode)
			modelweights[(x + (y*cols))*pars.c_nmixtures + mode] *= totalWeight;

		//make new mode if needed and exit

		if (!fitsPDF)
		{
			if(lastmode == pars.c_nmixtures) lastmode--;
			if (lastmode == 0)
				modelweights[(x + (y*cols))*pars.c_nmixtures + lastmode] = 1.f;
			else
			{
				modelweights[(x + (y*cols))*pars.c_nmixtures + lastmode] = alphaT;

				// renormalize all other weights

				for (int i = lastmode - 1; i >= 0 ; i--)
					modelweights[(x + (y*cols))*pars.c_nmixtures + i] *= alpha1;
			}

			// init

			rangemdl[(x + (y*cols))*pars.c_nmixtures + lastmode] = hasRange ? r_center : -1.0f;
			tempmdl[(x + (y*cols))*pars.c_nmixtures + lastmode] = temp;
			tempvar[(x + (y*cols))*pars.c_nmixtures + lastmode] = pars.c_varInit_t;
			rangevar[(x + (y*cols))*pars.c_nmixtures + lastmode] = pars.c_varInit_r;

			//sort
			//find the new place for it

			for (int i = lastmode - 1; i > 0; --i)
			{
				// check one up
				if (alphaT < modelweights[(i - 1) + pars.c_nmixtures*(x + y*cols)])
					break;

				//swap one up
				swap(modelweights, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
				swap(rangevar, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
				swap(tempvar, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
				swap(rangemdl, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
				swap(tempmdl, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
			}
		}
		//return inf if we don't have a range
		out_min[x + y*cols] = background ? NOPOINTVALUE : hasRange ? r_min : CUDART_INF_F;
		out_max[x + y*cols] = background ? NOPOINTVALUE : hasRange ? r_max : CUDART_INF_F;
	}

	__global__ void mixturegaussians_onlyTherm(const float * ranges_min, const float * ranges_max, const thermalType * temps, float * tempmdl, float * tempvar, 
		float * modelweights, float* out_min, float* out_max, int rows, int cols, float alphaT, float alpha1, float prune, DidoFusedAnalytics_3dBgSub_CUDA::bgrPars pars)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= cols || y >= rows)
			return;

		float r_min = ranges_min[x + y*cols];
		float r_max = ranges_max[x + y*cols];
		float temp = temps[x + y*cols];

		//calculate distances to the modes (+ sort)
		//here we need to go in descending order!!!

		bool background = false; // true - the pixel classified as background

		//internal:

		bool fitsPDF = false; //if it remains zero a new GMM mode will be added

		float totalWeight = 0.0f;

		//go through all modes
		int lastmode = 0;
		for (int mode = 0; mode < pars.c_nmixtures; ++mode)
		{
			int modeind = (x + (y*cols))*pars.c_nmixtures + mode;
			//skip modes with no weight
			//need only weight if fit is found
			if (modelweights[modeind] <= 0) continue;
			float weight = alpha1 * modelweights[modeind] + prune;
			lastmode++;
			//fit not found yet
			if (!fitsPDF)
			{
				//check if it belongs to some of the remaining modes
				float t_var = tempvar[modeind];
            
				//calculate difference and distance
				float t_diff = tempmdl[modeind] - temp;
				//weighted distance in both directions
				float dist2 = t_diff*t_diff;
				float bgthresh = pars.c_Tb*t_var;
				float genthresh =  pars.c_Tg*t_var;

				//background? - Tb - usually larger than Tg
				if (totalWeight < pars.c_TB && dist2 < bgthresh)
					background = true;

				//check fit
				if (dist2 < genthresh)
				{
					//belongs to the mode
					fitsPDF = true;

					//update distribution

					//update weight
					weight += alphaT;
					float k = alphaT / weight;

					//update variance
					float t_varnew = t_var + k * (t_diff*t_diff - t_var);
					//integrating the weighting against the probability of the observation

					//update means
					tempmdl[modeind] = tempmdl[modeind] - k * t_diff;
					
					//limit the variance
					t_varnew = (t_varnew < pars.c_varMin_t) ? pars.c_varMin_t : (t_varnew > pars.c_varMax_t)? pars.c_varMax_t : t_varnew;

					tempvar[modeind] = t_varnew;

					//sort
					//all other weights are at the same place and
					//only the matched (iModes) is higher -> just find the new place for it

					for (int i = mode; i > 0; --i)
					{
						//check one up
						if (weight < modelweights[(i - 1) + pars.c_nmixtures*(x + y*cols)])
							break;

						//swap one up
						swap(modelweights, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
						swap(tempvar, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
						swap(tempmdl, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
					}

					//belongs to the mode - bFitsPDF becomes 1
				}
			} // !fitsPDF

			//check prune
			if (weight < -prune)
			{
				weight = 0.0f;
				lastmode--;
			}

			modelweights[modeind] = weight; //update weight by the calculated value
			totalWeight += weight;
		}

		//renormalize weights

		totalWeight = totalWeight == 0 ? 1.f : 1.f / totalWeight;
		for (int mode = 0; mode < pars.c_nmixtures; ++mode)
			modelweights[(x + (y*cols))*pars.c_nmixtures + mode] *= totalWeight;

		//make new mode if needed and exit

		if (!fitsPDF)
		{
			if(lastmode == pars.c_nmixtures) lastmode--;
			if (lastmode == 0)
				modelweights[(x + (y*cols))*pars.c_nmixtures + lastmode] = 1.f;
			else
			{
				modelweights[(x + (y*cols))*pars.c_nmixtures + lastmode] = alphaT;

				// renormalize all other weights

				for (int i = lastmode - 1; i >= 0 ; i--)
					modelweights[(x + (y*cols))*pars.c_nmixtures + i] *= alpha1;
			}

			// init

			tempmdl[(x + (y*cols))*pars.c_nmixtures + lastmode] = temp;
			tempvar[(x + (y*cols))*pars.c_nmixtures + lastmode] = pars.c_varInit_t;

			//sort
			//find the new place for it

			for (int i = lastmode - 1; i > 0; --i)
			{
				// check one up
				if (alphaT < modelweights[(i - 1) + pars.c_nmixtures*(x + y*cols)])
					break;

				//swap one up
				swap(modelweights, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
				swap(tempvar, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
				swap(tempmdl, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
			}
		}
		//return inf if we don't have a range
		out_min[x + y*cols] = background ? NOPOINTVALUE : r_min;
		out_max[x + y*cols] = background ? NOPOINTVALUE : r_max;
	}

#define REGION_WIDTH 3
#define REGION_AREA (REGION_WIDTH*REGION_WIDTH)
#define REGION_HWIDTH (REGION_WIDTH/2)

	//the actual bgr - region based method
	__global__ void mixturegaussians_region(const float * ranges_min, const float * ranges_max, float * rangemdl, float * rangevar, const thermalType * temps,
		float * tempmdl, float * tempvar, float * modelweights, float* out_min, float* out_max, int rows, int cols, float alphaT, float alpha1, float prune, DidoFusedAnalytics_3dBgSub_CUDA::bgrPars pars)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= cols || y >= rows)
			return;

		float r_center[REGION_AREA], r_width =0, temp[REGION_AREA];
		for (int i = 0; i < REGION_AREA; i++)
		{
			int sx = x - REGION_HWIDTH + i % REGION_WIDTH;
			int sy = y - REGION_HWIDTH + i / REGION_WIDTH;
			if (sx < 0 || sy < 0 || sx >= cols || sy >= rows)
			{
				r_center[i] = 0;
				temp[i] = 0;
			}
			else
			{
				float r_min = ranges_min[sx + sy*cols];
				float r_max = ranges_max[sx + sy*cols];
				r_width += r_max - r_min;
				r_center[i] = (r_max + r_min) / 2;
				temp[i] = temps[sx + sy*cols];
			}
		}
		//normalise the variance of the range observations
		r_width /= REGION_AREA;
		//check if the observation is outside our maximum range		
		bool hasRange = r_center[4] > 0;


		//calculate distances to the modes (+ sort)
		//here we need to go in descending order!!!

		bool background = false; // true - the pixel classified as background

		//internal:

		bool fitsPDF = false; //if it remains zero a new GMM mode will be added

		float totalWeight = 0.0f;

		//go through all modes
		int lastmode = 0;
		for (int mode = 0; mode < pars.c_nmixtures; ++mode)
		{
			int modeind = (x + (y*cols))*pars.c_nmixtures + mode;
			//skip modes with no weight
			//need only weight if fit is found
			if (modelweights[modeind] <= 0) continue;
			float weight = alpha1 * modelweights[modeind] + prune;
			lastmode++;
			//fit not found yet
			if (!fitsPDF)
			{
				bool hasModelRange = rangemdl[modeind* REGION_AREA + REGION_WIDTH] > 0;
				//check if it belongs to some of the remaining modes
				float t_var = tempvar[modeind];
				//our observations of range are also gaussian distibuted, so we look at the distribution of the convolution
				float r_var = rangevar[modeind] + (r_width*r_width);
            
				//calculate difference and distance
				float t_diff[REGION_AREA], r_diff[REGION_AREA];
				for (int i = 0; i < REGION_AREA; i++)
				{
					t_diff[i] = tempmdl[modeind*REGION_AREA + i] - temp[i];
					r_diff[i] = rangemdl[modeind * REGION_AREA + i] - r_center[i];
				}

				//clculate the L2norm of the adjusted differences
				float tdist = 0, rdist = 0;
				for (int i = 0; i < REGION_AREA; i++)
				{
					tdist += fmin(t_diff[i] * t_diff[i], pars.c_errrorCap * t_var);
					rdist += fmin(r_diff[i] * r_diff[i], pars.c_errrorCap * r_var);
				}
				float dist2 = hasRange && hasModelRange ? tdist*r_var + rdist*t_var : tdist;
				//weighted distance in both directions
				float bgthresh = hasRange && hasModelRange ? pars.c_Tb * t_var * r_var *REGION_AREA : pars.c_Tb*t_var *REGION_AREA;
				float genthresh = hasRange && hasModelRange ? pars.c_Tg * t_var * r_var * REGION_AREA : pars.c_Tg*t_var *REGION_AREA;

				//background? - Tb - usually larger than Tg
				if (totalWeight < pars.c_TB && dist2 < bgthresh)
					background = true;

				//check fit
				if (dist2 < genthresh)
				{
					//belongs to the mode
					fitsPDF = true;

					//update distribution

					//update weight
					weight += alphaT;
					float k = alphaT / weight;

					//update variance
					float t_varnew = t_var + k * (tdist/ REGION_AREA + pars.c_t_varInflate - t_var);
					//integrating the weighting against the probability of the observation
					float r_varnew = rangevar[modeind] + hasRange && hasModelRange ? (k * ((r_width*r_width + 1)*(rdist/ REGION_AREA) + pars.c_r_varInflate - rangevar[modeind])) : 0;

					//update means
					for (int i = 0; i < REGION_AREA; i++)
					{
						tempmdl[modeind*REGION_AREA +i] = tempmdl[modeind * REGION_AREA + i] - k * t_diff[i];
						rangemdl[modeind*REGION_AREA + i] = rangemdl[modeind * REGION_AREA + i] > 0 ? rangemdl[modeind * REGION_AREA + i] - k *(r_diff[i]) : r_center[i];
					}


					//limit the variance
					t_varnew = (t_varnew < pars.c_varMin_t) ? pars.c_varMin_t : (t_varnew > pars.c_varMax_t)? pars.c_varMax_t : t_varnew;
					r_varnew = (r_varnew < pars.c_varMin_r) ? pars.c_varMin_r : (r_varnew > pars.c_varMax_r)? pars.c_varMax_r : r_varnew;

					rangevar[modeind] = r_varnew;
					tempvar[modeind] = t_varnew;

					//sort
					//all other weights are at the same place and
					//only the matched (iModes) is higher -> just find the new place for it

					for (int i = mode; i > 0; --i)
					{
						//check one up
						if (weight < modelweights[(i - 1) + pars.c_nmixtures*(x + y*cols)])
							break;

						//swap one up
						swap(modelweights, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
						swap(rangevar, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
						swap(tempvar, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));

						for (int k = 0; k < REGION_AREA; k++)
						{
							swap(rangemdl, ((i - 1) + pars.c_nmixtures*(x + y*cols)) * REGION_AREA + k, (i + pars.c_nmixtures*(x + y*cols)) * REGION_AREA + k);
							swap(tempmdl, ((i - 1) + pars.c_nmixtures*(x + y*cols)) *REGION_AREA + k, (i + pars.c_nmixtures*(x + y*cols)) * REGION_AREA + k);
						}
					}

					//belongs to the mode - bFitsPDF becomes 1
				}
			} // !fitsPDF

			//check prune
			if (weight < -prune)
			{
				weight = 0.0f;
				lastmode--;
			}

			modelweights[modeind] = weight; //update weight by the calculated value
			totalWeight += weight;
		}

		//renormalize weights

		totalWeight = totalWeight == 0 ? 1.f : 1.f / totalWeight;
		for (int mode = 0; mode < pars.c_nmixtures; ++mode)
			modelweights[(x + (y*cols))*pars.c_nmixtures + mode] *= totalWeight;

		//make new mode if needed and exit

		if (!fitsPDF)
		{
			if(lastmode == pars.c_nmixtures) lastmode--;
			if (lastmode == 0)
				modelweights[(x + (y*cols))*pars.c_nmixtures + lastmode] = 1.f;
			else
			{
				modelweights[(x + (y*cols))*pars.c_nmixtures + lastmode] = alphaT;

				// renormalize all other weights

				for (int i = lastmode - 1; i >= 0 ; i--)
					modelweights[(x + (y*cols))*pars.c_nmixtures + i] *= alpha1;
			}

			// init


			for (int k = 0; k < REGION_AREA; k++)
			{
				rangemdl[((x + (y*cols))*pars.c_nmixtures + lastmode)*REGION_AREA + k] = r_center[k];
				tempmdl[((x + (y*cols))*pars.c_nmixtures + lastmode)*REGION_AREA +k] = temp[k];
			}

			tempvar[(x + (y*cols))*pars.c_nmixtures + lastmode] = pars.c_varInit_t;
			rangevar[(x + (y*cols))*pars.c_nmixtures + lastmode] = pars.c_varInit_r;

			//sort
			//find the new place for it

			for (int i = lastmode - 1; i > 0; --i)
			{
				// check one up
				if (alphaT < modelweights[(i - 1) + pars.c_nmixtures*(x + y*cols)])
					break;

				//swap one up
				swap(modelweights, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
				swap(rangevar, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
				swap(tempvar, (i - 1) + pars.c_nmixtures*(x + y*cols),i + pars.c_nmixtures*(x + y*cols));
				for (int k = 0; k < REGION_AREA; k++)
				{
					swap(rangemdl, ((i - 1) + pars.c_nmixtures*(x + y*cols)) * REGION_AREA + k, (i + pars.c_nmixtures*(x + y*cols)) * REGION_AREA + k);
					swap(tempmdl, ((i - 1) + pars.c_nmixtures*(x + y*cols)) * REGION_AREA + k, (i + pars.c_nmixtures*(x + y*cols)) * REGION_AREA + k);
				}
			}
		}

		out_min[x + y*cols] = background ? NOPOINTVALUE :  hasRange ? ranges_min[x + y*cols]: CUDART_INF_F;
		out_max[x + y*cols] = background ? NOPOINTVALUE : hasRange ? ranges_max[x + y*cols] : CUDART_INF_F;

	}

}

//called whenever we change the constants 
static inline float getUnivarateThresh(float thresh)
{
	//these numbers are numerically estimated from the normal distributiobns in the range 1-6
	return pow(thresh, 4)*0.0016298f - pow(thresh, 3)*0.0080105f - pow(thresh, 2)*0.1293664f + thresh*1.3835517f - 0.6398407f;
}

DidoFusedAnalytics_3dBgSub_CUDA::DidoFusedAnalytics_3dBgSub_CUDA(DidoFusedAnalytics_3dBgSub_CUDA & cp): useRegion(cp.useRegion), scale(cp.scale)
{
	cols = cp.cols;
	rows = cp.rows;
	ct = cp.ct;
	history = cp.history;
	nsteps = cp.nsteps;
	pars = cp.pars;

	//cuda allocated variables
#if DIDOLIDAR_NOGPU

#else
	int modelsize = rows*cols*pars.c_nmixtures * sizeof(float);
	int regionsize = useRegion ? modelsize*REGION_AREA : modelsize;

	//allocate the model
	HANDLE_ERROR(cudaMalloc(&rangeModel, regionsize));
	HANDLE_ERROR(cudaMalloc(&rangevars, modelsize));
	HANDLE_ERROR(cudaMalloc(&tempmodel, regionsize));
	HANDLE_ERROR(cudaMalloc(&tempvars, modelsize));
	HANDLE_ERROR(cudaMalloc(&modelweights, modelsize));

	HANDLE_ERROR(cudaMemcpy(modelweights,cp.modelweights, modelsize,cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(rangeModel, cp.rangeModel, regionsize, cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(rangevars, cp.rangevars, modelsize, cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(tempmodel, cp.tempmodel, regionsize, cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(tempvars, cp.tempvars, modelsize, cudaMemcpyDeviceToDevice));
#endif
}

DidoFusedAnalytics_3dBgSub_CUDA::DidoFusedAnalytics_3dBgSub_CUDA(DidoFusedAnalytics_3dBgSub_CUDA && mv): useRegion(mv.useRegion), scale(mv.scale)
{
	cols = mv.cols;
	rows = mv.rows;
	ct = mv.ct;
	history = mv.history;
	nsteps = mv.nsteps;
	pars = mv.pars;

	modelweights = mv.modelweights;
	mv.modelweights = nullptr;
	rangeModel = mv.rangeModel;
	mv.rangeModel = nullptr;
	rangevars = mv.rangevars;
	mv.rangevars = nullptr;
	tempmodel = mv.tempmodel;
	mv.tempmodel = nullptr;
	tempvars = mv.tempvars;
	mv.tempvars = nullptr;

}

DidoFusedAnalytics_3dBgSub_CUDA::DidoFusedAnalytics_3dBgSub_CUDA(int _rows, int _cols, bool useregion, int _scale)
	: rows(_rows), cols(_cols), useRegion(useregion), scale(_scale)
{
   #if DIDOLIDAR_NOGPU

   #else
	int modelsize = (rows/scale)*(cols/scale)*pars.c_nmixtures*sizeof(float);
	int regionsize = useRegion ? modelsize*REGION_AREA : modelsize;
	//allocate the model
	HANDLE_ERROR(cudaMalloc(&rangeModel, regionsize));
	HANDLE_ERROR(cudaMalloc(&rangevars, modelsize));
	HANDLE_ERROR(cudaMalloc(&tempmodel, regionsize));
	HANDLE_ERROR(cudaMalloc(&tempvars, modelsize));
	HANDLE_ERROR(cudaMalloc(&modelweights, modelsize));

	pars.c_Tb_u = getUnivarateThresh(pars.c_Tb);
	pars.c_Tg_u = getUnivarateThresh(pars.c_Tg);
	#endif
}

DidoFusedAnalytics_3dBgSub_CUDA::~DidoFusedAnalytics_3dBgSub_CUDA()
{

   #if DIDOLIDAR_NOGPU

   #else
	//deallocate the model
	if(rangeModel != nullptr) (cudaFree(rangeModel));
	if(rangevars != nullptr) (cudaFree(rangevars));
	if(tempmodel != nullptr) (cudaFree(tempmodel));
	if(tempvars != nullptr) (cudaFree(tempvars));
	if(modelweights != nullptr) (cudaFree(modelweights));
	#endif
}

void DidoFusedAnalytics_3dBgSub_CUDA::apply(const thermalType * input_t, 
	const float * input_d_min, const float * input_d_max, float * out_min, float * out_max, float learningRate) 
{

#if DIDOLIDAR_NOGPU

#else
	nsteps++;
	float lr;
	//allocate the learning rate
	if(learningRate < 0)
	{
		lr = nsteps > history ? 1.f/history : 1.f/(nsteps);
	}
	else
	{
		lr = learningRate;
	}

	//downsample it
	float * l_d_min, *l_d_max, *l_o_max, *l_o_min;
	thermalType * l_therm;
	HANDLE_ERROR(cudaMalloc(&l_d_max, (cols / scale)*(rows / scale) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&l_d_min, (cols / scale)*(rows / scale) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&l_o_max, (cols / scale)*(rows / scale) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&l_o_min, (cols / scale)*(rows / scale) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&l_therm, (cols / scale)*(rows / scale) * sizeof(thermalType)));

	int blockdim = 16;
	dim3 grid((cols / scale) / blockdim + 1, (rows / scale) / blockdim + 1);
	dim3 block(blockdim, blockdim);
	if (scale == 1)
	{
		HANDLE_ERROR(cudaMemcpy(l_d_max, input_d_max, cols*rows*sizeof(float), cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaMemcpy(l_d_min, input_d_min, cols*rows * sizeof(float), cudaMemcpyDeviceToDevice));
	}
	else
	{
		bgrcuda::downsample<float> <<<grid, block >>> (input_d_max, l_d_max, rows, cols, scale);
		bgrcuda::downsample <float><<<grid, block >>> (input_d_min, l_d_min, rows, cols, scale);
		bgrcuda::downsample <thermalType> <<<grid, block >>> (input_t, l_therm, rows, cols, scale);
	}
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError());

	//run the bgr
	if(useRegion)
	{
		bgrcuda::mixturegaussians_region<<<grid, block>>>(l_d_min, l_d_max, rangeModel, rangevars, l_therm,
			tempmodel, tempvars, modelweights, l_o_min, l_o_max, rows/scale, cols/scale, lr, 1.0f - lr, -lr*ct, pars);
	}
	else
	{
		bgrcuda::mixturegaussians<<<grid, block>>>(l_d_min, l_d_max, rangeModel, rangevars, l_therm,
			tempmodel, tempvars, modelweights, l_o_min, l_o_max, rows/scale, cols/scale, lr, 1.0f - lr, -lr*ct, pars);
	}
	cudaDeviceSynchronize();
    HANDLE_ERROR(cudaGetLastError());

	//upsample it
	if (scale == 1)
	{
		HANDLE_ERROR(cudaMemcpy(out_max, l_o_max, cols*rows * sizeof(float), cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaMemcpy(out_min, l_o_min, cols*rows * sizeof(float), cudaMemcpyDeviceToDevice));
	}
	else
	{
		grid = dim3((cols) / blockdim + 1, (rows ) / blockdim + 1);
		bgrcuda::upsample <<<grid, block >>> (l_o_max, out_max, rows, cols, scale);
		bgrcuda::upsample<<<grid, block >>> (l_o_min, out_min, rows, cols, scale);
	}
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError());

	if (l_o_max) cudaFree(l_o_max);
	if (l_d_max) cudaFree(l_d_max);
	if (l_d_min) cudaFree(l_d_min);
	if (l_o_min) cudaFree(l_o_min);
	if (l_therm) cudaFree(l_therm);

#endif

}

void DidoFusedAnalytics_3dBgSub_CUDA::setHistory(int hist_)
{
	history = hist_;
}
void DidoFusedAnalytics_3dBgSub_CUDA::setBackgroundWeight(float TB)
{
	pars.c_TB = TB;
}

void DidoFusedAnalytics_3dBgSub_CUDA::setVariance(float initTempVar, float initRangeVar)
{
	//parameters for thermal space
	pars.c_varInit_t = initTempVar; // initial variance for new components
	pars.c_varMax_t = 5.0f * pars.c_varInit_t;
	pars.c_varMin_t = pars.c_varInit_t/1.5f;

	//params for range space
	pars.c_varInit_r = initRangeVar; // initial variance for new components
	pars.c_varMax_r = 5.0f * pars.c_varInit_r;
	pars.c_varMin_r = pars.c_varInit_r/4;

	pars.c_r_varInflate = initRangeVar / 3;
	pars.c_t_varInflate = initTempVar/3;
}
void DidoFusedAnalytics_3dBgSub_CUDA::setThresholds(float backgroundThresh, float generativeThresh)
{
	pars.c_Tb = backgroundThresh;
	pars.c_Tg  = generativeThresh;

	pars.c_Tb_u = getUnivarateThresh(pars.c_Tb);
	pars.c_Tg_u = getUnivarateThresh(pars.c_Tg);
}

DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly::DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly(DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly & cp): scale(cp.scale)
{
	cols = cp.cols;
	rows = cp.rows;
	ct = cp.ct;
	history = cp.history;
	nsteps = cp.nsteps;
	pars = cp.pars;

	//cuda allocated variables
#if DIDOLIDAR_NOGPU

#else
	int modelsize = rows*cols*pars.c_nmixtures * sizeof(float);
	int regionsize = modelsize;

	//allocate the model
	HANDLE_ERROR(cudaMalloc(&tempmodel, regionsize));
	HANDLE_ERROR(cudaMalloc(&tempvars, modelsize));
	HANDLE_ERROR(cudaMalloc(&modelweights, modelsize));
	HANDLE_ERROR(cudaMemcpy(modelweights,cp.modelweights, modelsize,cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(tempmodel, cp.tempmodel, regionsize, cudaMemcpyDeviceToDevice));
	HANDLE_ERROR(cudaMemcpy(tempvars, cp.tempvars, modelsize, cudaMemcpyDeviceToDevice));
#endif
}

DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly::DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly(DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly && mv): scale(mv.scale)
{
	cols = mv.cols;
	rows = mv.rows;
	ct = mv.ct;
	history = mv.history;
	nsteps = mv.nsteps;
	pars = mv.pars;

	modelweights = mv.modelweights;
	mv.modelweights = nullptr;
	tempmodel = mv.tempmodel;
	mv.tempmodel = nullptr;
	tempvars = mv.tempvars;
	mv.tempvars = nullptr;

}

DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly::DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly(int _rows, int _cols, int _scale)
	: rows(_rows), cols(_cols),  scale(_scale)
{
   #if DIDOLIDAR_NOGPU

   #else
	int modelsize = (rows/scale)*(cols/scale)*pars.c_nmixtures*sizeof(float);
	int regionsize = modelsize;
	//allocate the model
	HANDLE_ERROR(cudaMalloc(&tempmodel, regionsize));
	HANDLE_ERROR(cudaMalloc(&tempvars, modelsize));
	HANDLE_ERROR(cudaMalloc(&modelweights, modelsize));

	pars.c_Tb_u = getUnivarateThresh(pars.c_Tb);
	pars.c_Tg_u = getUnivarateThresh(pars.c_Tg);
	#endif
}

DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly::~DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly()
{

   #if DIDOLIDAR_NOGPU

   #else
	//deallocate the model
	if(tempmodel != nullptr) (cudaFree(tempmodel));
	if(tempvars != nullptr) (cudaFree(tempvars));
	if(modelweights != nullptr) (cudaFree(modelweights));
	#endif
}

void DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly::apply(const thermalType * input_t, 
	const float * input_d_min, const float * input_d_max, float * out_min, float * out_max, float learningRate) 
{

#if DIDOLIDAR_NOGPU

#else
	nsteps++;
	float lr;
	//allocate the learning rate
	if(learningRate < 0)
	{
		lr = nsteps > history ? 1.f/history : 1.f/(nsteps);
	}
	else
	{
		lr = learningRate;
	}

	//downsample it
	float * l_d_min, *l_d_max, *l_o_max, *l_o_min;
	thermalType * l_therm;
	HANDLE_ERROR(cudaMalloc(&l_d_max, (cols / scale)*(rows / scale) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&l_d_min, (cols / scale)*(rows / scale) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&l_o_max, (cols / scale)*(rows / scale) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&l_o_min, (cols / scale)*(rows / scale) * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&l_therm, (cols / scale)*(rows / scale) * sizeof(thermalType)));

	int blockdim = 16;
	dim3 grid((cols / scale) / blockdim + 1, (rows / scale) / blockdim + 1);
	dim3 block(blockdim, blockdim);
	if (scale == 1)
	{
		HANDLE_ERROR(cudaMemcpy(l_d_max, input_d_max, cols*rows*sizeof(float), cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaMemcpy(l_d_min, input_d_min, cols*rows * sizeof(float), cudaMemcpyDeviceToDevice));
	}
	else
	{
		bgrcuda::downsample<float> <<<grid, block >>> (input_d_max, l_d_max, rows, cols, scale);
		bgrcuda::downsample <float><<<grid, block >>> (input_d_min, l_d_min, rows, cols, scale);
		bgrcuda::downsample <thermalType> <<<grid, block >>> (input_t, l_therm, rows, cols, scale);
	}
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError());


		bgrcuda::mixturegaussians_onlyTherm<<<grid, block>>>(l_d_min, l_d_max, l_therm,
			tempmodel, tempvars, modelweights, l_o_min, l_o_max, rows/scale, cols/scale, lr, 1.0f - lr, -lr*ct, pars);
	cudaDeviceSynchronize();
    HANDLE_ERROR(cudaGetLastError());

	//upsample it
	if (scale == 1)
	{
		HANDLE_ERROR(cudaMemcpy(out_max, l_o_max, cols*rows * sizeof(float), cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaMemcpy(out_min, l_o_min, cols*rows * sizeof(float), cudaMemcpyDeviceToDevice));
	}
	else
	{
		grid = dim3((cols) / blockdim + 1, (rows ) / blockdim + 1);
		bgrcuda::upsample <<<grid, block >>> (l_o_max, out_max, rows, cols, scale);
		bgrcuda::upsample<<<grid, block >>> (l_o_min, out_min, rows, cols, scale);
	}
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError());

	if (l_o_max) cudaFree(l_o_max);
	if (l_d_max) cudaFree(l_d_max);
	if (l_d_min) cudaFree(l_d_min);
	if (l_o_min) cudaFree(l_o_min);
	if (l_therm) cudaFree(l_therm);

#endif

}

void DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly::setHistory(int hist_)
{
	history = hist_;
}
void DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly::setBackgroundWeight(float TB)
{
	pars.c_TB = TB;
}

void DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly::setVariance(float initTempVar, float initRangeVar)
{
	//parameters for thermal space
	pars.c_varInit_t = initTempVar; // initial variance for new components
	pars.c_varMax_t = 5.0f * pars.c_varInit_t;
	pars.c_varMin_t = pars.c_varInit_t/1.5f;

	//params for range space
	pars.c_varInit_r = initRangeVar; // initial variance for new components
	pars.c_varMax_r = 5.0f * pars.c_varInit_r;
	pars.c_varMin_r = pars.c_varInit_r/4;

	pars.c_r_varInflate = initRangeVar / 3;
	pars.c_t_varInflate = initTempVar/3;
}
void DidoFusedAnalytics_3dBgSub_CUDA_ThermalOnly::setThresholds(float backgroundThresh, float generativeThresh)
{
	pars.c_Tb = backgroundThresh;
	pars.c_Tg  = generativeThresh;

	pars.c_Tb_u = getUnivarateThresh(pars.c_Tb);
	pars.c_Tg_u = getUnivarateThresh(pars.c_Tg);
}


}