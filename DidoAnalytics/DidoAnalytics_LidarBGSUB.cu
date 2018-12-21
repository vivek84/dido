/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2017 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_LidarBGSUB.cu
* @author  	SL
* @version 	1
* @date    	2017-10-02
* @brief   	GPU based Background subtraction for the Lidar
*****************************************************************************
**/

/*
* Algorithm Description - This keeps a background model of the lidar points by first putting the data into a 2.5d represention clustering points in neary
* angular space into bins. Then each for each bin of the model that bin searches the neighbourhood of bins in the observation. If a point in the observation
* is within the fixed threshold, the model point is considered observed, else it is considered unobserved. Observed points have their weights increased,
* unobserved points have their weight decreased. Then for each point in the observation it searches the bins of the model in the neighbourhood. The highest weight
* of model points within the bin is mainatained. If this weight is above a threshold, the observation is considered to be a background point, else it is a foreground point.
* Then for each point that was observed that had no points in the model close to it, those points are added to the model. Finally the points in the model are sorted in
* weight order, and any model points with a weight below a threshold are discarded.
* The learning rate used for updating the weights follows the standard history pattern of 1./min(nframes, history).
*
* The clustering is done using DBSCAN. The points are again clustered into bins (this time broader), and each point searches it's neighbourhood of bins to decide if it is a core
* point. Then in the next step each point looks through it's neighbourhood and takes the lowest core parent index in that index, which is iteratively repeated several times.
* finally all points are allocated the roots of the resulting tree structure as a parent index, and then the clusters are formed into vectors on the CPU by a single insertion sort pass
*/

#include "global_defines.h"
#include "DidoAnalytics_LidarBGSUB.h"
#include "CUDA_Exception.h"
#include "math_constants.h"

#ifdef _WIN32
#include <ppl.h>
#include <concurrent_unordered_map.h>
#else
#include <unordered_map>
#endif


#define DEBUG_TIMINGS 0

#if DEBUG_TIMINGS
#include <chrono>
#include <iostream>
#endif


//error handling function
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		//		cudaDeviceReset();
		throw overview::CUDA_Exception(cudaGetErrorString(err), err, line, file);
	}
}
#define HANDLE_ERROR(err) {HandleError((err), __FILE__, __LINE__);} 

//data freeing simplification
inline void safe_Free(void * cudadata)
{
	if (cudadata != nullptr)HANDLE_ERROR(cudaFree(cudadata));
}

namespace overview
{
	namespace lbgsCUDA
	{
		//convenience function for swapping with
		__device__ void swap(float * array, int ind1, int ind2)
		{
			float tmp = array[ind1];
			array[ind1] = array[ind2];
			array[ind2] = tmp;
		}

#define TILT_DISTANCE 0.1f
		//as the lidar is in beams, it uses the actual distance in x y space, anda fixed multiplier on the distance in titlt space
		__device__ __forceinline__ float getDist(float p1, float t1, float r1, float p2, float t2, float r2)
		{
			__align__(8) float st1 = sinf(t1), st2 = sinf(t2);
			__align__(8) float tiltdist = TILT_DISTANCE * abs(t1 - t2);
			return r2 * r2*st2*st2 + r1 * r1*st1*st1 - 2 * r1*r2*st1*st2*cosf(p1 - p2) + tiltdist * tiltdist;
		}

		__global__ void collatePoints(LidarBin obsBox, DidoLidar_rangeData* obs, int npts, int nrows, int ncols, float binWidth, float binHeight)
		{
			//iterate over the input points
			const int index = threadIdx.x + blockIdx.x * blockDim.x;
			if (index < npts)
			{
				//work out which bin you should be in
				int x = (int)(obs[index].pan / binWidth) % ncols;
				int y = min(max((int)((obs[index].tilt - CUDART_PIO4_F) / binHeight), 0), nrows - 1);
				unsigned int binind = atomicAdd(&obsBox.npts[x + y * ncols], 1);
				if (binind < LIDARBGSUB_MAX_BIN_PTS)
				{
					unsigned int oind = binind + LIDARBGSUB_MAX_BIN_PTS * (x + y * ncols);
					obsBox.points_pan[oind] = obs[index].pan;
					obsBox.points_tilt[oind] = obs[index].tilt;
					obsBox.points_range[oind] = obs[index].range;
				}
			}
		}
		//we only search one above and below in tilt

		//updates the current model weights
		__global__ void bgsubKernel_pt1(LidarBin bgmodel, LidarBin obs, float * variances,
			float * weights, float threshold, float mindist_init, int nrows, int ncols, float lr, int searchWidth)
		{
			//one block per bin, using parallel threads for improved operation
			if (blockIdx.x >= ncols || blockIdx.y >= nrows)
				return;
			//correct the npoints for our inputs
			const int index = blockIdx.x + blockIdx.y*ncols;
			const int ind_y = blockIdx.y;

			//first proceed through the model and incriment or decrement depending if they are observed or not
			if (threadIdx.x < bgmodel.npts[index])
			{
				bool unobserved = true;
				bool unoccluded = true;
				bool lineobs = false; //checking if the packet is in the dataset (and no lost to occlusions/general IP stuff)
				float mdlpt_pan = bgmodel.points_pan[index*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];
				float mdlpt_tilt = bgmodel.points_tilt[index*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];
				float mdlpt_range = bgmodel.points_range[index*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];
				//tilt values are fixed so there's no value to vertical searching
				float mindist = variances[(index)*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];

				for (int i_x = -searchWidth; i_x <= searchWidth; i_x++)
				{
					//wrapping
					int ind_x = (i_x + blockIdx.x + ncols) % ncols;

					for (int j = 0; j < obs.npts[ind_x + ind_y * ncols] && j < LIDARBGSUB_MAX_BIN_PTS; j++)
					{
						float dist = getDist(obs.points_pan[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j],
							obs.points_tilt[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j],
							obs.points_range[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j], mdlpt_pan, mdlpt_tilt, mdlpt_range);
						if (dist < mindist * 3)
						{
							//		mindist = mindist + lr*(dist - mindist);
							unobserved = false;
						}
						else
							//is it in the line at all?
						{
							if (abs(obs.points_pan[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j] - mdlpt_pan) < 0.0002f)
							{
								lineobs = true;
								//is somethin closer and at the angle?
								if ((obs.points_range[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j] + 0.5f < mdlpt_range) &&
									(abs(obs.points_tilt[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j] - mdlpt_tilt) < 0.0001f))
								{
									unoccluded = false;
								}
							}
						}
					}

				}
				//update the point appropriately
				if (!unobserved || (unoccluded && lineobs))
				{
					weights[(index)*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x] = weights[(index)*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x] * (1.0f - lr) + (unobserved ? 0 : lr);
					//update my variance

					variances[(index)*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x] = min(max(mindist, mindist_init / 3), mindist_init * 3);

				}
			}
		}

		__global__ void bgsubKernel_pt2(LidarBin bgmodel, LidarBin obs, DidoLidar_rangeData* output, int * noutput, float * variances,
			float * weights, bool * addToModel, float threshold, int nrows, int ncols, int searchWidth)
		{
			//one block per bin, using parallel threads for improved operation
			if (blockIdx.x >= ncols || blockIdx.y >= nrows)
				return;
			//correct the npoints for our inputs
			const int index = blockIdx.x + blockIdx.y*ncols;
			const int ind_y = blockIdx.y;

			//then go through the observations and see if they are background and whether they are new
			//this does duplicate effort, but is needed to keep parallel determinism

			if (threadIdx.x < obs.npts[index])
			{
				float obsweight = -1.0f;
				bool newpoint = true;
				float obspt_pan = obs.points_pan[index*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];
				float obspt_tilt = obs.points_tilt[index*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];
				float obspt_range = obs.points_range[index*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];

				for (int i_x = -searchWidth; i_x <= searchWidth; i_x++)
				{
					//wrapping
					int ind_x = (i_x + blockIdx.x + ncols) % ncols;

					for (int j = 0; j < bgmodel.npts[ind_x + ind_y * ncols] && j < LIDARBGSUB_MAX_BIN_PTS; j++)
					{
						float dist = getDist(bgmodel.points_pan[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j],
							bgmodel.points_tilt[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j],
							bgmodel.points_range[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j],
							obspt_pan, obspt_tilt, obspt_range);
						float mindist = variances[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j];
						if (dist < 4 * mindist)
						{
							obsweight = max(obsweight, weights[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j]);
							if (dist < mindist * 3)
							{
								newpoint = false;
								//break;///its a sorted list (but this makes it slower due to awkwardness)
							}
						}
					}
				}
				//update the point appropriately

				//mark the point for output
				addToModel[index*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x] = newpoint;

				if (obsweight < threshold)
				{
					//put it in the output
					int outind = atomicAdd(noutput, 1);
					output[outind].range = obspt_range;
					output[outind].pan = obspt_pan;
					output[outind].tilt = obspt_tilt;
				}
			}

		}

		//sorts the models and culls the unobserved points and ones too close to each other
		__global__ void sortModels(LidarBin bgmodel, LidarBin obs, bool *addToModel, float * variances, float * weights, int nrows, int ncols, float lr, float mindist_init)
		{
			if (blockIdx.x >= ncols || blockIdx.y >= nrows) return;

			const int idx = (blockIdx.x + blockIdx.y*ncols);

			//produce an insertion vector
			__shared__ bool stillvalid[LIDARBGSUB_MAX_BIN_PTS];
			if (threadIdx.x < obs.npts[idx] && addToModel[idx*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x])
			{
				stillvalid[threadIdx.x] = true;
			}
			else stillvalid[threadIdx.x] = false;
			__syncthreads();
			//reduce it own by removing close vectors
			for (int i = 1; i < LIDARBGSUB_MAX_BIN_PTS / 2; i++)	//step size
			{
				int fidx = threadIdx.x + i * (threadIdx.x / i);
				int sidx = fidx + i;
				if (fidx < obs.npts[idx] && sidx < obs.npts[idx])
				{
					//compare and coalesc
					if (stillvalid[fidx] && stillvalid[sidx] && getDist(obs.points_pan[idx*LIDARBGSUB_MAX_BIN_PTS + fidx],
						obs.points_tilt[idx*LIDARBGSUB_MAX_BIN_PTS + fidx], obs.points_range[idx*LIDARBGSUB_MAX_BIN_PTS + fidx],
						obs.points_pan[idx*LIDARBGSUB_MAX_BIN_PTS + sidx], obs.points_tilt[idx*LIDARBGSUB_MAX_BIN_PTS + sidx], obs.points_range[idx*LIDARBGSUB_MAX_BIN_PTS + sidx]) < mindist_init)
					{
						stillvalid[sidx] = false;
					}
				}
				__syncthreads();
			}


			//then insert the remaining ones

			if (stillvalid[threadIdx.x])
			{
				//inputs  he new points here
				unsigned int npts = atomicAdd(&bgmodel.npts[idx], 1);
				if (npts < LIDARBGSUB_MAX_BIN_PTS)
				{
					int lidx = (idx)* LIDARBGSUB_MAX_BIN_PTS + npts;
					//add it to the model
					bgmodel.points_pan[lidx] = obs.points_pan[idx*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];
					bgmodel.points_tilt[lidx] = obs.points_tilt[idx*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];
					bgmodel.points_range[lidx] = obs.points_range[idx*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];
					weights[lidx] = lr;
					variances[lidx] = mindist_init;
				}
			}
			__syncthreads();
			//reset to max if we overflow
			if (threadIdx.x == 0) bgmodel.npts[idx] = min(bgmodel.npts[idx], (unsigned int)LIDARBGSUB_MAX_BIN_PTS);

			__syncthreads();

			// bubble sort the models (in parallel) so we can remove the worse
			__shared__ bool swapped;
			if (threadIdx.x == 0) swapped = true;
			while (swapped)
			{
				if (threadIdx.x == 0) swapped = false;
				if (threadIdx.x < bgmodel.npts[idx] / 2)
				{
					int sidx1 = idx * LIDARBGSUB_MAX_BIN_PTS + threadIdx.x * 2 + 1;
					int sidx2 = idx * LIDARBGSUB_MAX_BIN_PTS + threadIdx.x * 2;
					if (weights[sidx1] > weights[sidx2])
					{
						swap(weights, sidx1, sidx2);
						swap(variances, sidx1, sidx2);
						swap(bgmodel.points_pan, sidx1, sidx2);
						swap(bgmodel.points_tilt, sidx1, sidx2);
						swap(bgmodel.points_range, sidx1, sidx2);
						swapped = true;
					}

					sidx1 = idx * LIDARBGSUB_MAX_BIN_PTS + threadIdx.x * 2 + 2;
					sidx2 = idx * LIDARBGSUB_MAX_BIN_PTS + threadIdx.x * 2 + 1;
					if (threadIdx.x * 2 + 2 < bgmodel.npts[idx] && weights[sidx1] > weights[sidx2])
					{
						swap(weights, sidx1, sidx2);
						swap(variances, sidx1, sidx2);
						swap(bgmodel.points_pan, sidx1, sidx2);
						swap(bgmodel.points_tilt, sidx1, sidx2);
						swap(bgmodel.points_range, sidx1, sidx2);
						swapped = true;
					}
				}
				__syncthreads();
			}
			__syncthreads();
			//now remove any that have negative weights
			if (threadIdx.x < bgmodel.npts[idx])
			{
				if (weights[idx*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x] < lr / 4) atomicDec(&bgmodel.npts[idx], 0);
			}
		}

#define NO_POINT_PARENT -2
#define NON_CORE_PARENT -1

		__global__ void PDSCAN_init(const LidarBin fgpts, int* parents, bool *core, int nrows, int ncols, float mindist, int ncore, int searchwidth)
		{
			if (blockIdx.x >= ncols || blockIdx.y >= nrows) return;
			const int idx = (blockIdx.x + blockIdx.y*ncols);
			const int pind = idx * LIDARBGSUB_MAX_BIN_PTS + threadIdx.x;
			if (threadIdx.x < fgpts.npts[idx])
			{
				//search your neighbourhood to see how many neighbours you gave
				int nneighbours = 0;

				float obspt_pan = fgpts.points_pan[idx*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];
				float obspt_tilt = fgpts.points_tilt[idx*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];
				float obspt_range = fgpts.points_range[idx*LIDARBGSUB_MAX_BIN_PTS + threadIdx.x];

				for (int ind_y = (int)(blockIdx.y) - 2; ind_y <= blockIdx.y + 2; ind_y++)
				{
					if (ind_y >= 0 && ind_y < nrows)
					{
						for (int i_x = -searchwidth; i_x <= searchwidth; i_x++)
						{
							//wrapping
							int ind_x = (i_x + blockIdx.x + ncols) % ncols;

							for (int j = 0; j < fgpts.npts[ind_x + ind_y * ncols] && j < LIDARBGSUB_MAX_BIN_PTS; j++)
							{
								if (getDist(obspt_pan, obspt_tilt, obspt_range,
									fgpts.points_pan[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j],
									fgpts.points_tilt[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j],
									fgpts.points_range[(ind_x + ind_y * ncols)*LIDARBGSUB_MAX_BIN_PTS + j]) < mindist)
								{
									nneighbours++;
								}
							}
						}

					}
				}
				if (nneighbours >= ncore)
				{

					core[pind] = true;
					parents[pind] = pind;
				}
				else
				{
					core[pind] = false;
					parents[pind] = NON_CORE_PARENT;
				}
			}
			else if (threadIdx.x < LIDARBGSUB_MAX_BIN_PTS)
			{
				core[pind] = false;
				parents[pind] = NO_POINT_PARENT;
			}
		}

		__global__ void PDSCAN_local(const LidarBin fgpts, const int * parentsin, int * parentsout, const bool *core, int nrows, int ncols, float epsilon, int searchwidth)
		{
			if (blockIdx.x >= ncols || blockIdx.y >= nrows) return;
			const int idx = (blockIdx.x + blockIdx.y*ncols);
			const int pind = idx * LIDARBGSUB_MAX_BIN_PTS + threadIdx.x;
			if (threadIdx.x < fgpts.npts[idx])
			{
				//check every point in your region to populate your neigbourhood vector

				int my_parent = parentsin[pind];
				float obspt_pan = fgpts.points_pan[pind];
				float obspt_tilt = fgpts.points_tilt[pind];
				float obspt_range = fgpts.points_range[pind];

				for (int ind_y = (int)(blockIdx.y) - 2; ind_y <= blockIdx.y + 2; ind_y++)
				{
					if (ind_y >= 0 && ind_y < nrows)
					{
						for (int i_x = -searchwidth; i_x <= searchwidth; i_x++)
						{
							//wrapping
							int s_indx = ind_y * ncols + ((i_x + blockIdx.x + ncols) % ncols);

							for (int j = 0; j < fgpts.npts[s_indx] && j < LIDARBGSUB_MAX_BIN_PTS; j++)
							{
								if (core[s_indx*LIDARBGSUB_MAX_BIN_PTS + j] && getDist(obspt_pan, obspt_tilt, obspt_range,
									fgpts.points_pan[s_indx*LIDARBGSUB_MAX_BIN_PTS + j], fgpts.points_tilt[s_indx*LIDARBGSUB_MAX_BIN_PTS + j],
									fgpts.points_range[s_indx*LIDARBGSUB_MAX_BIN_PTS + j]) < epsilon)
								{
									//check parent
									my_parent = (core[(s_indx)*LIDARBGSUB_MAX_BIN_PTS + j] &&
										parentsin[(s_indx)*LIDARBGSUB_MAX_BIN_PTS + j] > my_parent) ?
										parentsin[(s_indx)*LIDARBGSUB_MAX_BIN_PTS + j] : my_parent;
								}
							}
						}
					}
				}
				parentsout[pind] = my_parent;
			}
		}
		//larger merges using atomics
		__global__ void PDSCAN_global(const LidarBin fgpts, const int * parentsin, int * parentsout, const bool *core, int rows, int cols, float epsilon, int searchwidth)
		{
			if (blockIdx.x >= cols || blockIdx.y >= rows) return;
			const int idx = (blockIdx.x + blockIdx.y*cols);
			const int pind = idx * LIDARBGSUB_MAX_BIN_PTS + threadIdx.x;
			if (threadIdx.x < fgpts.npts[idx])
			{
				//get your currecnt root
				int my_root = parentsin[pind];
				int it = 0;
				while (it < 10 && my_root >= 0 && my_root < rows*cols*LIDARBGSUB_MAX_BIN_PTS  && my_root != parentsin[my_root])
				{
					my_root = parentsin[my_root];
					it++;
				}

				float obspt_pan = fgpts.points_pan[pind];
				float obspt_tilt = fgpts.points_tilt[pind];
				float obspt_range = fgpts.points_range[pind];

				for (int ind_y = (int)(blockIdx.y) - 2; ind_y <= blockIdx.y + 2; ind_y++)
				{
					if (ind_y >= 0 && ind_y < rows)
					{
						for (int i_x = -searchwidth; i_x <= searchwidth; i_x++)
						{
							//wrapping
							int s_indx = ind_y * cols + ((i_x + blockIdx.x + cols) % cols);

							for (int j = 0; j < fgpts.npts[s_indx] && j < LIDARBGSUB_MAX_BIN_PTS; j++)
							{
								if (core[s_indx*LIDARBGSUB_MAX_BIN_PTS + j] && getDist(obspt_pan, obspt_tilt, obspt_range,
									fgpts.points_pan[s_indx*LIDARBGSUB_MAX_BIN_PTS + j], fgpts.points_tilt[s_indx*LIDARBGSUB_MAX_BIN_PTS + j],
									fgpts.points_range[s_indx*LIDARBGSUB_MAX_BIN_PTS + j]) < epsilon)
								{
									int otherroot = parentsin[s_indx*LIDARBGSUB_MAX_BIN_PTS + j];
									it = 0;
									while (it < 10 && otherroot >= 0 && otherroot < rows*cols && otherroot != parentsin[otherroot])
									{
										otherroot = parentsin[otherroot];
										it++;
									}

									if (otherroot > my_root)
									{
										atomicMax(&(parentsout[my_root < 0 ? pind : my_root]), otherroot);
										my_root = parentsout[my_root < 0 ? pind : my_root];
									}
								}
							}
						}
					}
				}
				atomicMax(&parentsout[pind], my_root);
			}

		}

		__global__ void setToRoot(const int * parentsin, int * parentsout, int npoints)
		{
			const int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index < npoints && parentsin[index] >= 0)
			{
				int head = parentsin[index];
				while (head >= 0 && head < npoints && head != parentsin[head])
					head = parentsin[head];
				parentsout[index] = head;
			}

		}


		std::vector<std::vector<DidoLidar_rangeData>> parseGraphsvec(LidarBin points, int * parents, int nbins)
		{
#ifdef _WIN32
			concurrency::concurrent_unordered_map<int, std::vector<DidoLidar_rangeData>> rootsinds;

			concurrency::parallel_for(0, nbins, [&](int i)
			{
				for (unsigned int j = 0; j < points.npts[i] && j < LIDARBGSUB_MAX_BIN_PTS; j++)
				{
					/*get it's root*/
					int root = parents[i*LIDARBGSUB_MAX_BIN_PTS + j];
					if (root < 0) continue;	//skip the special valued ones

					auto found = rootsinds.find(root);
					/*add it to that blob*/
					if (found != rootsinds.end())
					{
						found->second.push_back(DidoLidar_rangeData(points.points_range[i*LIDARBGSUB_MAX_BIN_PTS + j],
							points.points_pan[i*LIDARBGSUB_MAX_BIN_PTS + j], points.points_tilt[i*LIDARBGSUB_MAX_BIN_PTS + j]));
					}
					/*else create a new blob*/
					else
					{
						std::vector<DidoLidar_rangeData> temp;
						temp.push_back(DidoLidar_rangeData(points.points_range[i*LIDARBGSUB_MAX_BIN_PTS + j],
							points.points_pan[i*LIDARBGSUB_MAX_BIN_PTS + j], points.points_tilt[i*LIDARBGSUB_MAX_BIN_PTS + j]));
						rootsinds[root] = temp;
					}
				}

			}

			);
			//parse out the vectors
			std::vector<std::vector<DidoLidar_rangeData>> retval;
			for (auto & p : rootsinds)
			{
				retval.push_back(std::move(p.second));
			}
			return retval;

#else
			std::unordered_map<int, int> rootsinds;
			std::vector<std::vector<DidoLidar_rangeData>> retval;
			for (int i = 0; i < nbins; i++)
			{
				for (unsigned int j = 0; j < points.npts[i] && j < LIDARBGSUB_MAX_BIN_PTS; j++)
				{
					/*get it's root*/
					int root = parents[i*LIDARBGSUB_MAX_BIN_PTS + j];
					if (root < 0) continue;	//skip the special valued ones

					auto found = rootsinds.find(root);
					/*add it to that blob*/
					if (found != rootsinds.end())
					{
						retval[found->second].push_back(DidoLidar_rangeData(points.points_range[i*LIDARBGSUB_MAX_BIN_PTS + j],
							points.points_pan[i*LIDARBGSUB_MAX_BIN_PTS + j], points.points_tilt[i*LIDARBGSUB_MAX_BIN_PTS + j]));
					}
					/*else create a new blob*/
					else
					{
						rootsinds[root] = retval.size();
						std::vector<DidoLidar_rangeData> temp;
						temp.push_back(DidoLidar_rangeData(points.points_range[i*LIDARBGSUB_MAX_BIN_PTS + j],
							points.points_pan[i*LIDARBGSUB_MAX_BIN_PTS + j], points.points_tilt[i*LIDARBGSUB_MAX_BIN_PTS + j]));
						retval.push_back(temp);
					}
				}
			}
			return retval;
#endif
		}


	}

	DidoAnalytics_LidarBGSUB::DidoAnalytics_LidarBGSUB(float binw, float binh, float mindist, int history, float bgthreshold, float _eps, int _ncore) :
		binWidth(binw), binHeight(binh), sqmindist(mindist*mindist), hist(history), bgthresh(bgthreshold), epsilon(_eps), ncore(_ncore)
	{
		modelCols = abs((int)std::floor(2 * CUDART_PI_F / binw)) + 2;
		//no lidar in production I know of has a verticual FOV greater than 90' - maybe this should be a parameter?
		modelRows = abs((int)std::floor(CUDART_PIO2_F / binh)) + 2;
		searchWidth = (int)(mindist / binw) + 1;
		if (modelCols*modelRows < 1) throw std::runtime_error("the model size must be at least one in both dimensions");
#if DIDOLIDAR_NOGPU

#else
		bgmodel.allocate(modelCols*modelRows);
		HANDLE_ERROR(cudaMemset(bgmodel.npts, 0, modelCols*modelRows * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMalloc(&modelWeights, modelCols*modelRows * sizeof(float)*LIDARBGSUB_MAX_BIN_PTS));
		HANDLE_ERROR(cudaMemset(modelWeights, 0, modelCols*modelRows * sizeof(float)*LIDARBGSUB_MAX_BIN_PTS));
		HANDLE_ERROR(cudaMalloc(&variances, modelCols*modelRows * sizeof(float)*LIDARBGSUB_MAX_BIN_PTS));
		HANDLE_ERROR(cudaMemset(variances, 0, modelCols*modelRows * sizeof(float)*LIDARBGSUB_MAX_BIN_PTS));
#endif
	}


	DidoAnalytics_LidarBGSUB::~DidoAnalytics_LidarBGSUB()
	{
#if DIDOLIDAR_NOGPU

#else
		bgmodel.deallocate();
		safe_Free(modelWeights);
		safe_Free(variances);
#endif
	}



	std::vector<DidoLidar_rangeData> DidoAnalytics_LidarBGSUB::apply(const DidoLidar_rangeData* points, int npts, float learningRate)
	{
#if DIDOLIDAR_NOGPU
		std::vector<DidoLidar_rangeData> rval;
#else
		frameno++;
		float lr = (learningRate < 0) ? 1.0f / min(frameno, hist) : learningRate;

		if (lr != lr) throw std::runtime_error("learning rate was NaN");

#if DEBUG_TIMINGS
		auto prevtime = std::chrono::high_resolution_clock::now();
#endif

		//create GPU allocated data
		LidarBin d_obs;
		DidoLidar_rangeData * d_pts;
		int * d_nout;
		bool * d_addToModel;
		if (modelCols*modelRows < 1) throw std::runtime_error("the model size must be at least one in both dimensions");

		d_obs.allocate(modelCols*modelRows);
		HANDLE_ERROR(cudaMemset(d_obs.npts, 0, modelCols*modelRows * sizeof(unsigned int))); //make sure it's at zero
		HANDLE_ERROR(cudaMalloc(&d_pts, npts * sizeof(DidoLidar_rangeData)));
		HANDLE_ERROR(cudaMalloc(&d_nout, sizeof(int)));
		HANDLE_ERROR(cudaMalloc(&d_addToModel, modelCols*modelRows * sizeof(bool) * LIDARBGSUB_MAX_BIN_PTS));
		HANDLE_ERROR(cudaMemset(d_addToModel, 0, modelCols*modelRows * sizeof(bool) * LIDARBGSUB_MAX_BIN_PTS)); //make sure it's at zero
		HANDLE_ERROR(cudaMemcpy(d_pts, points, npts * sizeof(DidoLidar_rangeData), cudaMemcpyHostToDevice));

#if DEBUG_TIMINGS
		auto ts = std::chrono::high_resolution_clock::now();
		std::cout << "allocation took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		lbgsCUDA::collatePoints << <npts / 128 + 1, 128 >> >(d_obs, d_pts, npts, modelRows, modelCols, binWidth, binHeight);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());
		//run the bgsub
		dim3 grid(modelCols, modelRows);
		dim3 block(LIDARBGSUB_MAX_BIN_PTS, 1);


#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "colaltion took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		lbgsCUDA::bgsubKernel_pt1 << <grid, block >> >(bgmodel, d_obs, variances, modelWeights, bgthresh, sqmindist, modelRows, modelCols, lr, searchWidth);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());


#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "subtraction part 1 took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif

		lbgsCUDA::bgsubKernel_pt2 << <grid, block >> >(bgmodel, d_obs, d_pts, d_nout, variances, modelWeights, d_addToModel, bgthresh, modelRows, modelCols, searchWidth);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());


#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "subtraction pt2 took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		//parse the data to the output
		lbgsCUDA::sortModels << <grid, block >> >(bgmodel, d_obs, d_addToModel, variances, modelWeights, modelRows, modelCols, lr, sqmindist*0.75f);
		int * nout = (int*)malloc(sizeof(int));
		HANDLE_ERROR(cudaMemcpy(nout, d_nout, sizeof(int), cudaMemcpyDeviceToHost));
		std::vector<DidoLidar_rangeData> rval(nout[0]);
		HANDLE_ERROR(cudaMemcpy(rval.data(), d_pts, nout[0] * sizeof(DidoLidar_rangeData), cudaMemcpyDeviceToHost));
		if (nout) free(nout);
		cudaDeviceSynchronize();

#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "sorting took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		HANDLE_ERROR(cudaGetLastError());
		safe_Free(d_addToModel);
		d_obs.deallocate();
		safe_Free(d_pts);
		safe_Free(d_nout);
#endif
		return rval;
	}



	std::vector<std::vector<DidoLidar_rangeData>> DidoAnalytics_LidarBGSUB::applyAndCluster(const DidoLidar_rangeData * points, int npts, float learningRate)
	{
#if DIDOLIDAR_NOGPU
		std::vector<std::vector<DidoLidar_rangeData>> rval;
#else
		frameno++;
		float lr = (learningRate < 0) ? 1.0f / min(frameno, hist) : learningRate;

		if (lr != lr) throw std::runtime_error("learning rate was NaN");


#if DEBUG_TIMINGS
		auto prevtime = std::chrono::high_resolution_clock::now();
#endif

		//create GPU allocated data
		LidarBin d_obs;
		DidoLidar_rangeData * d_pts;
		int * d_nout;
		bool * d_addToModel;

		//clustering data
		LidarBin d_cluster_obs;
		int * d_cluster_parents_1, *d_cluster_parents_2;
		bool * d_cluster_core;

		float clusterWidth = min(binWidth * 4, epsilon / 3);
		float clusterHeight = min(binHeight * 4, epsilon / 3);
		int clusterCols = (int)std::floor(2 * CUDART_PI_F / clusterWidth) + 2;
		int clusterRows = (int)std::floor(binHeight*modelRows / clusterHeight) + 1;
		int clusterSearch = (int)(epsilon / clusterWidth) + 1;

		if (clusterCols*clusterRows < 1) throw std::runtime_error("the cluster size must be at least one in both dimensions");
		if (modelCols*modelRows < 1) throw std::runtime_error("the model size must be at least one in both dimensions");

		d_cluster_obs.allocate(clusterCols*clusterRows);
		HANDLE_ERROR(cudaMemset(d_cluster_obs.npts, 0, clusterCols*clusterRows * sizeof(unsigned int))); //make sure it's at zero
		HANDLE_ERROR(cudaMalloc(&d_cluster_core, clusterCols*clusterRows * sizeof(bool) * LIDARBGSUB_MAX_BIN_PTS));
		HANDLE_ERROR(cudaMalloc(&d_cluster_parents_1, clusterCols*clusterRows * sizeof(int) * LIDARBGSUB_MAX_BIN_PTS));
		HANDLE_ERROR(cudaMalloc(&d_cluster_parents_2, clusterCols*clusterRows * sizeof(int) * LIDARBGSUB_MAX_BIN_PTS));

		d_obs.allocate(modelRows*modelCols);
		HANDLE_ERROR(cudaMemset(d_obs.npts, 0, modelCols*modelRows * sizeof(unsigned int))); //make sure it's at zero
		HANDLE_ERROR(cudaMalloc(&d_pts, npts * sizeof(DidoLidar_rangeData)));
		HANDLE_ERROR(cudaMalloc(&d_nout, sizeof(int)));
		HANDLE_ERROR(cudaMalloc(&d_addToModel, modelCols*modelRows * sizeof(bool) * LIDARBGSUB_MAX_BIN_PTS));
		HANDLE_ERROR(cudaMemset(d_addToModel, 0, modelCols*modelRows * sizeof(bool) * LIDARBGSUB_MAX_BIN_PTS)); //make sure it's at zero
		HANDLE_ERROR(cudaMemcpy(d_pts, points, npts * sizeof(DidoLidar_rangeData), cudaMemcpyHostToDevice));


#if DEBUG_TIMINGS
		auto ts = std::chrono::high_resolution_clock::now();
		std::cout << "allocation took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		lbgsCUDA::collatePoints << <npts / 128 + 1, 128 >> >(d_obs, d_pts, npts, modelRows, modelCols, binWidth, binHeight);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());
		//run the bgsub
		dim3 grid(modelCols, modelRows);
		dim3 block(LIDARBGSUB_MAX_BIN_PTS, 1);


#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "colaltion took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		lbgsCUDA::bgsubKernel_pt1 << <grid, block >> >(bgmodel, d_obs, variances, modelWeights, bgthresh, sqmindist, modelRows, modelCols, lr, searchWidth);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "subtraction part 1 took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		lbgsCUDA::bgsubKernel_pt2 << <grid, block >> >(bgmodel, d_obs, d_pts, d_nout, variances, modelWeights, d_addToModel, bgthresh, modelRows, modelCols, searchWidth);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());


#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "bgsub pt 2 took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		//parse the data to the output
		lbgsCUDA::sortModels << <grid, block >> >(bgmodel, d_obs, d_addToModel, variances, modelWeights, modelRows, modelCols, lr, sqmindist*0.75f);

		int * nout = (int*)malloc(sizeof(int));
		HANDLE_ERROR(cudaMemcpy(nout, d_nout, sizeof(int), cudaMemcpyDeviceToHost));

		//collate the foreground points for clustering
		lbgsCUDA::collatePoints << <nout[0] / 128 + 1, 128 >> >(d_cluster_obs, d_pts, nout[0], clusterRows, clusterCols, clusterWidth, clusterHeight);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());


#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "sorting and clustering collation took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif

		dim3 clustergrid(clusterCols, clusterRows);
		//initialise the parents
		lbgsCUDA::PDSCAN_init << <clustergrid, block >> >(d_cluster_obs, d_cluster_parents_1, d_cluster_core, clusterRows, clusterCols, epsilon, ncore, clusterSearch);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "pdbscan init took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif
		//quickly do local updates
		lbgsCUDA::PDSCAN_local << <clustergrid, block >> >(d_cluster_obs, d_cluster_parents_1, d_cluster_parents_2, d_cluster_core, clusterRows, clusterCols, epsilon, clusterSearch);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "pdbscan local took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif

		//now update the roots
		lbgsCUDA::PDSCAN_global << <clustergrid, block >> >(d_cluster_obs, d_cluster_parents_2, d_cluster_parents_1, d_cluster_core, clusterRows, clusterCols, epsilon, clusterSearch);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());
		//twice so that we get everything
		lbgsCUDA::PDSCAN_global << <clustergrid, block >> >(d_cluster_obs, d_cluster_parents_1, d_cluster_parents_2, d_cluster_core, clusterRows, clusterCols, epsilon, clusterSearch);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());


#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "pdbscan global took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		lbgsCUDA::setToRoot << <clusterRows*clusterCols, LIDARBGSUB_MAX_BIN_PTS >> >(d_cluster_parents_2, d_cluster_parents_1, clusterRows*clusterCols*LIDARBGSUB_MAX_BIN_PTS);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());


#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "setting to root took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif

		//now drop it onto the cpu and compute the output
		LidarBin lbins;
		lbins.local_allocate(clusterRows*clusterCols);
		std::vector<int> lparents(clusterRows*clusterCols*LIDARBGSUB_MAX_BIN_PTS);

		HANDLE_ERROR(cudaMemcpy(lparents.data(), d_cluster_parents_1, clusterRows*clusterCols*LIDARBGSUB_MAX_BIN_PTS * sizeof(int), cudaMemcpyDeviceToHost));
		lbins.copyDownFrom(d_cluster_obs, clusterRows*clusterCols);

		auto rval = lbgsCUDA::parseGraphsvec(lbins, lparents.data(), clusterRows*clusterCols);
		lbins.local_deallocate();
		if (nout) free(nout);
		cudaDeviceSynchronize();


#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "generating output took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		HANDLE_ERROR(cudaGetLastError());
		safe_Free(d_cluster_parents_1);
		safe_Free(d_cluster_parents_2);
		d_cluster_obs.deallocate();
		safe_Free(d_cluster_core);
		safe_Free(d_addToModel);
		d_obs.deallocate();
		safe_Free(d_pts);
		safe_Free(d_nout);
#endif
		return rval;
	}

	std::vector<std::vector<DidoLidar_rangeData>> DidoAnalytics_LidarBGSUB::Cluster(const DidoLidar_rangeData * fgpts, size_t npts)
	{
#if DIDOLIDAR_NOGPU
		std::vector<std::vector<DidoLidar_rangeData>> rval;
#else
		//create GPU allocated data
		DidoLidar_rangeData * d_pts;

		//clustering data
		LidarBin d_cluster_obs;
		int * d_cluster_parents_1, *d_cluster_parents_2;
		bool * d_cluster_core;

		float clusterWidth = min(binWidth * 4, epsilon / 3);
		float clusterHeight = min(binHeight * 4, epsilon / 3);
		int clusterCols = (int)std::floor(2 * CUDART_PI_F / clusterWidth) + 2;
		int clusterRows = (int)std::floor(binHeight*modelRows / clusterHeight) + 1;
		int clusterSearch = (int)(epsilon / clusterWidth) + 1;

		if (clusterCols*clusterRows < 1) throw std::runtime_error("the cluster size must be at least one in both dimensions");


#if DEBUG_TIMINGS
		auto prevtime = std::chrono::high_resolution_clock::now();
#endif

		d_cluster_obs.allocate(clusterCols*clusterRows * sizeof(unsigned int));
		HANDLE_ERROR(cudaMemset(d_cluster_obs.npts, 0, clusterCols*clusterRows * sizeof(unsigned int))); //make sure it's at zero
		HANDLE_ERROR(cudaMalloc(&d_cluster_core, clusterCols*clusterRows * sizeof(bool) * LIDARBGSUB_MAX_BIN_PTS));
		HANDLE_ERROR(cudaMalloc(&d_cluster_parents_1, clusterCols*clusterRows * sizeof(int) * LIDARBGSUB_MAX_BIN_PTS));
		HANDLE_ERROR(cudaMalloc(&d_cluster_parents_2, clusterCols*clusterRows * sizeof(int) * LIDARBGSUB_MAX_BIN_PTS));

		HANDLE_ERROR(cudaMalloc(&d_pts, npts * sizeof(DidoLidar_rangeData)));
		HANDLE_ERROR(cudaMemcpy(d_pts, fgpts, npts * sizeof(DidoLidar_rangeData), cudaMemcpyHostToDevice));


#if DEBUG_TIMINGS
		auto ts = std::chrono::high_resolution_clock::now();
		std::cout << "cluster allocation took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		//collate the foreground points for clustering
		lbgsCUDA::collatePoints << <(npts / 128) + 1, 128 >> >(d_cluster_obs, d_pts, npts, clusterRows, clusterCols, clusterWidth, clusterHeight);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "cluster collation took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		dim3 block(LIDARBGSUB_MAX_BIN_PTS, 1);
		dim3 clustergrid(clusterCols, clusterRows);
		//initialise the parents
		lbgsCUDA::PDSCAN_init << <clustergrid, block >> >(d_cluster_obs, d_cluster_parents_1, d_cluster_core, clusterRows, clusterCols, epsilon, ncore, clusterSearch);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "pdbscan inti took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif


		//quickly do local updates
		lbgsCUDA::PDSCAN_local << <clustergrid, block >> >(d_cluster_obs, d_cluster_parents_1, d_cluster_parents_2, d_cluster_core, clusterRows, clusterCols, epsilon, clusterSearch);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "pdbscan local took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif
		//now update the roots
		lbgsCUDA::PDSCAN_global << <clustergrid, block >> >(d_cluster_obs, d_cluster_parents_2, d_cluster_parents_1, d_cluster_core, clusterRows, clusterCols, epsilon, clusterSearch);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());
		//twice so that we get everything
		lbgsCUDA::PDSCAN_global << <clustergrid, block >> >(d_cluster_obs, d_cluster_parents_1, d_cluster_parents_2, d_cluster_core, clusterRows, clusterCols, epsilon, clusterSearch);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "pdbscan global took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif

		lbgsCUDA::setToRoot << <clusterRows*clusterCols, LIDARBGSUB_MAX_BIN_PTS >> >(d_cluster_parents_2, d_cluster_parents_1, clusterRows*clusterCols*LIDARBGSUB_MAX_BIN_PTS);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());


#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "set to root took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif

		//now drop it onto the cpu and compute the output
		LidarBin lbins;
		lbins.local_allocate(clusterRows*clusterCols);
		std::vector<int> lparents(clusterRows*clusterCols*LIDARBGSUB_MAX_BIN_PTS);

		HANDLE_ERROR(cudaMemcpy(lparents.data(), d_cluster_parents_1, clusterRows*clusterCols*LIDARBGSUB_MAX_BIN_PTS * sizeof(int), cudaMemcpyDeviceToHost));
		lbins.copyDownFrom(d_cluster_obs, clusterRows*clusterCols);

		auto rval = lbgsCUDA::parseGraphsvec(lbins, lparents.data(), clusterRows*clusterCols);
		cudaDeviceSynchronize();


#if DEBUG_TIMINGS
		ts = std::chrono::high_resolution_clock::now();
		std::cout << "generating output took " << std::chrono::duration_cast<std::chrono::milliseconds>(ts - prevtime).count() << "ms" << std::endl;
		prevtime = ts;
#endif

		lbins.local_deallocate();
		HANDLE_ERROR(cudaGetLastError());
		safe_Free(d_cluster_parents_1);
		safe_Free(d_cluster_parents_2);
		d_cluster_obs.deallocate();
		safe_Free(d_cluster_core);
		safe_Free(d_pts);
#endif
		return rval;
	}




	__global__ void countPoints(LidarBin bins, unsigned char* out, int npts)
	{
		const int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < npts)
		{
			out[index] = (unsigned char)bins.npts[index];
		}
	}


	void DidoAnalytics_LidarBGSUB::dispayBgmodelNpts(unsigned char * out, int npts)
	{
		unsigned char * d_counts;
		HANDLE_ERROR(cudaMalloc(&d_counts, modelRows*modelCols * sizeof(unsigned char)));
		countPoints << <modelRows*modelCols / 128 + 1, 128 >> > (bgmodel, d_counts, modelRows*modelCols);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());
		HANDLE_ERROR(cudaMemcpy(out, d_counts, min(modelRows*modelCols, npts) * sizeof(unsigned char), cudaMemcpyDeviceToHost));
		safe_Free(d_counts);
	}

	void LidarBin::allocate(size_t nbins)
	{
		HANDLE_ERROR(cudaMalloc(&npts, nbins * sizeof(unsigned int)));
		HANDLE_ERROR(cudaMalloc(&points_pan, nbins*LIDARBGSUB_MAX_BIN_PTS * sizeof(float)));
		HANDLE_ERROR(cudaMalloc(&points_tilt, nbins*LIDARBGSUB_MAX_BIN_PTS * sizeof(float)));
		HANDLE_ERROR(cudaMalloc(&points_range, nbins*LIDARBGSUB_MAX_BIN_PTS * sizeof(float)));
	}
	void LidarBin::local_allocate(size_t nbins)
	{
		npts = (unsigned int *)malloc(nbins * sizeof(unsigned int));
		points_pan = (float*)malloc(nbins*LIDARBGSUB_MAX_BIN_PTS * sizeof(float));
		points_tilt = (float*)malloc(nbins*LIDARBGSUB_MAX_BIN_PTS * sizeof(float));
		points_range = (float*)malloc(nbins*LIDARBGSUB_MAX_BIN_PTS * sizeof(float));
	}
	void LidarBin::deallocate()
	{
		safe_Free(npts);
		safe_Free(points_pan);
		safe_Free(points_tilt);
		safe_Free(points_range);
	}
	void LidarBin::local_deallocate()
	{
		if (npts) free(npts);
		if (points_pan) free(points_pan);
		if (points_pan) free(points_tilt);
		if (points_pan) free(points_range);

	}
	void LidarBin::copyDownFrom(LidarBin & src, size_t nbins)
	{
		HANDLE_ERROR(cudaMemcpy(npts, src.npts, nbins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(points_pan, src.points_pan, nbins*LIDARBGSUB_MAX_BIN_PTS * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(points_tilt, src.points_tilt, nbins*LIDARBGSUB_MAX_BIN_PTS * sizeof(float), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(points_range, src.points_range, nbins*LIDARBGSUB_MAX_BIN_PTS * sizeof(float), cudaMemcpyDeviceToHost));
	}
}