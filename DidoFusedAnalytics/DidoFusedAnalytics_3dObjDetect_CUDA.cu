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



/**
*	description of the algorithm
*	collects the objects using PDSScan. This method searches the neighbourhood around each point and labels the point as core if it has above a threshold of neighbours
*	then points that are neighbours with core points are joined to the same cluster using a tree structure. This is done in parallel by repeated steps of searching the neighbourhood
*	and taking the lowest parent from amongst the parents of the core points in your neighbourhood until all the points in the block have collected their heads.
*	then each point is labelled with the root of it's tree. These points are then sorted by their parent index using merge sort. Finally this sorted list of points is formed into detections
*	in a two stage process, where first blocks of points are combined into sets of bounding boxes before this list of bounding boxes is then further combined on the CPU in the final pass to give the output

*/

#include "global_defines.h"
#include <vector>
#include "DidoFusedAnalytics_3dObjDetect_CUDA.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include "math.h"
#include "math_constants.h"
#include "cuda_profiler_api.h"
#include <chrono>


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

namespace objdetectCUDA
{
#if DIDOLIDAR_NOGPU

#else

//datatype for handling our things
	//prefer box of pointers to pointer to boxes in cuda
struct node
{
	bool * valid;
	bool * core;
	int * parentind;
	int * index;
};

//populates a box of nodes stored row major
__global__ void generateTrees(const float * fg_ranges_min, const float * fg_ranges_max, node outnodes, int rows, int cols)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x < cols && y < rows)
	{
		//work out if you are a node
		bool cond = (fg_ranges_max[x + cols*y] > 0 && fg_ranges_min[x + cols*y] > 0);
		outnodes.valid[x + cols*y] = cond;
		outnodes.parentind[x + cols*y] = cond ? x + cols*y : -1;
		outnodes.index[x + cols*y] = x + cols*y;
	}
}

//function to calculate the separation between ranges of ranges
__device__  inline float rangeDist(const float & a_min, const float & a_max, const float & b_min, const float & b_max)
{
	//look at the separation of thje averages

	//check if any of the points are at infinity (meaninbg there was no range observation)
	if(isfinite(a_min) && isfinite(a_max) && isfinite(b_min) && isfinite(b_max))
		return (abs((a_min + a_max) / 2 - (b_min + b_max) / 2) );
	else return 0;
}


//parallel DBSCAN using propagation
//this version wworks on a shred copy, but doens't seem to get every point (maybe we have to call it twice)
template <int eps, int bDim>
__global__ void PDSDBSCANInit__shared(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int l_len = (bDim + 2 * eps);
	const int l_size = l_len*l_len;
	__shared__ __align__(8) int l_parents[l_size];
	__shared__ __align__(8) float l_ranges_min[l_size];
	__shared__ __align__(8) float l_ranges_max[l_size];
	__shared__ __align__(4) bool l_valid[l_size];
	__shared__ __align__(4) bool l_core[l_size];

	//populate the local storage
	int flatind = threadIdx.y + threadIdx.x*blockDim.y;
	if (flatind < l_size / 2)
	{
		int xind = (cols + blockIdx.x * blockDim.x - eps + (flatind % l_len))%cols;
		int yind = blockIdx.y * blockDim.y - eps + (flatind / l_len);
		bool cond = yind >= 0 && yind < rows;
		l_ranges_min[flatind] = cond ? fg_ranges_min[yind*cols + xind] : -1.0f;
		l_ranges_max[flatind] = cond ? fg_ranges_max[yind*cols + xind] : -1.0f;
		l_valid[flatind] = cond ? nodes.valid[yind*cols + xind] : false;
		l_parents[flatind] = cond ? nodes.parentind[yind*cols + xind] : -1;
		l_core[flatind] = cond ? nodes.core[yind*cols + xind] : false;
		xind = (blockIdx.x * blockDim.x - eps + ((flatind + (l_size / 2)) % l_len));
		yind = blockIdx.y * blockDim.y - eps + ((flatind + (l_size / 2) )/ l_len);
		cond = yind >= 0 && yind < rows && xind >= 0 && xind < cols;
		l_ranges_min[flatind + (l_size / 2)] = cond ? fg_ranges_min[yind*cols + xind] : -1.0f;
		l_ranges_max[flatind + (l_size / 2)] = cond ? fg_ranges_max[yind*cols + xind] : -1.0f;
		l_valid[flatind + (l_size / 2)] = cond ? nodes.valid[yind*cols + xind] : false;
		l_parents[flatind + (l_size / 2)] = cond ? nodes.parentind[yind*cols + xind] : -1;
		l_core[flatind + (l_size / 2)] = cond ? nodes.core[yind*cols + xind] : false;
	}

	int l_ind = l_len*(threadIdx.y + eps) + (threadIdx.x + eps);
	if (x >= cols || y >= rows) return;
	if (l_valid[l_ind])
	{
		//check every point in your region to decide if you are a core point and populate your neigbourhood vector
		int nneighbours = 0;
		float myrange_min = l_ranges_min[l_ind];
		float myrange_max = l_ranges_max[l_ind];

		for (int i = -eps; i <= eps; i++)
		{
			int xind = eps + threadIdx.x + i;
			//manhattan distances are cheaper to compute
			for (int j = abs(i) - eps; j <= eps - abs(i); j++)
			{
				int t_lind = xind + l_len*(eps + threadIdx.y + j);
				//check range
				float sep = rangescaling*rangeDist(l_ranges_min[t_lind], l_ranges_max[t_lind],  myrange_min, myrange_max) + abs(i) + abs(j);
				nneighbours += l_valid[t_lind] && (sep <= eps);
			}
		}
		l_core[l_ind] = (nneighbours >= ncore);
		l_parents[l_ind] = l_core[l_ind] ? l_parents[l_ind] : -1;
		__syncthreads();

		__align__(4) int my_parent = l_parents[l_ind];
		for (int s = 0; s < bDim / eps; s++)
		{
			for (int i = -eps; i <= eps; i++)
			{
				int xind = eps + threadIdx.x + i;
				for (int j = abs(i) - eps; j <= eps - abs(i); j++)
				{
					int t_lind = xind + l_len*(eps + threadIdx.y + j);
					float sep = rangescaling*rangeDist(l_ranges_min[t_lind], l_ranges_max[t_lind], myrange_min, myrange_max) + abs(i) + abs(j);
					if ((sep <= eps) && l_valid[t_lind])
						my_parent = (l_core[t_lind] && (l_parents[t_lind] > my_parent)) ? l_parents[t_lind] : my_parent;
				}
			}
			l_parents[l_ind] = my_parent;
			__syncthreads();
		}
	}
	//drop values back to global
	nodes.core[x + y*cols] = l_core[l_ind];
	nodes.parentind[x + y*cols] = l_parents[l_ind];
} 

template <int eps>
__global__ void PDSDBSCANLocal(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore)
{
	const int maxnbour = eps * 2 * (eps + 1) + 1;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (!(x < cols && y < rows && nodes.valid[x + y*cols])) return;
	//check every point in your region to populate your neigbourhood vector
	__align__(4) bool nbhood[maxnbour];
	int nneighbours = 0;
	float myrange_min = fg_ranges_min[x + y*cols];
	float myrange_max = fg_ranges_max[x + y*cols];

	int my_parent = nodes.core[x + cols*y] ? nodes.parentind[x + cols*y] : -1;
	for (int i = -eps; i <= eps && nneighbours < maxnbour; i++)
	{
		int xind = (cols + x + i) % cols;
		//manhattan distances are cheaper to compute
		for (int j = abs(i) - eps; j <= eps - abs(i) && nneighbours < maxnbour; j++)
		{
			int yind = y + j;
			//bounds check
			if (yind >= 0 && yind < rows && nodes.valid[xind + cols*yind])
			{
				//check range
				float range = rangescaling*rangeDist(fg_ranges_min[xind + cols*yind], fg_ranges_max[xind + cols*yind], myrange_min, myrange_max) + abs(i) + abs(j);
				nbhood[nneighbours] = range <= eps;
				//check parent
				my_parent = (nbhood[nneighbours] && nodes.core[xind + cols*yind] &&
					(nodes.parentind[xind + cols*yind] > my_parent)) ? 
					  nodes.parentind[xind + cols*yind] : my_parent;
			}
			else
			{
				nbhood[nneighbours] = false;
			}
			nneighbours++;
		}
	}
	nodes.parentind[x + cols*y] = my_parent;
	__syncthreads();

	//get your currecnt root
	int my_root = my_parent;
	int it = 0;
	while ( it < 10 && my_root >=0 && my_root < rows*cols  && my_root != nodes.parentind[my_root])
	{
			my_root = nodes.parentind[my_root];
			it++;
	}

	nneighbours = 0;
	for (int i = -eps; i <= eps && nneighbours < maxnbour; i++)
	{
		int xind = (cols + x + i) % cols;
		//manhattan distances are cheaper to compute
		for (int j = abs(i) - eps; j <= eps - abs(i) && nneighbours < maxnbour; j++)
		{
			int yind = y + j;
			if (nbhood[nneighbours] && nodes.core[xind + cols*yind] && nodes.parentind[xind + cols*yind] != my_parent)
			{
				int otherroot = nodes.parentind[xind + cols*yind];
				it = 0;
				while (it < 10 && otherroot >= 0 && otherroot < rows*cols && otherroot != nodes.parentind[otherroot])
				{
					otherroot = nodes.parentind[otherroot];
					it++;
				}

				if (otherroot > my_root)
				{
					atomicMax(&(nodes.parentind[my_root < 0 ? x + y*cols : my_root]), otherroot);
					my_root = nodes.parentind[my_root < 0 ? x + y*cols : my_root];
				}
			}
			nneighbours++;
		}
	}
}

template <int eps>
__global__ void PDSDBSCANInit(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore)
{
	static const int maxnbour = eps * 2 * (eps + 1) + 1;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (!(x < cols && y < rows && nodes.valid[x + y*cols])) return;
	//check every point in your region to decide if you are a core point and populate your neigbourhood vector
	int nneighbours = 0;
	float myrange_min = fg_ranges_min[x + y*cols];
	float myrange_max = fg_ranges_max[x + y*cols];

	for (int i = -eps; i <= eps && nneighbours < maxnbour; i++)
	{
		int xind = (x + i) ;
		//manhattan distances are cheaper to compute
		for (int j = abs(i) - eps; j <= eps - abs(i) && nneighbours < maxnbour; j++)
		{
			int yind = y + j;
			//bounds check
			if (yind >= 0 && yind < rows && xind >= 0 && xind < cols && nodes.valid[xind + cols*yind])
			{
				//check range
				float range = rangescaling*rangeDist(fg_ranges_min[xind + cols*yind], fg_ranges_max[xind + cols*yind], myrange_min, myrange_max) + abs(i) + abs(j);
				if (range <= eps)
				{
					nneighbours++;
				}
			}
		}
	}
	nodes.core[x + y*cols] = (nneighbours >= ncore);
}

//template initialisation
template __global__ void PDSDBSCANLocal<5>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);

template __global__ void PDSDBSCANInit__shared<5, 32>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);
template __global__ void PDSDBSCANInit<5>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);

template __global__ void PDSDBSCANLocal<4>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);

template __global__ void PDSDBSCANInit__shared<4, 32>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);
template __global__ void PDSDBSCANInit<4>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);

template __global__ void PDSDBSCANLocal<3>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);

template __global__ void PDSDBSCANInit__shared<3, 32>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);
template __global__ void PDSDBSCANInit<3>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);

template __global__ void PDSDBSCANLocal<2>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);

template __global__ void PDSDBSCANInit__shared<2, 32>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);
template __global__ void PDSDBSCANInit<2>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);

template __global__ void PDSDBSCANLocal<1>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);

template __global__ void PDSDBSCANInit__shared<1, 32>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);
template __global__ void PDSDBSCANInit<1>(const float * fg_ranges_min, const float * fg_ranges_max, node nodes, int rows, int cols, float rangescaling, int ncore);

__global__ void g_setToHead(const node nodes, int * temphead, int rowxcols, int nnodes)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < nnodes)
	{
		int head = (index < rowxcols) ? nodes.parentind[index] : -1;
		while (head >= 0 && head < nnodes && head != nodes.parentind[head])
			head = nodes.parentind[head];
		temphead[index] = head;
	}
}

__global__ void  applysort(node arr, int* ptrs, int * parents, int nnodes)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < nnodes)
	{
		arr.parentind[index] = parents[index];
		arr.index[index] = ptrs[index];
	}
}


const int nOEMthreads = 1024;
const int nOEMshare = nOEMthreads * 2;

__global__ void OEMSort_kernel(int* arr, int * g_parents, int nnodes)
{
	//we iterate up until the len is our kernelsize, so we can use local (faster) synchronises
	const int idx = blockIdx.x * nOEMshare + threadIdx.x;
	if (threadIdx.x < nOEMthreads && idx < nnodes)
	{
		//cache into shared memory
		__shared__ __align__(8) int cache[nOEMshare];
		__shared__ __align__(8) int parents[nOEMshare];

		parents[threadIdx.x] = g_parents[idx];
		cache[threadIdx.x] = idx;
		parents[threadIdx.x + nOEMthreads] = idx + nOEMthreads < nnodes ? g_parents[idx + nOEMthreads] : -1 ;
		cache[threadIdx.x + nOEMthreads] = idx + nOEMthreads;
		__syncthreads();
		for (int len = 2; len <= blockDim.x; len *= 2)
		{
			int section_idx = threadIdx.x / (len / 2); //there are len/2 comparisons in each block
			int step_idx = threadIdx.x % (len / 2);
			int i = step_idx + section_idx*len;
			int stepped = i + (len / 2);
			if (stepped < nOEMshare)
			{
				//compareAndExchangeNodes(cache + i, cache + i + (len / 2));
				bool comp = parents[i] < parents[stepped];
				int tmp = cache[comp ? stepped : i];
				int p = parents[comp ? stepped : i];
				parents[stepped] = parents[comp ? i : stepped];
				parents[i] = p;
				cache[stepped] = cache[comp ? i : stepped];
				cache[i] = tmp;
			}
			__syncthreads();
			for (int step = len / 4; step > 0; step /= 2)
			{
				int start_idx = step_idx % step;
				int it_idx = step_idx / step;

				i = section_idx*len + start_idx + step + it_idx * 2 * step;
				stepped = i + step;
				if ((step * 2 + it_idx * 2 * step < len) && (stepped < nOEMshare))
				{
					bool comp = parents[i] < parents[stepped];
					int tmp = cache[comp ? stepped : i];
					int p = parents[comp ? stepped : i];
					parents[stepped] = parents[comp ? i : stepped];
					parents[i] = p;
					cache[stepped] = cache[comp ? i : stepped];
					cache[i] = tmp;
				}
				__syncthreads();
			}
		}
		//apply the cache back
		arr[idx] = cache[threadIdx.x];	//this actually is where this value will first be initialised
		g_parents[idx] = parents[threadIdx.x];
		if (idx + nOEMthreads < nnodes)
		{
		arr[idx + nOEMthreads] = cache[threadIdx.x + nOEMthreads];
		g_parents[idx + nOEMthreads] = parents[threadIdx.x + nOEMthreads];
		}
	}

}

__global__ void OddEvenMergeSort_a(int* cache, int* parents, int len, int nnodes)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int section_idx = idx / (len / 2); //there are len/2 comparisons in each block
	int step_idx = idx % (len / 2);
	int i = step_idx + section_idx*len;
	int stepped = i + (len / 2);
	if (stepped < nnodes)
	{
		//compareAndExchangeNodes(cache + i, cache + i + (len / 2));
		bool comp = parents[i] < parents[stepped];
		int tmp = cache[comp ? stepped : i];
		int p = parents[comp ? stepped : i];
		parents[stepped] = parents[comp ? i : stepped];
		parents[i] = p;
		cache[stepped] = cache[comp ? i : stepped];
		cache[i] = tmp;
	}
}

__global__ void OddEvenMergeSort_b(int* cache, int* parents, int len, int nnodes, int step)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int section_idx = idx / (len / 2); //there are len/2 comparisons in each block
	int step_idx = idx % (len / 2);
	int start_idx = step_idx % step;
	int it_idx = step_idx / step;

	int i = section_idx*len + start_idx + step + it_idx * 2 * step;
	int stepped = i + step;
	if ((step * 2 + it_idx * 2 * step < len) && (stepped < nnodes))
	{
		bool comp = parents[i] < parents[stepped];
		int tmp = cache[comp ? stepped : i];
		int p = parents[comp ? stepped : i];
		parents[stepped] = parents[comp ? i : stepped];
		parents[i] = p;
		cache[stepped] = cache[comp ? i : stepped];
		cache[i] = tmp;
	}

}


//temporary helper to check that we have sorted the input
__global__ void displayNodeList(node nodes, int* img, int count)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < count)
	{
		img[id] = nodes.parentind[id];
	}
}


int sortedTo(int * arr, int nvals)
{
	for(int i = 0; i < nvals - 1; i++)
	{
		if (arr[i] < arr[i + 1]) return i;
	}
	return nvals;
}

//this now invalidates .core and .valid in return for better performance
void OEMSort(node arr, int * parents, int nnodes, int rowsxcols)
{
	//allocate the temp array
	 __align__(4) int * ptrs = nullptr;
	try
	{
		g_setToHead <<<nnodes / 1024 + 1, 1024 >>> (arr, parents, rowsxcols, nnodes);

		HANDLE_ERROR(cudaMalloc(&ptrs, nnodes * sizeof(int)));
		cudaDeviceSynchronize();
		
		HANDLE_ERROR(cudaGetLastError());
		OEMSort_kernel <<<nnodes / (nOEMshare) + 1, nOEMthreads >>> (ptrs, parents,  nnodes);
		cudaDeviceSynchronize();

		HANDLE_ERROR(cudaGetLastError());
		for (int len = nOEMthreads * 2; len <= nnodes; len *= 2)
		{
			OddEvenMergeSort_a <<<nnodes / (nOEMthreads ) + 1, nOEMthreads >>> (ptrs, parents, len, nnodes);
		//	cudaDeviceSynchronize();
			for (int step = len / 4; step > 0; step /= 2)
			{
				OddEvenMergeSort_b <<<nnodes / (nOEMthreads) + 1, nOEMthreads >>> (ptrs, parents, len, nnodes, step);
			//	cudaDeviceSynchronize();
			}

		}
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

		applysort <<<nnodes / 64 + 1, 64 >>> (arr, ptrs, parents, nnodes);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

		if (ptrs != nullptr)HANDLE_ERROR(cudaFree(ptrs));
	}
	catch (CUDA_Exception e)
	{
		if(ptrs != nullptr)cudaFree(ptrs);
		throw e;
	}
}



//this produces an array of bounding boxes from the sorted nodes
//there are half as many bbs allocated as points
template<int nbbs>
__global__ void makeBBs_local(const node nodes, const float * fg_ranges_min, const float * fg_ranges_max, DidoFusedAnalytics_BoundingBox* bbs, int * bbparents, int * bbnnode,  int rows, int cols)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < rows*cols - 1)
	{
		//zero initialise the values

		__shared__ DidoFusedAnalytics_BoundingBox localbbs[nbbs];
		__shared__ __align__(4) int parents[nbbs];
		__shared__ __align__(4)  int nbbnodes[nbbs];
		if (threadIdx.x % 2 == 0)
		{
			parents[threadIdx.x / 2] = -1;
			nbbnodes[threadIdx.x / 2] = 0;
			localbbs[threadIdx.x / 2].x = 0;
			localbbs[threadIdx.x / 2].y = 0;
			localbbs[threadIdx.x / 2].max_x = 0;
			localbbs[threadIdx.x / 2].max_y = 0;
			localbbs[threadIdx.x / 2].avdepth = 0;
			localbbs[threadIdx.x / 2].uncertainty = 0;
		}

		int myparent = nodes.parentind[index];
		if (myparent >= 0 )
		{
			int myInd = nodes.index[index];
			int myx = myInd % cols;
			int myy = myInd / cols;
			if (myparent == nodes.parentind[index + 1] && ((threadIdx.x == 0) || nodes.parentind[index - 1] != myparent))
			{
				parents[threadIdx.x / 2] = myparent;
				localbbs[threadIdx.x / 2].x = myx;
				localbbs[threadIdx.x / 2].y = myy;
				localbbs[threadIdx.x / 2].max_x = myx;
				localbbs[threadIdx.x / 2].max_y = myy;
			}
			__syncthreads();
			for (int i = 0; i < nbbs; i++)
			{
				if (myparent == parents[i])
				{
					atomicMin(&(localbbs[i].x), myx);
					atomicMin(&(localbbs[i].y), myy);
					atomicMax(&(localbbs[i].max_x), myx);
					atomicMax(&(localbbs[i].max_y), myy);
					if(isfinite(fg_ranges_min[myInd]) && isfinite(fg_ranges_max[myInd])) 
					{
						atomicAdd(&(localbbs[i].avdepth), (fg_ranges_min[myInd] + fg_ranges_max[myInd])/2);
						atomicAdd(&(localbbs[i].uncertainty), (-fg_ranges_min[myInd] + fg_ranges_max[myInd]));
					}
					atomicAdd(&(nbbnodes[i]), 1);
					break;
				}
			}
		}
		__syncthreads();
		//return the values
		if (threadIdx.x % 2 == 0)
		{
			bbs[index / 2] = localbbs[threadIdx.x / 2];
			bbparents[index / 2] = parents[threadIdx.x / 2];
			bbnnode[index / 2] = nbbnodes[threadIdx.x / 2];
		}
	}
}

template __global__ void makeBBs_local<32>(const node nodes, const float * fg_ranges_min, const float * fg_ranges_max, DidoFusedAnalytics_BoundingBox* bbs, int * bbparents, int * bbnnodes, int rows, int cols);


__global__ void downsampleFG(const float * in_pano, float * out_pano, int ncols, int nrows, int scale)
{
	int index_x = (threadIdx.x + blockIdx.x * blockDim.x);
	int index_y = (threadIdx.y + blockIdx.y * blockDim.y);
	__shared__ float totals[16][16];	//the most we could need, for blocks of 32x32 with a scale of 2
	__shared__ int nadded[16][16];
	//work out which total is yours
	int totind_x = threadIdx.x / scale;
	int totind_y = threadIdx.y / scale;
	//initialise the totals
	bool topleft = (threadIdx.x % scale == 0) && (threadIdx.y % scale == 0);
	if (topleft)
	{
		totals[totind_x][totind_y] = 0;
		nadded[totind_x][totind_y] = 0;
	}
	__syncthreads();
	if (index_x / scale < ncols / scale && index_y / scale < nrows / scale)
	{
		if (in_pano[index_x + ncols*index_y] > 0)
		{
			atomicAdd(&totals[totind_x][totind_y], in_pano[index_x + ncols*index_y]);
			//count how many have contributed
			atomicAdd(&nadded[totind_x][totind_y], 1);
		}
		__syncthreads();
		if (topleft)
		{
			//normalise and put the value into the output
			out_pano[(index_x / scale) + (ncols / scale)*(index_y / scale)] =
				nadded[totind_x][totind_y] > 0 ? totals[totind_x][totind_y] / nadded[totind_x][totind_y] : -1.0f;
		}
	}
}



#endif

} //objdetectCUDA

std::vector<DidoFusedAnalytics_BoundingBox> DidoFusedAnalytics_3dObjDetect_CUDA::detectBlobs(const float * foreground_ranges_min, const float * foreground_ranges_max, int rows, int cols) const
{
	std::vector<DidoFusedAnalytics_BoundingBox> rval;
#if DIDOLIDAR_NOGPU

#else

	//auto start = std::chrono::high_resolution_clock::now();
	//allocate local data
	 objdetectCUDA::node m_nodes;
	__align__(16) DidoFusedAnalytics_BoundingBox* m_bbs;
	__align__(4) int * m_parents, *m_nnodes, * tmparray;
	__align__(4) float * d_fgr_min_scaled, * d_fgr_max_scaled;

	//host values
	DidoFusedAnalytics_BoundingBox* l_bbs;
	int * l_parents, *l_nnodes;

	const int blockdim = 32;

	int scaledRows = rows / workingscale;
	int scaledCols = cols / workingscale;

	//a;ways a power of two so we can sort easily
	int nnodes = 1 << int(ceil(log2(scaledRows*scaledCols)));
	int nbbs = (nnodes / 2);

	try
	{
		dim3 grid(((cols) / blockdim + 1), ((rows) / blockdim + 1));
		dim3 block(blockdim, blockdim);
		HANDLE_ERROR(cudaMalloc(&d_fgr_min_scaled,scaledRows*scaledCols*sizeof(float) ));
		HANDLE_ERROR(cudaMalloc(&d_fgr_max_scaled, scaledRows*scaledCols * sizeof(float)));
		if (workingscale == 1)
		{
			HANDLE_ERROR(cudaMemcpy(d_fgr_max_scaled, foreground_ranges_max, scaledRows*scaledCols * sizeof(float), cudaMemcpyDeviceToDevice));
			HANDLE_ERROR(cudaMemcpy(d_fgr_min_scaled, foreground_ranges_min, scaledRows*scaledCols * sizeof(float), cudaMemcpyDeviceToDevice));
		}
		else
		{
			objdetectCUDA::downsampleFG << <grid, block >> > (foreground_ranges_min, d_fgr_min_scaled, cols, rows, workingscale);
			objdetectCUDA::downsampleFG << <grid, block >> > (foreground_ranges_max, d_fgr_max_scaled, cols, rows, workingscale);
		}
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

		HANDLE_ERROR(cudaMalloc(&m_nodes.valid, nnodes * sizeof(bool)));
		HANDLE_ERROR(cudaMalloc(&m_nodes.core, nnodes * sizeof(bool)));
		HANDLE_ERROR(cudaMalloc(&m_nodes.parentind, nnodes * sizeof(int)));
		HANDLE_ERROR(cudaMalloc(&m_nodes.index, nnodes * sizeof(int)));

		grid = dim3(((scaledCols) / blockdim + 1), ((scaledRows) / blockdim + 1));
		block = dim3(blockdim, blockdim);
		objdetectCUDA::generateTrees <<<grid, block >>> (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

		
		//	auto gentree = std::chrono::high_resolution_clock::now();

#define SHARED_PDS 0
		HANDLE_ERROR(cudaGetLastError());
		switch(epsilon)
		{
		case 1:
#if SHARED_PDS
			objdetectCUDA::PDSDBSCANInit__shared<1, blockdim> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
				cudaDeviceSynchronize();

			HANDLE_ERROR(cudaGetLastError());
			objdetectCUDA::PDSDBSCANInit__shared<1, blockdim> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
#else
			//less safe version (may be needed?)

			objdetectCUDA::PDSDBSCANInit<1> <<<grid, block >>> (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
#endif
			HANDLE_ERROR(cudaMalloc(&tmparray, nnodes * sizeof(int)));
		
			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError());
			//auto pdsinit = std::chrono::high_resolution_clock::now();
			objdetectCUDA::PDSDBSCANLocal<2> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
			break;
		case 2:
#if SHARED_PDS
			objdetectCUDA::PDSDBSCANInit__shared<2, blockdim> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
				cudaDeviceSynchronize();

			HANDLE_ERROR(cudaGetLastError());
			objdetectCUDA::PDSDBSCANInit__shared<2, blockdim> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
#else
			//less safe version (may be needed?)

			objdetectCUDA::PDSDBSCANInit<2> <<<grid, block >>> (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
#endif
			HANDLE_ERROR(cudaMalloc(&tmparray, nnodes * sizeof(int)));
		
			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError());
			//auto pdsinit = std::chrono::high_resolution_clock::now();
			objdetectCUDA::PDSDBSCANLocal<3> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
			break;
		case 3:
#if SHARED_PDS
			objdetectCUDA::PDSDBSCANInit__shared<3, blockdim> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
				cudaDeviceSynchronize();

			HANDLE_ERROR(cudaGetLastError());
			objdetectCUDA::PDSDBSCANInit__shared<3, blockdim> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
#else
			//less safe version (may be needed?)

			objdetectCUDA::PDSDBSCANInit<3> <<<grid, block >>> (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
#endif
			HANDLE_ERROR(cudaMalloc(&tmparray, nnodes * sizeof(int)));
		
			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError());
			//auto pdsinit = std::chrono::high_resolution_clock::now();
			objdetectCUDA::PDSDBSCANLocal<4> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
			break;
		case 4:
#if SHARED_PDS
			objdetectCUDA::PDSDBSCANInit__shared<3, blockdim> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
			cudaDeviceSynchronize();

			HANDLE_ERROR(cudaGetLastError());
			objdetectCUDA::PDSDBSCANInit__shared<3, blockdim> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
#else
			//less safe version (may be needed?)

			objdetectCUDA::PDSDBSCANInit<4> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
#endif
			HANDLE_ERROR(cudaMalloc(&tmparray, nnodes * sizeof(int)));

			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError());
			//auto pdsinit = std::chrono::high_resolution_clock::now();
			objdetectCUDA::PDSDBSCANLocal<5> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
			break;
		case 5:
#if SHARED_PDS
			objdetectCUDA::PDSDBSCANInit__shared<3, blockdim> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
			cudaDeviceSynchronize();

			HANDLE_ERROR(cudaGetLastError());
			objdetectCUDA::PDSDBSCANInit__shared<3, blockdim> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
#else
			//less safe version (may be needed?)

			objdetectCUDA::PDSDBSCANInit<5> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
#endif
			HANDLE_ERROR(cudaMalloc(&tmparray, nnodes * sizeof(int)));

			cudaDeviceSynchronize();
			HANDLE_ERROR(cudaGetLastError());
			//auto pdsinit = std::chrono::high_resolution_clock::now();
			objdetectCUDA::PDSDBSCANLocal<5> << <grid, block >> > (d_fgr_min_scaled, d_fgr_max_scaled, m_nodes, scaledRows, scaledCols, rangeScaling, ncore);
			break;
		default:
			throw "only integer eps of 1 to 5 supported";
		}
		HANDLE_ERROR(cudaMalloc(&m_bbs, nbbs * sizeof(DidoFusedAnalytics_BoundingBox)));
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

	//	auto pdslocal = std::chrono::high_resolution_clock::now();
		//sort the nodes by parent
		objdetectCUDA::OEMSort(m_nodes, tmparray, nnodes, scaledRows*scaledCols);
		HANDLE_ERROR(cudaMalloc(&m_parents, nbbs * sizeof(int)));
		HANDLE_ERROR(cudaMalloc(&m_nnodes, nbbs * sizeof(int)));
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());
	//	auto oesort = std::chrono::high_resolution_clock::now();
		
		//make blobs from them

		objdetectCUDA::makeBBs_local<32> <<< nnodes / 64 + 1, 64 >>> (m_nodes, d_fgr_min_scaled, d_fgr_max_scaled, m_bbs, m_parents, m_nnodes, scaledRows, scaledCols);

		l_parents = (int*)malloc(nbbs * sizeof(int));
		l_nnodes = (int*)malloc(nbbs * sizeof(int));
		l_bbs = (DidoFusedAnalytics_BoundingBox*)malloc(nbbs * sizeof(DidoFusedAnalytics_BoundingBox));
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());
	//	auto makeb = std::chrono::high_resolution_clock::now();

		//copy down the data
		HANDLE_ERROR(cudaMemcpy(l_parents, m_parents, nbbs * sizeof(int), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(l_nnodes, m_nnodes, nbbs * sizeof(int), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(l_bbs, m_bbs, nbbs * sizeof(DidoFusedAnalytics_BoundingBox), cudaMemcpyDeviceToHost));

		int prevparent = -1;
		int prevnnodes = 0;
		for (int i = 0; i < nbbs ; i++)
		{
			if (l_nnodes[i] == 0) continue; //skip empty entries
			//breaks once we hit the -1s
			if (l_parents[i] < 0)
			{
				break;
			}

			//check if it was split
			if (l_parents[i] == prevparent)
			{
				rval.back().x = std::min(rval.back().x, l_bbs[i].x);
				rval.back().y = std::min(rval.back().y, l_bbs[i].y);
				rval.back().max_x = std::max(rval.back().max_x, l_bbs[i].max_x);
				rval.back().max_y = std::max(rval.back().max_y, l_bbs[i].max_y);
				rval.back().avdepth += l_bbs[i].avdepth;
				rval.back().uncertainty += l_bbs[i].uncertainty;
				prevnnodes += l_nnodes[i];
			}
			///else append to output
			else
			{
				if (!rval.empty())
				{
					//only keep ones that are big enough
					if (prevnnodes > minpoints)
					{
						rval.back().avdepth /= prevnnodes;
						rval.back().uncertainty /= prevnnodes;
						rval.back().x *= workingscale;
						rval.back().max_x *= workingscale;
						rval.back().y *= workingscale;
						rval.back().max_y *= workingscale;
						if(rval.back().x < 0 || rval.back().x > cols || rval.back().y < 0 || rval.back().y > rows ||
						rval.back().max_x < 0 || rval.back().max_x > cols || rval.back().max_y < 0 || rval.back().max_y > rows)
							rval.pop_back(); //sanity check
					}
					else rval.pop_back();

				}
				rval.push_back(l_bbs[i]);
				prevparent = l_parents[i];
				prevnnodes = l_nnodes[i];
			}
		}
		if (!rval.empty())
		{
			if (prevnnodes > minpoints)
			{
				rval.back().avdepth /= prevnnodes;
				rval.back().uncertainty /= prevnnodes;
				rval.back().x *= workingscale;
				rval.back().max_x *= workingscale;
				rval.back().y *= workingscale;
				rval.back().max_y *= workingscale;

				if (rval.back().x < 0 || rval.back().x > cols || rval.back().y < 0 || rval.back().y > rows ||
					rval.back().max_x < 0 || rval.back().max_x > cols || rval.back().max_y < 0 || rval.back().max_y > rows)
					rval.pop_back(); //sanity check
			}
			else
			{
				rval.pop_back();
			}
		}

	//	auto colb = std::chrono::high_resolution_clock::now();


		if (l_nnodes != nullptr) free(l_nnodes);
		if (l_bbs != nullptr) free(l_bbs);
		if (l_parents != nullptr) free(l_parents);

		//free local data
		if (d_fgr_min_scaled != nullptr) HANDLE_ERROR(cudaFree(d_fgr_min_scaled));
		if (d_fgr_max_scaled != nullptr) HANDLE_ERROR(cudaFree(d_fgr_max_scaled));
		if (tmparray != nullptr) HANDLE_ERROR(cudaFree(tmparray));
		if (m_nodes.valid != nullptr) HANDLE_ERROR(cudaFree(m_nodes.valid));
		if (m_nodes.core != nullptr) HANDLE_ERROR(cudaFree(m_nodes.core));
		if (m_nodes.parentind != nullptr) HANDLE_ERROR(cudaFree(m_nodes.parentind));
		if (m_nodes.index != nullptr) HANDLE_ERROR(cudaFree(m_nodes.index));
		if (m_nnodes != nullptr) HANDLE_ERROR(cudaFree(m_nnodes));
		if (m_bbs != nullptr) HANDLE_ERROR(cudaFree(m_bbs));
		if (m_parents != nullptr) HANDLE_ERROR(cudaFree(m_parents));

	}
	catch (CUDA_Exception e)
	{
		if (l_nnodes != nullptr) free(l_nnodes);
		if (l_bbs != nullptr) free(l_bbs);
		if (l_parents != nullptr) free(l_parents);
		//free local data in case of error and reset the context
		cudaDeviceReset();
		throw e;
	}
#endif

	//populate it

	return rval;
}
}