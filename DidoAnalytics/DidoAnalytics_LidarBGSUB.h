/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2013 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	DidoAnalytics_LidarBGSUB.h
* @author  	SL
* @version 	1
* @date    	2017-10-02
* @brief   	GPU based Background subtraction for the Lidar
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __DidoAnalytics_LidarBGSUB
#define __DidoAnalytics_LidarBGSUB


#pragma once
#include <vector>
/*defines ---------------------------*/


//alignment function
#ifndef MY_ALIGN
#if defined(__CUDACC__) // NVCC
#define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define MY_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif
#endif

namespace overview
{
	//container for the 2.5D arrangement of the dataset
	#define LIDARBGSUB_MAX_BIN_PTS 64
	struct LidarBin	//now a sret of pointers to the relevant bits
	{
		MY_ALIGN(8) unsigned int * npts;
		MY_ALIGN(8) float * points_pan, *points_tilt, *points_range;
		//memory management functions
		void allocate(size_t nbins);
		void local_allocate(size_t nbins);
		void deallocate();
		void local_deallocate();
		//memory copying
		void copyDownFrom(LidarBin & src, size_t nbins);
	};
	
	
	class DidoAnalytics_LidarBGSUB
	{
		LidarBin bgmodel;
		float * modelWeights, *variances;
		int modelRows, modelCols;
		//parameters of the system
		float binWidth, binHeight;
		int hist;
		float bgthresh;
		float sqmindist;
		
		//clustering paramters
		float epsilon;
		int ncore;

		//operational variables
		int frameno = 0;
		int searchWidth;
		
	public:
		virtual ~DidoAnalytics_LidarBGSUB();
		DidoAnalytics_LidarBGSUB(float binw, float binh, float mindist = 0.25f, int history = 80, float bgthreshold = 0.3f, float _eps = 0.8f, int _ncore = 3);

		//the input is on the host, not the device, for each of these

		//produces the foreground points from the given lidar frame
		std::vector<DidoLidar_rangeData> apply(const DidoLidar_rangeData* points, int npts, float learningRate = -1.0f);
		//applies the background subtraction then clusters the foreground points		
		std::vector<std::vector<DidoLidar_rangeData>> applyAndCluster(const DidoLidar_rangeData* points, int npts, float learningRate = -1.0f);

		//clusters the given set of foregorund points using DBSCAN
		std::vector<std::vector<DidoLidar_rangeData>> Cluster(const DidoLidar_rangeData * fgpts, size_t npts);
		std::vector<std::vector<DidoLidar_rangeData>> Cluster(std::vector<DidoLidar_rangeData> & fgpts) { return Cluster(fgpts.data(), fgpts.size()); }

		//produces a display of the current number of points in each of the model bins
		void dispayBgmodelNpts(unsigned char * out, int npts);

		int rows() { return modelRows; }
		int cols() { return modelCols; }

		void setHistory(int h) { hist = h; }
		void setBGThresh(float bgt) { bgthresh = bgt; }
	};
}
#endif
