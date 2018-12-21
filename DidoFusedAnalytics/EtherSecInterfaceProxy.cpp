//Proxy for EthersecInterface

#include "GlobalIncludes.h"
#include "EtherSecInterfaceProxy.h"

static cv::RNG rng;
static int xval = 10;
static int idshift = 0;

CESI_FrameOutput ThermalPanoramaDetect(const unsigned char * ThermalPanoGPUPtr, const float * RangesPanoGPUPtr, size_t rows, size_t cols)
{
   //populate it with a single detection for initial testing
   CESI_FrameOutput rval = {};   //zero initialise
   rval.m_FrameNumber = 2;
   rval.m_Camera[0].m_CameraNum = 1;
   rval.m_Camera[0].m_nTrackedObjects = 4;
   //the test detection
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_IsCurrent = true;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_IDInFrame = 1;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_GUID = 1234 + idshift;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_Depth = 30;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_BoundingBox = cv::Rect(xval,20,100,80);
	//get it to move
   xval += 5;
   if ((xval % 60) == 0) idshift += 0xf000;

   //it's behaviour classification
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_ESI_TrackingObjectNeualNetworkDetections.m_RunningWalking = 20;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_ESI_TrackingObjectNeualNetworkDetections.m_RunningWalkingDetectionInstances = 2;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_ESI_TrackingObjectNeualNetworkDetections.m_Crouching = 5;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_ESI_TrackingObjectNeualNetworkDetections.m_CrouchingDetectionInstances = 1;

   //it's CNN based classification
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_OpticalCassification.m_ObjProb_Type[0] = 0;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_OpticalCassification.m_ObjProb_Type[1] = 1;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_OpticalCassification.m_ObjProb_Type[2] = 2;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_OpticalCassification.m_ObjProb_Type[3] = 3;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_OpticalCassification.m_ObjProb_Type[4] = 4;

   rval.m_Camera[0].m_ESI_TrackingObject[0].m_OpticalCassification.m_ObjProb_Type_Confidence[0] = 0.2f;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_OpticalCassification.m_ObjProb_Type_Confidence[1] = 0.2f;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_OpticalCassification.m_ObjProb_Type_Confidence[2] = 0.3f;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_OpticalCassification.m_ObjProb_Type_Confidence[3] = 0.1f;
   rval.m_Camera[0].m_ESI_TrackingObject[0].m_OpticalCassification.m_ObjProb_Type_Confidence[4] = 0.1f;

   /*second test detection - same classifier*/
   rval.m_Camera[0].m_ESI_TrackingObject[1] = rval.m_Camera[0].m_ESI_TrackingObject[0];
   rval.m_Camera[0].m_ESI_TrackingObject[1].m_BoundingBox.y = 60;
   rval.m_Camera[0].m_ESI_TrackingObject[1].m_Depth = 120;
   rval.m_Camera[0].m_ESI_TrackingObject[1].m_GUID = 2345 + idshift;

   rval.m_Camera[0].m_ESI_TrackingObject[2] = rval.m_Camera[0].m_ESI_TrackingObject[0];
   rval.m_Camera[0].m_ESI_TrackingObject[2].m_BoundingBox.x = 60;
   rval.m_Camera[0].m_ESI_TrackingObject[2].m_Depth = rng.uniform(10.0f,200.0f);
   rval.m_Camera[0].m_ESI_TrackingObject[2].m_GUID = 345 + idshift;

   rval.m_Camera[0].m_ESI_TrackingObject[3] = rval.m_Camera[0].m_ESI_TrackingObject[0];
   rval.m_Camera[0].m_ESI_TrackingObject[3].m_BoundingBox.x = (int)rng.gaussian(30) + 100;
   rval.m_Camera[0].m_ESI_TrackingObject[3].m_Depth = 160;
   rval.m_Camera[0].m_ESI_TrackingObject[3].m_GUID = 45 + idshift;

   rval.m_Camera[1] = rval.m_Camera[0];
   rval.m_Camera[1].m_ESI_TrackingObject[0].m_GUID = 1 + idshift;
   rval.m_Camera[1].m_ESI_TrackingObject[0].m_BoundingBox.x += 5200;
   rval.m_Camera[1].m_ESI_TrackingObject[1].m_GUID = 2 + idshift;
   rval.m_Camera[1].m_ESI_TrackingObject[1].m_BoundingBox.x += 3200;
   rval.m_Camera[1].m_ESI_TrackingObject[2].m_GUID = 3 + idshift;
   rval.m_Camera[1].m_ESI_TrackingObject[2].m_BoundingBox.x += 2200;
   rval.m_Camera[1].m_ESI_TrackingObject[3].m_GUID = 4 + idshift;
   rval.m_Camera[1].m_ESI_TrackingObject[3].m_BoundingBox.x += 1200;
	return rval;
}

CESI_TrackingObjectOpticalCameraData OpticalImageDetect(cv::Mat * img, cv::Rect ROI)
{
	CESI_TrackingObjectOpticalCameraData rval = {};
   //return empty to see what happens
	rval.m_ObjProb_Type_Confidence[0] = 0.9f;
	rval.m_ObjProb_Type[0] = 0;
	rval.m_ObjProb_Type[1] = 1;
	rval.m_ObjProb_Type[2] = 4;
	rval.m_ObjProb_Type[3] = 3;
	rval.m_ObjProb_Type[4] = 2;
	return rval;
}
