//Proxy header file for the ethersec interface until we get the real one

#ifndef __ETHERSECIFACEPROXY
#define __ETHERSECIFACEPROXY



//What the numbers correspond to
enum CESI_ObjProb_Types
{
	human = 0,
	car,
	van,
	bus,
	goat,
	horse,
	unknown
};

//Specefies probablities of different classification types
struct CESI_TrackingObjectOpticalCameraData
{
	int m_ObjProb_Type[5];
	float m_ObjProb_Type_Confidence[5];
};


struct CESI_TrackingObjectNeualNetworkDetections
{
	int m_RunningWalking;
	int m_RunningWalkingDetectionInstances;

	
	int m_Crouching;
	int m_CrouchingDetectionInstances;

	int m_Crawl;
	int m_CrawlDetectionInstances;

	int m_BodyDrag;
	int m_BodyDragDetectionInstances;

	int m_LogRoll;
	int m_LogRollDetectionInstances;
};


struct CESI_TrackingObject
{
	/*Bounding box of the target in panorama coordinates*/
	cv::Rect m_BoundingBox;
	/*estimate of the depth to the target*/
	float m_Depth;

	/*Optical classification of the object class*/
	CESI_TrackingObjectOpticalCameraData m_OpticalCassification;
	
	/*Classification of target behaviour*/
	CESI_TrackingObjectNeualNetworkDetections m_ESI_TrackingObjectNeualNetworkDetections;
	
	/*unique identifier to allow association between frames*/
	int m_GUID;
	int m_IDInFrame;	
   /*is this track currently active?*/
   bool m_IsCurrent;
};

struct CESI_Camera
{
   /*the identifier for the camera*/
	int m_CameraNum;
	int m_nTrackedObjects;
	CESI_TrackingObject m_ESI_TrackingObject[8];
};

struct CESI_FrameOutput
{
	static const int nCams = 15;
	CESI_Camera m_Camera[nCams];
	int m_FrameNumber;
	long long m_TimeCompletedProcessing;
};

CESI_FrameOutput ThermalPanoramaDetect(const unsigned char * ThermalPanoGPUPtr, const float * RangesPanoGPUPtr, size_t rows, size_t cols);

CESI_TrackingObjectOpticalCameraData OpticalImageDetect(cv::Mat * img, cv::Rect ROI);
#endif
