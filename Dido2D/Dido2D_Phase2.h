/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview
* Limited. Any unauthorised use, reproduction or transfer of this
* program is strictly prohibited.
* Copyright 2018 Overview Limited. (Subject to limited
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	Dido2D_Phase2.h
* @author  	SL
* @version 	1
* @date    	2018-01-24
* @brief    Class that handles the reception of the data from the PTZ for phase 2
*****************************************************************************
**/

/* Define to prevent recursive inclusion -----*/
#ifndef __Dido2D_Phase2
#define __Dido2D_Phase2


#pragma once

#include "Dido2D.h"


/*defines ---------------------------*/


namespace overview
{
	class Dido2D_Phase2 : public Dido2D_iface
	{
	protected:

		virtual void loopfunction() override;

		//where to save the images to
		std::string saveloc;
		int imgID = 0;

	public:
		~Dido2D_Phase2();
		/*needs a pointer to the global store on construction*/
		Dido2D_Phase2(std::shared_ptr<OVTrack_Storage> _trackstore, std::string _videoAddress);
		Dido2D_Phase2(std::shared_ptr<OVTrack_Storage> _trackstore, std::shared_ptr<DidoVCap> _vcap);


	};
}
#endif
