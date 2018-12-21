/**  *****************************************************************************
* This program is the confidential and proprietary product of Overview    
* Limited. Any unauthorised use, reproduction or transfer of this         
* program is strictly prohibited.              
* Copyright 2013 Overview Limited. (Subject to limited                    
* distribution and restricted disclosure only.) All rights reserved.
*
* @file    	CUDA_Exception.h
* @author  	SL
* @version 	1
* @date    	2017-04-24
* @brief   	child of std::exception for handling CUDA errors with
 *****************************************************************************/

#include <exception>
#include <stdio.h>

namespace overview
{
class CUDA_Exception : public std::exception
{
	const char * description, *filename;
	int line, err;
public:
	CUDA_Exception(const char * desc, int errtype, int lineno, const char * file) :
		line(lineno), err(errtype), description(desc), filename(file) {
	}
	virtual  const char* what() const throw() override
	{
		static char status[256];
		snprintf(status, sizeof(status), "%s in line %d of %s", description, line, filename);
		return status;
	};
	int getErrType() { return err; }
};
}