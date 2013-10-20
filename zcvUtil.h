#ifndef _ZCVUTIL_H
#define _ZCVUTIL_H

#include "cv.h"
#include <stdio.h>

#ifndef PI
#ifdef M_PI
#define PI M_PI
#else
#define PI 3.141592653589793
#endif
#endif


namespace cv
{

	/**
	* \brief Stores an image and a corresponding binary mask.
	*/
	struct zcvIMask {
		IplImage* i;
		IplImage* m;
	};

	void zcvPrintMat( const CvMat* m );

	/**
	* zcvBGR2RGI converts a BGR-formatted image to RGI. The first two channels of
	* the resulting image are r- and g-chromaticity, and the third is intensity.
	*/
	void zcvBGR2RGI( IplImage* bgr, IplImage* rgi );

	/**
	* Does the same conversion as zcvBGR2RGI, but does not merge the image
	* channels at the end.
	*/
	void zcvBGR2RGIs( IplImage* bgr, IplImage* rout, IplImage* gout, IplImage* Iout );

}

#endif
