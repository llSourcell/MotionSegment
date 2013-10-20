#include "zcvUtil.h"


namespace cv
{

	void zcvPrintMat( const CvMat* m )
	{
		CvSize sz = cvGetSize( m );
		printf("(%d,%d):\n", sz.height, sz.width);
		for(int i=0; i<sz.height; i++){
			for(int j=0; j<sz.width; j++)
				printf(" %lf", cvmGet(m,i,j));
			printf("\n");
		}
	}

	void zcvBGR2RGI( IplImage* bgr, IplImage* rgi )
	{
		IplImage* ri = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_8U, 1 );
		IplImage* gi = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_8U, 1 );
		IplImage* Ii = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_8U, 1 );

		zcvBGR2RGIs( bgr, ri, gi, Ii );
		// build the output
		cvMerge(Ii, gi, ri, NULL, rgi);

		cvReleaseImage( &ri );
		cvReleaseImage( &gi );
		cvReleaseImage( &Ii );
	}

	void zcvBGR2RGIs( IplImage* bgr, IplImage* rout, IplImage* gout, IplImage* Iout )
	{
		IplImage* r;
		IplImage* g;
		IplImage* I;
		IplImage* ri;
		IplImage* gi;
		IplImage* Ii;

		int output_depth = rout->depth;
		if( !(rout->depth == gout->depth && gout->depth == Iout->depth) ){
			fprintf(stderr, "zcvBGR2RGIs: outputs must have same depth\n");
			exit(-1);
		}
		if( output_depth == IPL_DEPTH_8U ){
			ri = rout;
			gi = gout;
			Ii = Iout;
			r = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_32F, 1 );
			g = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_32F, 1 );
			I = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_32F, 1 );
		} else if( output_depth == IPL_DEPTH_32F ){
			ri = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_8U, 1 );
			gi = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_8U, 1 );
			Ii = cvCreateImage( cvGetSize(bgr), IPL_DEPTH_8U, 1 );
			r = rout;
			g = gout;
			I = Iout;
		} else {
			fprintf(stderr, "zcvBGR2RGIs: outputs depth must be either 8U or 32F\n");
			exit(-1);
		}
		// should really be able to do this with one call...
		//cvSplit( bgr, 0, gi, 0, 0 );
		//cvSplit( bgr, 0, 0, ri, 0 );
		cvSplit( bgr, Ii, gi, ri, 0 );

		//cvCvtColor( bgr, Ii, CV_BGR2GRAY );
		//cvConvertScale( Ii, I, 254.0/255.0, 1.0 );
		//cvConvertScale( I, Ii, 1.0, 0.0 );

		// switch to floats for division
		cvConvertScale( ri, r, 1.0, 0.0 );
		cvConvertScale( gi, g, 1.0, 0.0 );
		// manual intensity calc
		//cvSplit( bgr, Ii, 0, 0, 0 );
		cvConvertScale(Ii, I, 1.0, 0.0);
		cvAdd( I, r, I );
		cvAdd( I, g, I );
		// get rid of zeros
		cvConvertScale( I, I, 254.0/255.0, 1.0 );

		// Divide r and g by I
		cvDiv( r, I, r);
		cvDiv( g, I, g);
		cvConvertScale( r, r, 255.0, 0.0 );
		cvConvertScale( g, g, 255.0, 0.0 );
		if( output_depth == IPL_DEPTH_8U ){
			// go back to integers
			cvConvertScale( r, ri, 1.0, 0.0 );
			cvConvertScale( g, gi, 1.0, 0.0 );
			cvConvertScale( I, Ii, 1.0/3.0, 0.0 );
			// clean up
			cvReleaseImage( &r );
			cvReleaseImage( &g );
			cvReleaseImage( &I );
		} else {
			cvConvertScale( I, I, 1.0/3.0, 0.0 );
			// clean up
			cvReleaseImage( &ri );
			cvReleaseImage( &gi );
			cvReleaseImage( &Ii );
		}
	}

}

