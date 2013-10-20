#include "zcvGabor.h"



namespace cv
{

	zcvGabor::zcvGabor(){}

	/**
	* Builds the frequency-domain version of the gabor filters.
	* r and c are the number of rows and columns in the expected input.
	* These are used for choosing the optimal size DFT to produce for later
	* convolution.
	*/
	int zcvGabor::build_freq(int r, int c)
	{
		r_in = r;
		c_in = c;
		// first build a complex version of the filter
		CvMat* A = cvCreateMat( Real->rows, Real->cols, CV_32FC2 );
		cvMerge( Real, Imag, NULL, NULL, A );

		int dft_M = cvGetOptimalDFTSize( Width + r_in - 1 );
		int dft_N = cvGetOptimalDFTSize( Width + c_in - 1 );

		fDFT = cvCreateMat( dft_M, dft_N, CV_32FC2 );
		scratch = cvCreateMat( dft_M, dft_N, CV_32FC2 );
		CvMat tmp;

		// copy filter to fDFT and pad fDFT with zeros
		cvGetSubRect( fDFT, &tmp, cvRect(0,0,A->cols,A->rows));
		cvCopy( A, &tmp );
		cvGetSubRect( fDFT, &tmp, cvRect(A->cols,0,fDFT->cols - A->cols,A->rows));
		cvZero( &tmp );
		// no need to pad bottom part of fDFT with zeros because of
		// use nonzero_rows parameter in cvDFT() call below

		cvDFT( fDFT, fDFT, CV_DXT_FORWARD, A->rows );

		return 0;
	}

	void zcvGabor::DCTinput(CvMat* in, CvMat** out)
	{
		int dft_M = cvGetOptimalDFTSize( in->rows + Real->rows - 1 );
		int dft_N = cvGetOptimalDFTSize( in->cols + Real->cols - 1 );
		*out = cvCreateMat( dft_M, dft_N, CV_32FC2 );
		CvMat tmp;
		// Create a (zero) imaginary channel for the input
		CvSize sz = cvGetSize(in);
		CvMat* imin = cvCreateMat( sz.height, sz.width, CV_32FC1 );
		CvMat* input = cvCreateMat( sz.height, sz.width, CV_32FC2 );
		cvZero( imin );
		// build the input from the real and imaginary parts
		cvMerge( in, imin, NULL, NULL, input );

		cvGetSubRect( *out, &tmp, cvRect(0,0,in->cols,in->rows));
		cvCopy( input, &tmp );
		cvGetSubRect( *out, &tmp, cvRect(in->cols,0,(*out)->cols - in->cols,in->rows));
		cvZero( &tmp );
		// no need to pad bottom part of out with zeros because of
		// use nonzero_rows parameter in cvDFT() call below

		cvDFT( *out, *out, CV_DXT_FORWARD, in->rows );
	}

	void zcvGabor::Init(double dPhi, int iNu, double dSigma, double dF, int r, int c)
	{
		Init(dPhi, iNu, dSigma, dF);
		build_freq(r,c);
	}

	void zcvGabor::Init(double dPhi, int iNu, double dSigma, double dF)
	{
		bInitialised = false;
		bKernel = false;
		Sigma = dSigma;
		F = dF;

		Kmax = PI/2;

		// Absolute value of K
		K = Kmax / pow(F, (double)iNu);
		Phi = dPhi;
		bInitialised = true;
		Width = mask_width();
		Real = cvCreateMat( Width, Width, CV_32FC1);
		Imag = cvCreateMat( Width, Width, CV_32FC1);
		create_kernel();
	}

	int zcvGabor::fInit(int r, int c)
	{
		int ret;

		if( bInitialised && bKernel )
			ret = build_freq(r,c);
		else
			return 1;

		return ret;
	}

	void zcvGabor::fconvolve(CvMat* fin, CvMat* out)
	{
		CvMat* conv = cvCreateMat( r_in + Real->rows - 1, c_in + Real->cols - 1, CV_32FC2 );
		CvMat tmp;

		cvMulSpectrums( fin, fDFT, scratch, 0 );
		cvDFT( scratch, scratch, CV_DXT_INV_SCALE, conv->rows ); // calculate only the top part

		//cvGetSubRect( scratch, &tmp, cvRect(0,0,conv->cols,conv->rows) );
		//cvCopy( &tmp, conv );

		// Cut down to the size of the input and convert to magnitude
		CvMat* c1 = cvCreateMat( out->rows, out->cols, CV_32FC1 );
		cvGetSubRect( scratch, &tmp, cvRect(0,0,out->cols,out->rows) );
		cvSplit( &tmp, out, c1, NULL, NULL ); 
		cvMul( out, out, out );
		cvMul( c1, c1, c1 );
		cvAdd( out, c1, out );
	}

	void zcvGabor::create_kernel()
	{

		if (IsInit() == false) {perror("Error: The Object has not been initilised in creat_kernel()!\n");}
		else {
			CvMat *mReal, *mImag;
			mReal = cvCreateMat( Width, Width, CV_32FC1);
			mImag = cvCreateMat( Width, Width, CV_32FC1);

			/**************************** Gabor Function ****************************/ 
			int x, y;
			double dReal;
			double dImag;
			double dTemp1, dTemp2, dTemp3;

			for (int i = 0; i < Width; i++)
			{
				for (int j = 0; j < Width; j++)
				{
					x = i-(Width-1)/2;
					y = j-(Width-1)/2;
					dTemp1 = (pow(K,2)/pow(Sigma,2))*exp(-(pow((double)x,2)+pow((double)y,2))*pow(K,2)/(2*pow(Sigma,2)));
					dTemp2 = cos(K*cos(Phi)*x + K*sin(Phi)*y) - exp(-(pow(Sigma,2)/2));
					dTemp3 = sin(K*cos(Phi)*x + K*sin(Phi)*y);
					dReal = dTemp1*dTemp2;
					dImag = dTemp1*dTemp3; 
					//gan_mat_set_el(pmReal, i, j, dReal);
					//cvmSet( (CvMat*)mReal, i, j, dReal );
					cvSetReal2D((CvMat*)mReal, i, j, dReal );
					//gan_mat_set_el(pmImag, i, j, dImag);
					//cvmSet( (CvMat*)mImag, i, j, dImag );
					cvSetReal2D((CvMat*)mImag, i, j, dImag );

				} 
			}
			/**************************** Gabor Function ****************************/
			bKernel = true;
			cvCopy(mReal, Real, NULL);
			cvCopy(mImag, Imag, NULL);
			//printf("A %d x %d Gabor kernel with %f PI in arc is created.\n", Width, Width, Phi/PI);
			cvReleaseMat( &mReal );
			cvReleaseMat( &mImag );
		}
	}

}
