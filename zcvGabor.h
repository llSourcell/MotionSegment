#ifndef ZCVGABOR_H
#define ZCVGABOR_H

#include "cvgabor.h"


namespace cv
{
	class zcvGabor : public CvGabor{
	public:
		zcvGabor();
		~zcvGabor();

		void Init(double dPhi, int iNu, double dSigma, double dF, int r, int c);
		void Init(double dPhi, int iNu, double dSigma, double dF);
		int fInit(int r, int c);

		void convolve();
		void fconvolve(CvMat* fin, CvMat* conv);
		void DCTinput(CvMat* in, CvMat** out);

	protected:
		CvMat* fDFT;
		CvMat* scratch;
		int r_in, c_in;

		int build_freq(int r, int c);
		void create_kernel();

	};
}

#endif
