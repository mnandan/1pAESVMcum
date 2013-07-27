#ifndef _FILEINT_H
#define _FILEINT_H

#include <fstream>
#include <vector>

using namespace std;

typedef unsigned int UINT;

struct feat_T {
  UINT fNum;       // feature number
  double fVal;
};

struct dataVect_T {
  UINT numFeats;
  char label;
  double B;		//beta
  feat_T * F;	//features
};

class trainDat_T {
public:
	vector<dataVect_T> X;
	UINT totRpNum;
	UINT Xsize;
	double C;
	UINT maxF;			// largest index of features among all vectors
	double *w;
	double memUsed;
	double G;
	UINT maxNumFeats;	//maximum number of features in a vector
	UINT numVects;

	trainDat_T(double C1, double G1) {
		Xsize = 0;
		totRpNum = 0;
		C = C1;
		memUsed = 0;
		maxF = 0;
		w = NULL;
		G = G1;
		maxNumFeats = 0;
		numVects = 0;
	}

	int addVector(dataVect_T &Xtemp, UINT lineNum, double memReq) {
		if(memReq + memUsed > G)
			return -1;
		if(Xsize > lineNum + totRpNum)
			X[lineNum + totRpNum - 1] = Xtemp;
		else {
			X.push_back(Xtemp);
			memUsed += sizeof(dataVect_T);
			Xsize++;
		}
		numVects++;
		memUsed += memReq;
		return 0;
	}

	void remExtraData() {
		for (UINT ind = totRpNum; ind < Xsize; ind++) {
			if (X[ind].F != NULL) {
				delete[] X[ind].F;
				memUsed -= sizeof(feat_T)*X[ind].numFeats;
				X[ind].F = NULL;
				X[ind].numFeats = 0;
			}
		}
		numVects = totRpNum;
	}

	double createW(UINT maxF1) {
		if(maxF1 > maxF) {
			if(w!= NULL)
				delete [] w;
			maxF = maxF1;
			w = new double[maxF];
		}
		return maxF;
	}

	~trainDat_T() {
		for (UINT ind = 0; ind < Xsize; ind++) {
			if (X[ind].F != NULL)
				delete[] X[ind].F;
		}
		delete [] w;
	}
};


double getFileSize(char *fName);
int getBlockAE(ifstream & inpF, ofstream & outF, trainDat_T & trDat, double &fSize);

#endif
