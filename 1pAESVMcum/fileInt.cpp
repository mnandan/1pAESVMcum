#include <iostream>
#include <cstdlib>
#include <cstring>
#include "fileInt.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include "svmSolver.h"
#include "onePdRS.h"

using namespace std;

double getFileSize(char *fName) {
	ifstream inpF(fName);
	inpF.seekg(0, inpF.end);
	double fSize = (double) inpF.tellg();
	inpF.close();
	return fSize;
}

int getBlockAE(ifstream & inpF, ofstream & outMdl,
		trainDat_T & trDat,	double &fSize) {

	if (inpF.bad())
		return -1;

	double G = trDat.G;
	UINT maxF = 0;			// largest index of features among all vectors
	double maxNumFeats = 0;		//maximum number of features in a vector
	UINT lineNum = 0;
	double initMemUsed = trDat.memUsed;
	double initFpos = inpF.tellg();
	double FposRevert = initFpos;
	while (1) {
		string line;
		if (getline(inpF, line)) {
			char * lineC = new char[line.length() + 1];
			strcpy(lineC, line.c_str());
			char * labelC = strtok(lineC, " \t\n");
			if (labelC == NULL) {
				cout << "Error in input file read\n";
				return -1;
			}
			char *endPtr, *idx, *val;
			char label = (char) strtol(labelC, &endPtr, 10);
			queue<feat_T> tempQ;
			UINT numFeats = 0, fNum = 0;
			double fVal = 0;
			if (endPtr == labelC || *endPtr != '\0') {
				cout << "Error in input file read\n";
				return -1;
			}

			while (1) {
				idx = strtok(NULL, ":");
				val = strtok(NULL, " \t");

				if (val == NULL)
					break;

				fNum = (UINT) strtoll(idx, &endPtr, 10);
				if (endPtr == idx || *endPtr != '\0') {
					cout << "Error in input file read\n";
					return -1;
				}

				fVal = strtod(val, &endPtr);
				if (endPtr == val || (*endPtr != '\0' && !isspace(*endPtr))) {
					cout << "Error in input file read\n";
					return -1;
				}
				feat_T tempF;
				tempF.fNum = fNum;
				tempF.fVal = fVal;
				tempQ.push(tempF);
				numFeats++;
				maxF = max(maxF, fNum);
			}
			delete[] lineC;
			maxNumFeats = max(maxNumFeats, (double)numFeats);

			double memReq = (double) sizeof(dataVect_T) + sizeof(feat_T) * numFeats;

			dataVect_T Xtemp;
			Xtemp.numFeats = numFeats;
			Xtemp.label = label;
			Xtemp.B = 1.0;
			if (numFeats > 0) {
				Xtemp.F = new feat_T[numFeats];
				for (UINT ind = 0; ind < numFeats; ind++) {
					Xtemp.F[ind] = tempQ.front();
					tempQ.pop();
				}
				queue<feat_T> empty;
				swap(tempQ, empty);
			} else {
				Xtemp.F = NULL;
			}

			if (trDat.addVector(Xtemp, lineNum, memReq) == -1) {
				if (Xtemp.F != NULL)
					delete [] Xtemp.F;
				inpF.seekg(FposRevert);
				break;
			}
			lineNum++;
		}
		else {
			break;
		}
		FposRevert = inpF.tellg();
	}
	maxF++;
	maxF = trDat.createW(maxF);
	// solve AESVM for selected vectors
	svmSolverRp(trDat);
	if (inpF.eof()) {
		//write w to output model
		if(outMdl.is_open() && outMdl.good()) {
			outMdl<< "solver_type L2R_L1LOSS_SVC_DUAL\n"
					<< "nr_class 2\nlabel 1 -1\nnr_feature "<<maxF-1<<"\n"
					<< "bias -1\nw"<<endl;
			for(UINT fI = 1; fI < trDat.maxF; fI++)
				outMdl<< trDat.w[fI]<<endl;
		}
		else
			cerr<<"Cannot write to model file\n";

		return -1;
	}
	else {
		// calc number of rp vects
		// Remaining num of Rp vects that can fit in mem
		double numRpRem = (G/2.0 - initMemUsed)/ (maxNumFeats*sizeof(feat_T));
		//fraction of file read in iter. compared to total unprocessed data
		double currFpos = FposRevert;
		double fractionFileNow = (currFpos - initFpos)/fSize;
		double 	numRp = numRpRem*fractionFileNow;
		cout<<"\t"<<numRp<<"\t";
		if (numRp < 1) {
			cerr << "Not enough memory to store representative set.\n";
			exit(1);
		}
		// comp AE vects
		onePassDRS(trDat, numRp);
		fSize = fSize - (currFpos - initFpos);

		trDat.remExtraData();
	}
	return 0;
}
