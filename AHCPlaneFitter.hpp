// Copyright (C) 2014,2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <vector>
#include <set>
#include <queue>
#include <map>
#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

//#define DEBUG_CLUSTER
//#define DEBUG_CALC
//#define DEBUG_INIT
//#define EVAL_SPEED

#include "AHCTypes.hpp"
#include "AHCPlaneSeg.hpp"
#include "AHCParamSet.hpp"
#include "AHCUtils.hpp"

namespace ahc {
	using ahc::utils::Timer;
	using ahc::utils::pseudocolor;

	/**
	 *  \brief An example of Image3D struct as an adaptor for any kind of point cloud to be used by our ahc::PlaneFitter
	 *
	 *  \details A valid Image3D struct should implements the following three member functions:
	 *  1. int width()
	 *     return the #pixels for each row of the point cloud
	 *  2. int height()
	 *     return the #pixels for each column of the point cloud
	 *  3. bool get(const int i, const int j, double &x, double &y, double &z) const
	 *     access the xyz coordinate of the point at i-th-row j-th-column, return true if success and false otherwise (due to NaN depth or any other reasons)
	 */
	struct NullImage3D {
		int width() { return 0; }
		int height() { return 0; }
		//get point at row i, column j
		bool get(const int i, const int j, double &x, double &y, double &z) const { return false; }
	};

	//three types of erode operation for segmentation refinement
	enum ErodeType {
		ERODE_NONE=0,		//no erode
		ERODE_SEG_BORDER=1,	//erode only borders between two segments
		ERODE_ALL_BORDER=2	//erode all borders, either between two segments or between segment and "black"
	};

	/**
	 *  \brief ahc::PlaneFitter implements the Agglomerative Hierarchical Clustering based fast plane extraction
	 *
	 *  \details note: default parameters assume point's unit is mm
	 */
	template <class Image3D>
	struct PlaneFitter {
		/************************************************************************/
		/* Internal Classes                                                     */
		/************************************************************************/
		//for sorting PlaneSeg by size-decreasing order
		struct PlaneSegSizeCmp {
			bool operator()(const PlaneSeg::shared_ptr& a,
				const PlaneSeg::shared_ptr& b) const {
					return b->N < a->N;
			}
		};

		//for maintaining the Min MSE heap of PlaneSeg
		struct PlaneSegMinMSECmp {
			bool operator()(const PlaneSeg::shared_ptr& a,
				const PlaneSeg::shared_ptr& b) const {
				return b->mse < a->mse;
			}
		};
		typedef std::priority_queue<PlaneSeg::shared_ptr,
			std::vector<PlaneSeg::shared_ptr>,
			PlaneSegMinMSECmp> PlaneSegMinMSEQueue;

		/************************************************************************/
		/* Public Class Members                                                 */
		/************************************************************************/
		//input
		const Image3D *points;	//dim=<heightxwidthx3>, no ownership
		int width, height;		//witdth=#cols, height=#rows (size of the input point cloud)

		int maxStep;			//max number of steps for merging clusters
		int minSupport;			//min number of supporting point
		int windowWidth;		//make sure width is divisible by windowWidth
		int windowHeight;		//similarly for height and windowHeight
		bool doRefine;			//perform refinement of details or not
		ErodeType erodeType;

		ParamSet params;		//sets of parameters controlling dynamic thresholds T_mse, T_ang, T_dz

		//output
		ahc::shared_ptr<DisjointSet> ds;//with ownership, this disjoint set maintains membership of initial window/blocks during AHC merging
		std::vector<PlaneSeg::shared_ptr> extractedPlanes;//a set of extracted planes
		cv::Mat membershipImg;//segmentation map of the input pointcloud, membershipImg(i,j) records which plane (plid, i.e. plane id) this pixel/point (i,j) belongs to

		//intermediate
		std::map<int,int> rid2plid;		//extractedPlanes[rid2plid[rootid]].rid==rootid, i.e. rid2plid[rid] gives the idx of a plane in extractedPlanes
		std::vector<int> blkMap;	//(i,j) block belong to extractedPlanes[blkMap[i*Nh+j]]
		//std::vector<std::vector<int>> blkMembership; //blkMembership[i] contains all block id for extractedPlanes[i]
		bool dirtyBlkMbship;
		std::vector<cv::Vec3b> colors;
		std::vector<std::pair<int,int>> rfQueue;//for region grow/floodfill, p.first=pixidx, p.second=plid
		bool drawCoarseBorder;
		//std::vector<PlaneSeg::Stats> blkStats;
#if defined(DEBUG_INIT) || defined(DEBUG_CLUSTER)
		std::string saveDir;
#endif
#ifdef DEBUG_CALC
		std::vector<int>	numNodes;
		std::vector<int>	numEdges;
		std::vector<int>	mseNodeDegree;
		int maxIndvidualNodeDegree;
#endif

		
		// Structure to store history of the farthest plane
		struct FarthestPlaneHistory {
			cv::Vec3d center;
			cv::Vec3d normal;
			double mse;
			int N;

			FarthestPlaneHistory() : center(0, 0, 0), normal(0, 0, 0), mse(0), N(0) {}

			// Update with new plane using temporal filtering (weighted average)
			void updateWithNewPlane(const PlaneSeg& newPlane, double alpha) {
				this->center = (1.0 - alpha) * this->center + alpha * cv::Vec3d(newPlane.center[0], newPlane.center[1], newPlane.center[2]);
				this->normal = (1.0 - alpha) * this->normal + alpha * cv::Vec3d(newPlane.normal[0], newPlane.normal[1], newPlane.normal[2]);
				this->mse = (1.0 - alpha) * this->mse + alpha * newPlane.mse;
				this->N = (1.0 - alpha) * this->N + alpha * newPlane.N;
			}
		};
		
		
		/************************************************************************/
		/* Public Class Functions                                               */
		/************************************************************************/
		PlaneFitter() : points(0), width(0), height(0),
			maxStep(100000), minSupport(3000),
			windowWidth(10), windowHeight(10),
			doRefine(true), erodeType(ERODE_ALL_BORDER),
			dirtyBlkMbship(true), drawCoarseBorder(false)
		{
			static const unsigned char default_colors[10][3] =
			{
				{255, 0, 0},
				{255, 255, 0},
				{100, 20, 50},
				{0, 30, 255},
				{10, 255, 60},
				{80, 10, 100},
				{0, 255, 200},
				{10, 60, 60},
				{255, 0, 128},
				{60, 128, 128}
			};
			for(int i=0; i<10; ++i) {
				colors.push_back(cv::Vec3b(default_colors[i]));
			}
		}

		~PlaneFitter() {}

		/**
		 *  \brief clear/reset for next run
		 */
		void clear() {
			this->points=0;
			this->extractedPlanes.clear();
			ds.reset();
			rid2plid.clear();
			blkMap.clear();
			rfQueue.clear();
			//blkStats.clear();
			dirtyBlkMbship=true;
		}

		/**
		 *  \brief run AHC plane fitting on one frame of point cloud pointsIn
		 *
		 *  \param [in] pointsIn a frame of point cloud
		 *  \param [out] pMembership pointer to segmentation membership vector, each pMembership->at(i) is a vector of pixel indices that belong to the i-th extracted plane
		 *  \param [out] pSeg a 3-channel RGB image as another form of output of segmentation
		 *  \param [in] pIdxMap usually not needed (reserved for KinectSLAM to input pixel index map)
		 *  \param [in] verbose print out cluster steps and #planes or not
		 *  \return when compiled without EVAL_SPEED: 0 if pointsIn==0 and 1 otherwise; when compiled with EVAL_SPEED: total running time for this frame
		 *
		 *  \details this function corresponds to Algorithm 1 in our paper
		 */
		double run(const Image3D* pointsIn,
			std::vector<std::vector<int>>* pMembership=0,
			cv::Mat* pSeg=0,
			const std::vector<int> * const pIdxMap=0, bool verbose=true)
		{
			if(!pointsIn) return 0;
#ifdef EVAL_SPEED
			Timer timer(1000), timer2(1000);
			timer.tic(); timer2.tic();
#endif
			clear();
			this->points = pointsIn;
			this->height = points->height();
			this->width  = points->width();
			this->ds.reset(new DisjointSet((height/windowHeight)*(width/windowWidth)));

			PlaneSegMinMSEQueue minQ;
			this->initGraph(minQ);
#ifdef EVAL_SPEED
			timer.toctic("init time");
#endif
			int step=this->ahCluster(minQ);
#ifdef EVAL_SPEED
			timer.toctic("cluster time");
#endif
			if(doRefine) {
				this->refineDetails(pMembership, pIdxMap, pSeg);
#ifdef EVAL_SPEED
				timer.toctic("refine time");
#endif
			} else {
				if(pMembership) {
					this->findMembership(*pMembership, pIdxMap);
				}
				if(pSeg) {
					this->plotSegmentImage(pSeg, minSupport);
				}
#ifdef EVAL_SPEED
				timer.toctic("return time");
#endif
			}
			if(verbose) {
				std::cout<<"#step="<<step<<", #extractedPlanes="
					<<this->extractedPlanes.size()<<std::endl;
			}
#ifdef EVAL_SPEED
			return timer2.toc();
#endif
			return 1;
		}

		/**
		 *  \brief print out the current parameters
		 */
		void logParams() const {
#define TMP_LOG_VAR(var) << #var "="<<(var)<<"\n"
			std::cout<<"[PlaneFitter] Parameters:\n"
			TMP_LOG_VAR(width)
			TMP_LOG_VAR(height)
			TMP_LOG_VAR(mergeMSETolerance)
			TMP_LOG_VAR(initMSETolerance)
			TMP_LOG_VAR(depthSigmaFactor)
			TMP_LOG_VAR(similarityTh)
			TMP_LOG_VAR(finalMergeSimilarityTh)
			TMP_LOG_VAR(simTh_znear)
			TMP_LOG_VAR(simTh_zfar)
			TMP_LOG_VAR(simTh_angleMin)
			TMP_LOG_VAR(simTh_angleMax)
			TMP_LOG_VAR(depthChangeFactor)
			TMP_LOG_VAR(maxStep)
			TMP_LOG_VAR(minSupport)
			TMP_LOG_VAR(windowWidth)
			TMP_LOG_VAR(windowHeight)
			TMP_LOG_VAR(erodeType)
			TMP_LOG_VAR(doRefine)<<std::endl;
#undef TMP_LOG_VAR
		}

		/************************************************************************/
		/* Protected Class Functions                                            */
		/************************************************************************/
	protected:
		/**
		 *  \brief refine the coarse segmentation
		 *
		 *  \details this function corresponds to Algorithm 4 in our paper; note: plane parameters of each extractedPlanes in the PlaneSeg is NOT updated after this call since the new points added from region grow and points removed from block erosion are not properly reflected in the PlaneSeg
		 */
		void refineDetails(std::vector<std::vector<int>>* pMembership, const std::vector<int>* const pIdxMap, cv::Mat* pSeg) {
			if (pMembership == 0 && pSeg == 0) return;
			if (this->extractedPlanes.empty()) return;

			std::vector<bool> isValidExtractedPlane; // Tracks whether planes are valid
			this->findBlockMembership(isValidExtractedPlane);

			//this->floodFill(); // Refine segmentation using flood fill
			this->floodFillWithTemporalFiltering(membershipImg, 0.75);


			// **Apply Hole Filling Only to Valid Plane Regions**
			// Convert membershipImg to binary based on valid plane IDs
			cv::Mat binaryMembership = membershipImg.clone();
			binaryMembership.setTo(0, membershipImg < 0); // Set non-plane regions (negative values) to 0

			// Create a structuring element (adjust size as needed)
			int kernelSize = 8; // You can adjust this value
			cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));

			// Apply morphological closing only to the regions with valid plane IDs
			for (int planeID = 0; planeID < this->extractedPlanes.size(); ++planeID) {
				// Create a mask for the current plane
				cv::Mat planeMask = (membershipImg == planeID);

				// Convert to 8-bit image for morphological operation
				planeMask.convertTo(planeMask, CV_8UC1);

				// Apply morphological closing to fill small holes
				cv::morphologyEx(planeMask, planeMask, cv::MORPH_CLOSE, kernel);

				// Update the membershipImg with the filled plane
				membershipImg.setTo(planeID, planeMask); // Only update the pixels belonging to this plane
			}


			// Update the membership image and segmentation map
			std::vector<int> plidmap(this->extractedPlanes.size(), -1);
			int nFinalPlanes = 0;

			// Map only valid planes
			for (int i = 0; i < this->extractedPlanes.size(); ++i) {
				const PlaneSeg& op = *this->extractedPlanes[i];
				int np_rid = ds->Find(op.rid);
				if (np_rid == op.rid && isValidExtractedPlane[i]) { // Plane is unchanged, valid plane
					plidmap[i] = nFinalPlanes++;
				}
				else {
					plidmap[i] = -1; // This plane was eroded or merged
				}
			}


			// Scan the membership image and update the segmentation map
			if (pSeg) {
				pSeg->setTo(cv::Vec3b(0, 0, 0));
				const int nPixels = this->width * this->height;
				static const cv::Vec3b blackColor(0, 0, 0);
				for (int i = 0; i < nPixels; ++i) {
					pSeg->at<cv::Vec3b>(i) = blackColor;
					int& plid = membershipImg.at<int>(i);
					if (plid >= 0 && plidmap[plid] >= 0) {
						plid = plidmap[plid];
						pSeg->at<cv::Vec3b>(i) = this->colors[plid];
					}
					else {
						pSeg->at<cv::Vec3b>(i) = blackColor; // Mask out unrefined areas
					}
				}
			}

			// Update pMembership if needed
			if (pMembership) {
				pMembership->resize(nFinalPlanes);
				for (int i = 0; i < nFinalPlanes; ++i) {
					pMembership->at(i).reserve((int)(this->extractedPlanes[i]->N * 1.2f));
				}

				const int nPixels = this->width * this->height;
				for (int i = 0; i < nPixels; ++i) {
					int plid = membershipImg.at<int>(i);
					if (plid >= 0 && plidmap[plid] >= 0) {
						pMembership->at(plid).push_back(pIdxMap ? pIdxMap->at(i) : i);
					}
				}
			}
		}

		/**
		 *  \brief find out all valid 4-connect neighbours pixels of pixel (i,j)
		 *
		 *  \param [in] i row index of the center pixel
		 *  \param [in] j column index of the center pixel
		 *  \param [in] H height of the image
		 *  \param [in] W weight of the image
		 *  \param [out] nbs pixel id of all valid neighbours
		 *  \return number of valid neighbours
		 *
		 *  \details invalid 4-connect neighbours means out of image boundary
		 */
		static inline int getValid4Neighbor(
			const int i, const int j,
			const int H, const int W,
			int nbs[4])
		{
			// Define movement directions for 4 neighbors (left, right, up, down)
			const int directions[4][2] = {
				{0, -1},  // Left
				{0, 1},   // Right
				{-1, 0},  // Up
				{1, 0}    // Down
			};

			int cnt = 0;
			for (int k = 0; k < 4; ++k) {
				int ni = i + directions[k][0];
				int nj = j + directions[k][1];
				if (ni >= 0 && ni < H && nj >= 0 && nj < W) {
					nbs[cnt++] = ni * W + nj;
				}
			}
			return cnt;
		}

		/**
		 *  \brief find out pixel (pixX, pixY) belongs to which initial block/window
		 *
		 *  \param [in] pixX column index
		 *  \param [in] pixY row index
		 *  \return initial block id, or -1 if not in any block (usually because windowWidth%width!=0 or windowHeight%height!=0)
		 */
		inline int getBlockIdx(const int pixX, const int pixY) const {
			assert(pixX>=0 && pixY>=0 && pixX<this->width && pixY<this->height);
			const int Nw = this->width/this->windowWidth;
			const int Nh = this->height/this->windowHeight;
			const int by = pixY/this->windowHeight;
			const int bx = pixX/this->windowWidth;
			return (by<Nh && bx<Nw)?(by*Nw+bx):-1;
		}

		/**
		 *  \brief region grow from coarse segmentation boundaries
		 *
		 *  \details this function implemented line 14~25 of Algorithm 4 in our paper
		 */
		void floodFill()
		{
			std::vector<float> distMap(this->height*this->width,
				std::numeric_limits<float>::max());

			for(int k=0; k<(int)this->rfQueue.size(); ++k) {
				const int sIdx=rfQueue[k].first;
				const int seedy=sIdx/this->width;
				const int seedx=sIdx-seedy*this->width;
				const int plid=rfQueue[k].second;
				const PlaneSeg& pl = *extractedPlanes[plid];

				int nbs[4]={-1};
				const int Nnbs=this->getValid4Neighbor(seedy,seedx,this->height,this->width,nbs);
				for(int itr=0; itr<Nnbs; ++itr) {
					const int cIdx=nbs[itr];
					int& trail=membershipImg.at<int>(cIdx);
					if(trail<=-6) continue; //visited from 4 neighbors already, skip
					if(trail>=0 && trail==plid) continue; //if visited by the same plane, skip
					const int cy=cIdx/this->width;
					const int cx=cIdx-cy*this->width;
					const int blkid=this->getBlockIdx(cx,cy);
					if(blkid>=0 && this->blkMap[blkid]>=0) continue; //not in "black" block

					double pt[3]={0};
					float cdist=-1;
					if(this->points->get(cy,cx,pt[0],pt[1],pt[2]) &&
						std::pow(cdist=(float)std::abs(pl.signedDist(pt)),2)<9*pl.mse+1e-5) //point-plane distance within 3*std
					{
						if(trail>=0) {
							PlaneSeg& n_pl=*extractedPlanes[trail];
							if(pl.normalSimilarity(n_pl)>=params.T_ang(ParamSet::P_REFINE, pl.center[2])) {//potential for merging
								n_pl.connect(extractedPlanes[plid].get());
							}
						}
						float& old_dist=distMap[cIdx];
						if(cdist<old_dist) {
							trail=plid;
							old_dist=cdist;
							this->rfQueue.push_back(std::pair<int,int>(cIdx,plid));
						} else if(trail<0) {
							trail-=1;
						}
					} else {
						if(trail<0) trail-=1;
					}
				}
			}//for rfQueue
		}

		void floodFillWithTemporalFiltering(const cv::Mat& prevMembershipImg, float alpha = 0.5) {
			// prevMembershipImg is the segmentation map from the previous frame.
			std::vector<float> distMap(this->height * this->width, std::numeric_limits<float>::max());

			for (int k = 0; k < (int)this->rfQueue.size(); ++k) {
				const int sIdx = rfQueue[k].first;
				const int seedy = sIdx / this->width;
				const int seedx = sIdx - seedy * this->width;
				const int plid = rfQueue[k].second;
				const PlaneSeg& pl = *extractedPlanes[plid];

				int nbs[4] = { -1 };
				const int Nnbs = this->getValid4Neighbor(seedy, seedx, this->height, this->width, nbs);
				for (int itr = 0; itr < Nnbs; ++itr) {
					const int cIdx = nbs[itr];
					int& trail = membershipImg.at<int>(cIdx);
					if (trail <= -6) continue; // visited from 4 neighbors already, skip
					if (trail >= 0 && trail == plid) continue; // if visited by the same plane, skip

					const int prevLabel = prevMembershipImg.at<int>(cIdx);
					if (prevLabel == plid) {
						// Temporally stabilize: if the current plane matches the previous frame, prioritize it.
						trail = plid;
						continue;
					}

					double pt[3] = { 0 };
					float cdist = -1;
					if (this->points->get(seedy, seedx, pt[0], pt[1], pt[2]) &&
						std::pow(cdist = (float)std::abs(pl.signedDist(pt)), 2) < 9 * pl.mse + 1e-5) {

						float& old_dist = distMap[cIdx];
						if (cdist < old_dist) {
							// Apply temporal filtering by blending the current and previous segmentations
							trail = (alpha * plid + (1.0 - alpha) * prevLabel);
							old_dist = cdist;
							this->rfQueue.push_back(std::pair<int, int>(cIdx, plid));
						}
						else if (trail < 0) {
							trail -= 1;
						}
					}
					else {
						if (trail < 0) trail -= 1;
					}
				}
			}
		}

		/**
		 *  \brief erode each segment at initial block/window level
		 *
		 *  \param [in] isValidExtractedPlane coarsely extracted plane i is completely eroded if isValidExtractedPlane(i)==false
		 *
		 *  \details this function implements line 5~13 of Algorithm 4 in our paper, called by refineDetails; FIXME: after this ds is not updated, i.e. is dirtied
		 */
		/*void findBlockMembership(std::vector<bool>& isValidExtractedPlane) {
			rid2plid.clear();
			for (int plid = 0; plid < (int)extractedPlanes.size(); ++plid) {
				rid2plid.insert(std::pair<int, int>(extractedPlanes[plid]->rid, plid));
			}

			const int Nh = this->height / this->windowHeight;
			const int Nw = this->width / this->windowWidth;
			const int NptsPerBlk = this->windowHeight * this->windowWidth;

			membershipImg.create(height, width, CV_32SC1);
			membershipImg.setTo(-1);
			this->blkMap.resize(Nh * Nw);

			isValidExtractedPlane.resize(this->extractedPlanes.size(), false);
			for (int i = 0, blkid = 0; i < Nh; ++i) {
				for (int j = 0; j < Nw; ++j, ++blkid) {
					const int setid = ds->Find(blkid);
					const int setSize = ds->getSetSize(setid) * NptsPerBlk;
					if (setSize >= minSupport && !extractedPlanes[this->rid2plid[setid]]->nouse) { // Cluster large enough
						int nbs[4] = { -1 };
						const int nNbs = this->getValid4Neighbor(i, j, Nh, Nw, nbs);
						bool nbClsAllTheSame = true;
						for (int k = 0; k < nNbs && this->erodeType != ERODE_NONE; ++k) {
							if (ds->Find(nbs[k]) != setid &&
								(this->erodeType == ERODE_ALL_BORDER ||
									ds->getSetSize(nbs[k]) * NptsPerBlk >= minSupport)) {
								nbClsAllTheSame = false;
								break;
							}
						}
						const int plid = this->rid2plid[setid];
						if (nbClsAllTheSame) {
							this->blkMap[blkid] = plid;
							const int by = blkid / Nw;
							const int bx = blkid - by * Nw;
							membershipImg(cv::Range(by * windowHeight, (by + 1) * windowHeight),
								cv::Range(bx * windowWidth, (bx + 1) * windowWidth)).setTo(plid);
							isValidExtractedPlane[plid] = true;
						}
						else { // Erode border region or plane is not valid
							this->blkMap[blkid] = -1;
						}
					}
					else { // Too small cluster, i.e., "black" cluster
						this->blkMap[blkid] = -1;
					}

					// Save seed points for floodFill
					if (this->blkMap[blkid] < 0) { // Current block is not valid
						if (i > 0) {
							const int u_blkid = blkid - Nw;
							if (this->blkMap[u_blkid] >= 0) { // Up block is in border
								const int u_plid = this->blkMap[u_blkid];
								const int spixidx = (i * this->windowHeight - 1) * this->width + j * this->windowWidth;
								for (int k = 1; k < this->windowWidth; ++k) {
									this->rfQueue.push_back(std::pair<int, int>(spixidx + k, u_plid));
								}
							}
						}
						if (j > 0) {
							const int l_blkid = blkid - 1;
							if (this->blkMap[l_blkid] >= 0) { // Left block is in border
								const int l_plid = this->blkMap[l_blkid];
								const int spixidx = (i * this->windowHeight) * this->width + j * this->windowWidth - 1;
								for (int k = 0; k < this->windowHeight - 1; ++k) {
									this->rfQueue.push_back(std::pair<int, int>(spixidx + k * this->width, l_plid));
								}
							}
						}
					}
					else { // Current block is still valid
						const int plid = this->blkMap[blkid];
						if (i > 0) {
							const int u_blkid = blkid - Nw;
							if (this->blkMap[u_blkid] != plid) { // Up block is in border
								const int spixidx = (i * this->windowHeight) * this->width + j * this->windowWidth;
								for (int k = 0; k < this->windowWidth - 1; ++k) {
									this->rfQueue.push_back(std::pair<int, int>(spixidx + k, plid));
								}
							}
						}
						if (j > 0) {
							const int l_blkid = blkid - 1;
							if (this->blkMap[l_blkid] != plid) { // Left block is in border
								const int spixidx = (i * this->windowHeight) * this->width + j * this->windowWidth;
								for (int k = 1; k < this->windowHeight; ++k) {
									this->rfQueue.push_back(std::pair<int, int>(spixidx + k * this->width, plid));
								}
							}
						}
					}
				}
			}

			// Update plane equation
			for (int i = 0; i < (int)this->extractedPlanes.size(); ++i) {
				if (isValidExtractedPlane[i]) {
					if (this->extractedPlanes[i]->stats.N >= this->minSupport) {
						this->extractedPlanes[i]->update();
					}
				}
				else {
					this->extractedPlanes[i]->nouse = true;
				}
			}
		}*/

		void findBlockMembership(std::vector<bool>& isValidExtractedPlane) {
			rid2plid.clear();

			// Since we are only working with one plane, we can simplify this to handle only the farthest plane
			if (extractedPlanes.empty()) return; // No planes to process

			const PlaneSeg::shared_ptr farthest_plane = extractedPlanes[0]; // Assume the farthest plane is already determined
			rid2plid.insert(std::pair<int, int>(farthest_plane->rid, 0)); // Map its rid to 0 (as the only valid plane)

			const int Nh = this->height / this->windowHeight;
			const int Nw = this->width / this->windowWidth;
			const int NptsPerBlk = this->windowHeight * this->windowWidth;

			membershipImg.create(height, width, CV_32SC1);
			membershipImg.setTo(-1); // Set all to invalid by default
			this->blkMap.resize(Nh * Nw);

			isValidExtractedPlane.resize(1, false); // Only one plane is valid, so we only need space for one

			// Loop through each block in the image
			for (int i = 0, blkid = 0; i < Nh; ++i) {
				for (int j = 0; j < Nw; ++j, ++blkid) {
					const int setid = ds->Find(blkid);
					const int setSize = ds->getSetSize(setid) * NptsPerBlk;

					// Only process this block if it is large enough and belongs to the valid farthest plane
					if (setSize >= minSupport && !farthest_plane->nouse && ds->Find(farthest_plane->rid) == setid) {
						int nbs[4] = { -1 };
						const int nNbs = this->getValid4Neighbor(i, j, Nh, Nw, nbs);
						bool nbClsAllTheSame = true;

						// Check neighboring blocks to see if all belong to the same cluster
						for (int k = 0; k < nNbs && this->erodeType != ERODE_NONE; ++k) {
							if (ds->Find(nbs[k]) != setid &&
								(this->erodeType == ERODE_ALL_BORDER ||
									ds->getSetSize(nbs[k]) * NptsPerBlk >= minSupport)) {
								nbClsAllTheSame = false;
								break;
							}
						}

						if (nbClsAllTheSame) {
							this->blkMap[blkid] = 0; // Mark this block as part of the valid plane
							const int by = blkid / Nw;
							const int bx = blkid - by * Nw;
							membershipImg(cv::Range(by * windowHeight, (by + 1) * windowHeight),
								cv::Range(bx * windowWidth, (bx + 1) * windowWidth)).setTo(0); // Set pixels to the plane id
							isValidExtractedPlane[0] = true;
						}
						else {
							this->blkMap[blkid] = -1; // Erode the border region or invalid block
						}
					}
					else {
						this->blkMap[blkid] = -1; // Block is too small or invalid
					}

					// Save seed points for floodFill
					if (this->blkMap[blkid] < 0) { // Current block is not valid
						if (i > 0) {
							const int u_blkid = blkid - Nw;
							if (this->blkMap[u_blkid] >= 0) { // Up block is in border
								const int u_plid = this->blkMap[u_blkid];
								const int spixidx = (i * this->windowHeight - 1) * this->width + j * this->windowWidth;
								for (int k = 1; k < this->windowWidth; ++k) {
									this->rfQueue.push_back(std::pair<int, int>(spixidx + k, u_plid));
								}
							}
						}
						if (j > 0) {
							const int l_blkid = blkid - 1;
							if (this->blkMap[l_blkid] >= 0) { // Left block is in border
								const int l_plid = this->blkMap[l_blkid];
								const int spixidx = (i * this->windowHeight) * this->width + j * this->windowWidth - 1;
								for (int k = 0; k < this->windowHeight - 1; ++k) {
									this->rfQueue.push_back(std::pair<int, int>(spixidx + k * this->width, l_plid));
								}
							}
						}
					}
					else { // Current block is still valid
						const int plid = this->blkMap[blkid];
						if (i > 0) {
							const int u_blkid = blkid - Nw;
							if (this->blkMap[u_blkid] != plid) { // Up block is in border
								const int spixidx = (i * this->windowHeight) * this->width + j * this->windowWidth;
								for (int k = 0; k < this->windowWidth - 1; ++k) {
									this->rfQueue.push_back(std::pair<int, int>(spixidx + k, plid));
								}
							}
						}
						if (j > 0) {
							const int l_blkid = blkid - 1;
							if (this->blkMap[l_blkid] != plid) { // Left block is in border
								const int spixidx = (i * this->windowHeight) * this->width + j * this->windowWidth;
								for (int k = 1; k < this->windowHeight; ++k) {
									this->rfQueue.push_back(std::pair<int, int>(spixidx + k * this->width, plid));
								}
							}
						}
					}
				}
			}

			// Update the plane equation if the farthest plane is still valid
			/*if (isValidExtractedPlane[0] && farthest_plane->stats.N >= this->minSupport) {
				farthest_plane->update();
			}
			else {
				farthest_plane->nouse = true; // Mark the plane as invalid if it doesn't meet criteria
			}*/
		}

		void findBlockMembership() {
			if (!this->dirtyBlkMbship) return;
			this->dirtyBlkMbship = false;

			rid2plid.clear();
			for (int plid = 0; plid < (int)extractedPlanes.size(); ++plid) {
				rid2plid.insert(std::pair<int, int>(extractedPlanes[plid]->rid, plid));
			}

			const int Nh = this->height / this->windowHeight;
			const int Nw = this->width / this->windowWidth;
			const int NptsPerBlk = this->windowHeight * this->windowWidth;

			membershipImg.create(height, width, CV_32SC1);
			membershipImg.setTo(-1);
			this->blkMap.resize(Nh * Nw);

			for (int i = 0, blkid = 0; i < Nh; ++i) {
				for (int j = 0; j < Nw; ++j, ++blkid) {
					const int setid = ds->Find(blkid);
					const int setSize = ds->getSetSize(setid) * NptsPerBlk;
					if (setSize >= minSupport) {//cluster large enough
						const int plid = this->rid2plid[setid];
						this->blkMap[blkid] = plid;
						const int by = blkid / Nw;
						const int bx = blkid - by * Nw;
						membershipImg(cv::Range(by * windowHeight, (by + 1) * windowHeight),
							cv::Range(bx * windowWidth, (bx + 1) * windowWidth)).setTo(plid);
					}
					else {//too small cluster, i.e. "black" cluster
						this->blkMap[blkid] = -1;
					}//if setSize>=blkMinSupport
				}
			}//for blkik
		}

		//called by run when doRefine==false
		void findMembership(std::vector< std::vector<int> >& ret,
			const std::vector<int>* pIdxMap)
		{
			const int Nh   = this->height/this->windowHeight;
			const int Nw   = this->width/this->windowWidth;
			this->findBlockMembership();
			const int cnt = (int)extractedPlanes.size();
			ret.resize(cnt);
			for(int i=0; i<cnt; ++i) ret[i].reserve(extractedPlanes[i]->N);
			for(int i=0,blkid=0; i<Nh; ++i) {
				for(int j=0; j<Nw; ++j,++blkid) {
					const int plid=this->blkMap[blkid];
					if(plid<0) continue;
					for(int y=i*windowHeight; y<(i+1)*windowHeight; ++y) {
						for(int x=j*windowWidth; x<(j+1)*windowWidth; ++x) {
							const int pixIdx=x+y*width;
							ret[plid].push_back(pIdxMap?pIdxMap->at(pixIdx):pixIdx);
						}
					}
				}
			}
		}

		//called by run when doRefine==false
		void plotSegmentImage(cv::Mat* pSeg, const double supportTh)
		{
			if(pSeg==0) return;
			const int Nh   = this->height/this->windowHeight;
			const int Nw   = this->width/this->windowWidth;
			std::vector<int> ret;
			int cnt=0;

			std::vector<int>* pBlkid2plid;
			if(supportTh==this->minSupport) {
				this->findBlockMembership();
				pBlkid2plid=&(this->blkMap);
				cnt=(int)this->extractedPlanes.size();
			} else { //mainly for DEBUG_CLUSTER since then supportTh!=minSupport
				std::map<int, int> map; //map setid->cnt
				ret.resize(Nh*Nw);
				for(int i=0,blkid=0; i<Nh; ++i) {
					for(int j=0; j<Nw; ++j, ++blkid) {
						const int setid = ds->Find(blkid);
						const int setSize = ds->getSetSize(setid)*windowHeight*windowWidth;
						if(setSize>=supportTh) {
							std::map<int,int>::iterator fitr=map.find(setid);
							if(fitr==map.end()) {//found a new set id
								map.insert(std::pair<int,int>(setid,cnt));
								ret[blkid]=cnt;
								++cnt;
							} else {//found a existing set id
								ret[blkid]=fitr->second;
							}
						} else {//too small cluster, ignore
							ret[blkid]=-1;
						}
					}
				}
				pBlkid2plid=&ret;
			}
			std::vector<int>& blkid2plid=*pBlkid2plid;

			if(cnt>colors.size()) {
				std::vector<cv::Vec3b> tmpColors=pseudocolor(cnt-(int)colors.size());
				colors.insert(colors.end(), tmpColors.begin(), tmpColors.end());
			}
			cv::Mat& seg=*pSeg;
			for(int i=0,blkid=0; i<Nh; ++i) {
				for(int j=0; j<Nw; ++j,++blkid) {
					const int plid=blkid2plid[blkid];
					if(plid>=0) {
						seg(cv::Range(i*windowHeight,(i+1)*windowHeight),
							cv::Range(j*windowWidth, (j+1)*windowWidth)).setTo(colors[plid]);
					} else {
						seg(cv::Range(i*windowHeight,(i+1)*windowHeight),
							cv::Range(j*windowWidth, (j+1)*windowWidth)).setTo(cv::Vec3b(0,0,0));
					}
				}
			}
		}

#ifdef DEBUG_CLUSTER
		void floodFillColor(const int seedIdx, cv::Mat& seg, const cv::Vec3b& clr) {
			static const int step[8][2]={
				{1,0},{1,1},{0,1},{-1,1},
				{-1,0},{-1,-1},{0,-1},{1,-1}
			};
			const int Nh   = this->height/this->windowHeight;
			const int Nw   = this->width/this->windowWidth;

			std::vector<bool> visited(Nh*Nw, false);
			std::vector<int> idxStack;
			idxStack.reserve(Nh*Nw/10);
			idxStack.push_back(seedIdx);
			visited[seedIdx]=true;
			const int sy=seedIdx/Nw;
			const int sx=seedIdx-sy*Nw;
			seg(cv::Range(sy*windowHeight,(sy+1)*windowHeight),
				cv::Range(sx*windowWidth, (sx+1)*windowWidth)).setTo(clr);

			const int clsId=ds->Find(seedIdx);
			while(!idxStack.empty()) {
				const int sIdx=idxStack.back();
				idxStack.pop_back();
				const int seedy=sIdx/Nw;
				const int seedx=sIdx-seedy*Nw;
				for(int i=0; i<8; ++i) {
					const int cx=seedx+step[i][0];
					const int cy=seedy+step[i][1];
					if(0<=cx && cx<Nw && 0<=cy && cy<Nh) {
						const int cIdx=cx+cy*Nw;
						if(visited[cIdx]) continue; //if visited, skip
						visited[cIdx]=true;
						if(clsId==ds->Find(cIdx)) {//if same plane, move
							idxStack.push_back(cIdx);
							seg(cv::Range(cy*windowHeight,(cy+1)*windowHeight),
							cv::Range(cx*windowWidth, (cx+1)*windowWidth)).setTo(clr);
						}
					}
				}
			}//while
		}
#endif

#if defined(DEBUG_INIT) || defined(DEBUG_CLUSTER)
		cv::Mat dInit;
		cv::Mat dSeg;
		cv::Mat dGraph;
#endif

		/**
		 *  \brief initialize a graph from pointsIn
		 *
		 *  \param [in/out] minQ a min MSE queue of PlaneSegs
		 *
		 *  \details this function implements Algorithm 2 in our paper
		 */
		void initGraph(PlaneSegMinMSEQueue& minQ) {
			const int Nh   = this->height/this->windowHeight;
			const int Nw   = this->width/this->windowWidth;

			//1. init nodes
			std::vector<PlaneSeg::Ptr> G(Nh*Nw,0);
			//this->blkStats.resize(Nh*Nw);

#ifdef DEBUG_INIT
			dInit.create(this->height, this->width, CV_8UC3);
			dInit.setTo(cv::Vec3b(0,0,0));
#endif
			for(int i=0; i<Nh; ++i) {
				for(int j=0; j<Nw; ++j) {
					PlaneSeg::shared_ptr p( new PlaneSeg(
						*this->points, (i*Nw+j),
						i*this->windowHeight, j*this->windowWidth,
						this->width, this->height,
						this->windowWidth, this->windowHeight,
						this->params) );
					if(p->mse<params.T_mse(ParamSet::P_INIT, p->center[2])
						&& !p->nouse)
					{
						G[i*Nw+j]=p.get();
						minQ.push(p);
						//this->blkStats[i*Nw+j]=p->stats;
#ifdef DEBUG_INIT
						//const uchar cl=uchar(p->mse*255/dynThresh);
						//const cv::Vec3b clr(cl,cl,cl);
						dInit(cv::Range(i*windowHeight,(i+1)*windowHeight),
							cv::Range(j*windowWidth, (j+1)*windowWidth)).setTo(p->getColor(true));

						const int cx=j*windowWidth+0.5*(windowWidth-1);
						const int cy=i*windowHeight+0.5*(windowHeight-1);
						static const cv::Scalar blackColor(0,0,0,1);
						cv::circle(dInit, cv::Point(cx,cy), 1, blackColor, 2);
#endif
					} else {
						G[i*Nw+j]=0;
#ifdef DEBUG_INIT
						const int cx=j*windowWidth+0.5*(windowWidth-1);
						const int cy=i*windowHeight+0.5*(windowHeight-1);
						static const cv::Scalar blackColor(0,0,0,1);
						static const cv::Vec3b whiteColor(255,255,255);
						dInit(cv::Range(i*windowHeight,(i+1)*windowHeight),
							cv::Range(j*windowWidth, (j+1)*windowWidth)).setTo(whiteColor);

						switch(p->type) {
						case PlaneSeg::TYPE_NORMAL: //draw a big dot
							{
								static const cv::Scalar yellow(255,0,0,1);
								cv::circle(dInit, cv::Point(cx,cy), 3, yellow, 4);
								break;
							}
						case PlaneSeg::TYPE_MISSING_DATA: //draw an 'o'
							{
								static const cv::Scalar black(0,0,0,1);
								cv::circle(dInit, cv::Point(cx,cy), 3, black, 2);
								break;
							}
						case PlaneSeg::TYPE_DEPTH_DISCONTINUE: //draw an 'x'
							{
								static const cv::Scalar red(255,0,0,1);
								static const int len=4;
								cv::line(dInit, cv::Point(cx-len, cy-len), cv::Point(cx+len,cy+len), red, 2);
								cv::line(dInit, cv::Point(cx+len, cy-len), cv::Point(cx-len,cy+len), red, 2);
								break;
							}
						}
#endif
					}
				}
			}
#ifdef DEBUG_INIT
			//cv::applyColorMap(dInit, dInit,  cv::COLORMAP_COOL);
#endif
#ifdef DEBUG_CALC
			int nEdge=0;
			this->numEdges.clear();
			this->numNodes.clear();
#endif

			//2. init edges
			//first pass, connect neighbors from row direction
			for(int i=0; i<Nh; ++i) {
				for(int j=1; j<Nw; j+=2) {
					const int cidx=i*Nw+j;
					if(G[cidx-1]==0) { --j; continue; }
					if(G[cidx]==0) continue;
					if(j<Nw-1 && G[cidx+1]==0) { ++j; continue; }

					const double similarityTh=params.T_ang(ParamSet::P_INIT, G[cidx]->center[2]);
					if((j<Nw-1 && G[cidx-1]->normalSimilarity(*G[cidx+1])>=similarityTh) ||
						(j==Nw-1 && G[cidx]->normalSimilarity(*G[cidx-1])>=similarityTh)) {
							G[cidx]->connect(G[cidx-1]);
							if(j<Nw-1) G[cidx]->connect(G[cidx+1]);
#ifdef DEBUG_INIT
						const int cx=j*windowWidth+0.5*(windowWidth-1);
						const int cy=i*windowHeight+0.5*(windowHeight-1);
						const int rx=(j+1)*windowWidth+0.5*(windowWidth-1);
						const int lx=(j-1)*windowWidth+0.5*(windowWidth-1);
						static const cv::Scalar blackColor(0,0,0,1);
						cv::line(dInit, cv::Point(cx,cy), cv::Point(lx,cy),blackColor);
						if(j<Nw-1) cv::line(dInit, cv::Point(cx,cy), cv::Point(rx,cy),blackColor);
#endif
#ifdef DEBUG_CALC
						nEdge+=(j<Nw-1)?4:2;
#endif
					} else {//otherwise current block is in edge region
						--j;
					}
				}
			}
			//second pass, connect neighbors from column direction
			for(int j=0; j<Nw; ++j) {
				for(int i=1; i<Nh; i+=2) {
					const int cidx=i*Nw+j;
					if(G[cidx-Nw]==0) { --i; continue; }
					if(G[cidx]==0) continue;
					if(i<Nh-1 && G[cidx+Nw]==0) { ++i; continue; }

					const double similarityTh=params.T_ang(ParamSet::P_INIT, G[cidx]->center[2]);
					if((i<Nh-1 && G[cidx-Nw]->normalSimilarity(*G[cidx+Nw])>=similarityTh) ||
						(i==Nh-1 && G[cidx]->normalSimilarity(*G[cidx-Nw])>=similarityTh)) {
							G[cidx]->connect(G[cidx-Nw]);
							if(i<Nh-1) G[cidx]->connect(G[cidx+Nw]);
#ifdef DEBUG_INIT
						const int cx=j*windowWidth+0.5*(windowWidth-1);
						const int cy=i*windowHeight+0.5*(windowHeight-1);
						const int uy=(i-1)*windowHeight+0.5*(windowHeight-1);
						const int dy=(i+1)*windowHeight+0.5*(windowHeight-1);
						static const cv::Scalar blackColor(0,0,0,1);
						cv::line(dInit, cv::Point(cx,cy), cv::Point(cx,uy),blackColor);
						if(i<Nh-1) cv::line(dInit, cv::Point(cx,cy), cv::Point(cx,dy),blackColor);
#endif
#ifdef DEBUG_CALC
						nEdge+=(i<Nh-1)?4:2;
#endif
					} else {
						--i;
					}
				}
			}
#ifdef DEBUG_INIT
			static int cnt=0;
			cv::namedWindow("debug initGraph");
			cv::cvtColor(dInit,dInit,CV_RGB2BGR);
			cv::imshow("debug initGraph", dInit);
			std::stringstream ss;
			ss<<saveDir<<"/output/db_init"<<std::setw(5)<<std::setfill('0')<<cnt++<<".png";
			cv::imwrite(ss.str(), dInit);
#endif
#ifdef DEBUG_CALC
			this->numNodes.push_back(minQ.size());
			this->numEdges.push_back(nEdge);
			this->maxIndvidualNodeDegree=4;
			this->mseNodeDegree.clear();
#endif
		}

		/**
		 *  \brief main clustering step
		 *
		 *  \param [in] minQ a min MSE queue of PlaneSegs
		 *  \param [in] debug whether to collect some statistics when compiled with DEBUG_CALC
		 *  \return number of cluster steps
		 *
		 *  \details this function implements the Algorithm 3 in our paper
		 */

		 // Variable to store the history of the farthest plane
		FarthestPlaneHistory farthestPlaneHistory;

		// Smoothing factor (between 0 and 1)
		double alpha = 0.4;

		int ahCluster(PlaneSegMinMSEQueue& minQ, bool debug = true) {
			// Track the most distant plane
			float max_avg_depth = -std::numeric_limits<float>::infinity();
			PlaneSeg::shared_ptr farthest_plane = nullptr;

			int step = 0;
			while (!minQ.empty() && step <= maxStep) {
				PlaneSeg::shared_ptr p = minQ.top();
				minQ.pop();
				if (p->nouse) {
					continue;
				}

				PlaneSeg::shared_ptr cand_merge;
				PlaneSeg::Ptr cand_nb(0);
				PlaneSeg::NbSet::iterator itr = p->nbs.begin();
				for (; itr != p->nbs.end(); itr++) {
					PlaneSeg::Ptr nb = (*itr);
					if (p->normalSimilarity(*nb) < params.T_ang(ParamSet::P_MERGING, p->center[2])) continue;
					PlaneSeg::shared_ptr merge(new PlaneSeg(*p, *nb));
					if (cand_merge == 0 || cand_merge->mse > merge->mse || (cand_merge->mse == merge->mse && cand_merge->N < merge->mse)) {
						cand_merge = merge;
						cand_nb = nb;
					}
				}

				if (cand_merge != 0 && cand_merge->mse < params.T_mse(ParamSet::P_MERGING, cand_merge->center[2])) {
					minQ.push(cand_merge);
					cand_merge->mergeNbsFrom(*p, *cand_nb, *this->ds);
				}
				else {
					if (p->N >= this->minSupport) {
						// Calculate the average depth of the plane
						float sum_z = 0.0;
						int count_z = 0;
						for (int i = 0; i < p->N; ++i) {
							double x, y, z;
							if (this->points->get(p->rid / this->width, p->rid % this->width, x, y, z) && !std::isnan(z)) {
								sum_z += z;
								count_z++;
							}
						}
						if (count_z > 0) {
							float avg_depth = p->center[2];  // Use the z-component of the center as the average depth
							if (avg_depth > 0 && avg_depth > max_avg_depth) {
								max_avg_depth = avg_depth;
								farthest_plane = p;
								
								farthestPlaneHistory.updateWithNewPlane(*farthest_plane, alpha);

								// Update the farthest_plane's parameters with the filtered values
								farthest_plane->center[0] = farthestPlaneHistory.center[0];
								farthest_plane->center[1] = farthestPlaneHistory.center[1];
								farthest_plane->center[2] = farthestPlaneHistory.center[2];
								farthest_plane->normal[0] = farthestPlaneHistory.normal[0];
								farthest_plane->normal[1] = farthestPlaneHistory.normal[1];
								farthest_plane->normal[2] = farthestPlaneHistory.normal[2];
								farthest_plane->mse = farthestPlaneHistory.mse;
								farthest_plane->N = farthestPlaneHistory.N;
							}
						}
					}
					p->disconnectAllNbs();
				}
				++step;
			}

			while (!minQ.empty()) {//just check if any remaining PlaneSeg if maxstep reached
				const PlaneSeg::shared_ptr p = minQ.top();
				minQ.pop();
				if (p->nouse) {
					continue;
				}
				if (p->N >= this->minSupport) {
					// Calculate the average depth of the plane
					float sum_z = 0.0;
					int count_z = 0;
					for (int i = 0; i < p->N; ++i) {
						double x, y, z;
						if (this->points->get(p->rid / this->width, p->rid % this->width, x, y, z) && !std::isnan(z)) {
							sum_z += z;
							count_z++;
						}
					}
					if (count_z > 0) {
						float avg_depth = p->center[2];  // Use the z-component of the center as the average depth
						if (avg_depth > 0 && avg_depth > max_avg_depth) {
							max_avg_depth = avg_depth;
							farthest_plane = p;
							
							farthestPlaneHistory.updateWithNewPlane(*farthest_plane, alpha);

							// Update the farthest_plane's parameters with the filtered values
							farthest_plane->center[0] = farthestPlaneHistory.center[0];
							farthest_plane->center[1] = farthestPlaneHistory.center[1];
							farthest_plane->center[2] = farthestPlaneHistory.center[2];
							farthest_plane->normal[0] = farthestPlaneHistory.normal[0];
							farthest_plane->normal[1] = farthestPlaneHistory.normal[1];
							farthest_plane->normal[2] = farthestPlaneHistory.normal[2];
							farthest_plane->mse = farthestPlaneHistory.mse;
							farthest_plane->N = farthestPlaneHistory.N;
						}
					}
				}
				p->disconnectAllNbs();
			}

			if (farthest_plane != nullptr) {
				// Store the farthest plane in extractedPlanes
				this->extractedPlanes.clear();
				this->extractedPlanes.push_back(farthest_plane);
			}

			if (!this->extractedPlanes.empty()) {
				static PlaneSegSizeCmp sizecmp;
				std::sort(this->extractedPlanes.begin(),
					this->extractedPlanes.end(),
					sizecmp);
			}

			return step;
		}
	};//end of PlaneFitter
}//end of namespace ahc
