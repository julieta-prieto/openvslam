#include "openvslam/mapping_module.h"
#include "openvslam/global_optimization_module.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/frame.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/match/fuse.h"
#include "openvslam/util/converter.h"

#include "openvslam/IMU/configparam.h"
#include "openvslam/IMU/g2otypes.h"
#include <opencv2/core/eigen.hpp>

#include "openvslam/optimize/g2o/core/sparse_optimizer.h"
#include "openvslam/optimize/g2o/core/block_solver.h"
#include "openvslam/optimize/g2o/core/eigen_types.h"
#include "openvslam/optimize/g2o/solvers/linear_solver_eigen.h"
#include "openvslam/optimize/g2o/core/optimization_algorithm_gauss_newton.h"
#include "openvslam/optimize/g2o/core/optimization_algorithm_levenberg.h"
#include "openvslam/optimize/g2o/core/robust_kernel_impl.h"
#include "openvslam/optimize/g2o/types/types_sba.h"
#include "openvslam/optimize/g2o/solvers/linear_solver_cholmod.h"

#include <spdlog/spdlog.h>

namespace openvslam {

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
bool global_optimization_module::GetMapUpdateFlagForTracking()
{
    std::unique_lock<std::mutex> lock(mMutexMapUpdateFlag);
    return mbMapUpdateFlagForTracking;
}

void global_optimization_module::SetMapUpdateFlagInTracking(bool bflag)
{
    std::unique_lock<std::mutex> lock(mMutexMapUpdateFlag);
    mbMapUpdateFlagForTracking = bflag;
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

int global_optimization_module::PoseOptimization(data::frame *pFrame, data::keyframe* pLastKF, const IMUPreintegrator& imupreint, const cv::Mat& gw, const bool& bComputeMarg)
{
    #ifndef NOT_UPDATE_GYRO_BIAS
    static bool dbg2fopen=false;
    static ofstream dbg2fPoseOptPVRErr,dbg2fPoseOptBiasErr;
    if(!dbg2fopen)
    {
        dbg2fPoseOptPVRErr.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbg2fPoseOptPVRErr.txt");
        dbg2fPoseOptBiasErr.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbg2fPoseOptBiasErr.txt");
        if(dbg2fPoseOptPVRErr.is_open() && dbg2fPoseOptBiasErr.is_open())
        {
            std::cerr<<"file opened2"<<std::endl;
            dbg2fopen = true;
        }
        else
        {
            std::cerr<<"file open error in dbg2fPoseOpt"<<std::endl;
            dbg2fopen = false;
        }
        dbg2fPoseOptPVRErr<<std::fixed<<std::setprecision(6);
        dbg2fPoseOptBiasErr<<std::fixed<<std::setprecision(10);
    }
#endif

    // Extrinsics
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    // Gravity vector in world frame
    Vector3d GravityVec = util::converter::toVector3d(gw);

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    //linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    //g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    const int FramePVRId = 0;
    const int FrameBiasId = 1;
    const int LastKFPVRId = 2;
    const int LastKFBiasId = 3;

    // Set Frame vertex PVR/Bias
    g2o::VertexNavStatePVR * vNSFPVR = new g2o::VertexNavStatePVR();
    {
        vNSFPVR->setEstimate(pFrame->GetNavState());
        vNSFPVR->setId(FramePVRId);
        vNSFPVR->setFixed(false);
        optimizer.addVertex(vNSFPVR);
    }
    g2o::VertexNavStateBias * vNSFBias = new g2o::VertexNavStateBias();
    {
        vNSFBias->setEstimate(pFrame->GetNavState());
        vNSFBias->setId(FrameBiasId);
        vNSFBias->setFixed(false);
        optimizer.addVertex(vNSFBias);
    }

    // Set KeyFrame vertex PVR/Bias
    g2o::VertexNavStatePVR * vNSKFPVR = new g2o::VertexNavStatePVR();
    {
        vNSKFPVR->setEstimate(pLastKF->GetNavState());
        vNSKFPVR->setId(LastKFPVRId);
        vNSKFPVR->setFixed(true);
        optimizer.addVertex(vNSKFPVR);
    }
    g2o::VertexNavStateBias * vNSKFBias = new g2o::VertexNavStateBias();
    {
        vNSKFBias->setEstimate(pLastKF->GetNavState());
        vNSKFBias->setId(LastKFBiasId);
        vNSKFBias->setFixed(true);
        optimizer.addVertex(vNSKFBias);
    }

    // Set PVR edge between LastKF-Frame
    g2o::EdgeNavStatePVR* eNSPVR = new g2o::EdgeNavStatePVR();
    {
        eNSPVR->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastKFPVRId)));
        eNSPVR->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
        eNSPVR->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastKFBiasId)));
        eNSPVR->setMeasurement(imupreint);

        Matrix9d InvCovPVR = imupreint.getCovPVPhi().inverse() ;
        //eNSPVR->setInformation(InvCovPVR);
        Matrix9d addcov = Matrix9d::Identity();
        addcov.block<3,3>(0,0) *= 1e2;
        addcov.block<3,3>(3,3) *= 1;
        addcov.block<3,3>(6,6) *= 1e2;
        eNSPVR->setInformation(InvCovPVR + addcov);

        eNSPVR->SetParams(GravityVec);

        const float thHuberNavStatePVR = sqrt(21.666);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eNSPVR->setRobustKernel(rk);
        rk->setDelta(thHuberNavStatePVR);

        optimizer.addEdge(eNSPVR);
    }

    // Set Bias edge between LastKF-Frame
    g2o::EdgeNavStateBias* eNSBias = new g2o::EdgeNavStateBias();
    {
        eNSBias->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastKFBiasId)));
        eNSBias->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FrameBiasId)));
        eNSBias->setMeasurement(imupreint);

#ifndef NOT_UPDATE_GYRO_BIAS
        Matrix<double,6,6> InvCovBgaRW = Matrix<double,6,6>::Identity();
        InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
        InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE
#else
        Matrix<double,3,3> InvCovBgaRW = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE
#endif
        eNSBias->setInformation(InvCovBgaRW/imupreint.getDeltaTime());

        const float thHuberNavStateBias = sqrt(16.812);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eNSBias->setRobustKernel(rk);
        rk->setDelta(thHuberNavStateBias);

        optimizer.addEdge(eNSBias);
    }

    
    // Set MapPoint vertices
    const int N = pFrame->num_keypts_;

    std::vector<g2o::EdgeNavStatePVRPointXYZOnlyPose*> vpEdgesMono;
    std::vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    const float deltaMono = sqrt(5.991);

    {
        //std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);
        for(int i=0; i<N; i++)
        {
            data::landmark* pMP = pFrame->landmarks_[i];
            if(pMP)
            {
                // Monocular observation
                if(pFrame->stereo_x_right_[i]<0)
                {
                    nInitialCorrespondences++;
                    pFrame->outlier_flags_[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->undist_keypts_[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    //g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                    g2o::EdgeNavStatePVRPointXYZOnlyPose* e = new g2o::EdgeNavStatePVRPointXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->inv_level_sigma_sq_[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);
                    
                    e->SetParams(pFrame->fx,pFrame->fy,pFrame->cx,pFrame->cy,Rbc,Pbc,pMP->get_pos_in_world());

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else  // Stereo observation
                {
                    std::cerr<<"stereo shouldn't in poseoptimization"<<std::endl;
                }
            }
        }
    }

    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    //const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};

//    //Debug log
//    cout<<"total Points: "<<vpEdgesMono.size()<<endl;

    int nBad=0;

    for(size_t it=0; it<4; it++)
    {
        // Reset estimate for vertex
        vNSFPVR->setEstimate(pFrame->GetNavState());
        vNSFBias->setEstimate(pFrame->GetNavState());

        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeNavStatePVRPointXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->outlier_flags_[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->outlier_flags_[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->outlier_flags_[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        //        //Debug log
        //        cout<<nBad<<" bad Points in iter "<<it<<", rest points: "<<optimizer.edges().size()<<endl;

        if(optimizer.edges().size()<10)
            break;
    }

    // Debug log
    // if(eNSPVR->chi2()>21.666) cout<<"KF-F PVR edge chi2:"<<eNSPVR->chi2()<<endl;
    // if(eNSBias->chi2()>16.812) cout<<"KF-F Bias edge chi2:"<<eNSBias->chi2()<<endl;

#ifndef NOT_UPDATE_GYRO_BIAS
    {
        eNSPVR->computeError();
        Vector9d errPVR=eNSPVR->error();
        for(int n=0;n<9;n++)
            dbg2fPoseOptPVRErr<<errPVR[n]<<" ";
        dbg2fPoseOptPVRErr<<std::endl;

        eNSBias->computeError();
        Vector6d errBias=eNSBias->error();
        for(int n=0;n<6;n++)
            dbg2fPoseOptBiasErr<<errBias[n]<<" ";
        dbg2fPoseOptBiasErr<<std::endl;
    }
#endif

    // Recover optimized pose and return number of inliers
    g2o::VertexNavStatePVR* vNSPVR_recov = static_cast<g2o::VertexNavStatePVR*>(optimizer.vertex(FramePVRId));
    const NavState& nsPVR_recov = vNSPVR_recov->estimate();
    g2o::VertexNavStateBias* vNSBias_recov = static_cast<g2o::VertexNavStateBias*>(optimizer.vertex(FrameBiasId));
    const NavState& nsBias_recov = vNSBias_recov->estimate();
    NavState ns_recov = nsPVR_recov;
    ns_recov.Set_DeltaBiasGyr(nsBias_recov.Get_dBias_Gyr());
    ns_recov.Set_DeltaBiasAcc(nsBias_recov.Get_dBias_Acc());
    pFrame->SetNavState(ns_recov);
    pFrame->UpdatePoseFromNS(ConfigParam::GetMatTbc());

    // Compute marginalized Hessian H and B, H*x=B, H/B can be used as prior for next optimization in PoseOptimization
    if(bComputeMarg)
    {
        std::vector<g2o::OptimizableGraph::Vertex*> margVerteces;
        margVerteces.push_back(optimizer.vertex(FramePVRId));
        margVerteces.push_back(optimizer.vertex(FrameBiasId));

        //TODO: how to get the joint marginalized covariance of PVR&Bias
        g2o::SparseBlockMatrixXd spinv;
        optimizer.computeMarginals(spinv, margVerteces);
        // spinv include 2 blocks, 9x9-(0,0) for PVR, 6x6-(1,1) for Bias
#ifndef NOT_UPDATE_GYRO_BIAS
        Matrix<double,15,15> margCovInv = Matrix<double,15,15>::Zero();
        margCovInv.topLeftCorner(9,9) = spinv.block(0,0)->inverse();
        margCovInv.bottomRightCorner(6,6) = spinv.block(1,1)->inverse();
#else
        Matrix<double,12,12> margCovInv = Matrix<double,12,12>::Zero();
        margCovInv.topLeftCorner(9,9) = spinv.block(0,0)->inverse();
        margCovInv.bottomRightCorner(3,3) = spinv.block(1,1)->inverse();
#endif
        pFrame->mMargCovInv = margCovInv;
        pFrame->mNavStatePrior = ns_recov;

//        //Debug log
//        cout<<"marg result 1: "<<endl<<spinv<<endl;
//        cout<<"margCovInv: "<<endl<<pFrame->mMargCovInv<<endl;
//        cout<<"marg ns PV: "<<pFrame->mNavStatePrior.Get_P().transpose()<<" , "<<pFrame->mNavStatePrior.Get_V().transpose()<<endl;
//        cout<<"marg ns bg/dbg: "<<pFrame->mNavStatePrior.Get_BiasGyr().transpose()<<" , "<<pFrame->mNavStatePrior.Get_dBias_Gyr().transpose()<<endl;
//        cout<<"marg ns ba/dba: "<<pFrame->mNavStatePrior.Get_BiasAcc().transpose()<<" , "<<pFrame->mNavStatePrior.Get_dBias_Acc().transpose()<<endl;
    }

    //Test log
    if( (nsPVR_recov.Get_BiasGyr()-nsBias_recov.Get_BiasGyr()).norm()>1e-6 || (nsPVR_recov.Get_BiasAcc()-nsBias_recov.Get_BiasAcc()).norm()>1e-6 )
    {
        std::cerr<<"recovered bias gyr not equal for PVR/Bias vertex"<<std::endl<<nsPVR_recov.Get_BiasGyr().transpose()<<" / "<<nsBias_recov.Get_BiasGyr().transpose()<<std::endl;
        std::cerr<<"recovered bias acc not equal for PVR/Bias vertex"<<std::endl<<nsPVR_recov.Get_BiasAcc().transpose()<<" / "<<nsBias_recov.Get_BiasAcc().transpose()<<std::endl;
    }
    if( (ns_recov.Get_dBias_Gyr()-nsBias_recov.Get_dBias_Gyr()).norm()>1e-6 || (ns_recov.Get_dBias_Acc()-nsBias_recov.Get_dBias_Acc()).norm()>1e-6 )
    {
        std::cerr<<"recovered delta bias gyr not equal to Bias vertex"<<std::endl<<ns_recov.Get_dBias_Gyr().transpose()<<" / "<<nsBias_recov.Get_dBias_Gyr().transpose()<<std::endl;
        std::cerr<<"recovered delta bias acc not equal to Bias vertex"<<std::endl<<ns_recov.Get_dBias_Acc().transpose()<<" / "<<nsBias_recov.Get_dBias_Acc().transpose()<<std::endl;
    }

    return nInitialCorrespondences-nBad;
}

int global_optimization_module::PoseOptimization(data::frame *pFrame, data::frame* pLastFrame, const IMUPreintegrator& imupreint, const cv::Mat& gw, const bool& bComputeMarg)
{
    #ifndef NOT_UPDATE_GYRO_BIAS
    static bool dbg1fopen=false;
    static ofstream dbg1fPoseOptPVRErr,dbg1fPoseOptBiasErr,dbg1fPoseOptPriorErr;
    //if(dbgfop)
    if(!dbg1fopen)
    {cerr<<"try open 1"<<endl;
        dbg1fPoseOptPVRErr.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbg1fPoseOptPVRErr.txt");
        dbg1fPoseOptBiasErr.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbg1fPoseOptBiasErr.txt");
        dbg1fPoseOptPriorErr.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbg1fPoseOptPriorErr.txt");
        if(dbg1fPoseOptPVRErr.is_open() && dbg1fPoseOptBiasErr.is_open() && dbg1fPoseOptPriorErr.is_open() )
        {
            std::cerr<<"file opened1"<<std::endl;
            dbg1fopen = true;
        }
        else
        {
            std::cerr<<"file open error in dbg1fPoseOpt"<<std::endl;
            dbg1fopen = false;
        }
        dbg1fPoseOptPVRErr<<std::fixed<<std::setprecision(6);
        dbg1fPoseOptBiasErr<<std::fixed<<std::setprecision(10);
        dbg1fPoseOptPriorErr<<std::fixed<<std::setprecision(6);
    }
    #endif

    // Extrinsics
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    // Gravity vector in world frame
    Vector3d GravityVec = util::converter::toVector3d(gw);

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    //linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();
    linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    //g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    const int FramePVRId = 0;
    const int FrameBiasId = 1;
    const int LastFramePVRId = 2;
    const int LastFrameBiasId = 3;

    // Set Frame vertex PVR/Bias
    g2o::VertexNavStatePVR * vNSFPVR = new g2o::VertexNavStatePVR();
    {
        vNSFPVR->setEstimate(pFrame->GetNavState());
        vNSFPVR->setId(FramePVRId);
        vNSFPVR->setFixed(false);
        optimizer.addVertex(vNSFPVR);
    }
    g2o::VertexNavStateBias * vNSFBias = new g2o::VertexNavStateBias();
    {
        vNSFBias->setEstimate(pFrame->GetNavState());
        vNSFBias->setId(FrameBiasId);
        vNSFBias->setFixed(false);
        optimizer.addVertex(vNSFBias);
    }

    // Set LastFrame vertex
    g2o::VertexNavStatePVR * vNSFPVRlast = new g2o::VertexNavStatePVR();
    {
        vNSFPVRlast->setEstimate(pLastFrame->GetNavState());
        vNSFPVRlast->setId(LastFramePVRId);
        vNSFPVRlast->setFixed(false);
        optimizer.addVertex(vNSFPVRlast);
    }
    g2o::VertexNavStateBias * vNSFBiaslast = new g2o::VertexNavStateBias();
    {
        vNSFBiaslast->setEstimate(pLastFrame->GetNavState());
        vNSFBiaslast->setId(LastFrameBiasId);
        vNSFBiaslast->setFixed(false);
        optimizer.addVertex(vNSFBiaslast);
    }

    // Set prior edge for Last Frame, from mMargCovInv
    g2o::EdgeNavStatePriorPVRBias* eNSPrior = new g2o::EdgeNavStatePriorPVRBias();
    {
        eNSPrior->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFramePVRId)));
        eNSPrior->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFrameBiasId)));
        eNSPrior->setMeasurement(pLastFrame->mNavStatePrior);

        //eNSPrior->setInformation(pLastFrame->mMargCovInv);
#ifdef NOT_UPDATE_GYRO_BIAS
        Matrix<double,12,12> addcov = Matrix<double,12,12>::Identity();
        addcov.block<3,3>(0,0) *= 1e2;
        addcov.block<3,3>(3,3) *= 1;
        addcov.block<3,3>(6,6) *= 1e2;
        addcov.block<3,3>(9,9) *= 0;
        eNSPrior->setInformation(pLastFrame->mMargCovInv + addcov);
#else
        eNSPrior->setInformation(pLastFrame->mMargCovInv + Matrix<double,15,15>::Identity()*1e2);
#endif

        const float thHuberNavState = sqrt(30.5779);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eNSPrior->setRobustKernel(rk);
        rk->setDelta(thHuberNavState);

        optimizer.addEdge(eNSPrior);
    }

    // Set PVR edge between LastFrame-Frame
    g2o::EdgeNavStatePVR* eNSPVR = new g2o::EdgeNavStatePVR();
    {
        eNSPVR->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFramePVRId)));
        eNSPVR->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
        eNSPVR->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFrameBiasId)));
        eNSPVR->setMeasurement(imupreint);

        Matrix9d InvCovPVR = imupreint.getCovPVPhi().inverse() ;
        //eNSPVR->setInformation(InvCovPVR);
        Matrix9d addcov = Matrix9d::Identity();
        addcov.block<3,3>(0,0) *= 1e2;
        addcov.block<3,3>(3,3) *= 1;
        addcov.block<3,3>(6,6) *= 1e2;
        eNSPVR->setInformation(InvCovPVR + addcov);

        eNSPVR->SetParams(GravityVec);

        const float thHuberNavStatePVR = sqrt(21.666);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eNSPVR->setRobustKernel(rk);
        rk->setDelta(thHuberNavStatePVR);

        optimizer.addEdge(eNSPVR);
    }
    // Set Bias edge between LastFrame-Frame
    g2o::EdgeNavStateBias* eNSBias = new g2o::EdgeNavStateBias();
    {
        eNSBias->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFrameBiasId)));
        eNSBias->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FrameBiasId)));
        eNSBias->setMeasurement(imupreint);
#ifndef NOT_UPDATE_GYRO_BIAS
        Matrix<double,6,6> InvCovBgaRW = Matrix<double,6,6>::Identity();
        InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
        InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE
#else
        Matrix<double,3,3> InvCovBgaRW = Matrix3d::Identity()/IMUData::getAccBiasRW2();// Accelerometer bias random walk, covariance INVERSE
#endif
        eNSBias->setInformation(InvCovBgaRW/imupreint.getDeltaTime());

        const float thHuberNavStateBias = sqrt(16.812);
        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        eNSBias->setRobustKernel(rk);
        rk->setDelta(thHuberNavStateBias);

        optimizer.addEdge(eNSBias);
    }
/**/
    // Set MapPoint vertices
    const int Ncur = pFrame->num_keypts_;
    const int Nlast = pLastFrame->num_keypts_;

    std::vector<g2o::EdgeNavStatePVRPointXYZOnlyPose*> vpEdgesMono;
    std::vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(Ncur);
    vnIndexEdgeMono.reserve(Ncur);

    std::vector<g2o::EdgeNavStatePVRPointXYZOnlyPose*> vpEdgesMonoLast;
    std::vector<size_t> vnIndexEdgeMonoLast;
    vpEdgesMonoLast.reserve(Nlast);
    vnIndexEdgeMonoLast.reserve(Nlast);


    const float deltaMono = sqrt(5.991);
    {
        for(int i=0; i<Ncur; i++)
        {
            data::landmark* pMP = pFrame->landmarks_[i];
            if(pMP)
            {
                // Monocular observation
                if(pFrame->stereo_x_right_[i]<0)
                {
                    nInitialCorrespondences++;
                    pFrame->outlier_flags_[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    const cv::KeyPoint &kpUn = pFrame->undist_keypts_[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    //g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                    g2o::EdgeNavStatePVRPointXYZOnlyPose* e = new g2o::EdgeNavStatePVRPointXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(FramePVRId)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->inv_level_sigma_sq_[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->SetParams(pFrame->fx,pFrame->fy,pFrame->cx,pFrame->cy,Rbc,Pbc,pMP->get_pos_in_world());

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
                else  // Stereo observation
                {
                    std::cerr<<"stereo shouldn't in poseoptimization"<<std::endl;
                }
            }
        }

        // Add Point-Pose edges for last frame
        for(int i=0; i<Nlast; i++)
        {
            data::landmark* pMP = pLastFrame->landmarks_[i];
            if(pMP)
            {
                // Monocular observation
                if(pLastFrame->stereo_x_right_[i]<0)
                {
                    //nInitialCorrespondences++;
                    pLastFrame->outlier_flags_[i] = false;

                    Eigen::Matrix<double,2,1> obs;
                    const cv::KeyPoint &kpUn = pLastFrame->undist_keypts_[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    //g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                    g2o::EdgeNavStatePVRPointXYZOnlyPose* e = new g2o::EdgeNavStatePVRPointXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(LastFramePVRId)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pLastFrame->inv_level_sigma_sq_[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);
 
                    e->SetParams(pLastFrame->fx,pLastFrame->fy,pLastFrame->cx,pLastFrame->cy,Rbc,Pbc,pMP->get_pos_in_world());

                    optimizer.addEdge(e);

                    vpEdgesMonoLast.push_back(e);
                    vnIndexEdgeMonoLast.push_back(i);
                }
                else  // Stereo observation
                {
                    std::cerr<<"stereo shouldn't in poseoptimization"<<std::endl;
                }
            }
        }
    }
    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    //const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};

//    //Debug log
//    cout<<"total Points: "<<vpEdgesMono.size()<<endl;

    int nBad=0;
    int nBadLast=0;

    for(size_t it=0; it<4; it++)
    {
        // Reset estimates
        vNSFPVR->setEstimate(pFrame->GetNavState());
        vNSFBias->setEstimate(pFrame->GetNavState());
        vNSFPVRlast->setEstimate(pLastFrame->GetNavState());
        vNSFBiaslast->setEstimate(pLastFrame->GetNavState());
//cerr<<"before opt, ";
        //optimizer.setVerbose(true);
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

//cerr<<"after opt, iter: "<<it;
        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeNavStatePVRPointXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->outlier_flags_[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pFrame->outlier_flags_[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->outlier_flags_[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        nBadLast=0;
        for(size_t i=0, iend=vpEdgesMonoLast.size(); i<iend; i++)
        {
            g2o::EdgeNavStatePVRPointXYZOnlyPose* e = vpEdgesMonoLast[i];

            const size_t idx = vnIndexEdgeMonoLast[i];

            if(pLastFrame->outlier_flags_[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                pLastFrame->outlier_flags_[idx]=true;
                e->setLevel(1);
                nBadLast++;
            }
            else
            {
                pLastFrame->outlier_flags_[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        //        //Debug log
        //        cout<<nBad<<" bad Points in iter "<<it<<", rest points: "<<optimizer.edges().size()<<endl;
        //        cout<<nBadLast<<" bad Points of last Frame in iter "<<it<<endl;
        //        cout<<"NavState edge chi2: "<<eNS->chi2()<<endl;

        //if(vpEdgesMono.size() - nBad < 10)
        if(optimizer.edges().size()<10)
            break;
    }

    // Debug log
    // if(eNSPVR->chi2()>21.666) cout<<"F-F PVR edge chi2:"<<eNSPVR->chi2()<<endl;
    // if(eNSBias->chi2()>16.812) cout<<"F-F Bias edge chi2:"<<eNSBias->chi2()<<endl;
    // if(eNSPrior->chi2()>30.5779) cout<<"F-F Prior edge chi2:"<<eNSPrior->chi2()<<endl;
#ifndef NOT_UPDATE_GYRO_BIAS
    {
        eNSPVR->computeError();
        Vector9d errPVR=eNSPVR->error();
        for(int n=0;n<9;n++)
            dbg1fPoseOptPVRErr<<errPVR[n]<<" ";
        dbg1fPoseOptPVRErr<<std::endl;

        eNSBias->computeError();
        Vector6d errBias=eNSBias->error();
        for(int n=0;n<6;n++)
            dbg1fPoseOptBiasErr<<errBias[n]<<" ";
        dbg1fPoseOptBiasErr<<std::endl;

        eNSPrior->computeError();
        Vector15d errPrior=eNSPrior->error();
        for(int n=0;n<15;n++)
            dbg1fPoseOptPriorErr<<errPrior[n]<<" ";
        dbg1fPoseOptPriorErr<<std::endl;
    }
#endif

    // Recover optimized pose and return number of inliers
    g2o::VertexNavStatePVR* vNSPVR_recov = static_cast<g2o::VertexNavStatePVR*>(optimizer.vertex(FramePVRId));
    const NavState& nsPVR_recov = vNSPVR_recov->estimate();
    g2o::VertexNavStateBias* vNSBias_recov = static_cast<g2o::VertexNavStateBias*>(optimizer.vertex(FrameBiasId));
    const NavState& nsBias_recov = vNSBias_recov->estimate();
    NavState ns_recov = nsPVR_recov;
    ns_recov.Set_DeltaBiasGyr(nsBias_recov.Get_dBias_Gyr());
    ns_recov.Set_DeltaBiasAcc(nsBias_recov.Get_dBias_Acc());
    pFrame->SetNavState(ns_recov);
    pFrame->UpdatePoseFromNS(ConfigParam::GetMatTbc());

    // Compute marginalized Hessian H and B, H*x=B, H/B can be used as prior for next optimization in PoseOptimization
    if(bComputeMarg)
    {
        std::vector<g2o::OptimizableGraph::Vertex*> margVerteces;
        margVerteces.push_back(optimizer.vertex(FramePVRId));
        margVerteces.push_back(optimizer.vertex(FrameBiasId));
//
        //TODO: how to get the joint marginalized covariance of PVR&Bias
        g2o::SparseBlockMatrixXd spinv;
        optimizer.computeMarginals(spinv, margVerteces);
        //cout<<"marg result 2: "<<endl<<spinv<<endl;
        // spinv include 2 blocks, 9x9-(0,0) for PVR, 6x6-(1,1) for Bias
#ifndef NOT_UPDATE_GYRO_BIAS
        Matrix<double,15,15> margCov = Matrix<double,15,15>::Zero();
        margCov.topLeftCorner(9,9) = spinv.block(0,0)->eval();
        margCov.topRightCorner(9,6) = spinv.block(0,1)->eval();
        margCov.bottomLeftCorner(6,9) = spinv.block(1,0)->eval();
        margCov.bottomRightCorner(6,6) = spinv.block(1,1)->eval();
#else
        Matrix<double,12,12> margCov = Matrix<double,12,12>::Zero();
        margCov.topLeftCorner(9,9) = spinv.block(0,0)->eval();
        margCov.topRightCorner(9,3) = spinv.block(0,1)->eval();
        margCov.bottomLeftCorner(3,9) = spinv.block(1,0)->eval();
        margCov.bottomRightCorner(3,3) = spinv.block(1,1)->eval();
#endif
        //        margCovInv.topLeftCorner(9,9) = spinv.block(0,0)->inverse();
        //        margCovInv.bottomRightCorner(6,6) = spinv.block(1,1)->inverse();
        pFrame->mMargCovInv = margCov.inverse();
        pFrame->mNavStatePrior = ns_recov;

        //Debug log
        //cout<<"inv MargCov 2: "<<endl<<pFrame->mMargCovInv<<endl;
    }

    //Test log
    if( (nsPVR_recov.Get_BiasGyr()-nsBias_recov.Get_BiasGyr()).norm()>1e-6 || (nsPVR_recov.Get_BiasAcc()-nsBias_recov.Get_BiasAcc()).norm()>1e-6 )
    {
        std::cerr<<"1 recovered bias gyr not equal for PVR/Bias vertex"<<std::endl<<nsPVR_recov.Get_BiasGyr().transpose()<<" / "<<nsBias_recov.Get_BiasGyr().transpose()<<std::endl;
        std::cerr<<"1 recovered bias acc not equal for PVR/Bias vertex"<<std::endl<<nsPVR_recov.Get_BiasAcc().transpose()<<" / "<<nsBias_recov.Get_BiasAcc().transpose()<<std::endl;
    }
    if( (ns_recov.Get_dBias_Gyr()-nsBias_recov.Get_dBias_Gyr()).norm()>1e-6 || (ns_recov.Get_dBias_Acc()-nsBias_recov.Get_dBias_Acc()).norm()>1e-6 )
    {
        std::cerr<<"1 recovered delta bias gyr not equal to Bias vertex"<<std::endl<<ns_recov.Get_dBias_Gyr().transpose()<<" / "<<nsBias_recov.Get_dBias_Gyr().transpose()<<std::endl;
        std::cerr<<"1 recovered delta bias acc not equal to Bias vertex"<<std::endl<<ns_recov.Get_dBias_Acc().transpose()<<" / "<<nsBias_recov.Get_dBias_Acc().transpose()<<std::endl;
    }

    return nInitialCorrespondences-nBad;

}


void global_optimization_module::GlobalBundleAdjustmentNavState(data::map_database* pMap, const cv::Mat& gw, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
{
    std::vector<data::keyframe*> vpKFs = pMap->get_all_keyframes();
    std::vector<data::landmark*> vpMP = pMap->get_all_landmarks();

    // Extrinsics
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    // Gravity vector in world frame
    Vector3d GravityVec = util::converter::toVector3d(gw);

    std::vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    if(pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    long unsigned int maxKFid = 0;

    // Set KeyFrame vertices
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        data::keyframe* pKF = vpKFs[i];
        if(pKF->will_be_erased())
            continue;

        // PVR
        g2o::VertexNavStatePVR * vNSPVR = new g2o::VertexNavStatePVR();
        vNSPVR->setEstimate(pKF->GetNavState());
        vNSPVR->setId(pKF->id_*2);
        vNSPVR->setFixed(pKF->id_==0);
        optimizer.addVertex(vNSPVR);
        // Bias
        g2o::VertexNavStateBias * vNSBias = new g2o::VertexNavStateBias();
        vNSBias->setEstimate(pKF->GetNavState());
        vNSBias->setId(pKF->id_*2+1);
        vNSBias->setFixed(pKF->id_==0);
        optimizer.addVertex(vNSBias);

        if(pKF->id_*2+1>maxKFid)
            maxKFid=pKF->id_*2+1;
    }

    // Add NavState PVR/Bias edges
    const float thHuberNavStatePVR = sqrt(21.666);
    const float thHuberNavStateBias = sqrt(16.812);
    // Inverse covariance of bias random walk
#ifndef NOT_UPDATE_GYRO_BIAS
    Matrix<double,6,6> InvCovBgaRW = Matrix<double,6,6>::Identity();
    InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
    InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE
#else
    Matrix<double,3,3> InvCovBgaRW = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE
#endif

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        data::keyframe* pKF1 = vpKFs[i];
        if(pKF1->will_be_erased())
        {
            std::cout<<"pKF is bad in gBA, id "<<pKF1->id_<<std::endl;   //Debug log
            continue;
        }

        data::keyframe* pKF0 = pKF1->GetPrevKeyFrame();
        if(!pKF0)
        {
            if(pKF1->id_!=0) std::cerr<<"Previous KeyFrame is NULL?"<<std::endl;  //Test log
            continue;
        }

        // PVR edge
        {
            g2o::EdgeNavStatePVR * epvr = new g2o::EdgeNavStatePVR();
            epvr->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->id_)));
            epvr->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->id_)));
            epvr->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->id_+1)));
            epvr->setMeasurement(pKF1->GetIMUPreInt());

            Matrix9d InvCovPVR = pKF1->GetIMUPreInt().getCovPVPhi().inverse();
            epvr->setInformation(InvCovPVR);
            epvr->SetParams(GravityVec);

            if(bRobust)
            {
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                epvr->setRobustKernel(rk);
                rk->setDelta(thHuberNavStatePVR);
            }

            optimizer.addEdge(epvr);
        }
        // Bias edge
        {
            g2o::EdgeNavStateBias * ebias = new g2o::EdgeNavStateBias();
            ebias->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->id_+1)));
            ebias->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->id_+1)));
            ebias->setMeasurement(pKF1->GetIMUPreInt());

            ebias->setInformation(InvCovBgaRW/pKF1->GetIMUPreInt().getDeltaTime());

            if(bRobust)
            {
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                ebias->setRobustKernel(rk);
                rk->setDelta(thHuberNavStateBias);
            }

            optimizer.addEdge(ebias);
        }

    }

    const float thHuber2D = sqrt(5.99);

    // Set MapPoint vertices
    for(size_t i=0; i<vpMP.size(); i++)
    {
        data::landmark* pMP = vpMP[i];
        if(pMP->will_be_erased())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        cv::Mat aux;
        eigen2cv(pMP->get_pos_in_world(),aux);
        vPoint->setEstimate(util::converter::toVector3d(aux));
        const int id = pMP->id_+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

       const std::map<data::keyframe*,unsigned int> observations = pMP->get_observations();

        int nEdges = 0;
        //SET EDGES
        for(std::map<data::keyframe*,unsigned int>::const_iterator mit=observations.begin(); mit!=observations.end(); mit++)
        {

            data::keyframe* pKF = mit->first;
            if(pKF->will_be_erased() || 2*pKF->id_>maxKFid)
                continue;

            nEdges++;

            const cv::KeyPoint &kpUn = pKF->undist_keypts_[mit->second];

            if(pKF->stereo_x_right_[mit->second]<0)
            {
                Eigen::Matrix<double,2,1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeNavStatePVRPointXYZ* e = new g2o::EdgeNavStatePVRPointXYZ();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF->id_)));
                e->setMeasurement(obs);
                const float &invSigma2 = pKF->inv_level_sigma_sq_[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                if(bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                cv::Mat cam_param = ConfigParam::GetCamMatrix();
                float fx = cam_param.at<float>(0,0);
                float fy = cam_param.at<float>(1,1);
                float cx = cam_param.at<float>(0,2);
                float cy = cam_param.at<float>(1,2);
                e->SetParams(fx,fy,cx,cy,Rbc,Pbc);

                optimizer.addEdge(e);
            }
            else
            {
                std::cerr<<"Stereo not supported"<<std::endl;
            }
        }

        if(nEdges==0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i]=true;
        }
        else
        {
            vbNotIncludedMP[i]=false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    //Keyframes
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        data::keyframe* pKF = vpKFs[i];
        if(pKF->will_be_erased())
            continue;
        //g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->id_));
        //g2o::SE3Quat SE3quat = vSE3->estimate();
        g2o::VertexNavStatePVR* vNSPVR = static_cast<g2o::VertexNavStatePVR*>(optimizer.vertex(2*pKF->id_));
        g2o::VertexNavStateBias* vNSBias = static_cast<g2o::VertexNavStateBias*>(optimizer.vertex(2*pKF->id_+1));
        const NavState& nspvr = vNSPVR->estimate();
        const NavState& nsbias = vNSBias->estimate();
        NavState ns_recov = nspvr;
        ns_recov.Set_DeltaBiasGyr(nsbias.Get_dBias_Gyr());
        ns_recov.Set_DeltaBiasAcc(nsbias.Get_dBias_Acc());

        if(nLoopKF==0)
        {
            //pKF->SetPose(Converter::toCvMat(SE3quat));
            pKF->SetNavState(ns_recov);
            pKF->UpdatePoseFromNS(ConfigParam::GetMatTbc());
        }
        else
        {
            pKF->mNavStateGBA = ns_recov;

            //pKF->cam_pose_cw_after_loop_BA_.create(4,4,CV_32F);
            //Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
            cv::Mat Twb_ = cv::Mat::eye(4,4,CV_32F);
            util::converter::toCvMat(pKF->mNavStateGBA.Get_RotMatrix()).copyTo(Twb_.rowRange(0,3).colRange(0,3));
            util::converter::toCvMat(pKF->mNavStateGBA.Get_P()).copyTo(Twb_.rowRange(0,3).col(3));
            cv::Mat Twc_ = Twb_*ConfigParam::GetMatTbc();
            //pKF->cam_pose_cw_after_loop_BA_ = util::converter::toCvMatInverse(Twc_);
            cv2eigen(util::converter::toCvMatInverse(Twc_), pKF->cam_pose_cw_after_loop_BA_);

            pKF->loop_BA_identifier_ = nLoopKF;
        }
    }

    //Points
    for(size_t i=0; i<vpMP.size(); i++)
    {
        if(vbNotIncludedMP[i])
            continue;

        data::landmark* pMP = vpMP[i];

        if(pMP->will_be_erased())
            continue;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->id_+maxKFid+1));

        if(nLoopKF==0)
        {
            Vec3_t v;
            cv2eigen(util::converter::toCvMat(vPoint->estimate()), v);
            pMP->set_pos_in_world(v);

            //pMP->SetWorldPos(util::converter::toCvMat(vPoint->estimate()));
            pMP->update_normal_and_depth();
        }
        else
        {
            //pMP->mPosGBA.create(3,1,CV_32F);
            cv::Mat mat;
            mat.create(3,1,CV_32F);
            util::converter::toCvMat(vPoint->estimate()).copyTo(mat);
            cv2eigen(mat, pMP->pos_w_after_global_BA_);
            pMP->loop_BA_identifier_ = nLoopKF;
        }
    }

}




Vector3d global_optimization_module::OptimizeInitialGyroBias(const std::vector<data::frame>& vFrames)
{
    //size_t N = vpKFs.size();
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rcb = Tbc.topLeftCorner(3,3).transpose();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Add vertex of gyro bias, to optimizer graph
    g2o::VertexGyrBias * vBiasg = new g2o::VertexGyrBias();
    vBiasg->setEstimate(Eigen::Vector3d::Zero());
    vBiasg->setId(0);
    optimizer.addVertex(vBiasg);
    size_t N = vFrames.size();
    // Add unary edges for gyro bias vertex
    for(size_t i=0; N; i++)
    {
        // Only 19 edges between 20 Frames
        if(i==0)
            continue;

        const data::frame& Fi = vFrames[i-1];
        const data::frame& Fj = vFrames[i];


        cv::Mat Tiw;
        eigen2cv(Fi.cam_pose_cw_, Tiw);      // pose of previous KF
        Eigen::Matrix3d Rwci = util::converter::toMatrix3d(Tiw.rowRange(0,3).colRange(0,3).t());
        cv::Mat Tjw;      // pose of this KF
        eigen2cv(Fj.cam_pose_cw_, Tjw);
        Eigen::Matrix3d Rwcj = util::converter::toMatrix3d(Tjw.rowRange(0,3).colRange(0,3).t());

        //
        IMUPreintegrator imupreint;
        Fj.ComputeIMUPreIntSinceLastFrame(&Fi,imupreint);

        g2o::EdgeGyrBias * eBiasg = new g2o::EdgeGyrBias();
        eBiasg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        // measurement is not used in EdgeGyrBias
        eBiasg->dRbij = imupreint.getDeltaR();
        eBiasg->J_dR_bg = imupreint.getJRBiasg();
        eBiasg->Rwbi = Rwci*Rcb;
        eBiasg->Rwbj = Rwcj*Rcb;
        eBiasg->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(eBiasg);
    }
    // It's actualy a linear estimator, so 1 iteration is enough.
    //optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(1);

    g2o::VertexGyrBias * vBgEst = static_cast<g2o::VertexGyrBias*>(optimizer.vertex(0));

    return vBgEst->estimate();
}


Eigen::Vector3d global_optimization_module::OptimizeInitialGyroBias(const std::vector<cv::Mat>& vTwc, const std::vector<IMUPreintegrator>& vImuPreInt)
{
    int N = vTwc.size(); if(vTwc.size()!=vImuPreInt.size()) std::cerr<<"vTwc.size()!=vImuPreInt.size()"<<std::endl;
    Eigen::Matrix4d Tbc = ConfigParam::GetEigTbc();
    Eigen::Matrix3d Rcb = Tbc.topLeftCorner(3,3).transpose();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Add vertex of gyro bias, to optimizer graph
    g2o::VertexGyrBias * vBiasg = new g2o::VertexGyrBias();
    vBiasg->setEstimate(Eigen::Vector3d::Zero());
    vBiasg->setId(0);
    optimizer.addVertex(vBiasg);

    // Add unary edges for gyro bias vertex
    //for(std::vector<KeyFrame*>::const_iterator lit=vpKFs.begin(), lend=vpKFs.end(); lit!=lend; lit++)
    for(int i=0; i<N; i++)
    {
        // Ignore the first KF
        if(i==0)
            continue;

        const cv::Mat& Twi = vTwc[i-1];    // pose of previous KF
        Matrix3d Rwci = util::converter::toMatrix3d(Twi.rowRange(0,3).colRange(0,3));
        //Matrix3d Rwci = Twi.rotation_matrix();
        const cv::Mat& Twj = vTwc[i];        // pose of this KF
        Matrix3d Rwcj = util::converter::toMatrix3d(Twj.rowRange(0,3).colRange(0,3));
        //Matrix3d Rwcj =Twj.rotation_matrix();

        const IMUPreintegrator& imupreint = vImuPreInt[i];

        g2o::EdgeGyrBias * eBiasg = new g2o::EdgeGyrBias();
        eBiasg->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        // measurement is not used in EdgeGyrBias
        eBiasg->dRbij = imupreint.getDeltaR();
        eBiasg->J_dR_bg = imupreint.getJRBiasg();
        eBiasg->Rwbi = Rwci*Rcb;
        eBiasg->Rwbj = Rwcj*Rcb;
        //eBiasg->setInformation(Eigen::Matrix3d::Identity());
        eBiasg->setInformation(imupreint.getCovPVPhi().bottomRightCorner(3,3));
        optimizer.addEdge(eBiasg);
    }

    // It's actualy a linear estimator, so 1 iteration is enough.
    //optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(1);

    g2o::VertexGyrBias * vBgEst = static_cast<g2o::VertexGyrBias*>(optimizer.vertex(0));

    return vBgEst->estimate();
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
global_optimization_module::global_optimization_module(data::map_database* map_db, data::bow_database* bow_db,
                                                       data::bow_vocabulary* bow_vocab, const bool fix_scale, ConfigParam* pParams)
    : loop_detector_(new module::loop_detector(bow_db, bow_vocab, fix_scale)),
      loop_bundle_adjuster_(new module::loop_bundle_adjuster(map_db)),
      graph_optimizer_(new optimize::graph_optimizer(map_db, fix_scale)) {
    mpParams = pParams;
    spdlog::debug("CONSTRUCT: global_optimization_module");
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

global_optimization_module::~global_optimization_module() {
    abort_loop_BA();
    if (thread_for_loop_BA_) {
        thread_for_loop_BA_->join();
    }
    spdlog::debug("DESTRUCT: global_optimization_module");
}

void global_optimization_module::set_tracking_module(tracking_module* tracker) {
    tracker_ = tracker;
}

void global_optimization_module::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
    loop_bundle_adjuster_->set_mapping_module(mapper);
}

void global_optimization_module::enable_loop_detector() {
    spdlog::info("enable loop detector");
    loop_detector_->enable_loop_detector();
}

void global_optimization_module::disable_loop_detector() {
    spdlog::info("disable loop detector");
    loop_detector_->disable_loop_detector();
}

bool global_optimization_module::loop_detector_is_enabled() const {
    return loop_detector_->is_enabled();
}

void global_optimization_module::run() {
    spdlog::info("start global optimization module");

    is_terminated_ = false;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // check if termination is requested
        if (terminate_is_requested()) {
            // terminate and break
            terminate();
            break;
        }

        // check if pause is requested
        if (pause_is_requested()) {
            // pause and wait
            pause();
            // check if termination or reset is requested during pause
            while (is_paused() && !terminate_is_requested() && !reset_is_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }

        // check if reset is requested
        if (reset_is_requested()) {
            // reset and continue
            reset();
            continue;
        }

        // if the queue is empty, the following process is not needed
        if (!keyframe_is_queued()) {
            continue;
        }

        // dequeue the keyframe from the queue -> cur_keyfrm_
        {
            std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
            cur_keyfrm_ = keyfrms_queue_.front();
            keyfrms_queue_.pop_front();
        }

        // not to be removed during loop detection and correction
        cur_keyfrm_->set_not_to_be_erased();

        // pass the current keyframe to the loop detector
        loop_detector_->set_current_keyframe(cur_keyfrm_);

        // detect some loop candidate with BoW
        if (!loop_detector_->detect_loop_candidates()) {
            // could not find
            // allow the removal of the current keyframe
            cur_keyfrm_->set_to_be_erased();
            continue;
        }

        // validate candidates and select ONE candidate from them
        if (!loop_detector_->validate_candidates()) {
            // could not find
            // allow the removal of the current keyframe
            cur_keyfrm_->set_to_be_erased();
            continue;
        }

        correct_loop();
    }

    spdlog::info("terminate global optimization module");
}

void global_optimization_module::queue_keyframe(data::keyframe* keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    if (keyfrm->id_ != 0) {
        keyfrms_queue_.push_back(keyfrm);
    }
}

bool global_optimization_module::keyframe_is_queued() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return (!keyfrms_queue_.empty());
}

void global_optimization_module::correct_loop() {
    auto final_candidate_keyfrm = loop_detector_->get_selected_candidate_keyframe();

    spdlog::info("detect loop: keyframe {} - keyframe {}", final_candidate_keyfrm->id_, cur_keyfrm_->id_);
    loop_bundle_adjuster_->count_loop_BA_execution();

    // 0. pre-processing

    // 0-1. stop the mapping module and the previous loop bundle adjuster

    // pause the mapping module
    mapper_->request_pause();
    // abort the previous loop bundle adjuster
    if (thread_for_loop_BA_ || loop_bundle_adjuster_->is_running()) {
        abort_loop_BA();
    }
    // wait till the mapping module pauses
    while (!mapper_->is_paused()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // 0-2. update the graph

    cur_keyfrm_->graph_node_->update_connections();

    // 1. compute the Sim3 of the covisibilities of the current keyframe whose Sim3 is already estimated by the loop detector
    //    then, the covisibilities are moved to the corrected positions
    //    finally, landmarks observed in them are also moved to the correct position using the camera poses before and after camera pose correction

    // acquire the covisibilities of the current keyframe
    std::vector<data::keyframe*> curr_neighbors = cur_keyfrm_->graph_node_->get_covisibilities();
    curr_neighbors.push_back(cur_keyfrm_);

    // Sim3 camera poses BEFORE loop correction
    module::keyframe_Sim3_pairs_t Sim3s_nw_before_correction;
    // Sim3 camera poses AFTER loop correction
    module::keyframe_Sim3_pairs_t Sim3s_nw_after_correction;

    const auto g2o_Sim3_cw_after_correction = loop_detector_->get_Sim3_world_to_current();
    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        // camera pose of the current keyframe BEFORE loop correction
        const Mat44_t cam_pose_wc_before_correction = cur_keyfrm_->get_cam_pose_inv();

        // compute Sim3s BEFORE loop correction
        Sim3s_nw_before_correction = get_Sim3s_before_loop_correction(curr_neighbors);
        // compute Sim3s AFTER loop correction
        Sim3s_nw_after_correction = get_Sim3s_after_loop_correction(cam_pose_wc_before_correction, g2o_Sim3_cw_after_correction, curr_neighbors);

        // correct covibisibility landmark positions
        correct_covisibility_landmarks(Sim3s_nw_before_correction, Sim3s_nw_after_correction);
        // correct covisibility keyframe camera poses
        correct_covisibility_keyframes(Sim3s_nw_after_correction);
    }

    // 2. resolve duplications of landmarks caused by loop fusion

    const auto curr_match_lms_observed_in_cand = loop_detector_->current_matched_landmarks_observed_in_candidate();
    replace_duplicated_landmarks(curr_match_lms_observed_in_cand, Sim3s_nw_after_correction);

    // 3. extract the new connections created after loop fusion

    const auto new_connections = extract_new_connections(curr_neighbors);

    // 4. pose graph optimization
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    graph_optimizer_->optimize(final_candidate_keyfrm, cur_keyfrm_, Sim3s_nw_before_correction, Sim3s_nw_after_correction, new_connections);
    /*if(pLC)
    {
        pLC->SetMapUpdateFlagInTracking(true);
    }*/
    SetMapUpdateFlagInTracking(true);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // add a loop edge
    final_candidate_keyfrm->graph_node_->add_loop_edge(cur_keyfrm_);
    cur_keyfrm_->graph_node_->add_loop_edge(final_candidate_keyfrm);

    // 5. launch loop BA

    while (loop_bundle_adjuster_->is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
    if (thread_for_loop_BA_) {
        thread_for_loop_BA_->join();
        thread_for_loop_BA_.reset(nullptr);
    }
    thread_for_loop_BA_ = std::unique_ptr<std::thread>(new std::thread(&module::loop_bundle_adjuster::optimize, loop_bundle_adjuster_.get(), cur_keyfrm_->id_));
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // Map updated, set flag for Tracking
    SetMapUpdateFlagInTracking(true);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // 6. post-processing

    // resume the mapping module
    mapper_->resume();

    // set the loop fusion information to the loop detector
    loop_detector_->set_loop_correct_keyframe_id(cur_keyfrm_->id_);
}

module::keyframe_Sim3_pairs_t global_optimization_module::get_Sim3s_before_loop_correction(const std::vector<data::keyframe*>& neighbors) const {
    module::keyframe_Sim3_pairs_t Sim3s_nw_before_loop_correction;

    for (const auto neighbor : neighbors) {
        // camera pose of `neighbor` BEFORE loop correction
        const Mat44_t cam_pose_nw = neighbor->get_cam_pose();
        // create Sim3 from SE3
        const Mat33_t& rot_nw = cam_pose_nw.block<3, 3>(0, 0);
        const Vec3_t& trans_nw = cam_pose_nw.block<3, 1>(0, 3);
        const g2o::Sim3 Sim3_nw_before_correction(rot_nw, trans_nw, 1.0);
        Sim3s_nw_before_loop_correction[neighbor] = Sim3_nw_before_correction;
    }

    return Sim3s_nw_before_loop_correction;
}

module::keyframe_Sim3_pairs_t global_optimization_module::get_Sim3s_after_loop_correction(const Mat44_t& cam_pose_wc_before_correction,
                                                                                          const g2o::Sim3& g2o_Sim3_cw_after_correction,
                                                                                          const std::vector<data::keyframe*>& neighbors) const {
    module::keyframe_Sim3_pairs_t Sim3s_nw_after_loop_correction;

    for (auto neighbor : neighbors) {
        // camera pose of `neighbor` BEFORE loop correction
        const Mat44_t cam_pose_nw_before_correction = neighbor->get_cam_pose();
        // create the relative Sim3 from the current to `neighbor`
        const Mat44_t cam_pose_nc = cam_pose_nw_before_correction * cam_pose_wc_before_correction;
        const Mat33_t& rot_nc = cam_pose_nc.block<3, 3>(0, 0);
        const Vec3_t& trans_nc = cam_pose_nc.block<3, 1>(0, 3);
        const g2o::Sim3 Sim3_nc(rot_nc, trans_nc, 1.0);
        // compute the camera poses AFTER loop correction of the neighbors
        const g2o::Sim3 Sim3_nw_after_correction = Sim3_nc * g2o_Sim3_cw_after_correction;
        Sim3s_nw_after_loop_correction[neighbor] = Sim3_nw_after_correction;
    }

    return Sim3s_nw_after_loop_correction;
}

void global_optimization_module::correct_covisibility_landmarks(const module::keyframe_Sim3_pairs_t& Sim3s_nw_before_correction,
                                                                const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction) const {
    for (const auto& t : Sim3s_nw_after_correction) {
        auto neighbor = t.first;
        // neighbor->world AFTER loop correction
        const auto Sim3_wn_after_correction = t.second.inverse();
        // world->neighbor BEFORE loop correction
        const auto& Sim3_nw_before_correction = Sim3s_nw_before_correction.at(neighbor);

        const auto ngh_landmarks = neighbor->get_landmarks();
        for (auto lm : ngh_landmarks) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // avoid duplication
            if (lm->loop_fusion_identifier_ == cur_keyfrm_->id_) {
                continue;
            }
            lm->loop_fusion_identifier_ = cur_keyfrm_->id_;

            // correct position of `lm`
            const Vec3_t pos_w_before_correction = lm->get_pos_in_world();
            const Vec3_t pos_w_after_correction = Sim3_wn_after_correction.map(Sim3_nw_before_correction.map(pos_w_before_correction));
            lm->set_pos_in_world(pos_w_after_correction);
            // update geometry
            lm->update_normal_and_depth();

            // record the reference keyframe used in loop fusion of landmarks
            lm->ref_keyfrm_id_in_loop_fusion_ = neighbor->id_;
        }
    }
}

void global_optimization_module::correct_covisibility_keyframes(const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction) const {
    for (const auto& t : Sim3s_nw_after_correction) {
        auto neighbor = t.first;
        const auto Sim3_nw_after_correction = t.second;

        const auto s_nw = Sim3_nw_after_correction.scale();
        const Mat33_t rot_nw = Sim3_nw_after_correction.rotation().toRotationMatrix();
        const Vec3_t trans_nw = Sim3_nw_after_correction.translation() / s_nw;
        const Mat44_t cam_pose_nw = util::converter::to_eigen_cam_pose(rot_nw, trans_nw);
        neighbor->set_cam_pose(cam_pose_nw);

        // update graph
        neighbor->graph_node_->update_connections();
    }
}

void global_optimization_module::replace_duplicated_landmarks(const std::vector<data::landmark*>& curr_match_lms_observed_in_cand,
                                                              const module::keyframe_Sim3_pairs_t& Sim3s_nw_after_correction) const {
    // resolve duplications of landmarks between the current keyframe and the loop candidate
    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        for (unsigned int idx = 0; idx < cur_keyfrm_->num_keypts_; ++idx) {
            auto curr_match_lm_in_cand = curr_match_lms_observed_in_cand.at(idx);
            if (!curr_match_lm_in_cand) {
                continue;
            }

            auto lm_in_curr = cur_keyfrm_->get_landmark(idx);
            if (lm_in_curr) {
                // if the landmark corresponding `idx` exists,
                // replace it with `curr_match_lm_in_cand` (observed in the candidate)
                lm_in_curr->replace(curr_match_lm_in_cand);
            }
            else {
                // if landmark corresponding `idx` does not exists,
                // add association between the current keyframe and `curr_match_lm_in_cand`
                cur_keyfrm_->add_landmark(curr_match_lm_in_cand, idx);
                curr_match_lm_in_cand->add_observation(cur_keyfrm_, idx);
                curr_match_lm_in_cand->compute_descriptor();
            }
        }
    }

    // resolve duplications of landmarks between the current keyframe and the candidates of the loop candidate
    const auto curr_match_lms_observed_in_cand_covis = loop_detector_->current_matched_landmarks_observed_in_candidate_covisibilities();
    match::fuse fuser(0.8);
    for (const auto& t : Sim3s_nw_after_correction) {
        auto neighbor = t.first;
        const Mat44_t Sim3_nw_after_correction = util::converter::to_eigen_mat(t.second);

        // reproject the landmarks observed in the current keyframe to the neighbor,
        // then search duplication of the landmarks
        std::vector<data::landmark*> lms_to_replace(curr_match_lms_observed_in_cand_covis.size(), nullptr);
        fuser.detect_duplication(neighbor, Sim3_nw_after_correction, curr_match_lms_observed_in_cand_covis, 4, lms_to_replace);

        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
        // if any landmark duplication is found, replace it
        for (unsigned int i = 0; i < curr_match_lms_observed_in_cand_covis.size(); ++i) {
            auto lm_to_replace = lms_to_replace.at(i);
            if (lm_to_replace) {
                lm_to_replace->replace(curr_match_lms_observed_in_cand_covis.at(i));
            }
        }
    }
}

auto global_optimization_module::extract_new_connections(const std::vector<data::keyframe*>& covisibilities) const
    -> std::map<data::keyframe*, std::set<data::keyframe*>> {
    std::map<data::keyframe*, std::set<data::keyframe*>> new_connections;

    for (auto covisibility : covisibilities) {
        // acquire neighbors BEFORE loop fusion (because update_connections() is not called yet)
        const auto neighbors_before_update = covisibility->graph_node_->get_covisibilities();

        // call update_connections()
        covisibility->graph_node_->update_connections();
        // acquire neighbors AFTER loop fusion
        new_connections[covisibility] = covisibility->graph_node_->get_connected_keyframes();

        // remove covisibilities
        for (const auto keyfrm_to_erase : covisibilities) {
            new_connections.at(covisibility).erase(keyfrm_to_erase);
        }
        // remove nighbors before loop fusion
        for (const auto keyfrm_to_erase : neighbors_before_update) {
            new_connections.at(covisibility).erase(keyfrm_to_erase);
        }
    }

    return new_connections;
}

void global_optimization_module::request_reset() {
    {
        std::lock_guard<std::mutex> lock(mtx_reset_);
        reset_is_requested_ = true;
    }

    // BLOCK until reset
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mtx_reset_);
            if (!reset_is_requested_) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(3000));
    }
}

bool global_optimization_module::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

void global_optimization_module::reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    spdlog::info("reset global optimization module");
    keyfrms_queue_.clear();
    loop_detector_->set_loop_correct_keyframe_id(0);
    reset_is_requested_ = false;
}

void global_optimization_module::request_pause() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    pause_is_requested_ = true;
}

bool global_optimization_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_is_requested_;
}

bool global_optimization_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
}

void global_optimization_module::pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    spdlog::info("pause global optimization module");
    is_paused_ = true;
}

void global_optimization_module::resume() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);

    // if it has been already terminated, cannot resume
    if (is_terminated_) {
        return;
    }

    is_paused_ = false;
    pause_is_requested_ = false;

    spdlog::info("resume global optimization module");
}

void global_optimization_module::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}

bool global_optimization_module::is_terminated() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return is_terminated_;
}

bool global_optimization_module::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void global_optimization_module::terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    is_terminated_ = true;
}

bool global_optimization_module::loop_BA_is_running() const {
    return loop_bundle_adjuster_->is_running();
}

void global_optimization_module::abort_loop_BA() {
    loop_bundle_adjuster_->abort();
}

} // namespace openvslam
