#include "openvslam/type.h"
#include "openvslam/mapping_module.h"
#include "openvslam/global_optimization_module.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/match/fuse.h"
#include "openvslam/match/robust.h"
#include "openvslam/module/two_view_triangulator.h"
#include "openvslam/solve/essential_solver.h"

#include "openvslam/util/converter.h"
#include "openvslam/IMU/configparam.h"
#include <opencv2/core/eigen.hpp>

#include <unordered_set>
#include <thread>

#include <spdlog/spdlog.h>

namespace openvslam {

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
class KeyFrameInit
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double mTimeStamp;
    KeyFrameInit* mpPrevKeyFrame;
    cv::Mat Twc;
    IMUPreintegrator mIMUPreInt;
    std::vector<IMUData> mvIMUData;
    Vector3d bg;

    KeyFrameInit(data::keyframe& kf):
        mTimeStamp(kf.timestamp_), mpPrevKeyFrame(NULL), //Twc(util::converter::Mat44_t_to_cvMat4(kf.get_cam_pose_inv()).clone()),
        mIMUPreInt(kf.GetIMUPreInt()), mvIMUData(kf.GetVectorIMUData()), bg(0,0,0)
    {
        Mat44_t m = kf.get_cam_pose_inv();
        Twc = (cv::Mat_<float>(4,4) << m(0,0), m(0,1), m(0,2), m(0,3),
                                       m(1,0), m(1,1), m(1,2), m(1,3),
                                       m(2,0), m(2,1), m(2,2), m(2,3),
                                       m(3,0), m(3,1), m(3,2), m(3,3));
    }

    void ComputePreInt(void)
    {
        if(mpPrevKeyFrame == NULL)
        {
            return;
        }
        else
        {
            // Reset pre-integrator first
            mIMUPreInt.reset();

            if(mvIMUData.empty())
                return;

            // remember to consider the gap between the last KF and the first IMU
            {
                const IMUData& imu = mvIMUData.front();
                double dt = std::max(0., imu._t - mpPrevKeyFrame->mTimeStamp);
                mIMUPreInt.update(imu._g - bg,imu._a ,dt);  // Acc bias not considered here
            }
            // integrate each imu
            for(size_t i=0; i<mvIMUData.size(); i++)
            {
                const IMUData& imu = mvIMUData[i];
                double nextt;
                if(i==mvIMUData.size()-1)
                    nextt = mTimeStamp;         // last IMU, next is this KeyFrame
                else
                    nextt = mvIMUData[i+1]._t;  // regular condition, next is imu data

                // delta time
                double dt = std::max(0., nextt - imu._t);
                // update pre-integrator
                mIMUPreInt.update(imu._g - bg,imu._a ,dt);
            }
        }
    }
};

bool mapping_module::GetUpdatingInitPoses(void)
{
    std::unique_lock<std::mutex> lock(mMutexUpdatingInitPoses);
    return mbUpdatingInitPoses;
}

void mapping_module::SetUpdatingInitPoses(bool flag)
{
    std::unique_lock<std::mutex> lock(mMutexUpdatingInitPoses);
    mbUpdatingInitPoses = flag;
}

data::keyframe* mapping_module::GetMapUpdateKF()
{
    std::unique_lock<std::mutex> lock(mMutexMapUpdateFlag);
    return mpMapUpdateKF;
}

bool mapping_module::GetMapUpdateFlagForTracking()
{
    std::unique_lock<std::mutex> lock(mMutexMapUpdateFlag);
    return mbMapUpdateFlagForTracking;
}

void mapping_module::SetMapUpdateFlagInTracking(bool bflag)
{
    std::unique_lock<std::mutex> lock(mMutexMapUpdateFlag);
    mbMapUpdateFlagForTracking = bflag;
    if(bflag)
    {
        mpMapUpdateKF = cur_keyfrm_;
    }
}

bool mapping_module::GetVINSInited(void)
{
    std::unique_lock<std::mutex> lock(mMutexVINSInitFlag);
    return mbVINSInited;
}

void mapping_module::SetVINSInited(bool flag)
{
    std::unique_lock<std::mutex> lock(mMutexVINSInitFlag);
    mbVINSInited = flag;
}

bool mapping_module::GetFirstVINSInited(void)
{
    std::unique_lock<std::mutex> lock(mMutexFirstVINSInitFlag);
    return mbFirstVINSInited;
}

void mapping_module::SetFirstVINSInited(bool flag)
{
    std::unique_lock<std::mutex> lock(mMutexFirstVINSInitFlag);
    mbFirstVINSInited = flag;
}

cv::Mat mapping_module::GetGravityVec()
{
    return mGravityVec.clone();
}

cv::Mat mapping_module::GetRwiInit()
{
    return mRwiInit.clone();
}

void mapping_module::VINSInitThread()
{
    unsigned long initedid = 0;
    spdlog::info("start VINSInitThread");
    while(1)
    {
        if(data::keyframe::next_id_ > 2)
            if(!GetVINSInited() && cur_keyfrm_->id_ > initedid)
            {
                initedid = cur_keyfrm_->id_;

                bool tmpbool = TryInitVIO();
                if(tmpbool)
                {
                    break;
                }
            }
        usleep(3000);
        if(is_terminated())
            break;
    }
}

bool mapping_module::TryInitVIO(void)
{
    if(map_db_->get_num_keyframes()<=mnLocalWindowSize)
        return false;

    static bool fopened = false;
    static std::ofstream fgw,fscale,fbiasa,fcondnum,ftime,fbiasg;
    if(!fopened)
    {
        fgw.open("../../../tmp/gw.txt");
        fscale.open("../../../tmp/scale.txt");
        fbiasa.open("../../../tmp/biasa.txt");
        fcondnum.open("../../../tmp/condnum.txt");
        ftime.open("../../../tmp/computetime.txt");
        fbiasg.open("../../../tmp/biasg.txt");
        if(fgw.is_open() && fscale.is_open() && fbiasa.is_open() &&
                fcondnum.is_open() && ftime.is_open() && fbiasg.is_open())
            fopened = true;
        else
        {
            std::cerr<<"file open error in TryInitVIO"<<std::endl;
            fopened = false;
        }
        fgw<<std::fixed<<std::setprecision(6);
        fscale<<std::fixed<<std::setprecision(6);
        fbiasa<<std::fixed<<std::setprecision(6);
        fcondnum<<std::fixed<<std::setprecision(6);
        ftime<<std::fixed<<std::setprecision(6);
        fbiasg<<std::fixed<<std::setprecision(6);
    }

    //Optimizer::GlobalBundleAdjustemnt(mpMap, 10);

    // Extrinsics
    cv::Mat Tbc = ConfigParam::GetMatTbc();
    cv::Mat Rbc = Tbc.rowRange(0,3).colRange(0,3);
    cv::Mat pbc = Tbc.rowRange(0,3).col(3);
    cv::Mat Rcb = Rbc.t();
    cv::Mat pcb = -Rcb*pbc;

    #ifdef RUN_REALTIME
        // Wait KeyFrame Culling.
        // 1. if KeyFrame Culling is running, wait until finished.
        // 2. if KFs are being copied, then don't run KeyFrame Culling (in KeyFrameCulling function)
        while(GetFlagCopyInitKFs())
        {
            usleep(1000);
        }
    #endif

    SetFlagCopyInitKFs(true);

    // Use all KeyFrames in map to compute
    std::vector<data::keyframe*> vScaleGravityKF = map_db_->get_all_keyframes();
    int N = vScaleGravityKF.size();
    data::keyframe* pNewestKF = vScaleGravityKF[N-1];
    std::vector<cv::Mat> vTwc;
    std::vector<IMUPreintegrator> vIMUPreInt;
    // Store initialization-required KeyFrame data
    std::vector<KeyFrameInit*> vKFInit;

    for(int i=0;i<N;i++)
    {
        data::keyframe* pKF = vScaleGravityKF[i];
        Mat44_t mat = pKF->get_cam_pose_inv();
        cv::Mat vTwc_add;
        vTwc_add = (cv::Mat_<float>(4,4) << mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                                            mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                                            mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                                            mat(3,0), mat(3,1), mat(3,2), mat(3,3));
        vTwc.push_back(vTwc_add);
        vIMUPreInt.push_back(pKF->GetIMUPreInt());
        KeyFrameInit* pkfi = new KeyFrameInit (*pKF);
        if(i!=0)
        {
            pkfi->mpPrevKeyFrame = vKFInit[i-1];
        }
        vKFInit.push_back(pkfi);
    }

    SetFlagCopyInitKFs(false);

    // Step 1.
    // Try to compute initial gyro bias, using optimization with Gauss-Newton
    Vector3d bgest = global_optimization_module::OptimizeInitialGyroBias(vTwc,vIMUPreInt);
    //Vector3d bgest = Optimizer::OptimizeInitialGyroBias(vScaleGravityKF);

    // Update biasg and pre-integration in LocalWindow. Remember to reset back to zero
    for(int i=0;i<N;i++)
    {
        vKFInit[i]->bg = bgest;
    }
    for(int i=0;i<N;i++)
    {
        vKFInit[i]->ComputePreInt();
    }

    // Debug log
    //cout<<std::fixed<<std::setprecision(6)<<"estimated gyr bias: "<<bgest.transpose()<<endl;
    //std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase

//    // Update biasg and pre-integration in LocalWindow. Remember to reset back to zero
//    for(vector<KeyFrame*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
//    {
//        KeyFrame* pKF = *vit;
//        pKF->SetNavStateBiasGyr(bgest);
//    }
//    for(vector<KeyFrame*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
//    {
//        KeyFrame* pKF = *vit;
//        pKF->ComputePreInt();
//    }

    // Solve A*x=B for x=[s,gw] 4x1 vector
    cv::Mat A = cv::Mat::zeros(3*(N-2),4,CV_32F);
    cv::Mat B = cv::Mat::zeros(3*(N-2),1,CV_32F);
    cv::Mat I3 = cv::Mat::eye(3,3,CV_32F);

    // Step 2.
    // Approx Scale and Gravity vector in 'world' frame (first KF's camera frame)
    for(int i=0; i<N-2; i++)
    {
        //KeyFrameInit* pKF1 = vKFInit[i];//vScaleGravityKF[i];
        KeyFrameInit* pKF2 = vKFInit[i+1];
        KeyFrameInit* pKF3 = vKFInit[i+2];
        // Delta time between frames
        double dt12 = pKF2->mIMUPreInt.getDeltaTime();
        double dt23 = pKF3->mIMUPreInt.getDeltaTime();
        // Pre-integrated measurements
        cv::Mat dp12 = util::converter::toCvMat(pKF2->mIMUPreInt.getDeltaP());
        cv::Mat dv12 = util::converter::toCvMat(pKF2->mIMUPreInt.getDeltaV());
        cv::Mat dp23 = util::converter::toCvMat(pKF3->mIMUPreInt.getDeltaP());

        // Pose of camera in world frame
        cv::Mat Twc1 = vTwc[i].clone();//pKF1->GetPoseInverse();
        cv::Mat Twc2 = vTwc[i+1].clone();//pKF2->GetPoseInverse();
        cv::Mat Twc3 = vTwc[i+2].clone();//pKF3->GetPoseInverse();
        // Position of camera center
        cv::Mat pc1 = Twc1.rowRange(0,3).col(3);
        cv::Mat pc2 = Twc2.rowRange(0,3).col(3);
        cv::Mat pc3 = Twc3.rowRange(0,3).col(3);
        // Rotation of camera, Rwc
        cv::Mat Rc1 = Twc1.rowRange(0,3).colRange(0,3);
        cv::Mat Rc2 = Twc2.rowRange(0,3).colRange(0,3);
        cv::Mat Rc3 = Twc3.rowRange(0,3).colRange(0,3);

        // Stack to A/B matrix
        // lambda*s + beta*g = gamma
        cv::Mat lambda = (pc2-pc1)*dt23 + (pc2-pc3)*dt12;
        cv::Mat beta = 0.5*I3*(dt12*dt12*dt23 + dt12*dt23*dt23);
        cv::Mat gamma = (Rc3-Rc2)*pcb*dt12 + (Rc1-Rc2)*pcb*dt23 + Rc1*Rcb*dp12*dt23 - Rc2*Rcb*dp23*dt12 - Rc1*Rcb*dv12*dt12*dt23;
        lambda.copyTo(A.rowRange(3*i+0,3*i+3).col(0));
        beta.copyTo(A.rowRange(3*i+0,3*i+3).colRange(1,4));
        gamma.copyTo(B.rowRange(3*i+0,3*i+3));
        // Tested the formulation in paper, -gamma. Then the scale and gravity vector is -xx

        // Debug log
        //cout<<"iter "<<i<<endl;
    }

    // Use svd to compute A*x=B, x=[s,gw] 4x1 vector
    // A = u*w*vt, u*w*vt*x=B
    // Then x = vt'*winv*u'*B
    cv::Mat w,u,vt;
    // Note w is 4x1 vector by SVDecomp()
    // A is changed in SVDecomp() with cv::SVD::MODIFY_A for speed
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A);
    // Debug log
    //cout<<"u:"<<endl<<u<<endl;
    //cout<<"vt:"<<endl<<vt<<endl;
    //cout<<"w:"<<endl<<w<<endl;

    // Compute winv
    cv::Mat winv=cv::Mat::eye(4,4,CV_32F);
    for(int i=0;i<4;i++)
    {
        if(fabs(w.at<float>(i))<1e-10)
        {
            w.at<float>(i) += 1e-10;
            // Test log
            std::cerr<<"w(i) < 1e-10, w="<<std::endl<<w<<std::endl;
        }

        winv.at<float>(i,i) = 1./w.at<float>(i);
    }
    // Then x = vt'*winv*u'*B
    cv::Mat x = vt.t()*winv*u.t()*B;

    // x=[s,gw] 4x1 vector
    double sstar = x.at<float>(0);    // scale should be positive
    cv::Mat gwstar = x.rowRange(1,4);   // gravity should be about ~9.8

    // Debug log
    std::cout<<"scale sstar: "<<sstar<<std::endl;
    std::cout<<"gwstar: "<<gwstar.t()<<", |gwstar|="<<cv::norm(gwstar)<<std::endl;

    // Test log
    if(w.type()!=I3.type() || u.type()!=I3.type() || vt.type()!=I3.type())
        std::cerr<<"different mat type, I3,w,u,vt: "<<I3.type()<<","<<w.type()<<","<<u.type()<<","<<vt.type()<<std::endl;

    // Step 3.
    // Use gravity magnitude 9.8 as constraint
    // gI = [0;0;1], the normalized gravity vector in an inertial frame, NED type with no orientation.
    cv::Mat gI = cv::Mat::zeros(3,1,CV_32F);
    gI.at<float>(2) = 1;
    // Normalized approx. gravity vecotr in world frame
    cv::Mat gwn = gwstar/cv::norm(gwstar);
    // Debug log
    //cout<<"gw normalized: "<<gwn<<endl;

    // vhat = (gI x gw) / |gI x gw|
    cv::Mat gIxgwn = gI.cross(gwn);
    double normgIxgwn = cv::norm(gIxgwn);
    cv::Mat vhat = gIxgwn/normgIxgwn;
    double theta = std::atan2(normgIxgwn,gI.dot(gwn));
    // Debug log
    //cout<<"vhat: "<<vhat<<", theta: "<<theta*180.0/M_PI<<endl;

    Eigen::Vector3d vhateig = util::converter::toVector3d(vhat);
    Eigen::Matrix3d RWIeig = Sophus::SO3::exp(vhateig*theta).matrix();
    cv::Mat Rwi = util::converter::toCvMat(RWIeig);
    cv::Mat GI = gI*ConfigParam::GetG();//9.8012;
    // Solve C*x=D for x=[s,dthetaxy,ba] (1+2+3)x1 vector
    cv::Mat C = cv::Mat::zeros(3*(N-2),6,CV_32F);
    cv::Mat D = cv::Mat::zeros(3*(N-2),1,CV_32F);

    for(int i=0; i<N-2; i++)
    {
        //KeyFrameInit* pKF1 = vKFInit[i];//vScaleGravityKF[i];
        KeyFrameInit* pKF2 = vKFInit[i+1];
        KeyFrameInit* pKF3 = vKFInit[i+2];
        // Delta time between frames
        double dt12 = pKF2->mIMUPreInt.getDeltaTime();
        double dt23 = pKF3->mIMUPreInt.getDeltaTime();
        // Pre-integrated measurements
        cv::Mat dp12 = util::converter::toCvMat(pKF2->mIMUPreInt.getDeltaP());
        cv::Mat dv12 = util::converter::toCvMat(pKF2->mIMUPreInt.getDeltaV());
        cv::Mat dp23 = util::converter::toCvMat(pKF3->mIMUPreInt.getDeltaP());
        cv::Mat Jpba12 = util::converter::toCvMat(pKF2->mIMUPreInt.getJPBiasa());
        cv::Mat Jvba12 = util::converter::toCvMat(pKF2->mIMUPreInt.getJVBiasa());
        cv::Mat Jpba23 = util::converter::toCvMat(pKF3->mIMUPreInt.getJPBiasa());
        // Pose of camera in world frame
        cv::Mat Twc1 = vTwc[i].clone();//pKF1->GetPoseInverse();
        cv::Mat Twc2 = vTwc[i+1].clone();//pKF2->GetPoseInverse();
        cv::Mat Twc3 = vTwc[i+2].clone();//pKF3->GetPoseInverse();
        // Position of camera center
        cv::Mat pc1 = Twc1.rowRange(0,3).col(3);
        cv::Mat pc2 = Twc2.rowRange(0,3).col(3);
        cv::Mat pc3 = Twc3.rowRange(0,3).col(3);
        // Rotation of camera, Rwc
        cv::Mat Rc1 = Twc1.rowRange(0,3).colRange(0,3);
        cv::Mat Rc2 = Twc2.rowRange(0,3).colRange(0,3);
        cv::Mat Rc3 = Twc3.rowRange(0,3).colRange(0,3);
        // Stack to C/D matrix
        // lambda*s + phi*dthetaxy + zeta*ba = psi
        cv::Mat lambda = (pc2-pc1)*dt23 + (pc2-pc3)*dt12;
        cv::Mat phi = - 0.5*(dt12*dt12*dt23 + dt12*dt23*dt23)*Rwi*SkewSymmetricMatrix(GI);  // note: this has a '-', different to paper
        cv::Mat zeta = Rc2*Rcb*Jpba23*dt12 + Rc1*Rcb*Jvba12*dt12*dt23 - Rc1*Rcb*Jpba12*dt23;
        cv::Mat psi = (Rc1-Rc2)*pcb*dt23 + Rc1*Rcb*dp12*dt23 - (Rc2-Rc3)*pcb*dt12
                     - Rc2*Rcb*dp23*dt12 - Rc1*Rcb*dv12*dt23*dt12 - 0.5*Rwi*GI*(dt12*dt12*dt23 + dt12*dt23*dt23); // note:  - paper
        lambda.copyTo(C.rowRange(3*i+0,3*i+3).col(0));
        phi.colRange(0,2).copyTo(C.rowRange(3*i+0,3*i+3).colRange(1,3)); //only the first 2 columns, third term in dtheta is zero, here compute dthetaxy 2x1.
        zeta.copyTo(C.rowRange(3*i+0,3*i+3).colRange(3,6));
        psi.copyTo(D.rowRange(3*i+0,3*i+3));

        // Debug log
        //cout<<"iter "<<i<<endl;
    }

    // Use svd to compute C*x=D, x=[s,dthetaxy,ba] 6x1 vector
    // C = u*w*vt, u*w*vt*x=D
    // Then x = vt'*winv*u'*D
    cv::Mat w2,u2,vt2;
    // Note w2 is 6x1 vector by SVDecomp()
    // C is changed in SVDecomp() with cv::SVD::MODIFY_A for speed
    cv::SVDecomp(C,w2,u2,vt2,cv::SVD::MODIFY_A);
    // Debug log
    //cout<<"u2:"<<endl<<u2<<endl;
    //cout<<"vt2:"<<endl<<vt2<<endl;
    //cout<<"w2:"<<endl<<w2<<endl;

    // Compute winv
    cv::Mat w2inv=cv::Mat::eye(6,6,CV_32F);
    for(int i=0;i<6;i++)
    {
        if(fabs(w2.at<float>(i))<1e-10)
        {
            w2.at<float>(i) += 1e-10;
            // Test log
            std::cerr<<"w2(i) < 1e-10, w="<<std::endl<<w2<<std::endl;
        }

        w2inv.at<float>(i,i) = 1./w2.at<float>(i);
    }
    // Then y = vt'*winv*u'*D
    cv::Mat y = vt2.t()*w2inv*u2.t()*D;

    double s_ = y.at<float>(0);
    cv::Mat dthetaxy = y.rowRange(1,3);
    cv::Mat dbiasa_ = y.rowRange(3,6);
    Vector3d dbiasa_eig = util::converter::toVector3d(dbiasa_);

    // dtheta = [dx;dy;0]
    cv::Mat dtheta = cv::Mat::zeros(3,1,CV_32F);
    dthetaxy.copyTo(dtheta.rowRange(0,2));
    Eigen::Vector3d dthetaeig = util::converter::toVector3d(dtheta);
    // Rwi_ = Rwi*exp(dtheta)
    Eigen::Matrix3d Rwieig_ = RWIeig*Sophus::SO3::exp(dthetaeig).matrix();
    cv::Mat Rwi_ = util::converter::toCvMat(Rwieig_);

//    cout<<"time consumption: step1 "<<(t1-t0)/cv::getTickFrequency()*1000
//        <<"ms, step2 "<<(t2-t1)/cv::getTickFrequency()*1000
//        <<"ms, step3 "<<(t3-t2)/cv::getTickFrequency()*1000<<endl;

    //    // Debug log
//    if(cv::norm(dbiasa_)<1 && norm(dtheta*180/M_PI)<10)
    {
        //cout<<std::fixed<<std::setprecision(6)<<"estimated gyr bias: "<<bgest.transpose()<<endl;
        //std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
        cv::Mat gwbefore = Rwi*GI;
        //cout<<"gw with 9.8012 before optimization: "<<gwbefore<<endl;
        cv::Mat gwafter = Rwi_*GI;
        //cout<<"gw after optimization: "<<gwafter<<endl;
        //cout<<"----------estimated after optimization."<<endl;
        std::cout<<"Time: "<<pNewestKF->timestamp_ - mnStartTime<<", sstar: "<<sstar<<", s: "<<s_<<std::endl;
        //cout<<"biasa: "<<dbiasa_<<endl;
        //cout<<"dtheta: "<<dtheta*180/M_PI<<endl;
        //cout<<"condition number w0/w5: "<<w2.at<float>(0)/w2.at<float>(5)<<endl;
        //cout<<endl<<endl;


        fgw<<pNewestKF->timestamp_<<" "
           <<gwafter.at<float>(0)<<" "<<gwafter.at<float>(1)<<" "<<gwafter.at<float>(2)<<" "
           <<gwbefore.at<float>(0)<<" "<<gwbefore.at<float>(1)<<" "<<gwbefore.at<float>(2)<<" "
           <<std::endl;
        fscale<<pNewestKF->timestamp_<<" "
              <<s_<<" "<<sstar<<" "<<std::endl;
        fbiasa<<pNewestKF->timestamp_<<" "
              <<dbiasa_.at<float>(0)<<" "<<dbiasa_.at<float>(1)<<" "<<dbiasa_.at<float>(2)<<" "<<std::endl;
        fcondnum<<pNewestKF->timestamp_<<" "
                <<w2.at<float>(0)<<" "<<w2.at<float>(1)<<" "<<w2.at<float>(2)<<" "<<w2.at<float>(3)<<" "
                <<w2.at<float>(4)<<" "<<w2.at<float>(5)<<" "<<std::endl;
//        ftime<<pNewestKF->mTimeStamp<<" "
//             <<(t3-t0)/cv::getTickFrequency()*1000<<" "<<endl;
        fbiasg<<pNewestKF->timestamp_<<" "
              <<bgest(0)<<" "<<bgest(1)<<" "<<bgest(2)<<" "<<std::endl;

        std::ofstream fRwi("/home/jp/opensourcecode/ORB_SLAM2/tmp/Rwi.txt");
        fRwi<<Rwieig_(0,0)<<" "<<Rwieig_(0,1)<<" "<<Rwieig_(0,2)<<" "
            <<Rwieig_(1,0)<<" "<<Rwieig_(1,1)<<" "<<Rwieig_(1,2)<<" "
            <<Rwieig_(2,0)<<" "<<Rwieig_(2,1)<<" "<<Rwieig_(2,2)<<std::endl;
        fRwi.close();
    }

    // ********************************
    // Todo:
    // Add some logic or strategy to confirm init status
    bool bVIOInited = false;
    if(mbFirstTry)
    {
        mbFirstTry = false;
        mnStartTime = cur_keyfrm_->timestamp_;
    }
    if(pNewestKF->timestamp_ - mnStartTime >= ConfigParam::GetVINSInitTime())
    {
        bVIOInited = true;
    }

    // When failed. Or when you're debugging.
    // Reset biasg to zero, and re-compute imu-preintegrator.
    if(!bVIOInited)
    {
//        for(vector<KeyFrame*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
//        {
//            KeyFrame* pKF = *vit;
//            pKF->SetNavStateBiasGyr(Vector3d::Zero());
//            pKF->SetNavStateBiasAcc(Vector3d::Zero());
//            pKF->SetNavStateDeltaBg(Eigen::Vector3d::Zero());
//            pKF->SetNavStateDeltaBa(Eigen::Vector3d::Zero());
//        }
//        for(vector<KeyFrame*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
//        {
//            KeyFrame* pKF = *vit;
//            pKF->ComputePreInt();
//        }
    }
    else
    {
        // Set NavState , scale and bias for all KeyFrames
        // Scale
        double scale = s_;
        mnVINSInitScale = s_;
        // gravity vector in world frame
        cv::Mat gw = Rwi_*GI;
        mGravityVec = gw.clone();
        Vector3d gweig = util::converter::toVector3d(gw);
        mRwiInit = Rwi_.clone();

//        // For test
//        double scale = sstar;//s_;
//        mnVINSInitScale = sstar;//s_;
//        // gravity vector in world frame
//        cv::Mat gw = gwstar;//Rwi_*GI;
//        mGravityVec = gw;
//        Vector3d gweig = Converter::toVector3d(gw);
//        dbiasa_eig.setZero();

         // Update NavState for the KeyFrames not in vScaleGravityKF
        // Update Tcw-type pose for these KeyFrames, need mutex lock
        #ifdef RUN_REALTIME
                // Stop local mapping, and
                request_pause();


                // Wait until Local Mapping has effectively stopped
                while(!is_paused() && !is_terminated())
                {
                    usleep(1000);
                }
        #endif

        SetUpdatingInitPoses(true);
        {
            std::unique_lock<std::mutex> lock(map_db_->mtx_database_);

            int cnt=0;

            for(std::vector<data::keyframe*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++,cnt++)
            {
                data::keyframe* pKF = *vit;
                if(pKF->will_be_erased()) continue;
                if(pKF!=vScaleGravityKF[cnt]) std::cerr<<"pKF!=vScaleGravityKF[cnt], id: "<<pKF->id_<<" != "<<vScaleGravityKF[cnt]->id_<<std::endl;
                // Position and rotation of visual SLAM
                Mat44_t mat = pKF->get_cam_pose_inv();
                cv::Mat aux;
                aux = (cv::Mat_<float>(4,4) << mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                                            mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                                            mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                                            mat(3,0), mat(3,1), mat(3,2), mat(3,3));

                cv::Mat wPc = aux.rowRange(0,3).col(3);                   // wPc
                cv::Mat Rwc = aux.rowRange(0,3).colRange(0,3);            // Rwc
                // Set position and rotation of navstate
                cv::Mat wPb = scale*wPc + Rwc*pcb;
                pKF->SetNavStatePos(util::converter::toVector3d(wPb));
                pKF->SetNavStateRot(util::converter::toMatrix3d(Rwc*Rcb));
                // Update bias of Gyr & Acc
                pKF->SetNavStateBiasGyr(bgest);
                pKF->SetNavStateBiasAcc(dbiasa_eig);
                // Set delta_bias to zero. (only updated during optimization)
                pKF->SetNavStateDeltaBg(Eigen::Vector3d::Zero());
                pKF->SetNavStateDeltaBa(Eigen::Vector3d::Zero());
                // Step 4.
                // compute velocity
                if(pKF != vScaleGravityKF.back())
                {
                    data::keyframe* pKFnext = pKF->GetNextKeyFrame();
                    if(!pKFnext) std::cerr<<"pKFnext is NULL, cnt="<<cnt<<", pKFnext:"<<pKFnext<<std::endl;
                    if(pKFnext!=vScaleGravityKF[cnt+1]) std::cerr<<"pKFnext!=vScaleGravityKF[cnt+1], cnt="<<cnt<<", id: "<<pKFnext->id_<<" != "<<vScaleGravityKF[cnt+1]->id_<<std::endl;
                    // IMU pre-int between pKF ~ pKFnext
                    const IMUPreintegrator& imupreint = pKFnext->GetIMUPreInt();
                    // Time from this(pKF) to next(pKFnext)
                    double dt = imupreint.getDeltaTime();                                       // deltaTime
                    cv::Mat dp = util::converter::toCvMat(imupreint.getDeltaP());       // deltaP
                    cv::Mat Jpba = util::converter::toCvMat(imupreint.getJPBiasa());    // J_deltaP_biasa
                    Mat44_t mat = pKF->get_cam_pose_inv();
                    cv::Mat aux;
                    aux = (cv::Mat_<float>(4,4) << mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                                                mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                                                mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                                                mat(3,0), mat(3,1), mat(3,2), mat(3,3));

                    cv::Mat wPcnext = aux.rowRange(0,3).col(3);           // wPc next
                    cv::Mat Rwcnext = aux.rowRange(0,3).colRange(0,3);    // Rwc next

                    cv::Mat vel = - 1./dt*( scale*(wPc - wPcnext) + (Rwc - Rwcnext)*pcb + Rwc*Rcb*(dp + Jpba*dbiasa_) + 0.5*gw*dt*dt );
                    Eigen::Vector3d veleig = util::converter::toVector3d(vel);
                    pKF->SetNavStateVel(veleig);
                }
                else
                {
                    std::cerr<<"-----------here is the last KF in vScaleGravityKF------------"<<std::endl;
                    // If this is the last KeyFrame, no 'next' KeyFrame exists
                    data::keyframe* pKFprev = pKF->GetPrevKeyFrame();
                    if(!pKFprev) std::cerr<<"pKFprev is NULL, cnt="<<cnt<<std::endl;
                    if(pKFprev!=vScaleGravityKF[cnt-1]) std::cerr<<"pKFprev!=vScaleGravityKF[cnt-1], cnt="<<cnt<<", id: "<<pKFprev->id_<<" != "<<vScaleGravityKF[cnt-1]->id_<<std::endl;
                    const IMUPreintegrator& imupreint_prev_cur = pKF->GetIMUPreInt();
                    double dt = imupreint_prev_cur.getDeltaTime();
                    Eigen::Matrix3d Jvba = imupreint_prev_cur.getJVBiasa();
                    Eigen::Vector3d dv = imupreint_prev_cur.getDeltaV();
                    //
                    Eigen::Vector3d velpre = pKFprev->GetNavState().Get_V();
                    Eigen::Matrix3d rotpre = pKFprev->GetNavState().Get_RotMatrix();
                    Eigen::Vector3d veleig = velpre + gweig*dt + rotpre*( dv + Jvba*dbiasa_eig );
                    pKF->SetNavStateVel(veleig);
                }
            }

            // Re-compute IMU pre-integration at last. Should after usage of pre-int measurements.
            for(std::vector<data::keyframe*>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
            {
                data::keyframe* pKF = *vit;
                if(pKF->will_be_erased()) continue;
                pKF->ComputePreInt();
            }

            // Update poses (multiply metric scale)
            std::vector<data::keyframe*> mspKeyFrames = map_db_->get_all_keyframes();
            for(std::vector<data::keyframe*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
            {
                data::keyframe* pKF = *sit;
                Mat44_t mat = pKF->get_cam_pose();
                cv::Mat Tcw;
                Tcw = (cv::Mat_<float>(4,4) << mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                                                mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                                                mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                                                mat(3,0), mat(3,1), mat(3,2), mat(3,3));

                //cv::Mat Tcw = pKF->GetPose();
                cv::Mat tcw = Tcw.rowRange(0,3).col(3)*scale;
                tcw.copyTo(Tcw.rowRange(0,3).col(3));
                //pKF->SetPose(Tcw);
                Mat44_t cam_pose;
                cam_pose = util::converter::cvMat4_to_Mat44_t(Tcw);
                pKF->set_cam_pose(cam_pose);
            }
            std::vector<data::landmark*> mspMapPoints = map_db_->get_all_landmarks();
            for(std::vector<data::landmark*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
            {
                data::landmark* pMP = *sit;
                //pMP->SetWorldPos(pMP->GetWorldPos()*scale);
                pMP->UpdateScale(scale);
            }
            std::cout<<std::endl<<"... Map scale updated ..."<<std::endl<<std::endl;

            // Update NavStates
            if(pNewestKF!=cur_keyfrm_)
            {
                data::keyframe* pKF;

                // step1. bias&d_bias
                pKF = pNewestKF;
                do
                {
                    pKF = pKF->GetNextKeyFrame();

                    // Update bias of Gyr & Acc
                    pKF->SetNavStateBiasGyr(bgest);
                    pKF->SetNavStateBiasAcc(dbiasa_eig);
                    // Set delta_bias to zero. (only updated during optimization)
                    pKF->SetNavStateDeltaBg(Eigen::Vector3d::Zero());
                    pKF->SetNavStateDeltaBa(Eigen::Vector3d::Zero());
                }while(pKF!=cur_keyfrm_);

                // step2. re-compute pre-integration
                pKF = pNewestKF;
                do
                {
                    pKF = pKF->GetNextKeyFrame();

                    pKF->ComputePreInt();
                }while(pKF!=cur_keyfrm_);

                // step3. update pos/rot
                pKF = pNewestKF;
                do
                {
                    pKF = pKF->GetNextKeyFrame();

                    // Update rot/pos
                    // Position and rotation of visual SLAM
                    Mat44_t mat = pKF->get_cam_pose_inv();
                    cv::Mat aux;
                    aux = (cv::Mat_<float>(4,4) << mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                                                        mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                                                        mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                                                        mat(3,0), mat(3,1), mat(3,2), mat(3,3));
                    cv::Mat wPc = aux.rowRange(0,3).col(3);                   // wPc
                    cv::Mat Rwc = aux.rowRange(0,3).colRange(0,3);            // Rwc
                    cv::Mat wPb = wPc + Rwc*pcb;
                    pKF->SetNavStatePos(util::converter::toVector3d(wPb));
                    pKF->SetNavStateRot(util::converter::toMatrix3d(Rwc*Rcb));

                    //pKF->SetNavState();

                    if(pKF != cur_keyfrm_)
                    {
                        data::keyframe* pKFnext = pKF->GetNextKeyFrame();
                        // IMU pre-int between pKF ~ pKFnext
                        const IMUPreintegrator& imupreint = pKFnext->GetIMUPreInt();
                        // Time from this(pKF) to next(pKFnext)
                        double dt = imupreint.getDeltaTime();                                       // deltaTime
                        cv::Mat dp = util::converter::toCvMat(imupreint.getDeltaP());       // deltaP
                        cv::Mat Jpba = util::converter::toCvMat(imupreint.getJPBiasa());    // J_deltaP_biasa
                        Mat44_t mat = pKFnext->get_cam_pose_inv();
                        cv::Mat aux;
                        aux = (cv::Mat_<float>(4,4) << mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                                                            mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                                                            mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                                                            mat(3,0), mat(3,1), mat(3,2), mat(3,3));
                        cv::Mat wPcnext = aux.rowRange(0,3).col(3);           // wPc next
                        cv::Mat Rwcnext = aux.rowRange(0,3).colRange(0,3);    // Rwc next

                        cv::Mat vel = - 1./dt*( (wPc - wPcnext) + (Rwc - Rwcnext)*pcb + Rwc*Rcb*(dp + Jpba*dbiasa_) + 0.5*gw*dt*dt );
                        Eigen::Vector3d veleig = util::converter::toVector3d(vel);
                        pKF->SetNavStateVel(veleig);
                    }
                    else
                    {
                        // If this is the last KeyFrame, no 'next' KeyFrame exists
                        data::keyframe* pKFprev = pKF->GetPrevKeyFrame();
                        const IMUPreintegrator& imupreint_prev_cur = pKF->GetIMUPreInt();
                        double dt = imupreint_prev_cur.getDeltaTime();
                        Eigen::Matrix3d Jvba = imupreint_prev_cur.getJVBiasa();
                        Eigen::Vector3d dv = imupreint_prev_cur.getDeltaV();
                        //
                        Eigen::Vector3d velpre = pKFprev->GetNavState().Get_V();
                        Eigen::Matrix3d rotpre = pKFprev->GetNavState().Get_RotMatrix();
                        Eigen::Vector3d veleig = velpre + gweig*dt + rotpre*( dv + Jvba*dbiasa_eig );
                        pKF->SetNavStateVel(veleig);
                    }

                }while(pKF!=cur_keyfrm_);

            }

            std::cout<<std::endl<<"... Map NavState updated ..."<<std::endl<<std::endl;

            SetFirstVINSInited(true);
            SetVINSInited(true);

        }
        SetUpdatingInitPoses(false);

        #ifdef RUN_REALTIME

                resume();
        #endif



        // Run global BA after inited
        unsigned long nGBAKF = cur_keyfrm_->id_;
        //optimizer::GlobalBundleAdjustmentNavState(mpMap,mGravityVec,10,NULL,nGBAKF,false);
        std::cerr<<"finish global BA after vins init"<<std::endl;
        #ifdef RUN_REALTIME
        // Update pose
        // Stop local mapping, and

        request_pause();

        // Wait until Local Mapping has effectively stopped
        while(!is_paused() && !is_terminated())
        {
            usleep(1000);
        }


        cv::Mat cvTbc = ConfigParam::GetMatTbc();

        {
            std::unique_lock<std::mutex> lock(map_db_->mtx_database_);

            // Correct keyframes starting at map first keyframe
            std::list<data::keyframe*> lpKFtoCheck(map_db_->mvpKeyFrameOrigins.begin(),map_db_->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                data::keyframe* pKF = lpKFtoCheck.front();
                const std::set<data::keyframe*> sChilds = pKF->graph_node_->get_spanning_children();
                Mat44_t mat = pKF->get_cam_pose_inv();
                cv::Mat Twc;
                Twc = (cv::Mat_<float>(4,4) << mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                                                    mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                                                    mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                                                    mat(3,0), mat(3,1), mat(3,2), mat(3,3));
                //cv::Mat Twc = pKF->GetPoseInverse();
                const NavState& NS = pKF->GetNavState();
                //Debug log
                if(pKF->loop_BA_identifier_==nGBAKF)
                {
                    cv::Mat tTwb1 = Twc*ConfigParam::GetMatT_cb();
                    if((util::converter::toVector3d(tTwb1.rowRange(0,3).col(3))-NS.Get_P()).norm()>1e-6)
                        std::cout<<"Twc*Tcb != NavState for GBA KFs, id "<<pKF->id_<<": "<<tTwb1.rowRange(0,3).col(3).t()<<"/"<<NS.Get_P().transpose()<<std::endl;
                }
                else std::cout<<"pKF->loop_BA_identifier_ != nGBAKF???"<<std::endl;
                for(std::set<data::keyframe*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    data::keyframe* pChild = *sit;
                    if(pChild->loop_BA_identifier_!=nGBAKF)
                    {
                        std::cerr<<"correct KF after gBA in VI init: "<<pChild->id_<<std::endl;
                        Mat44_t mat = pChild->get_cam_pose();
                        cv::Mat aux;
                        aux = (cv::Mat_<float>(4,4) << mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                                                            mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                                                            mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                                                            mat(3,0), mat(3,1), mat(3,2), mat(3,3));

                        cv::Mat Tchildc = aux*Twc;
                        pChild->cam_pose_cw_after_loop_BA_ = util::converter::cvMat4_to_Mat44_t(Tchildc)*pChild->cam_pose_cw_after_loop_BA_;
                        //pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->loop_BA_identifier_=nGBAKF;

                        // Set NavStateGBA and correct the P/V/R
                        pChild->mNavStateGBA = pChild->GetNavState();
                        Mat44_t mat2 = pChild->cam_pose_cw_after_loop_BA_;
                        cv::Mat aux2;
                        aux2 = (cv::Mat_<float>(4,4) << mat2(0,0), mat2(0,1), mat2(0,2), mat2(0,3),
                                                            mat2(1,0), mat2(1,1), mat2(1,2), mat2(1,3),
                                                            mat2(2,0), mat2(2,1), mat2(2,2), mat2(2,3),
                                                            mat2(3,0), mat2(3,1), mat2(3,2), mat2(3,3));
                        cv::Mat TwbGBA = util::converter::toCvMatInverse(cvTbc*aux2);
                        Matrix3d RwbGBA = util::converter::toMatrix3d(TwbGBA.rowRange(0,3).colRange(0,3));
                        Vector3d PwbGBA = util::converter::toVector3d(TwbGBA.rowRange(0,3).col(3));
                        Matrix3d Rw1 = pChild->mNavStateGBA.Get_RotMatrix();
                        Vector3d Vw1 = pChild->mNavStateGBA.Get_V();
                        Vector3d Vw2 = RwbGBA*Rw1.transpose()*Vw1;   // bV1 = bV2 ==> Rwb1^T*wV1 = Rwb2^T*wV2 ==> wV2 = Rwb2*Rwb1^T*wV1
                        pChild->mNavStateGBA.Set_Pos(PwbGBA);
                        pChild->mNavStateGBA.Set_Rot(RwbGBA);
                        pChild->mNavStateGBA.Set_Vel(Vw2);
                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->cam_pose_cw_before_BA_ = pKF->get_cam_pose();
                //pKF->SetPose(pKF->mTcwGBA);
                pKF->mNavStateBefGBA = pKF->GetNavState();
                pKF->SetNavState(pKF->mNavStateGBA);
                pKF->UpdatePoseFromNS(cvTbc);

                lpKFtoCheck.pop_front();

                //Test log
                Mat44_t mat3 = pKF->get_cam_pose_inv();
                cv::Mat aux1;
                aux1 = (cv::Mat_<float>(4,4) << mat3(0,0), mat3(0,1), mat3(0,2), mat3(0,3),
                                                    mat3(1,0), mat3(1,1), mat3(1,2), mat3(1,3),
                                                    mat3(2,0), mat3(2,1), mat3(2,2), mat3(2,3),
                                                    mat3(3,0), mat3(3,1), mat3(3,2), mat3(3,3));
                cv::Mat tTwb = aux1*ConfigParam::GetMatT_cb();
                Vector3d tPwb = util::converter::toVector3d(tTwb.rowRange(0,3).col(3));
                if( (tPwb-pKF->GetNavState().Get_P()).norm()>1e-6 )
                    std::cerr<<"pKF PoseInverse Pwb != NavState.P ?"<<tPwb.transpose()<<"/"<<pKF->GetNavState().Get_P().transpose()<<std::endl;
            }

            // Correct MapPoints
            const std::vector<data::landmark*> vpMPs = map_db_->get_all_landmarks();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                data::landmark* pMP = vpMPs[i];

                if(pMP->will_be_erased())
                    continue;

                if(pMP->loop_BA_identifier_==nGBAKF)
                {
                    // If optimized by Global BA, just update
                    pMP->set_pos_in_world(pMP->pos_w_after_global_BA_);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    data::keyframe* pRefKF = pMP->get_ref_keyframe();

                    if(pRefKF->loop_BA_identifier_!=nGBAKF)
                        continue;

                    // Map to non-corrected camera
                    Mat44_t mat = pRefKF->cam_pose_cw_before_BA_;
                    cv::Mat aux;
                    aux = (cv::Mat_<float>(4,4) << mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                                                        mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                                                        mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                                                        mat(3,0), mat(3,1), mat(3,2), mat(3,3));

                    cv::Mat Rcw = aux.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = aux.rowRange(0,3).col(3);
                    Vec3_t vect = pMP->get_pos_in_world();
                    cv::Mat aux_vect;
                    eigen2cv(vect, aux_vect);
                    //cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;
                    cv::Mat Xc = Rcw*aux_vect+tcw;

                    // Backproject using corrected camera
                    Mat44_t mat2 = pRefKF->get_cam_pose_inv();
                    cv::Mat Twc;
                    Twc = (cv::Mat_<float>(4,4) << mat2(0,0), mat2(0,1), mat2(0,2), mat2(0,3),
                                                        mat2(1,0), mat2(1,1), mat2(1,2), mat2(1,3),
                                                        mat2(2,0), mat2(2,1), mat2(2,2), mat2(2,3),
                                                        mat2(3,0), mat2(3,1), mat2(3,2), mat2(3,3));
                    //cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);
                    Vec3_t vect2;
                    cv2eigen(Rwc*Xc+twc, vect2);
                    pMP->set_pos_in_world(vect2);
                    //pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }

            std::cout << "Map updated!" << std::endl;

            // Map updated, set flag for Tracking
            SetMapUpdateFlagInTracking(true);

            // Release LocalMapping
            resume();

        }

        #endif
        SetFlagInitGBAFinish(true);
    }

    for(int i=0;i<N;i++)
    {
        if(vKFInit[i])
            delete vKFInit[i];
    }

    return bVIOInited;
}


void mapping_module::AddToLocalWindow(data::keyframe* pKF)
{
    mlLocalKeyFrames.push_back(pKF);
    if(mlLocalKeyFrames.size() > mnLocalWindowSize)
    {
        mlLocalKeyFrames.pop_front();
    }
    else
    {
        data::keyframe* pKF0 = mlLocalKeyFrames.front();
        while(mlLocalKeyFrames.size() < mnLocalWindowSize && pKF0->GetPrevKeyFrame()!=NULL)
        {
            pKF0 = pKF0->GetPrevKeyFrame();
            mlLocalKeyFrames.push_front(pKF0);
        }
    }
}

void mapping_module::DeleteBadInLocalWindow(void)
{
    // Debug log
    //cout<<"start deletebadinlocal"<<endl;
    std::list<data::keyframe*>::iterator lit = mlLocalKeyFrames.begin();
    while(lit != mlLocalKeyFrames.end())
    {
        data::keyframe* pKF = *lit;
        if(!pKF)
            std::cout<<"pKF null?"<<std::endl;
        //cout<<"pKF id:"<<pKF->mnId<<". ";
        if(pKF->will_be_erased())
        {
            //Debug log
            //cout<<"KF "<<pKF->mnId<<" is bad, delted from local window"<<endl;
            lit = mlLocalKeyFrames.erase(lit);
        }
        else
        {
            //cout<<pKF->mnId<<", ";
            lit++;
        }
    }
    //cout<<"end deletebadinlocal"<<endl;
}

cv::Mat mapping_module::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

mapping_module::mapping_module(data::map_database* map_db, const bool is_monocular, ConfigParam* pParams)
    : local_map_cleaner_(new module::local_map_cleaner(is_monocular)), map_db_(map_db),
      local_bundle_adjuster_(new optimize::local_bundle_adjuster()), is_monocular_(is_monocular), mnLocalMapAbort(0) {
    spdlog::debug("CONSTRUCT: mapping_module");

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    mpParams = pParams;
    mnLocalWindowSize = ConfigParam::GetLocalWindowSize();
    std::cout<<"mnLocalWindowSize:"<<mnLocalWindowSize<<std::endl;

    mbVINSInited = false;
    mbFirstTry = true;
    mbFirstVINSInited = false;

    mbUpdatingInitPoses = false;
    mbCopyInitKFs = false;
    mbInitGBAFinish = false;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
}

mapping_module::~mapping_module() {
    spdlog::debug("DESTRUCT: mapping_module");
}

void mapping_module::set_tracking_module(tracking_module* tracker) {
    tracker_ = tracker;
}

void mapping_module::set_global_optimization_module(global_optimization_module* global_optimizer) {
    global_optimizer_ = global_optimizer;
}

void mapping_module::run() {
    spdlog::info("start mapping module");

    is_terminated_ = false;

    while (true) {
        // waiting time for the other threads
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // LOCK
        set_keyframe_acceptability(false);

        // check if termination is requested
        if (terminate_is_requested()) {
            // terminate and break
            terminate();
            break;
        }

        // check if pause is requested
        if (pause_is_requested()) {
            // if any keyframe is queued, all of them must be processed before the pause
            while (keyframe_is_queued()) {
                // create and extend the map with the new keyframe
                mapping_with_new_keyframe();
                if(GetFlagInitGBAFinish())
                {
                    // send the new keyframe to the global optimization module
                    global_optimizer_->queue_keyframe(cur_keyfrm_);
                }
                
            }
            // pause and wait
            pause();
            // check if termination or reset is requested during pause
            while (is_paused() && !terminate_is_requested() && !reset_is_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }

        // check if reset is requested
        if (reset_is_requested()) {
            // reset, UNLOCK and continue
            reset();
            set_keyframe_acceptability(true);
            continue;
        }

        // if the queue is empty, the following process is not needed
        if (!keyframe_is_queued()) {
            // UNLOCK and continue
            set_keyframe_acceptability(true);
            continue;
        }

        // create and extend the map with the new keyframe
        mapping_with_new_keyframe();
        if(GetFlagInitGBAFinish())
        {
            // send the new keyframe to the global optimization module
            global_optimizer_->queue_keyframe(cur_keyfrm_);

        }
        
        // LOCK end
        set_keyframe_acceptability(true);
    }

    spdlog::info("terminate mapping module");
}

void mapping_module::queue_keyframe(data::keyframe* keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    keyfrms_queue_.push_back(keyfrm);
    abort_local_BA_ = true;
}

unsigned int mapping_module::get_num_queued_keyframes() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return keyfrms_queue_.size();
}

bool mapping_module::keyframe_is_queued() const {
    std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
    return !keyfrms_queue_.empty();
}

bool mapping_module::get_keyframe_acceptability() const {
    return keyfrm_acceptability_;
}

void mapping_module::set_keyframe_acceptability(const bool acceptability) {
    keyfrm_acceptability_ = acceptability;
}

void mapping_module::abort_local_BA() {
    abort_local_BA_ = true;
}

void mapping_module::mapping_with_new_keyframe() {
    // dequeue
    {
        std::lock_guard<std::mutex> lock(mtx_keyfrm_queue_);
        // dequeue -> cur_keyfrm_
        cur_keyfrm_ = keyfrms_queue_.front();
        keyfrms_queue_.pop_front();
    }

    // set the origin keyframe
    local_map_cleaner_->set_origin_keyframe_id(map_db_->origin_keyfrm_->id_);

    // store the new keyframe to the database
    store_new_keyframe();

    // remove redundant landmarks
    local_map_cleaner_->remove_redundant_landmarks(cur_keyfrm_->id_);

    // triangulate new landmarks between the current frame and each of the covisibilities
    create_new_landmarks();

    if (keyframe_is_queued()) {
        return;
    }

    // detect and resolve the duplication of the landmarks observed in the current frame
    update_new_keyframe();

    if (keyframe_is_queued() || pause_is_requested()) {
        return;
    }

    // local bundle adjustment
    abort_local_BA_ = false;
    if (2 < map_db_->get_num_keyframes()) 
    {
        //local_bundle_adjuster_->optimize(cur_keyfrm_, &abort_local_BA_);
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        if(!GetVINSInited())
        {
            local_bundle_adjuster_->optimize(cur_keyfrm_, &abort_local_BA_, map_db_, this);
        }
        else
        {
#ifndef TRACK_WITH_IMU
            local_bundle_adjuster_->optimize(cur_keyfrm_, &abort_local_BA_, map_db_, this);
#else
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            local_bundle_adjuster_->LocalBundleAdjustmentNavState(cur_keyfrm_,mlLocalKeyFrames,&abort_local_BA_, map_db_, mGravityVec, this);
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double vlocalbatime2 = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

            if (abort_local_BA_) 
            {
                mnLocalMapAbort += 1;
                std::cout << "Local BA took: " << vlocalbatime2 << std::endl;
                std::cout << "Aborted " << mnLocalMapAbort << " times" << std::endl;
            }
#endif
        }
#ifndef RUN_REALTIME
        // Try to initialize VIO, if not inited
        if(!GetVINSInited())
        {
            bool tmpbool = TryInitVIO();
            SetVINSInited(tmpbool);
            if(tmpbool)
            {
                //mpMap->UpdateScale(mnVINSInitScale);
                //std::cout<<"start global BA"<<std::endl;
                //Optimizer::GlobalBundleAdjustmentNavState(mpMap,mGravityVec,20);
                SetFirstVINSInited(true);
                std::cout<<GetMapUpdateFlagForTracking()<<", "<<GetFirstVINSInited()<<std::endl;
            }
        }
#endif
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
    }
    local_map_cleaner_->remove_redundant_keyframes(cur_keyfrm_);
}

void mapping_module::store_new_keyframe() {
    // compute BoW feature vector
    cur_keyfrm_->compute_bow();

    // update graph
    const auto cur_lms = cur_keyfrm_->get_landmarks();
    for (unsigned int idx = 0; idx < cur_lms.size(); ++idx) {
        auto lm = cur_lms.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // if `lm` does not have the observation information from `cur_keyfrm_`,
        // add the association between the keyframe and the landmark
        if (lm->is_observed_in_keyframe(cur_keyfrm_)) {
            // if `lm` is correctly observed, make it be checked by the local map cleaner
            local_map_cleaner_->add_fresh_landmark(lm);
            continue;
        }

        // update connection
        lm->add_observation(cur_keyfrm_, idx);
        // update geometry
        lm->update_normal_and_depth();
        lm->compute_descriptor();
    }
    cur_keyfrm_->graph_node_->update_connections();

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // Delete bad KF in LocalWindow
    DeleteBadInLocalWindow();
    // Add Keyframe to LocalWindow
    AddToLocalWindow(cur_keyfrm_);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // store the new keyframe to the map database
    map_db_->add_keyframe(cur_keyfrm_);
}

void mapping_module::create_new_landmarks() {
    // get the covisibilities of `cur_keyfrm_`
    // in order to triangulate landmarks between `cur_keyfrm_` and each of the covisibilities
    constexpr unsigned int num_covisibilities = 10;
    const auto cur_covisibilities = cur_keyfrm_->graph_node_->get_top_n_covisibilities(num_covisibilities * (is_monocular_ ? 2 : 1));

    // lowe's_ratio will not be used
    match::robust robust_matcher(0.0, false);

    // camera center of the current keyframe
    const Vec3_t cur_cam_center = cur_keyfrm_->get_cam_center();

    for (unsigned int i = 0; i < cur_covisibilities.size(); ++i) {
        // if any keyframe is queued, abort the triangulation
        if (1 < i && keyframe_is_queued()) {
            return;
        }

        // get the neighbor keyframe
        auto ngh_keyfrm = cur_covisibilities.at(i);

        // camera center of the neighbor keyframe
        const Vec3_t ngh_cam_center = ngh_keyfrm->get_cam_center();

        // compute the baseline between the current and neighbor keyframes
        const Vec3_t baseline_vec = ngh_cam_center - cur_cam_center;
        const auto baseline_dist = baseline_vec.norm();
        if (is_monocular_) {
            // if the scene scale is much smaller than the baseline, abort the triangulation
            const float median_depth_in_ngh = ngh_keyfrm->compute_median_depth(true);
            if (baseline_dist < 0.02 * median_depth_in_ngh) {
                continue;
            }
        }
        else {
            // for stereo setups, it needs longer baseline than the stereo baseline
            if (baseline_dist < ngh_keyfrm->camera_->true_baseline_) {
                continue;
            }
        }

        // estimate matches between the current and neighbor keyframes,
        // then reject outliers using Essential matrix computed from the two camera poses

        // (cur bearing) * E_ngh_to_cur * (ngh bearing) = 0
        // const Mat33_t E_ngh_to_cur = solve::essential_solver::create_E_21(ngh_keyfrm, cur_keyfrm_);
        const Mat33_t E_ngh_to_cur = solve::essential_solver::create_E_21(ngh_keyfrm->get_rotation(), ngh_keyfrm->get_translation(),
                                                                          cur_keyfrm_->get_rotation(), cur_keyfrm_->get_translation());

        // vector of matches (idx in the current, idx in the neighbor)
        std::vector<std::pair<unsigned int, unsigned int>> matches;
        robust_matcher.match_for_triangulation(cur_keyfrm_, ngh_keyfrm, E_ngh_to_cur, matches);

        // triangulation
        triangulate_with_two_keyframes(cur_keyfrm_, ngh_keyfrm, matches);
    }
}

void mapping_module::triangulate_with_two_keyframes(data::keyframe* keyfrm_1, data::keyframe* keyfrm_2,
                                                    const std::vector<std::pair<unsigned int, unsigned int>>& matches) {
    const module::two_view_triangulator triangulator(keyfrm_1, keyfrm_2, 1.0);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned int i = 0; i < matches.size(); ++i) {
        const auto idx_1 = matches.at(i).first;
        const auto idx_2 = matches.at(i).second;

        // triangulate between idx_1 and idx_2
        Vec3_t pos_w;
        if (!triangulator.triangulate(idx_1, idx_2, pos_w)) {
            // failed
            continue;
        }
        // succeeded

        // create a landmark object
        auto lm = new data::landmark(pos_w, keyfrm_1, map_db_);

        lm->add_observation(keyfrm_1, idx_1);
        lm->add_observation(keyfrm_2, idx_2);

        keyfrm_1->add_landmark(lm, idx_1);
        keyfrm_2->add_landmark(lm, idx_2);

        lm->compute_descriptor();
        lm->update_normal_and_depth();

        map_db_->add_landmark(lm);
        // wait for redundancy check
#ifdef USE_OPENMP
#pragma omp critical
#endif
        {
            local_map_cleaner_->add_fresh_landmark(lm);
        }
    }
}

void mapping_module::update_new_keyframe() {
    // get the targets to check landmark fusion
    const auto fuse_tgt_keyfrms = get_second_order_covisibilities(is_monocular_ ? 20 : 10, 5);

    // resolve the duplication of landmarks between the current keyframe and the targets
    fuse_landmark_duplication(fuse_tgt_keyfrms);

    // update the geometries
    const auto cur_landmarks = cur_keyfrm_->get_landmarks();
    for (const auto lm : cur_landmarks) {
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        lm->compute_descriptor();
        lm->update_normal_and_depth();
    }

    // update the graph
    cur_keyfrm_->graph_node_->update_connections();
}

std::unordered_set<data::keyframe*> mapping_module::get_second_order_covisibilities(const unsigned int first_order_thr,
                                                                                    const unsigned int second_order_thr) {
    const auto cur_covisibilities = cur_keyfrm_->graph_node_->get_top_n_covisibilities(first_order_thr);

    std::unordered_set<data::keyframe*> fuse_tgt_keyfrms;
    fuse_tgt_keyfrms.reserve(cur_covisibilities.size() * 2);

    for (const auto first_order_covis : cur_covisibilities) {
        if (first_order_covis->will_be_erased()) {
            continue;
        }

        // check if the keyframe is aleady inserted
        if (static_cast<bool>(fuse_tgt_keyfrms.count(first_order_covis))) {
            continue;
        }

        fuse_tgt_keyfrms.insert(first_order_covis);

        // get the covisibilities of the covisibility of the current keyframe
        const auto ngh_covisibilities = first_order_covis->graph_node_->get_top_n_covisibilities(second_order_thr);
        for (const auto second_order_covis : ngh_covisibilities) {
            if (second_order_covis->will_be_erased()) {
                continue;
            }
            // "the covisibilities of the covisibility" contains the current keyframe
            if (*second_order_covis == *cur_keyfrm_) {
                continue;
            }

            fuse_tgt_keyfrms.insert(second_order_covis);
        }
    }

    return fuse_tgt_keyfrms;
}

void mapping_module::fuse_landmark_duplication(const std::unordered_set<data::keyframe*>& fuse_tgt_keyfrms) {
    match::fuse matcher;

    {
        // reproject the landmarks observed in the current keyframe to each of the targets, and acquire
        // - additional matches
        // - duplication of matches
        // then, add matches and solve duplication
        auto cur_landmarks = cur_keyfrm_->get_landmarks();
        for (const auto fuse_tgt_keyfrm : fuse_tgt_keyfrms) {
            matcher.replace_duplication(fuse_tgt_keyfrm, cur_landmarks);
        }
    }

    {
        // reproject the landmarks observed in each of the targets to each of the current frame, and acquire
        // - additional matches
        // - duplication of matches
        // then, add matches and solve duplication
        std::unordered_set<data::landmark*> candidate_landmarks_to_fuse;
        candidate_landmarks_to_fuse.reserve(fuse_tgt_keyfrms.size() * cur_keyfrm_->num_keypts_);

        for (const auto fuse_tgt_keyfrm : fuse_tgt_keyfrms) {
            const auto fuse_tgt_landmarks = fuse_tgt_keyfrm->get_landmarks();

            for (const auto lm : fuse_tgt_landmarks) {
                if (!lm) {
                    continue;
                }
                if (lm->will_be_erased()) {
                    continue;
                }

                if (static_cast<bool>(candidate_landmarks_to_fuse.count(lm))) {
                    continue;
                }
                candidate_landmarks_to_fuse.insert(lm);
            }
        }

        matcher.replace_duplication(cur_keyfrm_, candidate_landmarks_to_fuse);
    }
}

void mapping_module::request_reset() {
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

bool mapping_module::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

void mapping_module::reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    spdlog::info("reset mapping module");
    keyfrms_queue_.clear();
    local_map_cleaner_->reset();
    reset_is_requested_ = false;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    mlLocalKeyFrames.clear();

    // Add resetting init flags
    mbVINSInited = false;
    mbFirstTry = true;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

}

void mapping_module::request_pause() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    pause_is_requested_ = true;
    std::lock_guard<std::mutex> lock2(mtx_keyfrm_queue_);
    abort_local_BA_ = true;
}

bool mapping_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
}

bool mapping_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_is_requested_ && !force_to_run_;
}

void mapping_module::pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    spdlog::info("pause mapping module");
    is_paused_ = true;
}

bool mapping_module::set_force_to_run(const bool force_to_run) {
    std::lock_guard<std::mutex> lock(mtx_pause_);

    if (force_to_run && is_paused_) {
        return false;
    }

    force_to_run_ = force_to_run;
    return true;
}

void mapping_module::resume() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);

    // if it has been already terminated, cannot resume
    if (is_terminated_) {
        return;
    }

    is_paused_ = false;
    pause_is_requested_ = false;

    // clear the queue
    for (auto& new_keyframe : keyfrms_queue_) {
        delete new_keyframe;
    }
    keyfrms_queue_.clear();

    spdlog::info("resume mapping module");
}

void mapping_module::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}

bool mapping_module::is_terminated() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return is_terminated_;
}

bool mapping_module::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void mapping_module::terminate() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    std::lock_guard<std::mutex> lock2(mtx_terminate_);
    is_paused_ = true;
    is_terminated_ = true;
}

} // namespace openvslam
