#include "openvslam/config.h"
#include "openvslam/system.h"
#include "openvslam/tracking_module.h"
#include "openvslam/mapping_module.h"
#include "openvslam/global_optimization_module.h"
#include "openvslam/camera/base.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/data/bow_database.h"
#include "openvslam/feature/orb_extractor.h"
#include "openvslam/match/projection.h"
#include "openvslam/util/image_converter.h"

#include <opencv2/core/eigen.hpp>

#include <chrono>
#include <unordered_map>

#include <spdlog/spdlog.h>

namespace openvslam {

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
bool tracking_module::GetVINSInited()
{ 
    return mapper_->GetVINSInited(); 
}

cv::Mat tracking_module::GetGravityVec()
{
    return mapper_->GetGravityVec(); 
}

void tracking_module::RecomputeIMUBiasAndCurrentNavstate(NavState& nscur)
{
    size_t N = mv20FramesReloc.size();

    //Test log
    if(N!=20) std::cerr<<"Frame vector size not 20 to compute bias after reloc??? size: "<<mv20FramesReloc.size()<<std::endl;

    // Estimate gyr bias
    Vector3d bg = global_optimization_module::OptimizeInitialGyroBias(mv20FramesReloc);
    // Update gyr bias of Frames
    for(size_t i=0; i<N; i++)
    {
        data::frame& frame = mv20FramesReloc[i];
        //Test log
        if(frame.GetNavState().Get_BiasGyr().norm()!=0 || frame.GetNavState().Get_dBias_Gyr().norm()!=0)
            std::cerr<<"Frame "<<frame.id_<<" gyr bias or delta bias not zero???"<<std::endl;

        frame.SetNavStateBiasGyr(bg);
    }
    // Re-compute IMU pre-integration
    std::vector<IMUPreintegrator> v19IMUPreint;
    v19IMUPreint.reserve(20-1);
    for(size_t i=0; i<N; i++)
    {
        if(i==0)
            continue;

        const data::frame& Fi = mv20FramesReloc[i-1];
        const data::frame& Fj = mv20FramesReloc[i];

        IMUPreintegrator imupreint;
        Fj.ComputeIMUPreIntSinceLastFrame(&Fi,imupreint);
        v19IMUPreint.push_back(imupreint);
    }
    // Construct [A1;A2;...;AN] * ba = [B1;B2;.../BN], solve ba
    cv::Mat A = cv::Mat::zeros(3*(N-2),3,CV_32F);
    cv::Mat B = cv::Mat::zeros(3*(N-2),1,CV_32F);
    const cv::Mat& gw = mapper_->GetGravityVec();
    const cv::Mat& Tcb = ConfigParam::GetMatT_cb();

    for(int i=0; i<N-2; i++)
    {
        const data::frame& F1 = mv20FramesReloc[i];
        const data::frame& F2 = mv20FramesReloc[i+1];
        const data::frame& F3 = mv20FramesReloc[i+2];
        const IMUPreintegrator& PreInt12 = v19IMUPreint[i];
        const IMUPreintegrator& PreInt23 = v19IMUPreint[i+1];
        // Delta time between frames
        double dt12 = PreInt12.getDeltaTime();
        double dt23 = PreInt23.getDeltaTime();
        // Pre-integrated measurements
        cv::Mat dp12 = util::converter::toCvMat(PreInt12.getDeltaP());
        cv::Mat dv12 = util::converter::toCvMat(PreInt12.getDeltaV());
        cv::Mat dp23 = util::converter::toCvMat(PreInt23.getDeltaP());
        cv::Mat Jpba12 = util::converter::toCvMat(PreInt12.getJPBiasa());
        cv::Mat Jvba12 = util::converter::toCvMat(PreInt12.getJVBiasa());
        cv::Mat Jpba23 = util::converter::toCvMat(PreInt23.getJPBiasa());
        // Pose of body in world frame
        cv::Mat aux1;
        eigen2cv(F1.cam_pose_cw_, aux1);
        cv::Mat aux2;
        eigen2cv(F2.cam_pose_cw_, aux2);
        cv::Mat aux3;
        eigen2cv(F3.cam_pose_cw_, aux3);
        cv::Mat Twb1 = util::converter::toCvMatInverse(aux1)*Tcb;
        cv::Mat Twb2 = util::converter::toCvMatInverse(aux2)*Tcb;
        cv::Mat Twb3 = util::converter::toCvMatInverse(aux3)*Tcb;
        // Position of body, Pwb
        cv::Mat pb1 = Twb1.rowRange(0,3).col(3);
        cv::Mat pb2 = Twb2.rowRange(0,3).col(3);
        cv::Mat pb3 = Twb3.rowRange(0,3).col(3);
        // Rotation of body, Rwb
        cv::Mat Rb1 = Twb1.rowRange(0,3).colRange(0,3);
        cv::Mat Rb2 = Twb2.rowRange(0,3).colRange(0,3);
        //cv::Mat Rb3 = Twb3.rowRange(0,3).colRange(0,3);
        // Stack to A/B matrix
        // Ai * ba = Bi
        cv::Mat Ai = Rb1*Jpba12*dt23 - Rb2*Jpba23*dt12 - Rb1*Jvba12*dt12*dt23;
        cv::Mat Bi = (pb2-pb3)*dt12 + (pb2-pb1)*dt23 + Rb2*dp23*dt12 - Rb1*dp12*dt23 + Rb1*dv12*dt12*dt23 + 0.5*gw*(dt12*dt12*dt23+dt12*dt23*dt23);
        Ai.copyTo(A.rowRange(3*i+0,3*i+3));
        Bi.copyTo(B.rowRange(3*i+0,3*i+3));

        //Test log
        if(fabs(F2.timestamp_-F1.timestamp_-dt12)>1e-6 || fabs(F3.timestamp_-F2.timestamp_-dt23)>1e-6) std::cerr<<"delta time not right."<<std::endl;
    }

    // Use svd to compute A*x=B, x=ba 3x1 vector
    // A = u*w*vt, u*w*vt*x=B
    // Then x = vt'*winv*u'*B
    cv::Mat w2,u2,vt2;
    // Note w2 is 3x1 vector by SVDecomp()
    // A is changed in SVDecomp() with cv::SVD::MODIFY_A for speed
    cv::SVDecomp(A,w2,u2,vt2,cv::SVD::MODIFY_A);
    // Compute winv
    cv::Mat w2inv=cv::Mat::eye(3,3,CV_32F);
    for(int i=0;i<3;i++)
    {
        if(fabs(w2.at<float>(i))<1e-10)
        {
            w2.at<float>(i) += 1e-10;
            // Test log
            std::cerr<<"w2(i) < 1e-10, w="<<std::endl<<w2<<std::endl;
        }
        w2inv.at<float>(i,i) = 1./w2.at<float>(i);
    }
    // Then y = vt'*winv*u'*B
    cv::Mat ba_cv = vt2.t()*w2inv*u2.t()*B;
    Vector3d ba = util::converter::toVector3d(ba_cv);

    // Update acc bias
    for(size_t i=0; i<N; i++)
    {
        data::frame& frame = mv20FramesReloc[i];
        //Test log
        if(frame.GetNavState().Get_BiasAcc().norm()!=0 || frame.GetNavState().Get_dBias_Gyr().norm()!=0 || frame.GetNavState().Get_dBias_Acc().norm()!=0)
            std::cerr<<"Frame "<<frame.id_<<" acc bias or delta bias not zero???"<<std::endl;

        frame.SetNavStateBiasAcc(ba);
    }

    // Compute Velocity of the last 2 Frames
    Vector3d Pcur;
    Vector3d Vcur;
    Matrix3d Rcur;
    {
        data::frame& F1 = mv20FramesReloc[N-2];
        data::frame& F2 = mv20FramesReloc[N-1];
        const IMUPreintegrator& imupreint = v19IMUPreint.back();
        const double dt12 = imupreint.getDeltaTime();
        const Vector3d dp12 = imupreint.getDeltaP();
        const Vector3d gweig = util::converter::toVector3d(gw);
        const Matrix3d Jpba12 = imupreint.getJPBiasa();
        const Vector3d dv12 = imupreint.getDeltaV();
        const Matrix3d Jvba12 = imupreint.getJVBiasa();

        // Velocity of Previous Frame
        // P2 = P1 + V1*dt12 + 0.5*gw*dt12*dt12 + R1*(dP12 + Jpba*ba + Jpbg*0)
        cv::Mat aux4;
        eigen2cv(F1.cam_pose_cw_, aux4);
        cv::Mat aux5;
        eigen2cv(F2.cam_pose_cw_, aux5);
        cv::Mat Twb1 = util::converter::toCvMatInverse(aux4)*Tcb;
        cv::Mat Twb2 = util::converter::toCvMatInverse(aux5)*Tcb;
        Vector3d P1 = util::converter::toVector3d(Twb1.rowRange(0,3).col(3));
        /*Vector3d */Pcur = util::converter::toVector3d(Twb2.rowRange(0,3).col(3));
        Matrix3d R1 = util::converter::toMatrix3d(Twb1.rowRange(0,3).colRange(0,3));
        /*Matrix3d */Rcur = util::converter::toMatrix3d(Twb2.rowRange(0,3).colRange(0,3));
        Vector3d V1 = 1./dt12*( Pcur - P1 - 0.5*gweig*dt12*dt12 - R1*(dp12 + Jpba12*ba) );

        // Velocity of Current Frame
        Vcur = V1 + gweig*dt12 + R1*( dv12 + Jvba12*ba );

        // Test log
        if(F2.id_ != curr_frm_.id_) std::cerr<<"framecur.mnId != mCurrentFrame.mnId. why??"<<std::endl;
        if(fabs(F2.timestamp_-F1.timestamp_-dt12)>1e-6) std::cerr<<"timestamp not right?? in compute vel"<<std::endl;
    }

    // Set NavState of Current Frame, P/V/R/bg/ba/dbg/dba
    nscur.Set_Pos(Pcur);
    nscur.Set_Vel(Vcur);
    nscur.Set_Rot(Rcur);
    nscur.Set_BiasGyr(bg);
    nscur.Set_BiasAcc(ba);
    nscur.Set_DeltaBiasGyr(Vector3d::Zero());
    nscur.Set_DeltaBiasAcc(Vector3d::Zero());

    //mv20FramesReloc
}

bool tracking_module::TrackLocalMapWithIMU(bool bMapUpdated)
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    update_local_map();
    search_local_landmarks();
    data::frame backupCurrentFrame = curr_frm_;
    data::frame backupLastFrame = last_frm_;
    Mat44_t backupLastKeyFramePose = last_frm_.ref_keyfrm_->get_cam_pose();
    //eigen2cv(last_frm_.ref_keyfrm_->get_cam_pose(), backupLastKeyFramePose);

    // Map updated, optimize with last KeyFrame
    if(mapper_->GetFirstVINSInited() || bMapUpdated)
    {
        // Get initial pose from Last KeyFrame
        IMUPreintegrator imupreint = GetIMUPreIntSinceLastKF(&curr_frm_, last_frm_.ref_keyfrm_, mvIMUSinceLastKF);

        // Test log
        if(mapper_->GetFirstVINSInited() && !bMapUpdated) std::cerr<<"1-FirstVinsInit, but not bMapUpdated. shouldn't"<<std::endl;
        if(curr_frm_.GetNavState().Get_dBias_Acc().norm() > 1e-6) std::cerr<<"TrackLocalMapWithIMU current Frame dBias acc not zero"<<std::endl;
        if(last_frm_.GetNavState().Get_dBias_Gyr().norm() > 1e-6) std::cerr<<"TrackLocalMapWithIMU current Frame dBias gyr not zero"<<std::endl;


        global_optimization_module::PoseOptimization(&curr_frm_,last_frm_.ref_keyfrm_,imupreint,mapper_->GetGravityVec(),true);

        mbIsKeyframeTracked = true;
        //saveDebugStates("/home/sicong/VIORB_new/ORB_SLAM2/tmp/IMUpredic_aftertracklm.txt","../../../tmp/IMUpredict_aftertracklm.txt");
    }
    // Map not updated, optimize with last Frame
    else
    {
        // Get initial pose from Last Frame
        IMUPreintegrator imupreint = GetIMUPreIntSinceLastFrame(&curr_frm_, &last_frm_);
        global_optimization_module::PoseOptimization(&curr_frm_,last_frm_.ref_keyfrm_,imupreint,mapper_->GetGravityVec(),true);
        mbIsKeyframeTracked = false;
        //saveDebugStates("/home/sicong/VIORB_new/ORB_SLAM2/tmp/IMUpredic_aftertracklm.txt","../../../tmp/IMUpredict_aftertracklm.txt");
    }

    num_tracked_lms_ = 0;

    // Update MapPoints Statistics
    for(int i=0; i<curr_frm_.num_keypts_; i++)
    {
        if(curr_frm_.landmarks_[i])
        {
            if(!curr_frm_.outlier_flags_[i])
            {

                // the observation has been considered as inlier in the pose optimization
                assert(curr_frm_.landmarks_[i]->has_observation());
                // count up
                ++num_tracked_lms_;
                // increment the number of tracked frame
                curr_frm_.landmarks_[i]->increase_num_observed();
            }
            else 
                curr_frm_.landmarks_[i] = static_cast<data::landmark*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(curr_frm_.id_<last_reloc_frm_id_+camera_->fps_ && num_tracked_lms_<30)
        return false;

    if(num_tracked_lms_<15)
    {
        // revert 
        curr_frm_ = backupCurrentFrame;
        last_frm_ = backupLastFrame;
        last_frm_.ref_keyfrm_->set_cam_pose(backupLastKeyFramePose);

        return false;
    }
    else
        return true;
}

void tracking_module::PredictNavStateByIMU(bool bMapUpdated)
{
    if(!mapper_->GetVINSInited()) std::cerr<<"mpLocalMapper->GetVINSInited() not, shouldn't in PredictNavStateByIMU"<<std::endl;

    // Map updated, optimize with last KeyFrame
    if(mapper_->GetFirstVINSInited() || bMapUpdated)
    {
        if(mapper_->GetFirstVINSInited() && !bMapUpdated) std::cerr<<"2-FirstVinsInit, but not bMapUpdated. shouldn't"<<std::endl;
        //cout<<"tracking last kf"<<endl;
        saveIMUDataPerImage(true);
        mbIsKeyframeTracked = true;
        // Compute IMU Pre-integration
        mIMUPreIntInTrack = GetIMUPreIntSinceLastKF(&curr_frm_, last_frm_.ref_keyfrm_, mvIMUSinceLastKF);

        // Get initial NavState&pose from Last KeyFrame
        curr_frm_.SetInitialNavStateAndBias(last_frm_.ref_keyfrm_->GetNavState());
        curr_frm_.UpdateNavState(mIMUPreIntInTrack,util::converter::toVector3d(mapper_->GetGravityVec()));
        curr_frm_.UpdatePoseFromNS(ConfigParam::GetMatTbc());

        // Test log
        // Updated KF by Local Mapping. Should be the same as mpLastKeyFrame
        if(curr_frm_.GetNavState().Get_dBias_Acc().norm() > 1e-6) std::cerr<<"PredictNavStateByIMU1 current Frame dBias acc not zero"<<std::endl;
        if(curr_frm_.GetNavState().Get_dBias_Gyr().norm() > 1e-6) std::cerr<<"PredictNavStateByIMU1 current Frame dBias gyr not zero"<<std::endl;
    }
    // Map not updated, optimize with last Frame
    else
    {
        saveIMUDataPerImage(false);
        mbIsKeyframeTracked = false;

        // Compute IMU Pre-integration
        mIMUPreIntInTrack = GetIMUPreIntSinceLastFrame(&curr_frm_, &last_frm_);

        // Get initial pose from Last Frame
        curr_frm_.SetInitialNavStateAndBias(last_frm_.GetNavState());
        curr_frm_.UpdateNavState(mIMUPreIntInTrack,util::converter::toVector3d(mapper_->GetGravityVec()));
        curr_frm_.UpdatePoseFromNS(ConfigParam::GetMatTbc());

        // Test log
        if(curr_frm_.GetNavState().Get_dBias_Acc().norm() > 1e-6) std::cerr<<"PredictNavStateByIMU2 current Frame dBias acc not zero"<<std::endl;
        if(curr_frm_.GetNavState().Get_dBias_Gyr().norm() > 1e-6) std::cerr<<"PredictNavStateByIMU2 current Frame dBias gyr not zero"<<std::endl;
    }
}

bool tracking_module::TrackWithIMU(bool bMapUpdated)
{
    //ORBmatcher matcher(0.9,true);
    match::projection projection_matcher(0.9, true);

    // VINS has been inited in this function
    if(!mapper_->GetVINSInited()) std::cerr<<"local mapping VINS not inited. why call TrackWithIMU?"<<std::endl;

    // Predict NavState&Pose by IMU
    // And compute the IMU pre-integration for PoseOptimization
    PredictNavStateByIMU(bMapUpdated);
    //saveDebugStates("/home/sicong/VIORB_new/ORB_SLAM2/tmp/IMUpredic_beforetrack.txt","../../../tmp/IMUpredict_aftertracklm.txt");

    fill(curr_frm_.landmarks_.begin(),curr_frm_.landmarks_.end(),static_cast<data::landmark*>(NULL));

    // Project points seen in previous frame
    int th;
    if(camera_->setup_type_!=camera::setup_type_t::Stereo)
        th=15;
    else
        th=7;
    int nmatches = projection_matcher.match_current_and_last_frames(curr_frm_,last_frm_,th);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(curr_frm_.landmarks_.begin(),curr_frm_.landmarks_.end(),static_cast<data::landmark*>(NULL));
        nmatches = projection_matcher.match_current_and_last_frames(curr_frm_,last_frm_,2*th);
    }

    if(nmatches<20)
        return false;

    data::frame backupCurrentFrame = curr_frm_;
    data::frame backupLastFrame = last_frm_;
    Mat44_t backupLastKeyFramePose = last_frm_.ref_keyfrm_->get_cam_pose();

    // Pose optimization. false: no need to compute marginalized for current Frame
    if(mapper_->GetFirstVINSInited() || bMapUpdated)
    {
        global_optimization_module::PoseOptimization(&curr_frm_,last_frm_.ref_keyfrm_,mIMUPreIntInTrack,mapper_->GetGravityVec(),true);
        mbIsKeyframeTracked = true;
        //saveDebugStates("/home/sicong/VIORB_new/ORB_SLAM2/tmp/IMUpredic_aftertrackkf.txt","../../../tmp/IMUpredict_aftertracklm.txt");
    }
    else
    {
        global_optimization_module::PoseOptimization(&curr_frm_,&last_frm_,mIMUPreIntInTrack,mapper_->GetGravityVec(),false);
        mbIsKeyframeTracked = false;
        //saveDebugStates("/home/sicong/VIORB_new/ORB_SLAM2/tmp/IMUpredic_aftertrackkf.txt","../../../tmp/IMUpredict_aftertracklm.txt");
    }

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<curr_frm_.num_keypts_; i++)
    {
        if(curr_frm_.landmarks_[i])
        {
            if(curr_frm_.outlier_flags_[i])
            {
                data::landmark* pMP = curr_frm_.landmarks_[i];

                curr_frm_.landmarks_[i]=static_cast<data::landmark*>(NULL);
                curr_frm_.outlier_flags_[i]=false;
                pMP->is_observable_in_tracking_ = false;
                pMP->identifier_in_local_lm_search_ = curr_frm_.id_;
                nmatches--;
            }
            else if(curr_frm_.landmarks_[i]->num_observations()>0)
                nmatchesMap++;
        }
    }

    // return nmatchesMap>=10;
    if (nmatchesMap>=10)
    {
        return true;
    }
    else
    {
        // revert 
        curr_frm_ = backupCurrentFrame;
        last_frm_ = backupLastFrame;
        last_frm_.ref_keyfrm_->set_cam_pose(backupLastKeyFramePose);

        return false;
    }
}

IMUPreintegrator tracking_module::GetIMUPreIntSinceLastKF(data::frame* pCurF, data::keyframe* pLastKF, const std::vector<IMUData>& vIMUSInceLastKF)
{
    // Reset pre-integrator first
    IMUPreintegrator IMUPreInt;
    IMUPreInt.reset();

    Vector3d bg = pLastKF->GetNavState().Get_BiasGyr();
    Vector3d ba = pLastKF->GetNavState().Get_BiasAcc();

    // remember to consider the gap between the last KF and the first IMU
    {
        const IMUData& imu = vIMUSInceLastKF.front();
        double dt = imu._t - pLastKF->timestamp_;
        IMUPreInt.update(imu._g - bg, imu._a - ba, dt);

        // Test log
        if(dt < 0)
        {
            std::cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", last KF vs last imu time: "<<pLastKF->timestamp_<<" vs "<<imu._t<<std::endl;
            std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
        }
    }

    // integrate each imu
    for(size_t i=0; i<vIMUSInceLastKF.size(); i++)
    {
        const IMUData& imu = vIMUSInceLastKF[i];
        double nextt;
        if(i==vIMUSInceLastKF.size()-1)
            nextt = pCurF->timestamp_;         // last IMU, next is this KeyFrame
        else
            nextt = vIMUSInceLastKF[i+1]._t;  // regular condition, next is imu data

        // delta time
        double dt = nextt - imu._t;
        // update pre-integrator
        IMUPreInt.update(imu._g - bg, imu._a - ba, dt);


        // Test log
        if(dt <= 0)
        {
            std::cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this vs next time: "<<imu._t<<" vs "<<nextt<<std::endl;
            std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
        }
    }

    return IMUPreInt;
}


IMUPreintegrator tracking_module::GetIMUPreIntSinceLastFrame(data::frame* pCurF, data::frame* pLastF)
{
    // Reset pre-integrator first
    IMUPreintegrator IMUPreInt;
    IMUPreInt.reset();

    pCurF->ComputeIMUPreIntSinceLastFrame(pLastF,IMUPreInt);

    return IMUPreInt;
}


Mat44_t tracking_module::track_monocular_image_VI(const cv::Mat& img, const std::vector<IMUData> &vimu, const double timestamp, const cv::Mat& mask)
{
    mvIMUSinceLastKF.insert(mvIMUSinceLastKF.end(), vimu.begin(),vimu.end());

    const auto start = std::chrono::system_clock::now();

    // color conversion
    img_gray_ = img;
    util::convert_to_grayscale(img_gray_, camera_->color_order_);

    // create current frame object
    if (tracking_state_ == tracker_state_t::NotInitialized || tracking_state_ == tracker_state_t::Initializing) {
        curr_frm_ = data::frame(img_gray_, timestamp, vimu, ini_extractor_left_, bow_vocab_, camera_, cfg_->true_depth_thr_, mK, mask);
    }
    else {
        curr_frm_ = data::frame(img_gray_, timestamp, vimu, extractor_left_, bow_vocab_, camera_, cfg_->true_depth_thr_, mK, mask, last_frm_.ref_keyfrm_);
    }

    track();

    const auto end = std::chrono::system_clock::now();
    elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return curr_frm_.cam_pose_cw_;
}

void tracking_module::saveIMUDataPerImage(bool isKeyframe)
{
    static bool isFirst = true;
    // abbreviation
    // CVM: constant velocity model

    // save the state predicted by IMU
    std::ofstream f;
    char filename[] = "../../../tmp/IMUcount.txt";
    
    if (mapper_->GetVINSInited())
    {
        if (isFirst)
        {
            f.open(filename, std::ios_base::out);
        }
        else
        {
            f.open(filename, std::ios_base::app);
        }

        // Different operation, according to whether the map is updated
        if (isKeyframe)
        {
            // the prediction was done based on keyframe
            f << std::fixed;

            f << std::setprecision(6) << curr_frm_.timestamp_ << std::setprecision(7) << " ";
            f << mvIMUSinceLastKF.size() << " ";
            f << std::setprecision(6) << last_frm_.ref_keyfrm_->timestamp_ << std::setprecision(7) << " ";
            f << 0 << " ";
            f << std::endl;
            f.close();  
        }
        else 
        {
            // the prediction was done based on last frame
            f << std::fixed;

            f << std::setprecision(6) << curr_frm_.timestamp_ << std::setprecision(7) << " ";
            f << curr_frm_.mvIMUDataSinceLastFrame.size() << " ";
            f << std::setprecision(6) << last_frm_.timestamp_ << std::setprecision(7) << " ";
            f << 1 << " ";
            f << std::endl;
            f.close();  
        }

    }

    isFirst = false;
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

tracking_module::tracking_module(const std::shared_ptr<config>& cfg, system* system, data::map_database* map_db,
                                 data::bow_vocabulary* bow_vocab, data::bow_database* bow_db, ConfigParam* pParams)
    : cfg_(cfg), camera_(cfg->camera_), system_(system), map_db_(map_db), bow_vocab_(bow_vocab), bow_db_(bow_db),
      initializer_(cfg->camera_->setup_type_, map_db, bow_db, cfg->yaml_node_),
      frame_tracker_(camera_, 10), relocalizer_(bow_db_), pose_optimizer_(),
      keyfrm_inserter_(cfg_->camera_->setup_type_, cfg_->true_depth_thr_, map_db, bow_db, 0, cfg_->camera_->fps_) {
    spdlog::debug("CONSTRUCT: tracking_module");

    extractor_left_ = new feature::orb_extractor(cfg_->orb_params_);
    if (camera_->setup_type_ == camera::setup_type_t::Monocular) {
        ini_extractor_left_ = new feature::orb_extractor(cfg_->orb_params_);
        ini_extractor_left_->set_max_num_keypoints(ini_extractor_left_->get_max_num_keypoints() * 2);
    }
    if (camera_->setup_type_ == camera::setup_type_t::Stereo) {
        extractor_right_ = new feature::orb_extractor(cfg_->orb_params_);
    }

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    mbCreateNewKFAfterReloc = false;
    mbRelocBiasPrepare = false;
    mpParams = pParams;

    /*YAML::Node yaml_node = YAML::LoadFile(cfg_->config_file_path_);
    float fx = yaml_node["Camera.fx"].as<float>();
    float fy = yaml_node["Camera.fy"].as<float>();
    float cx = yaml_node["Camera.cx"].as<float>();
    float cy = yaml_node["Camera.cy"].as<float>();

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;*/
    cv::Mat K = ConfigParam::GetCamMatrix();
    K.copyTo(mK);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
}

tracking_module::~tracking_module() {
    delete extractor_left_;
    extractor_left_ = nullptr;
    delete extractor_right_;
    extractor_right_ = nullptr;
    delete ini_extractor_left_;
    ini_extractor_left_ = nullptr;

    spdlog::debug("DESTRUCT: tracking_module");
}


void tracking_module::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
    keyfrm_inserter_.set_mapping_module(mapper);
}

void tracking_module::set_global_optimization_module(global_optimization_module* global_optimizer) {
    global_optimizer_ = global_optimizer;
}

void tracking_module::set_mapping_module_status(const bool mapping_is_enabled) {
    std::lock_guard<std::mutex> lock(mtx_mapping_);
    mapping_is_enabled_ = mapping_is_enabled;
}

bool tracking_module::get_mapping_module_status() const {
    std::lock_guard<std::mutex> lock(mtx_mapping_);
    return mapping_is_enabled_;
}

std::vector<cv::KeyPoint> tracking_module::get_initial_keypoints() const {
    return initializer_.get_initial_keypoints();
}

std::vector<int> tracking_module::get_initial_matches() const {
    return initializer_.get_initial_matches();
}

Mat44_t tracking_module::track_monocular_image(const cv::Mat& img, const double timestamp, const cv::Mat& mask) {
    const auto start = std::chrono::system_clock::now();

    // color conversion
    img_gray_ = img;
    util::convert_to_grayscale(img_gray_, camera_->color_order_);

    // create current frame object
    if (tracking_state_ == tracker_state_t::NotInitialized || tracking_state_ == tracker_state_t::Initializing) {
        curr_frm_ = data::frame(img_gray_, timestamp, ini_extractor_left_, bow_vocab_, camera_, cfg_->true_depth_thr_, mask);
    }
    else {
        curr_frm_ = data::frame(img_gray_, timestamp, extractor_left_, bow_vocab_, camera_, cfg_->true_depth_thr_, mask);
    }

    track();

    const auto end = std::chrono::system_clock::now();
    elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return curr_frm_.cam_pose_cw_;
}

Mat44_t tracking_module::track_stereo_image(const cv::Mat& left_img_rect, const cv::Mat& right_img_rect, const double timestamp, const cv::Mat& mask) {
    const auto start = std::chrono::system_clock::now();

    // color conversion
    img_gray_ = left_img_rect;
    cv::Mat right_img_gray = right_img_rect;
    util::convert_to_grayscale(img_gray_, camera_->color_order_);
    util::convert_to_grayscale(right_img_gray, camera_->color_order_);

    // create current frame object
    curr_frm_ = data::frame(img_gray_, right_img_gray, timestamp, extractor_left_, extractor_right_, bow_vocab_, camera_, cfg_->true_depth_thr_, mask);

    track();

    const auto end = std::chrono::system_clock::now();
    elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return curr_frm_.cam_pose_cw_;
}

Mat44_t tracking_module::track_RGBD_image(const cv::Mat& img, const cv::Mat& depthmap, const double timestamp, const cv::Mat& mask) {
    const auto start = std::chrono::system_clock::now();

    // color and depth scale conversion
    img_gray_ = img;
    cv::Mat img_depth = depthmap;
    util::convert_to_grayscale(img_gray_, camera_->color_order_);
    util::convert_to_true_depth(img_depth, cfg_->depthmap_factor_);

    // create current frame object
    curr_frm_ = data::frame(img_gray_, img_depth, timestamp, extractor_left_, bow_vocab_, camera_, cfg_->true_depth_thr_, mask);

    track();

    const auto end = std::chrono::system_clock::now();
    elapsed_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return curr_frm_.cam_pose_cw_;
}

void tracking_module::reset() {
    spdlog::info("resetting system");

    initializer_.reset();
    keyfrm_inserter_.reset();

    mapper_->request_reset();
    global_optimizer_->request_reset();

    bow_db_->clear();
    map_db_->clear();

    data::frame::next_id_ = 0;
    data::keyframe::next_id_ = 0;
    data::landmark::next_id_ = 0;

    last_reloc_frm_id_ = 0;

    tracking_state_ = tracker_state_t::NotInitialized;
}

void tracking_module::track() {
    if (tracking_state_ == tracker_state_t::NotInitialized) {
        tracking_state_ = tracker_state_t::Initializing;
    }

    last_tracking_state_ = tracking_state_;

    // check if pause is requested
    check_and_execute_pause();
    while (is_paused()) {
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // LOCK the map database
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // Different operation, according to whether the map is updated
    bool bMapUpdated = false;
    if(mapper_->GetMapUpdateFlagForTracking())
    {
        bMapUpdated = true;
        mapper_->SetMapUpdateFlagInTracking(false);
    }
    if(global_optimizer_->GetMapUpdateFlagForTracking())
    {
        bMapUpdated = true;
        global_optimizer_->SetMapUpdateFlagInTracking(false);
    }
    if(curr_frm_.id_ == last_reloc_frm_id_ + 20)
    {
        bMapUpdated = true;
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    if (tracking_state_ == tracker_state_t::Initializing) {
        if (!initialize()) {
            return;
        }

        // update the reference keyframe, local keyframes, and local landmarks
        update_local_map();

        // pass all of the keyframes to the mapping module
        const auto keyfrms = map_db_->get_all_keyframes();
        for (const auto keyfrm : keyfrms) {
            mapper_->queue_keyframe(keyfrm);
        }

        // state transition to Tracking mode
        tracking_state_ = tracker_state_t::Tracking;
    }
    else {
        // apply replace of landmarks observed in the last frame
        apply_landmark_replace();
        // update the camera pose of the last frame
        // because the mapping module might optimize the camera pose of the last frame's reference keyframe
        update_last_frame();

        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        bool succeeded;
        #ifdef TRACK_WITH_IMU
                // If Visual-Inertial is initialized
                if(mapper_->GetVINSInited())
                {
                    // 20 Frames after reloc, track with only vision
                    if(mbRelocBiasPrepare)
                    {
                        succeeded = track_current_frame();
                    }
                    else
                    {
                        succeeded = TrackWithIMU(bMapUpdated);
                    }
                }
                // If Visual-Inertial not initialized, keep the same as pure-vslam
                else
        #endif
                {
                    succeeded = track_current_frame();
                }
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------

        // set the reference keyframe of the current frame
        curr_frm_.ref_keyfrm_ = ref_keyfrm_;

        // update the local map and optimize the camera pose of the current frame
        if (succeeded) {
            //-------------------------------------------------------------------------------------------
            //-------------------------------------------------------------------------------------------
            //-------------------------------------------------------------------------------------------
            #ifndef TRACK_WITH_IMU
                update_local_map();
                succeeded = optimize_current_frame_with_local_map();
            #else
                if(!mapper_->GetVINSInited())
                {
                    update_local_map();
                    succeeded = optimize_current_frame_with_local_map();
                }
                else
                {
                    if(mbRelocBiasPrepare)
                    {
                        // 20 Frames after reloc, track with only vision
                        update_local_map();
                        succeeded = optimize_current_frame_with_local_map();

                    }
                    else
                    {
                        //cout<<"tracking localmap with imu "<<trackingcounts++<<endl;
                        succeeded = TrackLocalMapWithIMU(bMapUpdated);
                    }
                }
            #endif
            //-------------------------------------------------------------------------------------------
            //-------------------------------------------------------------------------------------------
            //-------------------------------------------------------------------------------------------
        }

        // state transition
        //tracking_state_ = succeeded ? tracker_state_t::Tracking : tracker_state_t::Lost;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        if (succeeded)
        {
            tracking_state_ = tracker_state_t::Tracking;
            // Add Frames to re-compute IMU bias after reloc
            if(mbRelocBiasPrepare)
            {
                std::cout<<"call re-computing bias function here"<<std::endl;
                mv20FramesReloc.push_back(curr_frm_);

                // Before creating new keyframe
                // Use 20 consecutive frames to re-compute IMU bias
                if(curr_frm_.id_ == last_reloc_frm_id_+20-1)
                {
                    NavState nscur;
                    RecomputeIMUBiasAndCurrentNavstate(nscur);
                    // Update NavState of CurrentFrame
                    curr_frm_.SetNavState(nscur);
                    // Clear flag and Frame buffer
                    mbRelocBiasPrepare = false;
                    mv20FramesReloc.clear();

                    // Release LocalMapping. To ensure to insert new keyframe.
                    mapper_->resume();
                    // Create new KeyFrame
                    mbCreateNewKFAfterReloc = true;

                    //Debug log
                    std::cout<<"NavState recomputed."<<std::endl;
                    std::cout<<"V:"<<curr_frm_.GetNavState().Get_V().transpose()<<std::endl;
                    std::cout<<"bg:"<<curr_frm_.GetNavState().Get_BiasGyr().transpose()<<std::endl;
                    std::cout<<"ba:"<<curr_frm_.GetNavState().Get_BiasAcc().transpose()<<std::endl;
                    std::cout<<"dbg:"<<curr_frm_.GetNavState().Get_dBias_Gyr().transpose()<<std::endl;
                    std::cout<<"dba:"<<curr_frm_.GetNavState().Get_dBias_Acc().transpose()<<std::endl;
                }
            }
        }
        else
        {
            //tracking_state_ = tracker_state_t::Lost;
            if (mapper_->GetVINSInited() && !mbRelocBiasPrepare)
            {
                std::cout << std::fixed << std::setprecision(6);
                        
                if (tracking_state_ == tracker_state_t::Tracking)
                {
                    tracking_state_ = tracker_state_t::ImuOnlyTracking;
                    mTimestampLastLost = curr_frm_.timestamp_;

                    std::cout << "GOING TO IMU ONLY MODE!!!! " << curr_frm_.timestamp_ << std::endl;

                }
                else if (tracking_state_ == tracker_state_t::ImuOnlyTracking)
                {
                    std::cout<<"curr_frm_.timestamp_: "<<curr_frm_.timestamp_<< "mTimestampLastLost:" << mTimestampLastLost <<std::endl;
                    if (curr_frm_.timestamp_ - mTimestampLastLost > IMU_SAFE_WINDOW)
                    {
                        tracking_state_ = tracker_state_t::Lost;
                        std::cout << "QUITTING IMU ONLY MODE!!!! " << curr_frm_.timestamp_ << std::endl;
                    }
                }
            }
            else
            {
                tracking_state_ = tracker_state_t::Lost;
            }
            
            // Clear Frame vectors for reloc bias computation
            if(mv20FramesReloc.size()>0)
                mv20FramesReloc.clear();
        }
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------

        // update the motion model
        if (succeeded) {
            update_motion_model();
        }

        // update the frame statistics
        map_db_->update_frame_statistics(curr_frm_, tracking_state_ == tracker_state_t::Lost);

        // if tracking is failed within 5.0 sec after initialization, reset the system
        constexpr float init_retry_thr = 5.0;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        /*if (tracking_state_ == tracker_state_t::Lost
            && curr_frm_.id_ - initializer_.get_initial_frame_id() < camera_->fps_ * init_retry_thr)*/
        if (tracking_state_ == tracker_state_t::Lost && !mapper_->GetVINSInited())  
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        {
            spdlog::info("tracking lost within {} sec after initialization", init_retry_thr);
            system_->request_reset();
            return;
        }

        // show message if tracking has been lost
        if (last_tracking_state_ != tracker_state_t::Lost && tracking_state_ == tracker_state_t::Lost) {
            spdlog::info("tracking lost: frame {}", curr_frm_.id_);
        }

        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        // check to insert the new keyframe derived from the current frame
        if (succeeded && (new_keyframe_is_needed() || mbCreateNewKFAfterReloc)) {
            insert_new_keyframe();
        }
        // Clear flag
        if(succeeded && mbCreateNewKFAfterReloc)
        {
            mbCreateNewKFAfterReloc = false;
        }
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------

        // tidy up observations
        for (unsigned int idx = 0; idx < curr_frm_.num_keypts_; ++idx) {
            if (curr_frm_.landmarks_.at(idx) && curr_frm_.outlier_flags_.at(idx)) {
                curr_frm_.landmarks_.at(idx) = nullptr;
            }
        }

        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        // Clear First-Init flag
        if(succeeded && mapper_->GetFirstVINSInited())
        {
            mapper_->SetFirstVINSInited(false);
        }
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
    }

    // store the relative pose from the reference keyframe to the current frame
    // to update the camera pose at the beginning of the next tracking process
    if (curr_frm_.cam_pose_cw_is_valid_) {
        last_cam_pose_from_ref_keyfrm_ = curr_frm_.cam_pose_cw_ * curr_frm_.ref_keyfrm_->get_cam_pose_inv();
    }

    // update last frame
    last_frm_ = curr_frm_;
}

bool tracking_module::initialize() {
    // try to initialize with the current frame
    initializer_.initialize(curr_frm_, mvIMUSinceLastKF);
    

    // if map building was failed -> reset the map database
    if (initializer_.get_state() == module::initializer_state_t::Wrong) {
        // reset
        system_->request_reset();
        return false;
    }

    // if initializing was failed -> try to initialize with the next frame
    if (initializer_.get_state() != module::initializer_state_t::Succeeded) {
        return false;
    }

    // succeeded
    return true;
}

bool tracking_module::track_current_frame() {
    bool succeeded = false;
    if (tracking_state_ == tracker_state_t::Tracking) {
        // Tracking mode
        if (velocity_is_valid_ && last_reloc_frm_id_ + 2 < curr_frm_.id_) {
            // if the motion model is valid
            succeeded = frame_tracker_.motion_based_track(curr_frm_, last_frm_, velocity_);
        }
        if (!succeeded) {
            succeeded = frame_tracker_.bow_match_based_track(curr_frm_, last_frm_, ref_keyfrm_);
        }
        if (!succeeded) {
            succeeded = frame_tracker_.robust_match_based_track(curr_frm_, last_frm_, ref_keyfrm_);
        }
    }
    else {
        // Lost mode
        // try to relocalize
        succeeded = relocalizer_.relocalize(curr_frm_);
        if (succeeded) {
            //-------------------------------------------------------------------------------------------
            //-------------------------------------------------------------------------------------------
            //-------------------------------------------------------------------------------------------
            if(!mapper_->GetVINSInited()) std::cerr<<"VINS not inited? why."<<std::endl;
            mbRelocBiasPrepare = true;
            //-------------------------------------------------------------------------------------------
            //-------------------------------------------------------------------------------------------
            //-------------------------------------------------------------------------------------------
            last_reloc_frm_id_ = curr_frm_.id_;
        }
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        else if(tracking_state_ == tracker_state_t::ImuOnlyTracking)
        {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Trying to track with IMU only " << curr_frm_.timestamp_ << std::endl;

            succeeded = TrackWithIMU(true);
            if (!succeeded)
            {
                succeeded = TrackLocalMapWithIMU(true);
            }
            else
            {
                std::cout << "Recovered with IMU only (previous keyframe)!!! " << curr_frm_.timestamp_ << std::endl;
            }

            cv::Mat mCamPose;
            eigen2cv(curr_frm_.cam_pose_cw_, mCamPose);
            map_db_->AddIMUTrackedFrames(mCamPose);
            //mpMapDrawer->SetCurrentCameraPose(curr_frm_.mTcw);

            if (succeeded)
            {
                tracking_state_ = tracker_state_t::Tracking;
                std::cout << "Recovered with IMU only (local map)!!! " << curr_frm_.timestamp_ << std::endl;
            }
        }
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
    }
    return succeeded;
}

void tracking_module::update_motion_model() {
    if (last_frm_.cam_pose_cw_is_valid_) {
        Mat44_t last_frm_cam_pose_wc = Mat44_t::Identity();
        last_frm_cam_pose_wc.block<3, 3>(0, 0) = last_frm_.get_rotation_inv();
        last_frm_cam_pose_wc.block<3, 1>(0, 3) = last_frm_.get_cam_center();
        velocity_is_valid_ = true;
        velocity_ = curr_frm_.cam_pose_cw_ * last_frm_cam_pose_wc;
    }
    else {
        velocity_is_valid_ = false;
        velocity_ = Mat44_t::Identity();
    }
}

void tracking_module::apply_landmark_replace() {
    for (unsigned int idx = 0; idx < last_frm_.num_keypts_; ++idx) {
        auto lm = last_frm_.landmarks_.at(idx);
        if (!lm) {
            continue;
        }

        auto replaced_lm = lm->get_replaced();
        if (replaced_lm) {
            last_frm_.landmarks_.at(idx) = replaced_lm;
        }
    }
}

void tracking_module::update_last_frame() {
    auto last_ref_keyfrm = last_frm_.ref_keyfrm_;
    if (!last_ref_keyfrm) {
        return;
    }
    last_frm_.set_cam_pose(last_cam_pose_from_ref_keyfrm_ * last_ref_keyfrm->get_cam_pose());
}

bool tracking_module::optimize_current_frame_with_local_map() {
    // acquire more 2D-3D matches by reprojecting the local landmarks to the current frame
    search_local_landmarks();

    // optimize the pose
    pose_optimizer_.optimize(curr_frm_);

    // count up the number of tracked landmarks
    num_tracked_lms_ = 0;
    for (unsigned int idx = 0; idx < curr_frm_.num_keypts_; ++idx) {
        auto lm = curr_frm_.landmarks_.at(idx);
        if (!lm) {
            continue;
        }

        if (!curr_frm_.outlier_flags_.at(idx)) {
            // the observation has been considered as inlier in the pose optimization
            assert(lm->has_observation());
            // count up
            ++num_tracked_lms_;
            // increment the number of tracked frame
            lm->increase_num_observed();
        }
        else {
            // the observation has been considered as outlier in the pose optimization
            // remove the observation
            curr_frm_.landmarks_.at(idx) = nullptr;
        }
    }

    constexpr unsigned int num_tracked_lms_thr = 20;

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //modificado num_tracked_lms_ <2 * num_tracked_lms_thr por <30
    //-------------------------------------------------------------------------------------------
    // if recently relocalized, use the more strict threshold
    if (curr_frm_.id_ < last_reloc_frm_id_ + camera_->fps_ && num_tracked_lms_ < 30) {
        spdlog::debug("local map tracking failed: {} matches < {}", num_tracked_lms_, 30);
        return false;
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // check the threshold of the number of tracked landmarks
    if (num_tracked_lms_ < num_tracked_lms_thr) {
        spdlog::debug("local map tracking failed: {} matches < {}", num_tracked_lms_, num_tracked_lms_thr);
        return false;
    }

    return true;
}

void tracking_module::update_local_map() {
    update_local_keyframes();
    update_local_landmarks();

    map_db_->set_local_landmarks(local_landmarks_);
}

void tracking_module::update_local_keyframes() {
    constexpr unsigned int max_num_local_keyfrms = 60;

    // count the number of sharing landmarks between the current frame and each of the neighbor keyframes
    // key: keyframe, value: number of sharing landmarks
    std::unordered_map<data::keyframe*, unsigned int> keyfrm_weights;
    for (unsigned int idx = 0; idx < curr_frm_.num_keypts_; ++idx) {
        auto lm = curr_frm_.landmarks_.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            curr_frm_.landmarks_.at(idx) = nullptr;
            continue;
        }

        const auto observations = lm->get_observations();
        for (auto obs : observations) {
            ++keyfrm_weights[obs.first];
        }
    }

    if (keyfrm_weights.empty()) {
        return;
    }

    // set the aforementioned keyframes as local keyframes
    // and find the nearest keyframe
    unsigned int max_weight = 0;
    data::keyframe* nearest_covisibility = nullptr;

    local_keyfrms_.clear();
    local_keyfrms_.reserve(4 * keyfrm_weights.size());

    for (auto& keyfrm_weight : keyfrm_weights) {
        auto keyfrm = keyfrm_weight.first;
        const auto weight = keyfrm_weight.second;

        if (keyfrm->will_be_erased()) {
            continue;
        }

        local_keyfrms_.push_back(keyfrm);

        // avoid duplication
        keyfrm->local_map_update_identifier = curr_frm_.id_;

        // update the nearest keyframe
        if (max_weight < weight) {
            max_weight = weight;
            nearest_covisibility = keyfrm;
        }
    }

    // add the second-order keyframes to the local landmarks
    auto add_local_keyframe = [this](data::keyframe* keyfrm) {
        if (!keyfrm) {
            return false;
        }
        if (keyfrm->will_be_erased()) {
            return false;
        }
        // avoid duplication
        if (keyfrm->local_map_update_identifier == curr_frm_.id_) {
            return false;
        }
        keyfrm->local_map_update_identifier = curr_frm_.id_;
        local_keyfrms_.push_back(keyfrm);
        return true;
    };
    for (auto iter = local_keyfrms_.cbegin(); iter != local_keyfrms_.cend(); ++iter) {
        if (max_num_local_keyfrms < local_keyfrms_.size()) {
            break;
        }

        auto keyfrm = *iter;

        // covisibilities of the neighbor keyframe
        const auto neighbors = keyfrm->graph_node_->get_top_n_covisibilities(10);
        for (auto neighbor : neighbors) {
            if (add_local_keyframe(neighbor)) {
                break;
            }
        }

        // children of the spanning tree
        const auto spanning_children = keyfrm->graph_node_->get_spanning_children();
        for (auto child : spanning_children) {
            if (add_local_keyframe(child)) {
                break;
            }
        }

        // parent of the spanning tree
        auto parent = keyfrm->graph_node_->get_spanning_parent();
        add_local_keyframe(parent);

        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        data::keyframe* pPrevKF = keyfrm->GetPrevKeyFrame();
        if(pPrevKF)
        {
            if(pPrevKF->will_be_erased()) std::cerr<<"pPrevKF is bad in UpdateLocalKeyFrames()?????"<<std::endl;
            if(pPrevKF->local_map_update_identifier!=curr_frm_.id_)
            {
                local_keyfrms_.push_back(pPrevKF);
                pPrevKF->local_map_update_identifier=curr_frm_.id_;
            }
        }

        data::keyframe* pNextKF = keyfrm->GetNextKeyFrame();
        if(pNextKF)
        {
            if(pNextKF->will_be_erased()) std::cerr<<"pNextKF is bad in UpdateLocalKeyFrames()?????"<<std::endl;
            if(pNextKF->local_map_update_identifier!=curr_frm_.id_)
            {
                local_keyfrms_.push_back(pNextKF);
                pNextKF->local_map_update_identifier=curr_frm_.id_;
            }
        }
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
    }

    // update the reference keyframe with the nearest one
    if (nearest_covisibility) {
        ref_keyfrm_ = nearest_covisibility;
        curr_frm_.ref_keyfrm_ = ref_keyfrm_;
    }
}

void tracking_module::update_local_landmarks() {
    local_landmarks_.clear();

    for (auto keyfrm : local_keyfrms_) {
        const auto lms = keyfrm->get_landmarks();

        for (auto lm : lms) {
            if (!lm) {
                continue;
            }
            if (lm->will_be_erased()) {
                continue;
            }

            // avoid duplication
            if (lm->identifier_in_local_map_update_ == curr_frm_.id_) {
                continue;
            }
            lm->identifier_in_local_map_update_ = curr_frm_.id_;

            local_landmarks_.push_back(lm);
        }
    }
}

void tracking_module::search_local_landmarks() {
    // select the landmarks which can be reprojected from the ones observed in the current frame
    for (auto lm : curr_frm_.landmarks_) {
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // this landmark cannot be reprojected
        // because already observed in the current frame
        lm->is_observable_in_tracking_ = false;
        lm->identifier_in_local_lm_search_ = curr_frm_.id_;

        // this landmark is observable from the current frame
        lm->increase_num_observable();
    }

    bool found_proj_candidate = false;
    // temporary variables
    Vec2_t reproj;
    float x_right;
    unsigned int pred_scale_level;
    for (auto lm : local_landmarks_) {
        // avoid the landmarks which cannot be reprojected (== observed in the current frame)
        if (lm->identifier_in_local_lm_search_ == curr_frm_.id_) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        // check the observability
        if (curr_frm_.can_observe(lm, 0.5, reproj, x_right, pred_scale_level)) {
            // pass the temporary variables
            lm->reproj_in_tracking_ = reproj;
            lm->x_right_in_tracking_ = x_right;
            lm->scale_level_in_tracking_ = pred_scale_level;

            // this landmark can be reprojected
            lm->is_observable_in_tracking_ = true;

            // this landmark is observable from the current frame
            lm->increase_num_observable();

            found_proj_candidate = true;
        }
        else {
            // this landmark cannot be reprojected
            lm->is_observable_in_tracking_ = false;
        }
    }

    if (!found_proj_candidate) {
        return;
    }

    // acquire more 2D-3D matches by projecting the local landmarks to the current frame
    match::projection projection_matcher(0.8);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    const float margin = (curr_frm_.id_ < last_reloc_frm_id_ + 2||curr_frm_.timestamp_ - mTimestampLastLost < IMU_SAFE_WINDOW)
                             ? 20.0
                             : ((camera_->setup_type_ == camera::setup_type_t::RGBD)
                                    ? 10.0
                                    : 5.0);
    int matcherscount;
    matcherscount = projection_matcher.match_frame_and_landmarks(curr_frm_, local_landmarks_, margin);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
}

bool tracking_module::new_keyframe_is_needed() const {
    if (!mapping_is_enabled_) {
        return false;
    }

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // While updating initial poses
    if(mapper_->GetUpdatingInitPoses())
    {
        std::cerr<<"mpLocalMapper->GetUpdatingInitPoses, no new KF"<<std::endl;
        return false;
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // cannnot insert the new keyframe in a second after relocalization
    const auto num_keyfrms = map_db_->get_num_keyframes();
    if (cfg_->camera_->fps_ < num_keyfrms && curr_frm_.id_ < last_reloc_frm_id_ + cfg_->camera_->fps_) {
        return false;
    }

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // Do not insert keyframes if bias is not computed in VINS mode
    if(mbRelocBiasPrepare/* && mpLocalMapper->GetVINSInited()*/)
        return false;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // check the new keyframe is needed
    return keyfrm_inserter_.new_keyframe_is_needed(curr_frm_, num_tracked_lms_, *ref_keyfrm_);
}

void tracking_module::insert_new_keyframe() {
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    keyfrm_inserter_.set_tracking_module(this);
    ref_keyfrm_public_ = ref_keyfrm_;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // insert the new keyframe
    const auto ref_keyfrm = keyfrm_inserter_.insert_new_keyframe(curr_frm_);

    // set the reference keyframe with the new keyframe
    ref_keyfrm_ = ref_keyfrm ? ref_keyfrm : ref_keyfrm_;
    curr_frm_.ref_keyfrm_ = ref_keyfrm_;
}

void tracking_module::request_pause() {
    std::lock_guard<std::mutex> lock1(mtx_pause_);
    pause_is_requested_ = true;
}

bool tracking_module::pause_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return pause_is_requested_;
}

bool tracking_module::is_paused() const {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    return is_paused_;
}

void tracking_module::resume() {
    std::lock_guard<std::mutex> lock(mtx_pause_);

    is_paused_ = false;
    pause_is_requested_ = false;

    spdlog::info("resume tracking module");
}

bool tracking_module::check_and_execute_pause() {
    std::lock_guard<std::mutex> lock(mtx_pause_);
    if (pause_is_requested_) {
        is_paused_ = true;
        spdlog::info("pause tracking module");
        return true;
    }
    else {
        return false;
    }
}

} // namespace openvslam
