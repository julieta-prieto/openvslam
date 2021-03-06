#include "openvslam/camera/perspective.h"
#include "openvslam/camera/fisheye.h"
#include "openvslam/camera/equirectangular.h"
#include "openvslam/data/common.h"
#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/feature/orb_extractor.h"
#include "openvslam/match/stereo.h"

#include <thread>

#include <spdlog/spdlog.h>

namespace openvslam {
namespace data {

std::atomic<unsigned int> frame::next_id_{0};

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

float frame::fx;
float frame::fy;
float frame::cx;
float frame::cy;

void frame::ComputeIMUPreIntSinceLastFrame(const frame* pLastF, IMUPreintegrator& IMUPreInt) const
{
    // Reset pre-integrator first
    IMUPreInt.reset();

    const std::vector<IMUData>& vIMUSInceLastFrame = mvIMUDataSinceLastFrame;

    Vector3d bg = pLastF->GetNavState().Get_BiasGyr();
    Vector3d ba = pLastF->GetNavState().Get_BiasAcc();

    // remember to consider the gap between the last KF and the first IMU
    {
        const IMUData& imu = vIMUSInceLastFrame.front();
        double dt = imu._t - pLastF->timestamp_;
        IMUPreInt.update(imu._g - bg, imu._a - ba, dt);

        // Test log
        if(dt < 0)
        {
            std::cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this frame vs last imu time: "<<pLastF->timestamp_<<" vs "<<imu._t<<std::endl;
            std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
        }
    }
    // integrate each imu
    for(size_t i=0; i<vIMUSInceLastFrame.size(); i++)
    {
        const IMUData& imu = vIMUSInceLastFrame[i];
        double nextt;
        if(i==vIMUSInceLastFrame.size()-1)
            nextt = timestamp_;         // last IMU, next is this KeyFrame
        else
            nextt = vIMUSInceLastFrame[i+1]._t;  // regular condition, next is imu data

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
}

void frame::UpdatePoseFromNS(const cv::Mat &Tbc)
{
    cv::Mat Rbc = Tbc.rowRange(0,3).colRange(0,3).clone();
    cv::Mat Pbc = Tbc.rowRange(0,3).col(3).clone();

    cv::Mat Rwb = util::converter::toCvMat(mNavState.Get_RotMatrix());
    cv::Mat Pwb = util::converter::toCvMat(mNavState.Get_P());

    cv::Mat Rcw = (Rwb*Rbc).t();
    cv::Mat Pwc = Rwb*Pbc + Pwb;
    cv::Mat Pcw = -Rcw*Pwc;

    cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
    Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
    Pcw.copyTo(Tcw.rowRange(0,3).col(3));

    // Convert the opencv matrix to a Mat44_t openvslam standard matrix
    Mat44_t cam_pose;
    cam_pose = util::converter::cvMat4_to_Mat44_t(Tcw);

    set_cam_pose(cam_pose);
}

void frame::UpdateNavState(const IMUPreintegrator& imupreint, const Vector3d& gw)
{
    util::converter::updateNS(mNavState,imupreint,gw);
}

const NavState& frame::GetNavState(void) const
{
    return mNavState;
}

void frame::SetInitialNavStateAndBias(const NavState& ns)
{
    mNavState = ns;
    // Set bias as bias+delta_bias, and reset the delta_bias term
    mNavState.Set_BiasGyr(ns.Get_BiasGyr()+ns.Get_dBias_Gyr());
    mNavState.Set_BiasAcc(ns.Get_BiasAcc()+ns.Get_dBias_Acc());
    mNavState.Set_DeltaBiasGyr(Vector3d::Zero());
    mNavState.Set_DeltaBiasAcc(Vector3d::Zero());
}


void frame::SetNavStateBiasGyr(const Vector3d &bg)
{
    mNavState.Set_BiasGyr(bg);
}

void frame::SetNavStateBiasAcc(const Vector3d &ba)
{
    mNavState.Set_BiasAcc(ba);
}

void frame::SetNavState(const NavState& ns)
{
    mNavState = ns;
}

frame::frame(const cv::Mat& img_gray, const double &timestamp, const std::vector<IMUData> &vimu, feature::orb_extractor* extractor, bow_vocabulary* bow_vocab,
             camera::base* camera, const float depth_thr, cv::Mat &K, const cv::Mat& mask, data::keyframe* pLastKF)
    :id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor), extractor_right_(nullptr),
      timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr), mK(K.clone())
{
    // Copy IMU data
    mvIMUDataSinceLastFrame = vimu;

    // Get ORB scale
    update_orb_info();

    // Extract ORB feature
    extract_orb(img_gray, mask);
    num_keypts_ = keypts_.size();
    if (keypts_.empty()) {
        spdlog::warn("frame {}: cannot extract any keypoints", id_);
        return;
    }

    // Undistort keypoints
    camera_->undistort_keypoints(keypts_, undist_keypts_);

    // Ignore stereo parameters
    stereo_x_right_ = std::vector<float>(num_keypts_, -1);
    depths_ = std::vector<float>(num_keypts_, -1);

    // Convert to bearing vector
    camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

    // Initialize association with 3D points
    landmarks_ = std::vector<landmark*>(num_keypts_, nullptr);
    outlier_flags_ = std::vector<bool>(num_keypts_, false);

    // Assign all the keypoints into grid
    assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);

    //Añadido por mi
    fx = K.at<float>(0,0);
    fy = K.at<float>(1,1);
    cx = K.at<float>(0,2);
    cy = K.at<float>(1,2);
}

frame::frame(){}

//Copy Constructor
frame::frame(const frame &frame)
    :id_(frame.id_), /*next_id_(frame.next_id_), */bow_vocab_(frame.bow_vocab_), extractor_(frame.extractor_),
    extractor_right_(frame.extractor_right_), timestamp_(frame.timestamp_), camera_(frame.camera_), depth_thr_(frame.depth_thr_), num_keypts_(frame.num_keypts_),
    keypts_(frame.keypts_), keypts_right_(frame.keypts_right_), undist_keypts_(frame.undist_keypts_), 
    bearings_(frame.bearings_), stereo_x_right_(frame.stereo_x_right_), depths_(frame.depths_), bow_vec_(frame.bow_vec_), bow_feat_vec_(frame.bow_feat_vec_),
    descriptors_(frame.descriptors_), descriptors_right_(frame.descriptors_right_), 
    landmarks_(frame.landmarks_), outlier_flags_(frame.outlier_flags_), keypt_indices_in_cells_(frame.keypt_indices_in_cells_),
    cam_pose_cw_is_valid_(frame.cam_pose_cw_is_valid_), cam_pose_cw_(frame.cam_pose_cw_), 
    ref_keyfrm_(frame.ref_keyfrm_), num_scale_levels_(frame.num_scale_levels_), scale_factor_(frame.scale_factor_),
    log_scale_factor_(frame.log_scale_factor_), scale_factors_(frame.scale_factors_), 
    inv_scale_factors_(frame.inv_scale_factors_), level_sigma_sq_(frame.level_sigma_sq_), inv_level_sigma_sq_(frame.inv_level_sigma_sq_),
    rot_cw_(frame.rot_cw_), trans_cw_(frame.trans_cw_), rot_wc_(frame.rot_wc_), cam_center_(frame.cam_center_)
{
    mvIMUDataSinceLastFrame = frame.mvIMUDataSinceLastFrame;
    mNavState = frame.GetNavState();
    mMargCovInv = frame.mMargCovInv;
    mNavStatePrior = frame.mNavStatePrior;
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

frame::frame(const cv::Mat& img_gray, const double timestamp,
             feature::orb_extractor* extractor, bow_vocabulary* bow_vocab,
             camera::base* camera, const float depth_thr,
             const cv::Mat& mask)
    : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor), extractor_right_(nullptr),
      timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr) {
    // Get ORB scale
    update_orb_info();

    // Extract ORB feature
    extract_orb(img_gray, mask);
    num_keypts_ = keypts_.size();
    if (keypts_.empty()) {
        spdlog::warn("frame {}: cannot extract any keypoints", id_);
    }

    // Undistort keypoints
    camera_->undistort_keypoints(keypts_, undist_keypts_);

    // Ignore stereo parameters
    stereo_x_right_ = std::vector<float>(num_keypts_, -1);
    depths_ = std::vector<float>(num_keypts_, -1);

    // Convert to bearing vector
    camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

    // Initialize association with 3D points
    landmarks_ = std::vector<landmark*>(num_keypts_, nullptr);
    outlier_flags_ = std::vector<bool>(num_keypts_, false);

    // Assign all the keypoints into grid
    assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
}

frame::frame(const cv::Mat& left_img_gray, const cv::Mat& right_img_gray, const double timestamp,
             feature::orb_extractor* extractor_left, feature::orb_extractor* extractor_right,
             bow_vocabulary* bow_vocab, camera::base* camera, const float depth_thr,
             const cv::Mat& mask)
    : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor_left), extractor_right_(extractor_right),
      timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr) {
    // Get ORB scale
    update_orb_info();

    // Extract ORB feature
    std::thread thread_left(&frame::extract_orb, this, left_img_gray, mask, image_side::Left);
    std::thread thread_right(&frame::extract_orb, this, right_img_gray, mask, image_side::Right);
    thread_left.join();
    thread_right.join();
    num_keypts_ = keypts_.size();
    if (keypts_.empty()) {
        spdlog::warn("frame {}: cannot extract any keypoints", id_);
    }

    // Undistort keypoints
    camera_->undistort_keypoints(keypts_, undist_keypts_);

    // Estimate depth with stereo match
    match::stereo stereo_matcher(extractor_left->image_pyramid_, extractor_right_->image_pyramid_,
                                 keypts_, keypts_right_, descriptors_, descriptors_right_,
                                 scale_factors_, inv_scale_factors_,
                                 camera->focal_x_baseline_, camera_->true_baseline_);
    stereo_matcher.compute(stereo_x_right_, depths_);

    // Convert to bearing vector
    camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

    // Initialize association with 3D points
    landmarks_ = std::vector<landmark*>(num_keypts_, nullptr);
    outlier_flags_ = std::vector<bool>(num_keypts_, false);

    // Assign all the keypoints into grid
    assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
}

frame::frame(const cv::Mat& img_gray, const cv::Mat& img_depth, const double timestamp,
             feature::orb_extractor* extractor, bow_vocabulary* bow_vocab,
             camera::base* camera, const float depth_thr,
             const cv::Mat& mask)
    : id_(next_id_++), bow_vocab_(bow_vocab), extractor_(extractor), extractor_right_(nullptr),
      timestamp_(timestamp), camera_(camera), depth_thr_(depth_thr) {
    // Get ORB scale
    update_orb_info();

    // Extract ORB feature
    extract_orb(img_gray, mask);
    num_keypts_ = keypts_.size();
    if (keypts_.empty()) {
        spdlog::warn("frame {}: cannot extract any keypoints", id_);
    }

    // Undistort keypoints
    camera_->undistort_keypoints(keypts_, undist_keypts_);

    // Calculate disparity from depth
    compute_stereo_from_depth(img_depth);

    // Convert to bearing vector
    camera->convert_keypoints_to_bearings(undist_keypts_, bearings_);

    // Initialize association with 3D points
    landmarks_ = std::vector<landmark*>(num_keypts_, nullptr);
    outlier_flags_ = std::vector<bool>(num_keypts_, false);

    // Assign all the keypoints into grid
    assign_keypoints_to_grid(camera_, undist_keypts_, keypt_indices_in_cells_);
}

void frame::set_cam_pose(const Mat44_t& cam_pose_cw) {
    cam_pose_cw_is_valid_ = true;
    cam_pose_cw_ = cam_pose_cw;
    update_pose_params();
}

void frame::set_cam_pose(const g2o::SE3Quat& cam_pose_cw) {
    set_cam_pose(util::converter::to_eigen_mat(cam_pose_cw));
}

void frame::update_pose_params() {
    rot_cw_ = cam_pose_cw_.block<3, 3>(0, 0);
    rot_wc_ = rot_cw_.transpose();
    trans_cw_ = cam_pose_cw_.block<3, 1>(0, 3);
    cam_center_ = -rot_cw_.transpose() * trans_cw_;
}

Vec3_t frame::get_cam_center() const {
    return cam_center_;
}

Mat33_t frame::get_rotation_inv() const {
    return rot_wc_;
}

void frame::update_orb_info() {
    num_scale_levels_ = extractor_->get_num_scale_levels();
    scale_factor_ = extractor_->get_scale_factor();
    log_scale_factor_ = std::log(scale_factor_);
    scale_factors_ = extractor_->get_scale_factors();
    inv_scale_factors_ = extractor_->get_inv_scale_factors();
    level_sigma_sq_ = extractor_->get_level_sigma_sq();
    inv_level_sigma_sq_ = extractor_->get_inv_level_sigma_sq();
}

void frame::compute_bow() {
    if (bow_vec_.empty()) {
#ifdef USE_DBOW2
        bow_vocab_->transform(util::converter::to_desc_vec(descriptors_), bow_vec_, bow_feat_vec_, 4);
#else
        bow_vocab_->transform(descriptors_, 4, bow_vec_, bow_feat_vec_);
#endif
    }
}

bool frame::can_observe(landmark* lm, const float ray_cos_thr,
                        Vec2_t& reproj, float& x_right, unsigned int& pred_scale_level) const {
    const Vec3_t pos_w = lm->get_pos_in_world();

    const bool in_image = camera_->reproject_to_image(rot_cw_, trans_cw_, pos_w, reproj, x_right);
    if (!in_image) {
        return false;
    }

    const Vec3_t cam_to_lm_vec = pos_w - cam_center_;
    const auto cam_to_lm_dist = cam_to_lm_vec.norm();
    if (!lm->is_inside_in_orb_scale(cam_to_lm_dist)) {
        return false;
    }

    const Vec3_t obs_mean_normal = lm->get_obs_mean_normal();
    const auto ray_cos = cam_to_lm_vec.dot(obs_mean_normal) / cam_to_lm_dist;
    if (ray_cos < ray_cos_thr) {
        return false;
    }

    pred_scale_level = lm->predict_scale_level(cam_to_lm_dist, this);
    return true;
}

std::vector<unsigned int> frame::get_keypoints_in_cell(const float ref_x, const float ref_y, const float margin, const int min_level, const int max_level) const {
    return data::get_keypoints_in_cell(camera_, undist_keypts_, keypt_indices_in_cells_, ref_x, ref_y, margin, min_level, max_level);
}

Vec3_t frame::triangulate_stereo(const unsigned int idx) const {
    assert(camera_->setup_type_ != camera::setup_type_t::Monocular);

    switch (camera_->model_type_) {
        case camera::model_type_t::Perspective: {
            auto camera = static_cast<camera::perspective*>(camera_);

            const float depth = depths_.at(idx);
            if (0.0 < depth) {
                const float x = undist_keypts_.at(idx).pt.x;
                const float y = undist_keypts_.at(idx).pt.y;
                const float unproj_x = (x - camera->cx_) * depth * camera->fx_inv_;
                const float unproj_y = (y - camera->cy_) * depth * camera->fy_inv_;
                const Vec3_t pos_c{unproj_x, unproj_y, depth};

                // Convert from camera coordinates to world coordinates
                return rot_wc_ * pos_c + cam_center_;
            }
            else {
                return Vec3_t::Zero();
            }
        }
        case camera::model_type_t::Fisheye: {
            auto camera = static_cast<camera::fisheye*>(camera_);

            const float depth = depths_.at(idx);
            if (0.0 < depth) {
                const float x = undist_keypts_.at(idx).pt.x;
                const float y = undist_keypts_.at(idx).pt.y;
                const float unproj_x = (x - camera->cx_) * depth * camera->fx_inv_;
                const float unproj_y = (y - camera->cy_) * depth * camera->fy_inv_;
                const Vec3_t pos_c{unproj_x, unproj_y, depth};

                // Convert from camera coordinates to world coordinates
                return rot_wc_ * pos_c + cam_center_;
            }
            else {
                return Vec3_t::Zero();
            }
        }
        case camera::model_type_t::Equirectangular: {
            throw std::runtime_error("Not implemented: Stereo or RGBD of equirectangular camera model");
        }
    }

    return Vec3_t::Zero();
}

void frame::extract_orb(const cv::Mat& img, const cv::Mat& mask, const image_side& img_side) {
    switch (img_side) {
        case image_side::Left: {
            extractor_->extract(img, mask, keypts_, descriptors_);
            break;
        }
        case image_side::Right: {
            extractor_right_->extract(img, mask, keypts_right_, descriptors_right_);
            break;
        }
    }
}

void frame::compute_stereo_from_depth(const cv::Mat& right_img_depth) {
    assert(camera_->setup_type_ == camera::setup_type_t::RGBD);

    // Initialize with invalid value
    stereo_x_right_ = std::vector<float>(num_keypts_, -1);
    depths_ = std::vector<float>(num_keypts_, -1);

    for (unsigned int idx = 0; idx < num_keypts_; idx++) {
        const auto& keypt = keypts_.at(idx);
        const auto& undist_keypt = undist_keypts_.at(idx);

        const float x = keypt.pt.x;
        const float y = keypt.pt.y;

        const float depth = right_img_depth.at<float>(y, x);

        if (depth <= 0) {
            continue;
        }

        depths_.at(idx) = depth;
        stereo_x_right_.at(idx) = undist_keypt.pt.x - camera_->focal_x_baseline_ / depth;
    }
}

} // namespace data
} // namespace openvslam
