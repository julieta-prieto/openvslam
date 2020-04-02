#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/tracking_module.h"
#include "openvslam/mapping_module.h"
#include "openvslam/global_optimization_module.h"
#include "openvslam/camera/base.h"
#include "openvslam/data/camera_database.h"
#include "openvslam/data/map_database.h"
#include "openvslam/data/bow_database.h"
#include "openvslam/io/trajectory_io.h"
#include "openvslam/io/map_database_io.h"
#include "openvslam/publish/map_publisher.h"
#include "openvslam/publish/frame_publisher.h"

#include <thread>

#include <spdlog/spdlog.h>

#include <time.h>

#include "openvslam/IMU/configparam.h"
#include <opencv2/core/eigen.hpp>

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
bool has_suffix(const std::string &str, const std::string &suffix) {
  std::size_t index = str.find(suffix, str.size() - suffix.size());
  return (index != std::string::npos);
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

namespace openvslam {
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

bool system::bLocalMapAcceptKF()
{
    return (mapper_->get_keyframe_acceptability() && !mapper_->is_paused());
    //return mpLocalMapper->ForsyncCheckNewKeyFrames();
}

void system::GetSLAMData(SLAMData& slamdata)
{
    //slamdata.TrackingStatus = tracker_->mState;
    //if(slamdata.TrackingStatus > 0)
    //{
        cv::Mat pose;
        eigen2cv(tracker_->curr_frm_.cam_pose_cw_, pose);
        if(!pose.empty())
        {
            slamdata.Tcw = pose.clone();
        }
        slamdata.VINSInitFlag = mapper_->GetVINSInited();
        if(slamdata.VINSInitFlag)
        {
            slamdata.gw = mapper_->GetGravityVec();
            slamdata.Rwi = mapper_->GetRwiInit();
        }
        slamdata.Timestamp = tracker_->curr_frm_.timestamp_;
    //}
}

void system::SaveKeyFrameTrajectoryNavState(const std::string &filename)
{
    std::cout << std::endl << "Saving keyframe NavState to " << filename << " ..." << std::endl;

    std::vector<data::keyframe*> vpKFs = map_db_->get_all_keyframes();
    std::sort(vpKFs.begin(), vpKFs.end(), [&](data::keyframe* keyfrm_1, data::keyframe* keyfrm_2) {
        return *keyfrm_1 < *keyfrm_2;
    });
    //sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f;
    f.open(filename.c_str());
    f << std::fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        data::keyframe* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->will_be_erased())
            continue;

        Eigen::Vector3d P = pKF->GetNavState().Get_P();
        Eigen::Vector3d V = pKF->GetNavState().Get_V();
        Eigen::Quaterniond q = pKF->GetNavState().Get_R().unit_quaternion();
        Eigen::Vector3d bg = pKF->GetNavState().Get_BiasGyr();
        Eigen::Vector3d ba = pKF->GetNavState().Get_BiasAcc();
        Eigen::Vector3d dbg = pKF->GetNavState().Get_dBias_Gyr();
        Eigen::Vector3d dba = pKF->GetNavState().Get_dBias_Acc();
        f << std::setprecision(6) << pKF->timestamp_ << std::setprecision(7) << " ";
        f << P(0) << " " << P(1) << " " << P(2) << " ";
        f << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " ";
        f << V(0) << " " << V(1) << " " << V(2) << " ";
        f << bg(0)+dbg(0) << " " << bg(1)+dbg(1) << " " << bg(2)+dbg(2) << " ";
        f << ba(0)+dba(0) << " " << ba(1)+dba(1) << " " << ba(2)+dba(2) << " ";
//        f << bg(0) << " " << bg(1) << " " << bg(2) << " ";
//        f << ba(0) << " " << ba(1) << " " << ba(2) << " ";
//        f << dbg(0) << " " << dbg(1) << " " << dbg(2) << " ";
//        f << dba(0) << " " << dba(1) << " " << dba(2) << " ";
        f << std::endl;
    }

    f.close();
    std::cout << std::endl << "NavState trajectory saved!" << std::endl;
}

Mat44_t system::feed_monocular_frame_VI(const cv::Mat& img, const std::vector<IMUData> &vimu, const double timestamp, const cv::Mat& mask)
{
    // CONTINUAR A PARTIR DE AQUI EL LUNES!!!!
    assert(camera_->setup_type_ == camera::setup_type_t::Monocular);

    check_reset_request();

    const Mat44_t cam_pose_cw = tracker_->track_monocular_image_VI(img, vimu, timestamp, mask);

    frame_publisher_->update(tracker_);
    if (tracker_->tracking_state_ == tracker_state_t::Tracking) {
        map_publisher_->set_current_cam_pose(cam_pose_cw);
    }

    return cam_pose_cw;
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------


system::system(const std::shared_ptr<config>& cfg, const std::string& vocab_file_path)
    : cfg_(cfg), camera_(cfg->camera_) {
    spdlog::debug("CONSTRUCT: system");

    std::cout << R"(  ___               __   _____ _      _   __  __ )" << std::endl;
    std::cout << R"( / _ \ _ __  ___ _ _\ \ / / __| |    /_\ |  \/  |)" << std::endl;
    std::cout << R"(| (_) | '_ \/ -_) ' \\ V /\__ \ |__ / _ \| |\/| |)" << std::endl;
    std::cout << R"( \___/| .__/\___|_||_|\_/ |___/____/_/ \_\_|  |_|)" << std::endl;
    std::cout << R"(      |_|                                        )" << std::endl;
    std::cout << std::endl;
    std::cout << "Copyright (C) 2019," << std::endl;
    std::cout << "National Institute of Advanced Industrial Science and Technology (AIST)" << std::endl;
    std::cout << "All rights reserved." << std::endl;
    std::cout << std::endl;
    std::cout << "This is free software," << std::endl;
    std::cout << "and you are welcome to redistribute it under certain conditions." << std::endl;
    std::cout << "See the LICENSE file." << std::endl;
    std::cout << std::endl;

    // show configuration
    std::cout << *cfg_ << std::endl;

    // load ORB vocabulary
    spdlog::info("loading ORB vocabulary: {}", vocab_file_path);
#ifdef USE_DBOW2
    bow_vocab_ = new data::bow_vocabulary();
    try {
        bow_vocab_->loadFromBinaryFile(vocab_file_path);
    }
    catch (const std::exception& e) {
        spdlog::critical("wrong path to vocabulary");
        delete bow_vocab_;
        bow_vocab_ = nullptr;
        exit(EXIT_FAILURE);
    }
#else
    bow_vocab_ = new fbow::Vocabulary();
    bow_vocab_->readFromFile(vocab_file_path);
    if (!bow_vocab_->isValid()) {
        spdlog::critical("wrong path to vocabulary");
        delete bow_vocab_;
        bow_vocab_ = nullptr;
        exit(EXIT_FAILURE);
    }
#endif
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    ConfigParam config(cfg_->config_file_path_);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    
    // database
    cam_db_ = new data::camera_database(camera_);
    map_db_ = new data::map_database();
    bow_db_ = new data::bow_database(bow_vocab_);

    // frame and map publisher
    frame_publisher_ = std::shared_ptr<publish::frame_publisher>(new publish::frame_publisher(cfg_, map_db_));
    map_publisher_ = std::shared_ptr<publish::map_publisher>(new publish::map_publisher(cfg_, map_db_));

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // Cambiados los constructores
    //-------------------------------------------------------------------------------------------
    // tracking module
    tracker_ = new tracking_module(cfg_, this, map_db_, bow_vocab_, bow_db_, &config);
    // mapping module
    mapper_ = new mapping_module(map_db_, camera_->setup_type_ == camera::setup_type_t::Monocular, &config);
    // global optimization module
    global_optimizer_ = new global_optimization_module(map_db_, bow_db_, bow_vocab_, camera_->setup_type_ != camera::setup_type_t::Monocular, &config);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // connect modules each other
    tracker_->set_mapping_module(mapper_);
    tracker_->set_global_optimization_module(global_optimizer_);
    mapper_->set_tracking_module(tracker_);
    mapper_->set_global_optimization_module(global_optimizer_);
    global_optimizer_->set_tracking_module(tracker_);
    global_optimizer_->set_mapping_module(mapper_);

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    #ifdef RUN_REALTIME
    //Thread for VINS initialization
    mptLocalMappingVIOInit = new std::thread(&openvslam::mapping_module::VINSInitThread, mapper_);
    #endif

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
}

system::~system() {
    global_optimization_thread_.reset(nullptr);
    delete global_optimizer_;
    global_optimizer_ = nullptr;

    mapping_thread_.reset(nullptr);
    delete mapper_;
    mapper_ = nullptr;

    delete tracker_;
    tracker_ = nullptr;

    delete bow_db_;
    bow_db_ = nullptr;
    delete map_db_;
    map_db_ = nullptr;
    delete cam_db_;
    cam_db_ = nullptr;
    delete bow_vocab_;
    bow_vocab_ = nullptr;

    spdlog::debug("DESTRUCT: system");
}

void system::startup(const bool need_initialize) {
    spdlog::info("startup SLAM system");
    system_is_running_ = true;

    if (!need_initialize) {
        tracker_->tracking_state_ = tracker_state_t::Lost;
    }

    mapping_thread_ = std::unique_ptr<std::thread>(new std::thread(&openvslam::mapping_module::run, mapper_));
    global_optimization_thread_ = std::unique_ptr<std::thread>(new std::thread(&openvslam::global_optimization_module::run, global_optimizer_));
}

void system::shutdown() {
    // terminate the other threads
    mapper_->request_terminate();
    global_optimizer_->request_terminate();
    // wait until they stop
    while (!mapper_->is_terminated()
           || !global_optimizer_->is_terminated()
           || global_optimizer_->loop_BA_is_running()) {
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // wait until the threads stop
    mapping_thread_->join();
    global_optimization_thread_->join();

    spdlog::info("shutdown SLAM system");
    system_is_running_ = false;
}

void system::save_frame_trajectory(const std::string& path, const std::string& format) const {
    pause_other_threads();
    io::trajectory_io trajectory_io(map_db_);
    trajectory_io.save_frame_trajectory(path, format);
    resume_other_threads();
}

void system::save_keyframe_trajectory(const std::string& path, const std::string& format) const {
    pause_other_threads();
    io::trajectory_io trajectory_io(map_db_);
    trajectory_io.save_keyframe_trajectory(path, format);
    resume_other_threads();
}

void system::load_map_database(const std::string& path) const {
    pause_other_threads();
    io::map_database_io map_db_io(cam_db_, map_db_, bow_db_, bow_vocab_);
    map_db_io.load_message_pack(path);
    resume_other_threads();
}

void system::save_map_database(const std::string& path) const {
    pause_other_threads();
    io::map_database_io map_db_io(cam_db_, map_db_, bow_db_, bow_vocab_);
    map_db_io.save_message_pack(path);
    resume_other_threads();
}

const std::shared_ptr<publish::map_publisher> system::get_map_publisher() const {
    return map_publisher_;
}

const std::shared_ptr<publish::frame_publisher> system::get_frame_publisher() const {
    return frame_publisher_;
}

void system::enable_mapping_module() {
    std::lock_guard<std::mutex> lock(mtx_mapping_);
    if (!system_is_running_) {
        spdlog::critical("please call system::enable_mapping_module() after system::startup()");
    }
    // resume the mapping module
    mapper_->resume();
    // inform to the tracking module
    tracker_->set_mapping_module_status(true);
}

void system::disable_mapping_module() {
    std::lock_guard<std::mutex> lock(mtx_mapping_);
    if (!system_is_running_) {
        spdlog::critical("please call system::disable_mapping_module() after system::startup()");
    }
    // pause the mapping module
    mapper_->request_pause();
    // wait until it stops
    while (!mapper_->is_paused()) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
    // inform to the tracking module
    tracker_->set_mapping_module_status(false);
}

bool system::mapping_module_is_enabled() const {
    return !mapper_->is_paused();
}

void system::enable_loop_detector() {
    std::lock_guard<std::mutex> lock(mtx_loop_detector_);
    global_optimizer_->enable_loop_detector();
}

void system::disable_loop_detector() {
    std::lock_guard<std::mutex> lock(mtx_loop_detector_);
    global_optimizer_->disable_loop_detector();
}

bool system::loop_detector_is_enabled() const {
    return global_optimizer_->loop_detector_is_enabled();
}

bool system::loop_BA_is_running() const {
    return global_optimizer_->loop_BA_is_running();
}

void system::abort_loop_BA() {
    global_optimizer_->abort_loop_BA();
}

Mat44_t system::feed_monocular_frame(const cv::Mat& img, const double timestamp, const cv::Mat& mask) {
    assert(camera_->setup_type_ == camera::setup_type_t::Monocular);

    check_reset_request();

    const Mat44_t cam_pose_cw = tracker_->track_monocular_image(img, timestamp, mask);

    frame_publisher_->update(tracker_);
    
    if (tracker_->tracking_state_ == tracker_state_t::Tracking) {
        map_publisher_->set_current_cam_pose(cam_pose_cw);
    }

    return cam_pose_cw;
}

Mat44_t system::feed_stereo_frame(const cv::Mat& left_img, const cv::Mat& right_img, const double timestamp, const cv::Mat& mask) {
    assert(camera_->setup_type_ == camera::setup_type_t::Stereo);

    check_reset_request();

    const Mat44_t cam_pose_cw = tracker_->track_stereo_image(left_img, right_img, timestamp, mask);

    frame_publisher_->update(tracker_);
    if (tracker_->tracking_state_ == tracker_state_t::Tracking) {
        map_publisher_->set_current_cam_pose(cam_pose_cw);
    }

    return cam_pose_cw;
}

Mat44_t system::feed_RGBD_frame(const cv::Mat& rgb_img, const cv::Mat& depthmap, const double timestamp, const cv::Mat& mask) {
    assert(camera_->setup_type_ == camera::setup_type_t::RGBD);

    check_reset_request();

    const Mat44_t cam_pose_cw = tracker_->track_RGBD_image(rgb_img, depthmap, timestamp, mask);

    frame_publisher_->update(tracker_);
    if (tracker_->tracking_state_ == tracker_state_t::Tracking) {
        map_publisher_->set_current_cam_pose(cam_pose_cw);
    }

    return cam_pose_cw;
}

void system::pause_tracker() {
    tracker_->request_pause();
}

bool system::tracker_is_paused() const {
    return tracker_->is_paused();
}

void system::resume_tracker() {
    tracker_->resume();
}

void system::request_reset() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    reset_is_requested_ = true;
}

bool system::reset_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    return reset_is_requested_;
}

void system::request_terminate() {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    terminate_is_requested_ = true;
}

bool system::terminate_is_requested() const {
    std::lock_guard<std::mutex> lock(mtx_terminate_);
    return terminate_is_requested_;
}

void system::check_reset_request() {
    std::lock_guard<std::mutex> lock(mtx_reset_);
    if (reset_is_requested_) {
        tracker_->reset();
        reset_is_requested_ = false;
    }
}

void system::pause_other_threads() const {
    // pause the mapping module
    if (mapper_ && !mapper_->is_terminated()) {
        mapper_->request_pause();
        while (!mapper_->is_paused() && !mapper_->is_terminated()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    }
    // pause the global optimization module
    if (global_optimizer_ && !global_optimizer_->is_terminated()) {
        global_optimizer_->request_pause();
        while (!global_optimizer_->is_paused() && !global_optimizer_->is_terminated()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    }
}

void system::resume_other_threads() const {
    // resume the global optimization module
    if (global_optimizer_) {
        global_optimizer_->resume();
    }
    // resume the mapping module
    if (mapper_) {
        mapper_->resume();
    }
}

} // namespace openvslam
