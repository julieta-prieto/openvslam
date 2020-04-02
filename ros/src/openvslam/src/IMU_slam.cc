#ifdef USE_PANGOLIN_VIEWER
#include <pangolin_viewer/viewer.h>
#elif USE_SOCKET_PUBLISHER
#include <socket_publisher/publisher.h>
#endif
#include <openvslam/system.h>
#include <openvslam/config.h>
#include <openvslam/data/keyframe.h>
#include <openvslam/type.h>
#include <iostream>
#include <chrono>
#include <numeric>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
//
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <nav_msgs/Odometry.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Geometry>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <opencv2/core/eigen.hpp>
//
#include <tf/transform_broadcaster.h>
#include <Eigen/Geometry> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>
#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif
#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

std_msgs::Header header;

void mono_tracking(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path,
                   const std::string& mask_img_path, const bool eval_log, const std::string& map_db_path) {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);
    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup();
    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif
    std::vector<double> track_times;
    const auto tp_0 = std::chrono::steady_clock::now();
    // initialize this node
    const ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    // run the SLAM as subscriber
    image_transport::Subscriber sub = it.subscribe("camera/image_raw", 1, [&](const sensor_msgs::ImageConstPtr& msg) {
        const auto tp_1 = std::chrono::steady_clock::now();
        const auto timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(tp_1 - tp_0).count();
        // input the current frame and estimate the camera pose
        SLAM.feed_monocular_frame(cv_bridge::toCvShare(msg, "bgr8")->image, timestamp, mask);
        // Get the camera position

        const auto tp_2 = std::chrono::steady_clock::now();
        const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
        track_times.push_back(track_time);
    });
    // run the viewer in another thread
#ifdef USE_PANGOLIN_VIEWER
    std::thread thread([&]() {
        viewer.run();
        if (SLAM.terminate_is_requested()) {
            // wait until the loop BA is finished
            while (SLAM.loop_BA_is_running()) {
                std::this_thread::sleep_for(std::chrono::microseconds(5000));
            }
            ros::shutdown();
        }
    });
#elif USE_SOCKET_PUBLISHER
    std::thread thread([&]() {
        publisher.run();
        if (SLAM.terminate_is_requested()) {
            // wait until the loop BA is finished
            while (SLAM.loop_BA_is_running()) {
                std::this_thread::sleep_for(std::chrono::microseconds(5000));
            }
            ros::shutdown();
        }
    });
#endif
    ros::spin();
    // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
    viewer.request_terminate();
    thread.join();
#elif USE_SOCKET_PUBLISHER
    publisher.request_terminate();
    thread.join();
#endif
    // shutdown the SLAM process
    SLAM.shutdown();
    if (eval_log) {
        // output the trajectories for evaluation
        SLAM.save_frame_trajectory("frame_trajectory.txt", "TUM");
        SLAM.save_keyframe_trajectory("keyframe_trajectory.txt", "TUM");
        // output the tracking times for evaluation
        std::ofstream ofs("track_times.txt", std::ios::out);
        if (ofs.is_open()) {
            for (const auto track_time : track_times) {
                ofs << track_time << std::endl;
            }
            ofs.close();
        }
    }
    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path);
    }
    if (track_times.size()) {
        std::sort(track_times.begin(), track_times.end());
        const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
        std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
        std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
    }
}

geometry_msgs::PoseStamped convert_cam_to_world(openvslam::Mat44_t cam_pose)
{
    //if (pose.empty())
    //    return;

    const openvslam::Mat33_t& rot = cam_pose.block<3, 3>(0, 0);
    const openvslam::Vec3_t& trans = cam_pose.block<3, 1>(0, 3);
    const openvslam::Quat_t quat = openvslam::Quat_t(rot);

    cv::Mat pose = (cv::Mat_<float>(4,4) <<   rot(0,0),rot(0,1),rot(0,2), trans(0),
                                     rot(1,0), rot(1,1),rot(1,2), trans(1),
                                     rot(2,0),rot(2,1), rot(2,2), trans(2),
                                     0, 0, 0, 1);
    
    /*cv::Mat pose = cv::Mat::eye(4,4, CV_32F);

    pose.at<float>(0,0) = rot(0,0);
    pose.at<float>(0,1) = rot(0,1);
    pose.at<float>(0,2) = rot(0,2);
    pose.at<float>(0,3) = trans(0);
    pose.at<float>(1,0) = rot(1,0);
    pose.at<float>(1,1) = rot(1,1);
    pose.at<float>(1,2) = rot(1,2);
    pose.at<float>(1,3) = trans(1);
    pose.at<float>(2,0) = rot(2,0);
    pose.at<float>(2,1) = rot(2,1);
    pose.at<float>(2,2) = rot(2,2);
    pose.at<float>(2,3) = trans(2);*/

    //std::cout << pose.at<float>(0,0) << std::endl;
    /* global left handed coordinate system */
    static cv::Mat pose_prev = cv::Mat::eye(4,4, CV_32F);
    static cv::Mat world_lh = cv::Mat::eye(4,4, CV_32F);
    // matrix to flip signs of sinus in rotation matrix, not sure why we need to do that
    static const cv::Mat flipSign = (cv::Mat_<float>(4,4) <<   1,-1,-1, 1,
                                     -1, 1,-1, 1,
                                     -1,-1, 1, 1,
                                     1, 1, 1, 1);

    //prev_pose * T = pose
    cv::Mat translation =  (pose * pose_prev.inv()).mul(flipSign);
    world_lh = world_lh * translation;
    pose_prev = pose.clone();

    tf::Matrix3x3 tf3d;
    tf3d.setValue(pose.at<float>(0,0), pose.at<float>(0,1), pose.at<float>(0,2),
                  pose.at<float>(1,0), pose.at<float>(1,1), pose.at<float>(1,2),
                  pose.at<float>(2,0), pose.at<float>(2,1), pose.at<float>(2,2));

    tf::Vector3 cameraTranslation_rh( world_lh.at<float>(0,3),world_lh.at<float>(1,3), - world_lh.at<float>(2,3) );

    //rotate 270deg about x and 270deg about x to get ENU: x forward, y left, z up
    const tf::Matrix3x3 rotation270degXZ(   0, 1, 0,
                                            0, 0, 1,
                                            1, 0, 0);

    static tf::TransformBroadcaster br;

    tf::Matrix3x3 globalRotation_rh = tf3d;
    tf::Vector3 globalTranslation_rh = cameraTranslation_rh * rotation270degXZ;

    tf::Quaternion tfqt;
    globalRotation_rh.getRotation(tfqt);

    double aux = tfqt[0];
    tfqt[0]=-tfqt[2];
    tfqt[2]=tfqt[1];
    tfqt[1]=aux;

    tf::Transform transform;
    transform.setOrigin(globalTranslation_rh);
    transform.setRotation(tfqt);

    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "camera_link"));

    geometry_msgs::PoseStamped orb_slam_pose;
    orb_slam_pose.header.stamp = ros::Time::now();
    orb_slam_pose.header.frame_id = "map";

    tf::poseTFToMsg(transform, orb_slam_pose.pose);
    //pub.publish(orb_slam_pose);
    return orb_slam_pose;
}

void callback(const sensor_msgs::ImageConstPtr& left_image, const sensor_msgs::ImageConstPtr& right_image, std::vector<double> &track_times,
              const std::chrono::_V2::steady_clock::time_point tp_0, openvslam::system *SLAM, ros::Publisher odom_pub){
  
    const auto tp_1 = std::chrono::steady_clock::now();
    const auto timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(tp_1 - tp_0).count();
    openvslam::Mat44_t cam_pose;
    cam_pose = SLAM->feed_stereo_frame(cv_bridge::toCvShare(left_image, "mono8")->image, cv_bridge::toCvShare(right_image, "mono8")->image,timestamp); //feed mask in the end [optinal]
    //cv::Mat pose_cv;
    //eigen2cv(cam_pose, pose_cv);
    geometry_msgs::PoseStamped pose_world_frame;
    pose_world_frame = convert_cam_to_world(cam_pose);
    nav_msgs::Odometry odom;
    odom.header = pose_world_frame.header;
    odom.pose.pose = pose_world_frame.pose;
    odom_pub.publish(odom);


    /*
    // Separate the pose message
    const openvslam::Mat33_t& rot = cam_pose.block<3, 3>(0, 0);
    const openvslam::Vec3_t& trans = cam_pose.block<3, 1>(0, 3);
    const openvslam::Quat_t quat = openvslam::Quat_t(rot);
    // Fill the odometry message
    nav_msgs::Odometry odom;
    odom.header = header;
    odom.header.frame_id = "world";
    odom.pose.pose.position.x = trans(0);
    odom.pose.pose.position.y = trans(1);
    odom.pose.pose.position.z = trans(2);
    odom.pose.pose.orientation = toMsg(quat);
    // Publish the odometry
    odom_pub.publish(odom);
    */
    const auto tp_2 = std::chrono::steady_clock::now();
    const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
    track_times.push_back(track_time);
}
void callbackLeftInfo(const sensor_msgs::Image& info){
    header = info.header;
}
void stereo_tracking(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path,
                   const std::string& mask_img_path, const bool eval_log, const std::string& map_db_path) {
    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup();

    std::string save_file_path;
    if (ros::param::get("~save_file_path", save_file_path))
    {
        ROS_WARN("Using keyframes path file: %s", save_file_path.c_str());
    }
    else
    {
        ROS_ERROR("Failed to get path file to save the keyframes. Please check and try again.");
        ros::shutdown();
    }

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif
    std::vector<double> track_times;
    const auto tp_0 = std::chrono::steady_clock::now();
    // initialize this node
    ros::NodeHandle nh;  

    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("openvslam_odom",1);
    ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("openvslam_pose",1);
    ros::Subscriber camera_info = nh.subscribe("/left/image_rect", 1, callbackLeftInfo);

    message_filters::Subscriber<sensor_msgs::Image> left_image_sub(nh, "/left/image_rect", 1);
    message_filters::Subscriber<sensor_msgs::Image> right_image_sub(nh, "/right/image_rect", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MyApproxSyncPolicy;
    message_filters::Synchronizer<MyApproxSyncPolicy> sync(MyApproxSyncPolicy(10), left_image_sub, right_image_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2, track_times, tp_0, &SLAM, odom_pub));

    // run the viewer in another thread
#ifdef USE_PANGOLIN_VIEWER
    std::thread thread([&]() {
        viewer.run();
        if (SLAM.terminate_is_requested()) {
            // wait until the loop BA is finished
            while (SLAM.loop_BA_is_running()) {
                std::this_thread::sleep_for(std::chrono::microseconds(5000));
            }
            ros::shutdown();
        }
    });
#elif USE_SOCKET_PUBLISHER
    std::thread thread([&]() {
        publisher.run();
        if (SLAM.terminate_is_requested()) {
            // wait until the loop BA is finished
            while (SLAM.loop_BA_is_running()) {
                std::this_thread::sleep_for(std::chrono::microseconds(5000));
            }
            ros::shutdown();
        }
    });
#endif
    ros::spin();
    // automatically close the viewer
#ifdef USE_PANGOLIN_VIEWER
    viewer.request_terminate();
    thread.join();
#elif USE_SOCKET_PUBLISHER
    publisher.request_terminate();
    thread.join();
#endif
    // shutdown the SLAM process
    SLAM.shutdown();
    if (eval_log) {
        // output the trajectories for evaluation
        SLAM.save_frame_trajectory(save_file_path+"frame_trajectory.txt", "TUM");
        SLAM.save_keyframe_trajectory(save_file_path+"keyframe_trajectory.txt", "TUM");
        // output the tracking times for evaluation
        std::ofstream ofs(save_file_path+"track_times.txt", std::ios::out);
        if (ofs.is_open()) {
            for (const auto track_time : track_times) {
                ofs << track_time << std::endl;
            }
            ofs.close();
        }
    }
    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path);
    }
    if (track_times.size()) {
        std::sort(track_times.begin(), track_times.end());
        const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
        std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
        std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;
    }
}
int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif
    ros::init(argc, argv, "run_slam");
    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto setting_file_path = op.add<popl::Value<std::string>>("c", "config", "setting file path");
    auto mask_img_path = op.add<popl::Value<std::string>>("", "mask", "mask image path", "");
    auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
    auto eval_log = op.add<popl::Switch>("", "eval-log", "store trajectory and tracking times for evaluation");
    auto map_db_path = op.add<popl::Value<std::string>>("", "map-db", "store a map database at this path after SLAM", "");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !setting_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode->is_set()) {
        spdlog::set_level(spdlog::level::debug);
    }
    else {
        spdlog::set_level(spdlog::level::info);
    }
    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(setting_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif
    // run tracking
    if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
        mono_tracking(cfg, vocab_file_path->value(), mask_img_path->value(), eval_log->is_set(), map_db_path->value());
    }
    else if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Stereo) {
        std::cout << "Using stereo.." << std::endl;
        stereo_tracking(cfg, vocab_file_path->value(), mask_img_path->value(), eval_log->is_set(), map_db_path->value());
    }
    else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }
#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif
    return EXIT_SUCCESS;
}

