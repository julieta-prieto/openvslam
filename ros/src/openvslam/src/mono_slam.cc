#ifdef USE_PANGOLIN_VIEWER
#include <pangolin_viewer/viewer.h>
#elif USE_SOCKET_PUBLISHER
#include <socket_publisher/publisher.h>
#endif

#include <openvslam/system.h>
#include <openvslam/config.h>

#include <iostream>
#include <chrono>
#include <numeric>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
#include "MsgSync/MsgSynchronizer.h"
#include "MsgSync/MsgSynchronizer.cpp"
#include "openvslam/IMU/imudata.h"
#include "openvslam/IMU/configparam.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <boost/foreach.hpp>

#include "/home/julieta/openvslam/ros/src/openvslam/msg_gen/cpp/include/ORB_VIO/viorb_msg.h"
#include "openvslam/util/converter.h"

//#undef RUN_REALTIME
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void PublishSLAMData(ros::Publisher& slamdatapub, const openvslam::SLAMData& slamdata)
{
    ORB_VIO::viorb_msg slamdatamsg;
    slamdatamsg.header.stamp = ros::Time(slamdata.Timestamp);
    // VINSInitFlag
    slamdatamsg.VINSInitFlag.data = slamdata.VINSInitFlag;
    // TrackStatus
    slamdatamsg.TrackStatus.data = slamdata.TrackingStatus;
    // -1-notready,0-noimage,1-initing,2-OK
    if(slamdata.TrackingStatus > 1 && slamdata.VINSInitFlag==true)
    {
        cv::Mat Rwi = slamdata.Rwi;

        // Qwi
        Eigen::Matrix<double,3,3> eigRwi = openvslam::util::converter::toMatrix3d(Rwi);
        Eigen::Quaterniond qwi(eigRwi);
        geometry_msgs::Quaternion qwimsg;
        qwimsg.x = qwi.x();
        qwimsg.y = qwi.y();
        qwimsg.z = qwi.z();
        qwimsg.w = qwi.w();
        slamdatamsg.Qwi = qwimsg;

        // gw
        geometry_msgs::Point gw;
        gw.x = slamdata.gw.at<float>(0);
        gw.y = slamdata.gw.at<float>(1);
        gw.z = slamdata.gw.at<float>(2);
        slamdatamsg.gw = gw;

        // Tic(Ric&tic)
        cv::Mat Riw = Rwi.t();

        cv::Mat Rcw = slamdata.Tcw.rowRange(0,3).colRange(0,3).clone();
        cv::Mat tcw = slamdata.Tcw.rowRange(0,3).col(3).clone();
        cv::Mat Rwc = Rcw.t();
        cv::Mat twc = -Rwc*tcw;

        cv::Mat tic = Riw*twc;
        cv::Mat Ric = Riw*Rwc;

        Eigen::Matrix<double,3,3> eigRic = openvslam::util::converter::toMatrix3d(Ric);
        Eigen::Quaterniond qic(eigRic);

        geometry_msgs::Quaternion qicmsg;
        qicmsg.x = qic.x();
        qicmsg.y = qic.y();
        qicmsg.z = qic.z();
        qicmsg.w = qic.w();

        geometry_msgs::Point ticmsg;
        ticmsg.x = tic.at<float>(0);
        ticmsg.y = tic.at<float>(1);
        ticmsg.z = tic.at<float>(2);

        geometry_msgs::Pose Ticmsg;
        Ticmsg.orientation = qicmsg;
        Ticmsg.position = ticmsg;

        slamdatamsg.Tic = Ticmsg;
    }

    slamdatapub.publish(slamdatamsg);
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

void mono_tracking(const std::shared_ptr<openvslam::config>& cfg, const std::string& vocab_file_path,
                   const std::string& mask_img_path, const bool eval_log, const std::string& map_db_path) {
    // load the mask image
    const cv::Mat mask = mask_img_path.empty() ? cv::Mat{} : cv::imread(mask_img_path, cv::IMREAD_GRAYSCALE);

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    openvslam::ConfigParam config(cfg->config_file_path_);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

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

    /*std::vector<double> track_times;
    const auto tp_0 = std::chrono::steady_clock::now();*/

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    /**
     * @brief added data sync
     */
    ros::NodeHandle nh;
    double imageMsgDelaySec = config.GetImageDelayToIMU();
    openvslam::MsgSynchronizer msgsync(imageMsgDelaySec);
    
#ifdef RUN_REALTIME
    ros::Subscriber imagesub = nh.subscribe(config._imageTopic, 2, &openvslam::MsgSynchronizer::imageCallback, &msgsync);
    ros::Subscriber imusub = nh.subscribe(config._imuTopic, 200, &openvslam::MsgSynchronizer::imuCallback, &msgsync);
#endif
    sensor_msgs::ImageConstPtr imageMsg;
    std::vector<sensor_msgs::ImuConstPtr> vimuMsg;

    // 3dm imu output per g. 1g=9.80665 according to datasheet
    const double g3dm = 9.80665;
    const bool bAccMultiply98 = config.GetAccMultiply9p8();

    #ifndef RUN_REALTIME
        std::string bagfile = config._bagfile;
        rosbag::Bag bag;
        bag.open(bagfile,rosbag::bagmode::Read);

        std::vector<std::string> topics;
        std::string imutopic = config._imuTopic;
        std::string imagetopic = config._imageTopic;
        topics.push_back(imagetopic);
        topics.push_back(imutopic);

        rosbag::View view(bag, rosbag::TopicQuery(topics));
    #endif

    openvslam::SLAMData slamdata;
    ros::Publisher slamdatapub = nh.advertise<ORB_VIO::viorb_msg>("VIORB/SLAMData", 10);

    ros::Rate r(1000);

    #ifdef RUN_REALTIME
    ROS_WARN("Run realtime");
    while(ros::ok())
    {
#else
    ROS_WARN("Run non-realtime");
    BOOST_FOREACH(rosbag::MessageInstance const m, view)
    {
        sensor_msgs::ImuConstPtr simu = m.instantiate<sensor_msgs::Imu>();
        if(simu!=NULL)
            msgsync.imuCallback(simu);
        sensor_msgs::ImageConstPtr simage = m.instantiate<sensor_msgs::Image>();
        if(simage!=NULL)
            msgsync.imageCallback(simage);
#endif

        bool bdata = msgsync.getRecentMsgs(imageMsg,vimuMsg);
        if(bdata)
        {
            //if(msgsync.getImageMsgSize()>=2) ROS_ERROR("image queue size: %d",msgsync.getImageMsgSize());
            std::vector<openvslam::IMUData> vimuData;
            //ROS_INFO("image time: %.3f",imageMsg->header.stamp.toSec());
            for(unsigned int i=0;i<vimuMsg.size();i++)
            {
                sensor_msgs::ImuConstPtr imuMsg = vimuMsg[i];
                double ax = imuMsg->linear_acceleration.x;
                double ay = imuMsg->linear_acceleration.y;
                double az = imuMsg->linear_acceleration.z;
                if(bAccMultiply98)
                {
                    ax *= g3dm;
                    ay *= g3dm;
                    az *= g3dm;
                }
                openvslam::IMUData imudata(imuMsg->angular_velocity.x,imuMsg->angular_velocity.y,imuMsg->angular_velocity.z,
                                ax,ay,az,imuMsg->header.stamp.toSec());
                vimuData.push_back(imudata);
                //ROS_INFO("imu time: %.3f",vimuMsg[i]->header.stamp.toSec());
            }

            // Copy the ros image message to cv::Mat.
            cv_bridge::CvImageConstPtr cv_ptr;
            try
            {
                cv_ptr = cv_bridge::toCvShare(imageMsg);
            }
            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            // Consider delay of image message
            //SLAM.TrackMonocular(cv_ptr->image, imageMsg->header.stamp.toSec() - imageMsgDelaySec);
            // Below to test relocalizaiton
            cv::Mat im = cv_ptr->image.clone();
            {
                // To test relocalization
                static double startT=-1;
                if(startT<0)
                    startT = imageMsg->header.stamp.toSec();
                //if(imageMsg->header.stamp.toSec() > startT+25 && imageMsg->header.stamp.toSec() < startT+25.3)
                if(imageMsg->header.stamp.toSec() < startT+config._testDiscardTime)
                    im = cv::Mat::zeros(im.rows,im.cols,im.type());
            }
            //***C++11 Style:***
            //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            SLAM.feed_monocular_frame_VI(im, vimuData, imageMsg->header.stamp.toSec() - imageMsgDelaySec);
            //std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
            //ROS_INFO_STREAM( "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl );

            SLAM.GetSLAMData(slamdata);
            PublishSLAMData(slamdatapub,slamdata);

            //SLAM.TrackMonoVI(cv_ptr->image, vimuData, imageMsg->header.stamp.toSec() - imageMsgDelaySec);
            //cv::imshow("image",cv_ptr->image);//
#ifndef RUN_REALTIME
            // Wait local mapping end.
            bool bstop = false;
            
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            while(!SLAM.bLocalMapAcceptKF())
            {
                if(!ros::ok())
                {
                    bstop=true;
                }
            };
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            double vlocalbatime2 = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            ROS_INFO("Waited: %f", vlocalbatime2);
            if(bstop)
                break;
#endif
        }

        //cv::waitKey(1);

        ros::spinOnce();
        r.sleep();
        if(!ros::ok())
            break;
    }

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

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
/*
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
    }*/
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
    else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
}
