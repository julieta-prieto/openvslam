#ifndef CONFIGPARAM_H
#define CONFIGPARAM_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
#include <yaml-cpp/yaml.h>
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

#define TRACK_WITH_IMU

#define RUN_REALTIME

namespace openvslam
{

class ConfigParam
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ConfigParam(std::string configfile);

    double _testDiscardTime;

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    static cv::Mat GetCamMatrix();
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    static Eigen::Matrix4d GetEigTbc();
    static cv::Mat GetMatTbc();
    static Eigen::Matrix4d GetEigT_cb();
    static cv::Mat GetMatT_cb();
    static int GetLocalWindowSize();
    static double GetImageDelayToIMU();
    static bool GetAccMultiply9p8();

    static double GetG(){return _g;}

    std::string _bagfile;
    std::string _imageTopic;
    std::string _imuTopic;

    static double GetVINSInitTime(){return _nVINSInitTime;}

private:
    static Eigen::Matrix4d _EigTbc;
    static cv::Mat _MatTbc;
    static Eigen::Matrix4d _EigTcb;
    static cv::Mat _MatTcb;
    static int _LocalWindowSize;
    static double _ImageDelayToIMU;
    static bool _bAccMultiply9p8;

    static double _g;
    static double _nVINSInitTime;

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    static cv::Mat cam_mat;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
};

}

#endif // CONFIGPARAM_H
