#include "configparam.h"

namespace openvslam
{
double ConfigParam::_g = 9.810;

Eigen::Matrix4d ConfigParam::_EigTbc = Eigen::Matrix4d::Identity();
cv::Mat ConfigParam::_MatTbc = cv::Mat::eye(4,4,CV_32F);
Eigen::Matrix4d ConfigParam::_EigTcb = Eigen::Matrix4d::Identity();
cv::Mat ConfigParam::_MatTcb = cv::Mat::eye(4,4,CV_32F);
int ConfigParam::_LocalWindowSize = 10;
double ConfigParam::_ImageDelayToIMU = 0;
bool ConfigParam::_bAccMultiply9p8 = false;
double ConfigParam::_nVINSInitTime = 15;

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
cv::Mat ConfigParam::cam_mat = cv::Mat::eye(3,3,CV_32F);
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

ConfigParam::ConfigParam(std::string configfile)
{
    YAML::Node fSettings = YAML::LoadFile(configfile);
    std::cout<<std::endl<<std::endl<<"Parameters: "<<std::endl;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    float fx = (float) fSettings["Camera.fx"].as<double>();
    float fy = (float) fSettings["Camera.fy"].as<double>();
    float cx = (float) fSettings["Camera.cx"].as<double>();
    float cy = (float) fSettings["Camera.cy"].as<double>();

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(cam_mat);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    
    _testDiscardTime = fSettings["test.DiscardTime"].as<double>();
    _nVINSInitTime = fSettings["test.VINSInitTime"].as<double>();
    std::cout<<"VINS initialize time: "<<_nVINSInitTime<<std::endl;
    std::cout<<"Discart time in test data: "<<_testDiscardTime<<std::endl;

    _bagfile = fSettings["bagfile"].as<std::string>();
    std::cout<<"open rosbag: "<<_bagfile<<std::endl;
    _imuTopic = fSettings["imutopic"].as<std::string>();
    _imageTopic = fSettings["imagetopic"].as<std::string>();
    std::cout<<"imu topic: "<<_imuTopic<<std::endl;
    std::cout<<"image topic: "<<_imageTopic<<std::endl;

    _LocalWindowSize = fSettings["LocalMapping.LocalWindowSize"].as<int>();
    std::cout<<"local window size: "<<_LocalWindowSize<<std::endl;

    _ImageDelayToIMU = fSettings["Camera.delaytoimu"].as<double>();
    std::cout<<"timestamp image delay to imu: "<<_ImageDelayToIMU<<std::endl;

    {
        const std::vector<double> Tbc_ = fSettings["Camera.Tbc"].as<std::vector<double>>();
        //cv::FileNode Tbc_ = fSettings["Camera.Tbc"];
        Eigen::Matrix<double,3,3> R;
        R << Tbc_[0], Tbc_[1], Tbc_[2],
                Tbc_[4], Tbc_[5], Tbc_[6],
                Tbc_[8], Tbc_[9], Tbc_[10];
        Eigen::Quaterniond qr(R);
        R = qr.normalized().toRotationMatrix();
        Eigen::Matrix<double,3,1> t( Tbc_[3], Tbc_[7], Tbc_[11]);
        _EigTbc = Eigen::Matrix4d::Identity();
        _EigTbc.block<3,3>(0,0) = R;
        _EigTbc.block<3,1>(0,3) = t;
        _MatTbc = cv::Mat::eye(4,4,CV_32F);
        for(int i=0;i<4;i++)
            for(int j=0;j<4;j++)
                _MatTbc.at<float>(i,j) = _EigTbc(i,j);

        _EigTcb = Eigen::Matrix4d::Identity();
        _EigTcb.block<3,3>(0,0) = R.transpose();
        _EigTcb.block<3,1>(0,3) = -R.transpose()*t;
        for(int i=0;i<4;i++)
            for(int j=0;j<4;j++)
                _MatTcb.at<float>(i,j) = _EigTcb(i,j);

        // Tbc_[0], Tbc_[1], Tbc_[2], Tbc_[3], Tbc_[4], Tbc_[5], Tbc_[6], Tbc_[7], Tbc_[8], Tbc_[9], Tbc_[10], Tbc_[11], Tbc_[12], Tbc_[13], Tbc_[14], Tbc_[15];
        std::cout<<"Tbc inited:"<<std::endl<<_EigTbc<<std::endl<<_MatTbc<<std::endl;
        std::cout<<"Tcb inited:"<<std::endl<<_EigTcb<<std::endl<<_MatTcb<<std::endl;
        std::cout<<"Tbc*Tcb:"<<std::endl<<_EigTbc*_EigTcb<<std::endl<<_MatTbc*_MatTcb<<std::endl;
    }

    {
        int tmpBool = fSettings["IMU.multiplyG"].as<int>();
        _bAccMultiply9p8 = (tmpBool != 0);
        std::cout<<"whether acc*9.8? 0/1: "<<_bAccMultiply9p8<<std::endl;
    }
}

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
cv::Mat ConfigParam::GetCamMatrix()
{
    return cam_mat;
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

Eigen::Matrix4d ConfigParam::GetEigTbc()
{
    return _EigTbc;
}

cv::Mat ConfigParam::GetMatTbc()
{
    return _MatTbc.clone();
}

Eigen::Matrix4d ConfigParam::GetEigT_cb()
{
    return _EigTcb;
}

cv::Mat ConfigParam::GetMatT_cb()
{
    return _MatTcb.clone();
}

int ConfigParam::GetLocalWindowSize()
{
    return _LocalWindowSize;
}

double ConfigParam::GetImageDelayToIMU()
{
    return _ImageDelayToIMU;
}

bool ConfigParam::GetAccMultiply9p8()
{
    return _bAccMultiply9p8;
}

}
