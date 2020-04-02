#include "openvslam/util/converter.h"

namespace openvslam {
namespace util {

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
void converter::updateNS(NavState& ns, const IMUPreintegrator& imupreint, const Vector3d& gw)
{
    Matrix3d dR = imupreint.getDeltaR();
    Vector3d dP = imupreint.getDeltaP();
    Vector3d dV = imupreint.getDeltaV();
    double dt = imupreint.getDeltaTime();

    Vector3d Pwbpre = ns.Get_P();
    Matrix3d Rwbpre = ns.Get_RotMatrix();
    Vector3d Vwbpre = ns.Get_V();

    Matrix3d Rwb = Rwbpre * dR;
    Vector3d Pwb = Pwbpre + Vwbpre*dt + 0.5*gw*dt*dt + Rwbpre*dP;
    Vector3d Vwb = Vwbpre + gw*dt + Rwbpre*dV;

    // Here assume that the pre-integration is re-computed after bias updated, so the bias term is ignored
    ns.Set_Pos(Pwb);
    ns.Set_Vel(Vwb);
    ns.Set_Rot(Rwb);

    // Test log
    if(ns.Get_dBias_Gyr().norm()>1e-6 || ns.Get_dBias_Acc().norm()>1e-6) std::cerr<<"delta bias in updateNS is not zero"<<ns.Get_dBias_Gyr().transpose()<<", "<<ns.Get_dBias_Acc().transpose()<<std::endl;
}

cv::Mat converter::toCvMatInverse(const cv::Mat &Tcw)
{
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    cv::Mat twc = -Rwc*tcw;

    cv::Mat Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    twc.copyTo(Twc.rowRange(0,3).col(3));

    return Twc.clone();
}

cv::Mat converter::toCvMat(const g2o::SE3Quat &SE3)
{
    Eigen::Matrix<double,4,4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}

cv::Mat converter::toCvMat(const g2o::Sim3 &Sim3)
{
    Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
    Eigen::Vector3d eigt = Sim3.translation();
    double s = Sim3.scale();
    return toCvSE3(s*eigR,eigt);
}

cv::Mat converter::toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat converter::toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat converter::toCvMat(const Eigen::Matrix<double,3,1> &m)
{
    cv::Mat cvMat(3,1,CV_32F);
    for(int i=0;i<3;i++)
            cvMat.at<float>(i)=m(i);

    return cvMat.clone();
}

cv::Mat converter::toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t)
{
    cv::Mat cvMat = cv::Mat::eye(4,4,CV_32F);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            cvMat.at<float>(i,j)=R(i,j);
        }
    }
    for(int i=0;i<3;i++)
    {
        cvMat.at<float>(i,3)=t(i);
    }

    return cvMat.clone();
}

Eigen::Matrix<double,3,1> converter::toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<double,3,1> converter::toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double,3,1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double,3,3> converter::toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

Mat44_t converter::cvMat4_to_Mat44_t(const cv::Mat& cvMat4)
{
    Mat44_t M;
    M << cvMat4.at<float>(0,0), cvMat4.at<float>(0,1), cvMat4.at<float>(0,2), cvMat4.at<float>(0,3),
         cvMat4.at<float>(1,0), cvMat4.at<float>(1,1), cvMat4.at<float>(1,2), cvMat4.at<float>(1,3),
         cvMat4.at<float>(2,0), cvMat4.at<float>(2,1), cvMat4.at<float>(2,2), cvMat4.at<float>(2,3),
         cvMat4.at<float>(3,0), cvMat4.at<float>(3,1), cvMat4.at<float>(3,2), cvMat4.at<float>(3,3);
    return M;
}

cv::Mat Mat44_t_to_cvMat4(const Mat44_t& mat)
{
    cv::Mat cvMat;
    cvMat = (cv::Mat_<float>(4,4) << mat(0,0), mat(0,1), mat(0,2), mat(0,3),
                                     mat(1,0), mat(1,1), mat(1,2), mat(1,3),
                                     mat(2,0), mat(2,1), mat(2,2), mat(2,3),
                                     mat(3,0), mat(3,1), mat(3,2), mat(3,3));
    return cvMat.clone();
}
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

std::vector<cv::Mat> converter::to_desc_vec(const cv::Mat& desc) {
    std::vector<cv::Mat> desc_vec(desc.rows);
    for (int i = 0; i < desc.rows; ++i) {
        desc_vec.at(i) = desc.row(i);
    }
    return desc_vec;
}

g2o::SE3Quat converter::to_g2o_SE3(const Mat44_t& cam_pose) {
    const Mat33_t rot = cam_pose.block<3, 3>(0, 0);
    const Vec3_t trans = cam_pose.block<3, 1>(0, 3);
    return g2o::SE3Quat{rot, trans};
}

Mat44_t converter::to_eigen_mat(const g2o::SE3Quat& g2o_SE3) {
    return g2o_SE3.to_homogeneous_matrix();
}

Mat44_t converter::to_eigen_mat(const g2o::Sim3& g2o_Sim3) {
    Mat44_t cam_pose = Mat44_t::Identity();
    cam_pose.block<3, 3>(0, 0) = g2o_Sim3.scale() * g2o_Sim3.rotation().toRotationMatrix();
    cam_pose.block<3, 1>(0, 3) = g2o_Sim3.translation();
    return cam_pose;
}

Mat44_t converter::to_eigen_cam_pose(const Mat33_t& rot, const Vec3_t& trans) {
    Mat44_t cam_pose = Mat44_t::Identity();
    cam_pose.block<3, 3>(0, 0) = rot;
    cam_pose.block<3, 1>(0, 3) = trans;
    return cam_pose;
}

Vec3_t converter::to_angle_axis(const Mat33_t& rot_mat) {
    const Eigen::AngleAxisd angle_axis(rot_mat);
    return angle_axis.axis() * angle_axis.angle();
}

Mat33_t converter::to_rot_mat(const Vec3_t& angle_axis) {
    Eigen::Matrix3d rot_mat;
    const double angle = angle_axis.norm();
    if (angle <= 1e-5) {
        rot_mat = Eigen::Matrix3d::Identity();
    }
    else {
        const Eigen::Vector3d axis = angle_axis / angle;
        rot_mat = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    }
    return rot_mat;
}

Mat33_t converter::to_skew_symmetric_mat(const Vec3_t& vec) {
    Mat33_t skew;
    skew << 0, -vec(2), vec(1),
        vec(2), 0, -vec(0),
        -vec(1), vec(0), 0;
    return skew;
}

} // namespace util
} // namespace openvslam
