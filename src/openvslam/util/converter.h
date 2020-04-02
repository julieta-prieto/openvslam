#ifndef OPENVSLAM_UTIL_CONVERTER_H
#define OPENVSLAM_UTIL_CONVERTER_H

#include "openvslam/type.h"

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include "openvslam/IMU/NavState.h"
#include "openvslam/IMU/IMUPreintegrator.h"

namespace openvslam {
namespace util {

class converter {
public:
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    static void updateNS(NavState& ns, const IMUPreintegrator& imupreint, const Vector3d& gw);
    static cv::Mat toCvMatInverse(const cv::Mat &T12);
    static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
    static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
    static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
    static cv::Mat toCvMat(const Eigen::Matrix3d &m);
    static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
    static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);

    static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
    static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
    static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);

    //CREACION MIA
    static Mat44_t cvMat4_to_Mat44_t(const cv::Mat& cvMat4);
    static cv::Mat Mat44_t_to_cvMat4(const Mat44_t& mat);

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    //! descriptor vector
    static std::vector<cv::Mat> to_desc_vec(const cv::Mat& desc);

    //! to SE3 of g2o
    static g2o::SE3Quat to_g2o_SE3(const Mat44_t& cam_pose);

    //! to Eigen::Mat/Vec
    static Mat44_t to_eigen_mat(const g2o::SE3Quat& g2o_SE3);
    static Mat44_t to_eigen_mat(const g2o::Sim3& g2o_Sim3);
    static Mat44_t to_eigen_cam_pose(const Mat33_t& rot, const Vec3_t& trans);

    //! from/to angle axis
    static Vec3_t to_angle_axis(const Mat33_t& rot_mat);
    static Mat33_t to_rot_mat(const Vec3_t& angle_axis);

    //! to homogeneous coordinates
    template<typename T>
    static Vec3_t to_homogeneous(const cv::Point_<T>& pt) {
        return Vec3_t{pt.x, pt.y, 1.0};
    }

    //! to skew symmetric matrix
    static Mat33_t to_skew_symmetric_mat(const Vec3_t& vec);
};

} // namespace util
} // namespace openvslam

#endif // OPENVSLAM_UTIL_CONVERTER_H
