#include "openvslam/optimize/g2o/se3/shot_vertex_container.h"
#include "openvslam/util/converter.h"

namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {

shot_vertex_container::shot_vertex_container(const unsigned int offset, const unsigned int num_reserve)
    : offset_(offset) {
    vtx_container_.reserve(num_reserve);
}

shot_vertex* shot_vertex_container::create_vertex(const unsigned int id, const Mat44_t& cam_pose_cw, const bool is_constant) {
    // vertexを作成
    const auto vtx_id = offset_ + id;
    auto vtx = new shot_vertex();
    vtx->setId(vtx_id);
    std::cout << "DEBUG 4" << std::endl;
    std::cout << cam_pose_cw(0, 0) << std::endl;
    std::cout << cam_pose_cw(0, 1) << std::endl;
    std::cout << cam_pose_cw(0, 2) << std::endl;
    std::cout << cam_pose_cw(0, 3) << std::endl;
    std::cout << cam_pose_cw(1, 0) << std::endl;
    std::cout << cam_pose_cw(1, 1) << std::endl;
    std::cout << cam_pose_cw(1, 2) << std::endl;
    std::cout << cam_pose_cw(1, 3) << std::endl;
    std::cout << cam_pose_cw(2, 0) << std::endl;
    std::cout << cam_pose_cw(2, 1) << std::endl;
    std::cout << cam_pose_cw(2, 2) << std::endl;
    std::cout << cam_pose_cw(2, 3) << std::endl;
    std::cout << cam_pose_cw(3, 0) << std::endl;
    std::cout << cam_pose_cw(3, 1) << std::endl;
    std::cout << cam_pose_cw(3, 2) << std::endl;
    std::cout << cam_pose_cw(3, 3) << std::endl;
    vtx->setEstimate(util::converter::to_g2o_SE3(cam_pose_cw));
    std::cout << "DEBUG 5" << std::endl;
    vtx->setFixed(is_constant);
    // databaseに登録
    vtx_container_[id] = vtx;
    // max IDを更新
    if (max_vtx_id_ < vtx_id) {
        max_vtx_id_ = vtx_id;
    }
    // 作成したvertexをreturn
    return vtx;
}

} // namespace se3
} // namespace g2o
} // namespace optimize
} // namespace openvslam
