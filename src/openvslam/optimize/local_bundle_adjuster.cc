#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/optimize/local_bundle_adjuster.h"
#include "openvslam/optimize/g2o/landmark_vertex_container.h"
#include "openvslam/optimize/g2o/se3/shot_vertex_container.h"
#include "openvslam/optimize/g2o/se3/reproj_edge_wrapper.h"
#include "openvslam/util/converter.h"

#include <unordered_map>

#include "openvslam/IMU/g2otypes.h"
#include <opencv2/core/eigen.hpp>

#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <g2o/core/optimizable_graph.h>

namespace openvslam {

/*void VertexNavStatePVR(data::keyframe* local_keyfrm, int idKF, ::g2o::SparseOptimizer& optimizer)
{
    // Vertex of PVR
    g2o::VertexNavStatePVR * vNSPVR = new g2o::VertexNavStatePVR();
    vNSPVR->setEstimate(local_keyfrm->GetNavState());
    vNSPVR->setId(idKF);
    vNSPVR->setFixed(false);
    optimizer.addVertex(vNSPVR);
}

void VertexNavStateBias(data::keyframe* local_keyfrm, int idKF, ::g2o::SparseOptimizer& optimizer)
{
    // Vertex of Bias
    g2o::VertexNavStateBias * vNSBias = new g2o::VertexNavStateBias();
    vNSBias->setEstimate(local_keyfrm->GetNavState());
    vNSBias->setId(idKF+1);
    vNSBias->setFixed(false);
    optimizer.addVertex(vNSBias);
}*/

namespace optimize {

local_bundle_adjuster::local_bundle_adjuster(const unsigned int num_first_iter,
                                             const unsigned int num_second_iter)
    : num_first_iter_(num_first_iter), num_second_iter_(num_second_iter) {}


void local_bundle_adjuster::optimize(openvslam::data::keyframe* curr_keyfrm, bool* const force_stop_flag, data::map_database* map_db, mapping_module* mapper_) const {
    // 1. local/fixed keyframes, local landmarksを集計する

    // correct local keyframes of the current keyframe
    std::unordered_map<unsigned int, data::keyframe*> local_keyfrms;

    local_keyfrms[curr_keyfrm->id_] = curr_keyfrm;
    const auto curr_covisibilities = curr_keyfrm->graph_node_->get_covisibilities();
    for (auto local_keyfrm : curr_covisibilities) {
        if (!local_keyfrm) {
            continue;
        }
        if (local_keyfrm->will_be_erased()) {
            continue;
        }

        local_keyfrms[local_keyfrm->id_] = local_keyfrm;
    }

    // correct local landmarks seen in local keyframes
    std::unordered_map<unsigned int, data::landmark*> local_lms;

    for (auto local_keyfrm : local_keyfrms) {
        const auto landmarks = local_keyfrm.second->get_landmarks();
        for (auto local_lm : landmarks) {
            if (!local_lm) {
                continue;
            }
            if (local_lm->will_be_erased()) {
                continue;
            }

            // 重複を避ける
            if (local_lms.count(local_lm->id_)) {
                continue;
            }

            local_lms[local_lm->id_] = local_lm;
        }
    }

    // fixed keyframes: keyframes which observe local landmarks but which are NOT in local keyframes
    std::unordered_map<unsigned int, data::keyframe*> fixed_keyfrms;

    for (auto local_lm : local_lms) {
        const auto observations = local_lm.second->get_observations();
        for (auto& obs : observations) {
            auto fixed_keyfrm = obs.first;
            if (!fixed_keyfrm) {
                continue;
            }
            if (fixed_keyfrm->will_be_erased()) {
                continue;
            }

            // local keyframesに属しているときは追加しない
            if (local_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            // 重複を避ける
            if (fixed_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            fixed_keyfrms[fixed_keyfrm->id_] = fixed_keyfrm;
        }
    }

    // 2. optimizerを構築

    auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = ::g2o::make_unique<::g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    ::g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    // 3. keyframeをg2oのvertexに変換してoptimizerにセットする

    // shot vertexのcontainer
    g2o::se3::shot_vertex_container keyfrm_vtx_container(0, local_keyfrms.size() + fixed_keyfrms.size());
    // vertexに変換されたkeyframesを保存しておく
    std::unordered_map<unsigned int, data::keyframe*> all_keyfrms;

    // local keyframesをoptimizerにセット
    for (auto& id_local_keyfrm_pair : local_keyfrms) {
        auto local_keyfrm = id_local_keyfrm_pair.second;

        all_keyfrms.emplace(id_local_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(local_keyfrm, local_keyfrm->id_ == 0);
        optimizer.addVertex(keyfrm_vtx);
    }

    // fixed keyframesをoptimizerにセット
    for (auto& id_fixed_keyfrm_pair : fixed_keyfrms) {
        auto fixed_keyfrm = id_fixed_keyfrm_pair.second;

        all_keyfrms.emplace(id_fixed_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(fixed_keyfrm, true);
        optimizer.addVertex(keyfrm_vtx);
    }

    // 4. keyframeとlandmarkのvertexをreprojection edgeで接続する

    // landmark vertexのcontainer
    g2o::landmark_vertex_container lm_vtx_container(keyfrm_vtx_container.get_max_vertex_id() + 1, local_lms.size());

    // reprojection edgeのcontainer
    using reproj_edge_wrapper = g2o::se3::reproj_edge_wrapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> reproj_edge_wraps;
    reproj_edge_wraps.reserve(all_keyfrms.size() * local_lms.size());

    // 有意水準5%のカイ2乗値
    // 自由度n=2
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // 自由度n=3
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    for (auto& id_local_lm_pair : local_lms) {
        auto local_lm = id_local_lm_pair.second;

        // landmarkをg2oのvertexに変換してoptimizerにセットする
        auto lm_vtx = lm_vtx_container.create_vertex(local_lm, false);
        optimizer.addVertex(lm_vtx);

        const auto observations = local_lm->get_observations();
        for (const auto& obs : observations) {
            auto keyfrm = obs.first;
            auto idx = obs.second;
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }

            const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
            const auto& undist_keypt = keyfrm->undist_keypts_.at(idx);
            const float x_right = keyfrm->stereo_x_right_.at(idx);
            const float inv_sigma_sq = keyfrm->inv_level_sigma_sq_.at(undist_keypt.octave);
            const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                         ? sqrt_chi_sq_2D
                                         : sqrt_chi_sq_3D;
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, local_lm, lm_vtx,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
            reproj_edge_wraps.push_back(reproj_edge_wrap);
            optimizer.addEdge(reproj_edge_wrap.edge_);
        }
    }

    // 5. 1回目の最適化を実行

    if (force_stop_flag) {
        if (*force_stop_flag) {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(num_first_iter_);

    // 6. アウトライア除去をして2回目の最適化を実行

    bool run_robust_BA = true;

    if (force_stop_flag) {
        if (*force_stop_flag) {
            run_robust_BA = false;
        }
    }

    if (run_robust_BA) {
        for (auto& reproj_edge_wrap : reproj_edge_wraps) {
            auto edge = reproj_edge_wrap.edge_;

            auto local_lm = reproj_edge_wrap.lm_;
            if (local_lm->will_be_erased()) {
                continue;
            }

            if (reproj_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }

            edge->setRobustKernel(nullptr);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(num_second_iter_);
    }

    // 7. アウトライアを集計する

    std::vector<std::pair<data::keyframe*, data::landmark*>> outlier_observations;
    outlier_observations.reserve(reproj_edge_wraps.size());

    for (auto& reproj_edge_wrap : reproj_edge_wraps) {
        auto edge = reproj_edge_wrap.edge_;

        auto local_lm = reproj_edge_wrap.lm_;
        if (local_lm->will_be_erased()) {
            continue;
        }

        if (reproj_edge_wrap.is_monocular_) {
            if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
        else {
            if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
    }

    // 8. 情報を更新

    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        if (!outlier_observations.empty()) {
            for (auto& outlier_obs : outlier_observations) {
                auto keyfrm = outlier_obs.first;
                auto lm = outlier_obs.second;
                keyfrm->erase_landmark(lm);
                lm->erase_observation(keyfrm);
            }
        }

        for (auto id_local_keyfrm_pair : local_keyfrms) {
            auto local_keyfrm = id_local_keyfrm_pair.second;

            auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(local_keyfrm);
            local_keyfrm->set_cam_pose(keyfrm_vtx->estimate());
        }

        for (auto id_local_lm_pair : local_lms) {
            auto local_lm = id_local_lm_pair.second;

            auto lm_vtx = lm_vtx_container.get_vertex(local_lm);
            local_lm->set_pos_in_world(lm_vtx->estimate());
            local_lm->update_normal_and_depth();
        }
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    if(mapper_)
    {
        mapper_->SetMapUpdateFlagInTracking(true);
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
}


void local_bundle_adjuster::LocalBundleAdjustmentNavState(data::keyframe *pCurKF, std::list<data::keyframe*> &local_keyfrms, bool* const force_stop_flag, data::map_database* map_db, cv::Mat& gw, mapping_module* mapper_) const
{
    #ifndef NOT_UPDATE_GYRO_BIAS
    static bool dbgLBAfopen=false;
    static ofstream dbgLBAPVRErr,dbgLBABiasErr,dbgLBAPVRErr2,dbgLBABiasErr2;
    if(!dbgLBAfopen)
    {
        dbgLBAPVRErr.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbgLBAPVRErr.txt");
        dbgLBABiasErr.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbgLBABiasErr.txt");
        dbgLBAPVRErr2.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbgLBAPVRErr2.txt");
        dbgLBABiasErr2.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbgLBABiasErr2.txt");
        if(dbgLBAPVRErr.is_open() && dbgLBABiasErr.is_open() && dbgLBAPVRErr2.is_open() && dbgLBABiasErr2.is_open())
        {
            std::cerr<<"file opened."<<std::endl;
            dbgLBAfopen = true;
        }
        else
        {
            std::cerr<<"file open error in dbgLBAfopen"<<std::endl;
            dbgLBAfopen = false;
        }
        dbgLBAPVRErr<<std::fixed<<std::setprecision(6);
        dbgLBABiasErr<<std::fixed<<std::setprecision(10);
        dbgLBAPVRErr2<<std::fixed<<std::setprecision(6);
        dbgLBABiasErr2<<std::fixed<<std::setprecision(10);
    }
    #endif

    // Check current KeyFrame in local window
    if(pCurKF != local_keyfrms.back())
        std::cerr<<"pCurKF != local_keyfrms.back. check"<<std::endl;

    // Extrinsics
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    // Gravity vector in world frame
    Vector3d GravityVec = util::converter::toVector3d(gw);

    // 1. local/fixed keyframes, local landmarksを集計する

    // correct local keyframes of the current keyframe
    //std::unordered_map<unsigned int, data::keyframe*> local_keyfrms;

    //local_keyfrms[curr_keyfrm->id_] = curr_keyfrm;
    //const auto curr_covisibilities = curr_keyfrm->graph_node_->get_covisibilities();
    for(std::list<data::keyframe*>::const_iterator lit=local_keyfrms.begin(), lend=local_keyfrms.end(); lit!=lend; lit++)
    {
        data::keyframe* pKFi = *lit;
        pKFi->mnBALocalForKF = pCurKF->id_;
    }

    // correct local landmarks seen in local keyframes
    std::list<data::landmark*> local_lms;
    //std::unordered_map<unsigned int, data::landmark*> local_lms;
    for(std::list<data::keyframe*>::const_iterator lit=local_keyfrms.begin() , lend=local_keyfrms.end(); lit!=lend; lit++)
    {
        std::vector<data::landmark*> landmarks = (*lit)->get_landmarks();
        for(std::vector<data::landmark*>::iterator vit=landmarks.begin(), vend=landmarks.end(); vit!=vend; vit++)
        {
            data::landmark* pMP = *vit;
            if(pMP)
                if(!pMP->will_be_erased())
                    if(pMP->mnBALocalForKF!=pCurKF->id_)
                    {
                        local_lms.push_back(pMP);
                        pMP->mnBALocalForKF=pCurKF->id_;
                    }
        }
    }

    // fixed keyframes: keyframes which observe local landmarks but which are NOT in local keyframes
    std::list<data::keyframe*> fixed_keyfrms;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    data::keyframe* pKFPrevLocal = local_keyfrms.front()->GetPrevKeyFrame();
    //data::keyframe* pKFPrevLocal;// = lLocalKeyFrames.front()->GetPrevKeyFrame();
    if(pKFPrevLocal)
    {
        // Test log
        if(pKFPrevLocal->will_be_erased()) std::cerr<<"KeyFrame before local window is bad?"<<std::endl;
        if(pKFPrevLocal->mnBAFixedForKF == pCurKF->id_) std::cerr<<"KeyFrame before local, has been added to lFixedKF?"<<std::endl;
        if(pKFPrevLocal->mnBALocalForKF == pCurKF->id_) std::cerr<<"KeyFrame before local, has been added to lLocalKF?"<<std::endl;

        pKFPrevLocal->mnBAFixedForKF = pCurKF->id_;
        if(!pKFPrevLocal->will_be_erased())
        {
            fixed_keyfrms.push_back(pKFPrevLocal);
        }
        else
            std::cerr<<"pKFPrevLocal is Bad?"<<std::endl;
    }
    // Test log
    else {std::cerr<<"pKFPrevLocal is NULL?"<<std::endl;}
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    for(std::list<data::landmark*>::iterator lit=local_lms.begin(), lend=local_lms.end(); lit!=lend; lit++)
    {
        std::map<data::keyframe*,unsigned int> observations = (*lit)->get_observations();
        for(std::map<data::keyframe*,unsigned int>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            data::keyframe* fixed_keyfrm = mit->first;
            if(fixed_keyfrm->mnBALocalForKF!=pCurKF->id_ && fixed_keyfrm->mnBAFixedForKF!=pCurKF->id_)
            {
                fixed_keyfrm->mnBAFixedForKF=pCurKF->id_;
                if(!fixed_keyfrm->will_be_erased())
                    fixed_keyfrms.push_back(fixed_keyfrm);
            }
        }
    }

    // 2. optimizerを構築

    auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolverX::PoseMatrixType>>();
    auto block_solver = ::g2o::make_unique<::g2o::BlockSolverX>(std::move(linear_solver));
    auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    ::g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    unsigned long maxKFid = 0;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // 3. keyframeをg2oのvertexに変換してoptimizerにセットする

    // shot vertexのcontainer
    g2o::se3::shot_vertex_container keyfrm_vtx_container(0, local_keyfrms.size() + fixed_keyfrms.size());
    // vertexに変換されたkeyframesを保存しておく
    std::unordered_map<unsigned int, data::keyframe*> all_keyfrms;

    // local keyframesをoptimizerにセット
    for(std::list<data::keyframe*>::const_iterator lit=local_keyfrms.begin(), lend=local_keyfrms.end(); lit!=lend; lit++)
    {
        data::keyframe* local_keyfrm = *lit;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        int idKF = local_keyfrm->id_*2;
        /*VertexNavStatePVR(local_keyfrm, idKF, optimizer);
        VertexNavStateBias(local_keyfrm, idKF, optimizer);*/
        // Vertex of PVR
        {
            ::g2o::VertexNavStatePVR * vNSPVR = new ::g2o::VertexNavStatePVR();
            vNSPVR->setEstimate(local_keyfrm->GetNavState());
            vNSPVR->setId(idKF);
            vNSPVR->setFixed(false);
            optimizer.addVertex(vNSPVR);
        }
        // Vertex of Bias
        {
            ::g2o::VertexNavStateBias * vNSBias = new ::g2o::VertexNavStateBias();
            vNSBias->setEstimate(local_keyfrm->GetNavState());
            vNSBias->setId(idKF+1);
            vNSBias->setFixed(false);
            optimizer.addVertex(vNSBias);
        }
        if(idKF + 1 > maxKFid)
            maxKFid = idKF + 1;
        // Test log
        if(local_keyfrm->id_ == 0) std::cerr<<"local_keyfrm->id_ == 0, shouldn't in LocalBA of NavState"<<std::endl;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------

        /*all_keyfrms.emplace(id_local_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(local_keyfrm, local_keyfrm->id_ == 0);
        optimizer.addVertex(keyfrm_vtx);*/
    }

    // fixed keyframesをoptimizerにセット
    for(std::list<data::keyframe*>::iterator lit=fixed_keyfrms.begin(), lend=fixed_keyfrms.end(); lit!=lend; lit++)
    {
        data::keyframe* fixed_keyfrm = *lit;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        int idKF = fixed_keyfrm->id_*2;
        // For common fixed KeyFrames, only add PVR vertex
        {
            ::g2o::VertexNavStatePVR * vNSPVR = new ::g2o::VertexNavStatePVR();
            vNSPVR->setEstimate(fixed_keyfrm->GetNavState());
            vNSPVR->setId(idKF);
            vNSPVR->setFixed(true);
            optimizer.addVertex(vNSPVR);
        }
        // For Local-Window-Previous KeyFrame, add Bias vertex
        if(fixed_keyfrm == pKFPrevLocal)
        {
            ::g2o::VertexNavStateBias * vNSBias = new ::g2o::VertexNavStateBias();
            vNSBias->setEstimate(fixed_keyfrm->GetNavState());
            vNSBias->setId(idKF+1);
            vNSBias->setFixed(true);
            optimizer.addVertex(vNSBias);
        }

        if(idKF + 1 > maxKFid)
            maxKFid = idKF + 1;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------

        /*all_keyfrms.emplace(id_fixed_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(fixed_keyfrm, true);
        optimizer.addVertex(keyfrm_vtx);*/
    }

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // Edges between KeyFrames in Local Window
    // and
    // Edges between 1st KeyFrame of Local Window and its previous (fixed)KeyFrame - pKFPrevLocal
    std::vector<::g2o::EdgeNavStatePVR*> vpEdgesNavStatePVR;
    std::vector<::g2o::EdgeNavStateBias*> vpEdgesNavStateBias;
    // Use chi2inv() in MATLAB to compute the value corresponding to 0.95/0.99 prob. w.r.t 15DOF: 24.9958/30.5779
    // 12.592/16.812 for 0.95/0.99 6DoF
    // 16.919/21.666 for 0.95/0.99 9DoF
    //const float thHuberNavState = sqrt(30.5779);
    const float thHuberNavStatePVR = sqrt(21.666);
    const float thHuberNavStateBias = sqrt(16.812);
    // Inverse covariance of bias random walk
#ifndef NOT_UPDATE_GYRO_BIAS
    Matrix<double,6,6> InvCovBgaRW = Matrix<double,6,6>::Identity();
    InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
    InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE
#else
    Matrix<double,3,3> InvCovBgaRW = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE
#endif

    for(std::list<data::keyframe*>::const_iterator lit=local_keyfrms.begin(), lend=local_keyfrms.end(); lit!=lend; lit++)
    
    for (auto& local_keyfrm : local_keyfrms) 
    {
        data::keyframe* pKF1 = *lit; // Current KF, store the IMU pre-integration between previous-current
        data::keyframe* pKF0 = pKF1->GetPrevKeyFrame();   // Previous KF

        // PVR edge
        {
            ::g2o::EdgeNavStatePVR * epvr = new ::g2o::EdgeNavStatePVR();
            epvr->setVertex(0, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->id_)));
            epvr->setVertex(1, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->id_)));
            epvr->setVertex(2, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->id_+1)));
            epvr->setMeasurement(pKF1->GetIMUPreInt());

            Matrix9d InvCovPVR = pKF1->GetIMUPreInt().getCovPVPhi().inverse();
            epvr->setInformation(InvCovPVR);
            epvr->SetParams(GravityVec);

            ::g2o::RobustKernelHuber* rk = new ::g2o::RobustKernelHuber;
            epvr->setRobustKernel(rk);
            rk->setDelta(thHuberNavStatePVR);

            optimizer.addEdge(epvr);
            vpEdgesNavStatePVR.push_back(epvr);
        }
        // Bias edge
        {
            ::g2o::EdgeNavStateBias * ebias = new ::g2o::EdgeNavStateBias();
            ebias->setVertex(0, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->id_+1)));
            ebias->setVertex(1, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->id_+1)));
            ebias->setMeasurement(pKF1->GetIMUPreInt());

            ebias->setInformation(InvCovBgaRW/pKF1->GetIMUPreInt().getDeltaTime());

            ::g2o::RobustKernelHuber* rk = new ::g2o::RobustKernelHuber;
            ebias->setRobustKernel(rk);
            rk->setDelta(thHuberNavStateBias);

            optimizer.addEdge(ebias);
            vpEdgesNavStateBias.push_back(ebias);
        }

        // Test log
        if(pKF1->GetIMUPreInt().getDeltaTime() < 1e-3)
        {
            std::cerr<<"IMU pre-integrator delta time between 2 KFs too small: "<<pKF1->GetIMUPreInt().getDeltaTime()<<std::endl;
            std::cerr<<"No EdgeNavState added"<<std::endl;
            continue;
        }
        // Lo dejo comentado porque solo imprime por pantalla
        /*if(lit == lLocalKeyFrames.begin())
        {
            // First KF in Local Window, link (fixed) pKFPrevLocal
            if(pKF0 != pKFPrevLocal) cerr<<"pKF0 != pKFPrevLocal for 1st KF in Local Window, id: "<<pKF0->mnId<<","<<pKFPrevLocal->mnId<<endl;
        }
        else
        {
            // KFs in Local Window, link another local KF
        }*/
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // 4. keyframeとlandmarkのvertexをreprojection edgeで接続する

    // landmark vertexのcontainer
    g2o::landmark_vertex_container lm_vtx_container(keyfrm_vtx_container.get_max_vertex_id() + 1, local_lms.size());

    /*// reprojection edgeのcontainer
    using reproj_edge_wrapper = g2o::se3::reproj_edge_wrapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> reproj_edge_wraps;
    reproj_edge_wraps.reserve(all_keyfrms.size() * local_lms.size());*/

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // Set MapPoint vertices
    const int nExpectedSize = (local_keyfrms.size() + fixed_keyfrms.size())*local_lms.size();

    std::vector<::g2o::EdgeNavStatePVRPointXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    std::vector<data::keyframe*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    std::vector<data::landmark*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);
    const float thHuberMono = sqrt(5.991);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // 有意水準5%のカイ2乗値
    // 自由度n=2
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // 自由度n=3
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    for(std::list<data::landmark*>::iterator lit=local_lms.begin(), lend=local_lms.end(); lit!=lend; lit++)
    {
        //auto local_lm = id_local_lm_pair.second;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        data::landmark* pMP = *lit;
        ::g2o::VertexSBAPointXYZ* vPoint = new ::g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->get_pos_in_world());
        int id = pMP->id_+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        cv::Mat Pw;
        eigen2cv(pMP->get_pos_in_world(), Pw);
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        /*
        // landmarkをg2oのvertexに変換してoptimizerにセットする
        auto lm_vtx = lm_vtx_container.create_vertex(local_lm, false);
        optimizer.addVertex(lm_vtx);
        */
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------

        const std::map<data::keyframe*,unsigned int> observations = pMP->get_observations();
        for(std::map<data::keyframe*,unsigned int>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            //auto keyfrm = obs.first;
            data::keyframe* pKFi = mit->first;
            if (!pKFi->will_be_erased()) {
                const cv::KeyPoint &kpUn = pKFi->undist_keypts_[mit->second];
                

                // Monocular observation
                if(pKFi->stereo_x_right_[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ::g2o::EdgeNavStatePVRPointXYZ* e = new ::g2o::EdgeNavStatePVRPointXYZ();

                    e->setVertex(0, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKFi->id_)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->inv_level_sigma_sq_[kpUn.octave];

                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    ::g2o::RobustKernelHuber* rk = new ::g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    
                    cv::Mat cam_param = ConfigParam::GetCamMatrix();
                    float fx = cam_param.at<float>(0,0);
                    float fy = cam_param.at<float>(1,1);
                    float cx = cam_param.at<float>(0,2);
                    float cy = cam_param.at<float>(1,2);
                    e->SetParams(fx,fy,cx,cy,Rbc,Pbc);

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else
                {
                    // Test log
                   std:: cerr<<"Stereo not supported yet, why here?? check."<<std::endl;
                }
            }
            /*
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }

            const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
            const auto& undist_keypt = keyfrm->undist_keypts_.at(idx);
            const float x_right = keyfrm->stereo_x_right_.at(idx);
            const float inv_sigma_sq = keyfrm->inv_level_sigma_sq_.at(undist_keypt.octave);
            const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                         ? sqrt_chi_sq_2D
                                         : sqrt_chi_sq_3D;
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, local_lm, lm_vtx,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
            reproj_edge_wraps.push_back(reproj_edge_wrap);
            optimizer.addEdge(reproj_edge_wrap.edge_);*/
        }
    }


    // 5. 1回目の最適化を実行

    if (force_stop_flag) {
        if (*force_stop_flag) {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(num_first_iter_);

    // 6. アウトライア除去をして2回目の最適化を実行

    bool run_robust_BA = true;

    if (force_stop_flag) {
        if (*force_stop_flag) {
            run_robust_BA = false;
        }
    }

    if (run_robust_BA) {
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ::g2o::EdgeNavStatePVRPointXYZ* e = vpEdgesMono[i];
            data::landmark* pMP = vpMapPointEdgeMono[i];

            if(pMP->will_be_erased())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        // Check inlier observations
        int tmpcnt=0;
        for(size_t i=0, iend=vpEdgesNavStatePVR.size(); i<iend; i++)
        {
            ::g2o::EdgeNavStatePVR* e = vpEdgesNavStatePVR[i];
            if(e->chi2()>21.666)
            {
                //e->setLevel(1);
                //cout<<"1 PVRedge "<<i<<", chi2 "<<e->chi2()<<". ";
                tmpcnt++;
            }
            //e->setRobustKernel(0);

        #ifndef NOT_UPDATE_GYRO_BIAS
                e->computeError();
                Vector9d err=e->error();
                for(int n=0;n<9;n++)
                    dbgLBAPVRErr<<err[n]<<" ";
                dbgLBAPVRErr<<endl;
        #endif
        }

        if(tmpcnt>0)
            std::cout<<std::endl;

        // Optimize again without the outliers
        optimizer.initializeOptimization();
        optimizer.optimize(num_second_iter_);
        /*for (auto& reproj_edge_wrap : reproj_edge_wraps) {
            auto edge = reproj_edge_wrap.edge_;

            auto local_lm = reproj_edge_wrap.lm_;
            if (local_lm->will_be_erased()) {
                continue;
            }

            if (reproj_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }

            edge->setRobustKernel(nullptr);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(num_second_iter_);*/
    }

    // 7. アウトライアを集計する
    std::vector<std::pair<data::keyframe*, data::landmark*>> outlier_observations;
    //outlier_observations.reserve(reproj_edge_wraps.size());
    outlier_observations.reserve(vpEdgesMono.size());
    double PosePointchi2=0, PosePosechi2=0, BiasBiaschi2=0;

    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        ::g2o::EdgeNavStatePVRPointXYZ* e = vpEdgesMono[i];
        data::landmark* pMP = vpMapPointEdgeMono[i];

        if(pMP->will_be_erased())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            data::keyframe* pKFi = vpEdgeKFMono[i];
            outlier_observations.push_back(std::make_pair(pKFi,pMP));
        }

        PosePointchi2 += e->chi2();
    }

     // Debug log
    // Check inlier observations
    int tmpcnt2=0;
    for(size_t i=0, iend=vpEdgesNavStatePVR.size(); i<iend; i++)
    {
        ::g2o::EdgeNavStatePVR* e = vpEdgesNavStatePVR[i];
        if(e->chi2()>21.666)
        {
            //cout<<"2 PVRedge "<<i<<", chi2 "<<e->chi2()<<". ";
            tmpcnt2++;
        }

    #ifndef NOT_UPDATE_GYRO_BIAS
        e->computeError();
        Vector9d err=e->error();
        for(int n=0;n<9;n++)
            dbgLBAPVRErr2<<err[n]<<" ";
        dbgLBAPVRErr2<<endl;
    #endif
    }

        //cout<<endl<<"edge bias ns bad: ";
    for(size_t i=0, iend=vpEdgesNavStateBias.size(); i<iend; i++)
    {
        ::g2o::EdgeNavStateBias* e = vpEdgesNavStateBias[i];
        if(e->chi2()>16.812)
        {
            //cout<<"2 Bias edge "<<i<<", chi2 "<<e->chi2()<<". ";
            tmpcnt2++;
        }

        #ifndef NOT_UPDATE_GYRO_BIAS
            e->computeError();
            Vector6d err=e->error();
            for(int n=0;n<6;n++)
                dbgLBABiasErr2<<err[n]<<" ";
            dbgLBABiasErr2<<endl;
        #endif
    }
    if(tmpcnt2>0)
        std::cout<<std::endl;
    /*
    for (auto& reproj_edge_wrap : reproj_edge_wraps) {
        auto edge = reproj_edge_wrap.edge_;

        auto local_lm = reproj_edge_wrap.lm_;
        if (local_lm->will_be_erased()) {
            continue;
        }

        if (reproj_edge_wrap.is_monocular_) {
            if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
        else {
            if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
    }*/

    // 8. 情報を更新

    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        if (!outlier_observations.empty()) {
            for (auto& outlier_obs : outlier_observations) {
                auto keyfrm = outlier_obs.first;
                auto lm = outlier_obs.second;
                keyfrm->erase_landmark(lm);
                lm->erase_observation(keyfrm);
            }
        }
        for(std::list<data::keyframe*>::const_iterator lit=local_keyfrms.begin(), lend=local_keyfrms.end(); lit!=lend; lit++)
        {
            data::keyframe* pKFi = *lit;
            ::g2o::VertexNavStatePVR* vNSPVR = static_cast<::g2o::VertexNavStatePVR*>(optimizer.vertex(2*pKFi->id_));
            ::g2o::VertexNavStateBias* vNSBias = static_cast<::g2o::VertexNavStateBias*>(optimizer.vertex(2*pKFi->id_+1));
            // In optimized navstate, bias not changed, delta_bias not zero, should be added to bias
            const NavState& optPVRns = vNSPVR->estimate();
            const NavState& optBiasns = vNSBias->estimate();
            NavState primaryns = pKFi->GetNavState();
            // Update NavState
            pKFi->SetNavStatePos(optPVRns.Get_P());
            pKFi->SetNavStateVel(optPVRns.Get_V());
            pKFi->SetNavStateRot(optPVRns.Get_R());
            //if(optBiasns.Get_dBias_Acc().norm()<1e-2 && optBiasns.Get_BiasGyr().norm()<1e-4)
            //{
            pKFi->SetNavStateDeltaBg(optBiasns.Get_dBias_Gyr());
            pKFi->SetNavStateDeltaBa(optBiasns.Get_dBias_Acc());

            // Update pose Tcw
            pKFi->UpdatePoseFromNS(ConfigParam::GetMatTbc());

            // Test log
            if( (primaryns.Get_BiasGyr() - optPVRns.Get_BiasGyr()).norm() > 1e-6 || (primaryns.Get_BiasGyr() - optBiasns.Get_BiasGyr()).norm() > 1e-6 )
                std::cerr<<"gyr bias change in optimization?"<<std::endl;
            if( (primaryns.Get_BiasAcc() - optPVRns.Get_BiasAcc()).norm() > 1e-6 || (primaryns.Get_BiasAcc() - optBiasns.Get_BiasAcc()).norm() > 1e-6 )
                std::cerr<<"acc bias change in optimization?"<<std::endl;
        }
        for(std::list<data::landmark*>::const_iterator lit=local_lms.begin(), lend=local_lms.end(); lit!=lend; lit++)
        {
            data::landmark* pMP = *lit;
            ::g2o::VertexSBAPointXYZ* vPoint = static_cast<::g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->id_+maxKFid+1));
            pMP->set_pos_in_world(vPoint->estimate());
            pMP->update_normal_and_depth();
        }

        /*
        for (auto id_local_keyfrm_pair : local_keyfrms) {
            auto local_keyfrm = id_local_keyfrm_pair.second;

            auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(local_keyfrm);
            local_keyfrm->set_cam_pose(keyfrm_vtx->estimate());
        }

        for (auto id_local_lm_pair : local_lms) {
            auto local_lm = id_local_lm_pair.second;

            auto lm_vtx = lm_vtx_container.get_vertex(local_lm);
            local_lm->set_pos_in_world(lm_vtx->estimate());
            local_lm->update_normal_and_depth();
        }*/
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    if(mapper_)
    {
        mapper_->SetMapUpdateFlagInTracking(true);
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
}










void local_bundle_adjuster::LocalBundleAdjustmentNavState(data::keyframe *pCurKF, std::unordered_map<unsigned int, data::keyframe*> &local_keyfrms, bool* const force_stop_flag, data::map_database* map_db, cv::Mat& gw, mapping_module* mapper_) const
{
    #ifndef NOT_UPDATE_GYRO_BIAS
    static bool dbgLBAfopen=false;
    static ofstream dbgLBAPVRErr,dbgLBABiasErr,dbgLBAPVRErr2,dbgLBABiasErr2;
    if(!dbgLBAfopen)
    {
        dbgLBAPVRErr.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbgLBAPVRErr.txt");
        dbgLBABiasErr.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbgLBABiasErr.txt");
        dbgLBAPVRErr2.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbgLBAPVRErr2.txt");
        dbgLBABiasErr2.open("/home/jp/opensourcecode/ORB_SLAM2/tmp/dbgLBABiasErr2.txt");
        if(dbgLBAPVRErr.is_open() && dbgLBABiasErr.is_open() && dbgLBAPVRErr2.is_open() && dbgLBABiasErr2.is_open())
        {
            std::cerr<<"file opened."<<std::endl;
            dbgLBAfopen = true;
        }
        else
        {
            std::cerr<<"file open error in dbgLBAfopen"<<std::endl;
            dbgLBAfopen = false;
        }
        dbgLBAPVRErr<<std::fixed<<std::setprecision(6);
        dbgLBABiasErr<<std::fixed<<std::setprecision(10);
        dbgLBAPVRErr2<<std::fixed<<std::setprecision(6);
        dbgLBABiasErr2<<std::fixed<<std::setprecision(10);
    }
    #endif

    // Check current KeyFrame in local window
    /*if(pCurKF != lLocalKeyFrames.back())
        std::cerr<<"pCurKF != lLocalKeyFrames.back. check"<<std::endl;*/

    // Extrinsics
    Matrix4d Tbc = ConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    // Gravity vector in world frame
    Vector3d GravityVec = util::converter::toVector3d(gw);

    // 1. local/fixed keyframes, local landmarksを集計する

    // correct local keyframes of the current keyframe
    //std::unordered_map<unsigned int, data::keyframe*> local_keyfrms;

    //local_keyfrms[curr_keyfrm->id_] = curr_keyfrm;
    //const auto curr_covisibilities = curr_keyfrm->graph_node_->get_covisibilities();
    /*for (auto local_keyfrm : local_keyfrms) {
        auto pKFi = local_keyfrm.second;
        pKFi-> = pCurKF->id_;
    }*/

    // correct local landmarks seen in local keyframes
    std::unordered_map<unsigned int, data::landmark*> local_lms;

    for (auto local_keyfrm : local_keyfrms) {
        const auto landmarks = local_keyfrm.second->get_landmarks();
        for (auto local_lm : landmarks) {
            if (!local_lm) {
                continue;
            }
            if (local_lm->will_be_erased()) {
                continue;
            }

            // 重複を避ける
            if (local_lms.count(local_lm->id_)) {
                continue;
            }

            local_lms[local_lm->id_] = local_lm;
        }
    }

    // fixed keyframes: keyframes which observe local landmarks but which are NOT in local keyframes
    std::unordered_map<unsigned int, data::keyframe*> fixed_keyfrms;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    data::keyframe* pKFPrevLocal = fixed_keyfrms.begin()->second->GetPrevKeyFrame();
    //data::keyframe* pKFPrevLocal;// = lLocalKeyFrames.front()->GetPrevKeyFrame();
    if(pKFPrevLocal)
    {
        // Test log
        if(pKFPrevLocal->will_be_erased()) std::cerr<<"KeyFrame before local window is bad?"<<std::endl;
        /*if(pKFPrevLocal->mnBAFixedForKF == pCurKF->id_) std::cerr<<"KeyFrame before local, has been added to lFixedKF?"<<std::endl;
        if(pKFPrevLocal->mnBALocalForKF == pCurKF->id_) std::cerr<<"KeyFrame before local, has been added to lLocalKF?"<<std::endl;*/

        //pKFPrevLocal->mnBAFixedForKF = pCurKF->id_;
        if(!pKFPrevLocal->will_be_erased())
        {
            fixed_keyfrms[pKFPrevLocal->id_] = pKFPrevLocal;
        }
        else
            std::cerr<<"pKFPrevLocal is Bad?"<<std::endl;
    }
    // Test log
    else {std::cerr<<"pKFPrevLocal is NULL?"<<std::endl;}
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    for (auto local_lm : local_lms) {
        const auto observations = local_lm.second->get_observations();
        for (auto& obs : observations) {
            auto fixed_keyfrm = obs.first;
            if (!fixed_keyfrm) {
                continue;
            }
            if (fixed_keyfrm->will_be_erased()) {
                continue;
            }

            // local keyframesに属しているときは追加しない
            if (local_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            // 重複を避ける
            if (fixed_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            fixed_keyfrms[fixed_keyfrm->id_] = fixed_keyfrm;
        }
    }

    // 2. optimizerを構築

    auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolverX::PoseMatrixType>>();
    auto block_solver = ::g2o::make_unique<::g2o::BlockSolverX>(std::move(linear_solver));
    auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    ::g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    unsigned long maxKFid = 0;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // 3. keyframeをg2oのvertexに変換してoptimizerにセットする

    // shot vertexのcontainer
    g2o::se3::shot_vertex_container keyfrm_vtx_container(0, local_keyfrms.size() + fixed_keyfrms.size());
    // vertexに変換されたkeyframesを保存しておく
    std::unordered_map<unsigned int, data::keyframe*> all_keyfrms;

    // local keyframesをoptimizerにセット
    for (auto& id_local_keyfrm_pair : local_keyfrms) {
        auto local_keyfrm = id_local_keyfrm_pair.second;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        int idKF = local_keyfrm->id_*2;
        /*VertexNavStatePVR(local_keyfrm, idKF, optimizer);
        VertexNavStateBias(local_keyfrm, idKF, optimizer);*/
        // Vertex of PVR
        {
            ::g2o::VertexNavStatePVR * vNSPVR = new ::g2o::VertexNavStatePVR();
            vNSPVR->setEstimate(local_keyfrm->GetNavState());
            vNSPVR->setId(idKF);
            vNSPVR->setFixed(false);
            optimizer.addVertex(vNSPVR);
        }
        // Vertex of Bias
        {
            ::g2o::VertexNavStateBias * vNSBias = new ::g2o::VertexNavStateBias();
            vNSBias->setEstimate(local_keyfrm->GetNavState());
            vNSBias->setId(idKF+1);
            vNSBias->setFixed(false);
            optimizer.addVertex(vNSBias);
        }
        if(idKF + 1 > maxKFid)
            maxKFid = idKF + 1;
        // Test log
        if(local_keyfrm->id_ == 0) std::cerr<<"local_keyfrm->id_ == 0, shouldn't in LocalBA of NavState"<<std::endl;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------

        /*all_keyfrms.emplace(id_local_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(local_keyfrm, local_keyfrm->id_ == 0);
        optimizer.addVertex(keyfrm_vtx);*/
    }

    // fixed keyframesをoptimizerにセット
    for (auto& id_fixed_keyfrm_pair : fixed_keyfrms) {
        auto fixed_keyfrm = id_fixed_keyfrm_pair.second;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        int idKF = fixed_keyfrm->id_*2;
        // For common fixed KeyFrames, only add PVR vertex
        {
            ::g2o::VertexNavStatePVR * vNSPVR = new ::g2o::VertexNavStatePVR();
            vNSPVR->setEstimate(fixed_keyfrm->GetNavState());
            vNSPVR->setId(idKF);
            vNSPVR->setFixed(true);
            optimizer.addVertex(vNSPVR);
        }
        // For Local-Window-Previous KeyFrame, add Bias vertex
        if(fixed_keyfrm == pKFPrevLocal)
        {
            ::g2o::VertexNavStateBias * vNSBias = new ::g2o::VertexNavStateBias();
            vNSBias->setEstimate(fixed_keyfrm->GetNavState());
            vNSBias->setId(idKF+1);
            vNSBias->setFixed(true);
            optimizer.addVertex(vNSBias);
        }

        if(idKF + 1 > maxKFid)
            maxKFid = idKF + 1;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------

        /*all_keyfrms.emplace(id_fixed_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(fixed_keyfrm, true);
        optimizer.addVertex(keyfrm_vtx);*/
    }

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // Edges between KeyFrames in Local Window
    // and
    // Edges between 1st KeyFrame of Local Window and its previous (fixed)KeyFrame - pKFPrevLocal
    std::vector<::g2o::EdgeNavStatePVR*> vpEdgesNavStatePVR;
    std::vector<::g2o::EdgeNavStateBias*> vpEdgesNavStateBias;
    // Use chi2inv() in MATLAB to compute the value corresponding to 0.95/0.99 prob. w.r.t 15DOF: 24.9958/30.5779
    // 12.592/16.812 for 0.95/0.99 6DoF
    // 16.919/21.666 for 0.95/0.99 9DoF
    //const float thHuberNavState = sqrt(30.5779);
    const float thHuberNavStatePVR = sqrt(21.666);
    const float thHuberNavStateBias = sqrt(16.812);
    // Inverse covariance of bias random walk
#ifndef NOT_UPDATE_GYRO_BIAS
    Matrix<double,6,6> InvCovBgaRW = Matrix<double,6,6>::Identity();
    InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
    InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE
#else
    Matrix<double,3,3> InvCovBgaRW = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE
#endif

    for (auto& local_keyfrm : local_keyfrms) 
    {
        auto pKF1 = local_keyfrm.second;                      // Current KF, store the IMU pre-integration between previous-current
        data::keyframe* pKF0 = pKF1->GetPrevKeyFrame();   // Previous KF

        // PVR edge
        {
            ::g2o::EdgeNavStatePVR * epvr = new ::g2o::EdgeNavStatePVR();
            epvr->setVertex(0, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->id_)));
            epvr->setVertex(1, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->id_)));
            epvr->setVertex(2, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->id_+1)));
            epvr->setMeasurement(pKF1->GetIMUPreInt());

            Matrix9d InvCovPVR = pKF1->GetIMUPreInt().getCovPVPhi().inverse();
            epvr->setInformation(InvCovPVR);
            epvr->SetParams(GravityVec);

            ::g2o::RobustKernelHuber* rk = new ::g2o::RobustKernelHuber;
            epvr->setRobustKernel(rk);
            rk->setDelta(thHuberNavStatePVR);

            optimizer.addEdge(epvr);
            vpEdgesNavStatePVR.push_back(epvr);
        }
        // Bias edge
        {
            ::g2o::EdgeNavStateBias * ebias = new ::g2o::EdgeNavStateBias();
            ebias->setVertex(0, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF0->id_+1)));
            ebias->setVertex(1, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKF1->id_+1)));
            ebias->setMeasurement(pKF1->GetIMUPreInt());

            ebias->setInformation(InvCovBgaRW/pKF1->GetIMUPreInt().getDeltaTime());

            ::g2o::RobustKernelHuber* rk = new ::g2o::RobustKernelHuber;
            ebias->setRobustKernel(rk);
            rk->setDelta(thHuberNavStateBias);

            optimizer.addEdge(ebias);
            vpEdgesNavStateBias.push_back(ebias);
        }

        // Test log
        if(pKF1->GetIMUPreInt().getDeltaTime() < 1e-3)
        {
            std::cerr<<"IMU pre-integrator delta time between 2 KFs too small: "<<pKF1->GetIMUPreInt().getDeltaTime()<<std::endl;
            std::cerr<<"No EdgeNavState added"<<std::endl;
            continue;
        }
        // Lo dejo comentado porque solo imprime por pantalla
        /*if(lit == lLocalKeyFrames.begin())
        {
            // First KF in Local Window, link (fixed) pKFPrevLocal
            if(pKF0 != pKFPrevLocal) cerr<<"pKF0 != pKFPrevLocal for 1st KF in Local Window, id: "<<pKF0->mnId<<","<<pKFPrevLocal->mnId<<endl;
        }
        else
        {
            // KFs in Local Window, link another local KF
        }*/
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // 4. keyframeとlandmarkのvertexをreprojection edgeで接続する

    // landmark vertexのcontainer
    g2o::landmark_vertex_container lm_vtx_container(keyfrm_vtx_container.get_max_vertex_id() + 1, local_lms.size());

    /*// reprojection edgeのcontainer
    using reproj_edge_wrapper = g2o::se3::reproj_edge_wrapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> reproj_edge_wraps;
    reproj_edge_wraps.reserve(all_keyfrms.size() * local_lms.size());*/

    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // Set MapPoint vertices
    const int nExpectedSize = (local_keyfrms.size() + fixed_keyfrms.size())*local_lms.size();

    std::vector<::g2o::EdgeNavStatePVRPointXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    std::vector<data::keyframe*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    std::vector<data::landmark*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);
    const float thHuberMono = sqrt(5.991);
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

    // 有意水準5%のカイ2乗値
    // 自由度n=2
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // 自由度n=3
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    for (auto& id_local_lm_pair : local_lms) {
        //auto local_lm = id_local_lm_pair.second;
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        auto pMP = id_local_lm_pair.second;
        ::g2o::VertexSBAPointXYZ* vPoint = new ::g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->get_pos_in_world());
        int id = pMP->id_+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        cv::Mat Pw;
        eigen2cv(pMP->get_pos_in_world(), Pw);
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        /*
        // landmarkをg2oのvertexに変換してoptimizerにセットする
        auto lm_vtx = lm_vtx_container.create_vertex(local_lm, false);
        optimizer.addVertex(lm_vtx);
        */
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------
        //-------------------------------------------------------------------------------------------

        const auto observations = pMP->get_observations();
        for (const auto& obs : observations) {
            //auto keyfrm = obs.first;
            data::keyframe* pKFi = obs.first;
            auto idx = obs.second;
            if (!pKFi->will_be_erased()) {
                const cv::KeyPoint &kpUn = pKFi->undist_keypts_[idx];
                

                // Monocular observation
                if(pKFi->stereo_x_right_[idx]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    ::g2o::EdgeNavStatePVRPointXYZ* e = new ::g2o::EdgeNavStatePVRPointXYZ();

                    e->setVertex(0, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<::g2o::OptimizableGraph::Vertex*>(optimizer.vertex(2*pKFi->id_)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->inv_level_sigma_sq_[kpUn.octave];

                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    ::g2o::RobustKernelHuber* rk = new ::g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);
                    
                    cv::Mat cam_param = ConfigParam::GetCamMatrix();
                    float fx = cam_param.at<float>(0,0);
                    float fy = cam_param.at<float>(1,1);
                    float cx = cam_param.at<float>(0,2);
                    float cy = cam_param.at<float>(1,2);
                    e->SetParams(fx,fy,cx,cy,Rbc,Pbc);

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else
                {
                    // Test log
                   std:: cerr<<"Stereo not supported yet, why here?? check."<<std::endl;
                }
            }
            /*
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }

            const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
            const auto& undist_keypt = keyfrm->undist_keypts_.at(idx);
            const float x_right = keyfrm->stereo_x_right_.at(idx);
            const float inv_sigma_sq = keyfrm->inv_level_sigma_sq_.at(undist_keypt.octave);
            const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                         ? sqrt_chi_sq_2D
                                         : sqrt_chi_sq_3D;
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, local_lm, lm_vtx,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
            reproj_edge_wraps.push_back(reproj_edge_wrap);
            optimizer.addEdge(reproj_edge_wrap.edge_);*/
        }
    }


    // 5. 1回目の最適化を実行

    if (force_stop_flag) {
        if (*force_stop_flag) {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(num_first_iter_);

    // 6. アウトライア除去をして2回目の最適化を実行

    bool run_robust_BA = true;

    if (force_stop_flag) {
        if (*force_stop_flag) {
            run_robust_BA = false;
        }
    }

    if (run_robust_BA) {
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            ::g2o::EdgeNavStatePVRPointXYZ* e = vpEdgesMono[i];
            data::landmark* pMP = vpMapPointEdgeMono[i];

            if(pMP->will_be_erased())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        // Check inlier observations
        int tmpcnt=0;
        for(size_t i=0, iend=vpEdgesNavStatePVR.size(); i<iend; i++)
        {
            ::g2o::EdgeNavStatePVR* e = vpEdgesNavStatePVR[i];
            if(e->chi2()>21.666)
            {
                //e->setLevel(1);
                //cout<<"1 PVRedge "<<i<<", chi2 "<<e->chi2()<<". ";
                tmpcnt++;
            }
            //e->setRobustKernel(0);

        #ifndef NOT_UPDATE_GYRO_BIAS
                e->computeError();
                Vector9d err=e->error();
                for(int n=0;n<9;n++)
                    dbgLBAPVRErr<<err[n]<<" ";
                dbgLBAPVRErr<<endl;
        #endif
        }

        if(tmpcnt>0)
            std::cout<<std::endl;

        // Optimize again without the outliers
        optimizer.initializeOptimization();
        optimizer.optimize(num_second_iter_);
        /*for (auto& reproj_edge_wrap : reproj_edge_wraps) {
            auto edge = reproj_edge_wrap.edge_;

            auto local_lm = reproj_edge_wrap.lm_;
            if (local_lm->will_be_erased()) {
                continue;
            }

            if (reproj_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }

            edge->setRobustKernel(nullptr);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(num_second_iter_);*/
    }

    // 7. アウトライアを集計する
    std::vector<std::pair<data::keyframe*, data::landmark*>> outlier_observations;
    //outlier_observations.reserve(reproj_edge_wraps.size());
    outlier_observations.reserve(vpEdgesMono.size());
    double PosePointchi2=0, PosePosechi2=0, BiasBiaschi2=0;

    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        ::g2o::EdgeNavStatePVRPointXYZ* e = vpEdgesMono[i];
        data::landmark* pMP = vpMapPointEdgeMono[i];

        if(pMP->will_be_erased())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            data::keyframe* pKFi = vpEdgeKFMono[i];
            outlier_observations.push_back(std::make_pair(pKFi,pMP));
        }

        PosePointchi2 += e->chi2();
    }

     // Debug log
    // Check inlier observations
    int tmpcnt2=0;
    for(size_t i=0, iend=vpEdgesNavStatePVR.size(); i<iend; i++)
    {
        ::g2o::EdgeNavStatePVR* e = vpEdgesNavStatePVR[i];
        if(e->chi2()>21.666)
        {
            //cout<<"2 PVRedge "<<i<<", chi2 "<<e->chi2()<<". ";
            tmpcnt2++;
        }

    #ifndef NOT_UPDATE_GYRO_BIAS
        e->computeError();
        Vector9d err=e->error();
        for(int n=0;n<9;n++)
            dbgLBAPVRErr2<<err[n]<<" ";
        dbgLBAPVRErr2<<endl;
    #endif
    }

        //cout<<endl<<"edge bias ns bad: ";
    for(size_t i=0, iend=vpEdgesNavStateBias.size(); i<iend; i++)
    {
        ::g2o::EdgeNavStateBias* e = vpEdgesNavStateBias[i];
        if(e->chi2()>16.812)
        {
            //cout<<"2 Bias edge "<<i<<", chi2 "<<e->chi2()<<". ";
            tmpcnt2++;
        }

        #ifndef NOT_UPDATE_GYRO_BIAS
            e->computeError();
            Vector6d err=e->error();
            for(int n=0;n<6;n++)
                dbgLBABiasErr2<<err[n]<<" ";
            dbgLBABiasErr2<<endl;
        #endif
    }
    if(tmpcnt2>0)
        std::cout<<std::endl;
    /*
    for (auto& reproj_edge_wrap : reproj_edge_wraps) {
        auto edge = reproj_edge_wrap.edge_;

        auto local_lm = reproj_edge_wrap.lm_;
        if (local_lm->will_be_erased()) {
            continue;
        }

        if (reproj_edge_wrap.is_monocular_) {
            if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
        else {
            if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
    }*/

    // 8. 情報を更新

    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        if (!outlier_observations.empty()) {
            for (auto& outlier_obs : outlier_observations) {
                auto keyfrm = outlier_obs.first;
                auto lm = outlier_obs.second;
                keyfrm->erase_landmark(lm);
                lm->erase_observation(keyfrm);
            }
        }

        for (auto id_local_keyfrm_pair : local_keyfrms) {
            auto pKFi = id_local_keyfrm_pair.second;
            ::g2o::VertexNavStatePVR* vNSPVR = static_cast<::g2o::VertexNavStatePVR*>(optimizer.vertex(2*pKFi->id_));
            ::g2o::VertexNavStateBias* vNSBias = static_cast<::g2o::VertexNavStateBias*>(optimizer.vertex(2*pKFi->id_+1));
            // In optimized navstate, bias not changed, delta_bias not zero, should be added to bias
            const NavState& optPVRns = vNSPVR->estimate();
            const NavState& optBiasns = vNSBias->estimate();
            NavState primaryns = pKFi->GetNavState();
            // Update NavState
            pKFi->SetNavStatePos(optPVRns.Get_P());
            pKFi->SetNavStateVel(optPVRns.Get_V());
            pKFi->SetNavStateRot(optPVRns.Get_R());
            //if(optBiasns.Get_dBias_Acc().norm()<1e-2 && optBiasns.Get_BiasGyr().norm()<1e-4)
            //{
            pKFi->SetNavStateDeltaBg(optBiasns.Get_dBias_Gyr());
            pKFi->SetNavStateDeltaBa(optBiasns.Get_dBias_Acc());

            // Update pose Tcw
            pKFi->UpdatePoseFromNS(ConfigParam::GetMatTbc());

            // Test log
            if( (primaryns.Get_BiasGyr() - optPVRns.Get_BiasGyr()).norm() > 1e-6 || (primaryns.Get_BiasGyr() - optBiasns.Get_BiasGyr()).norm() > 1e-6 )
                std::cerr<<"gyr bias change in optimization?"<<std::endl;
            if( (primaryns.Get_BiasAcc() - optPVRns.Get_BiasAcc()).norm() > 1e-6 || (primaryns.Get_BiasAcc() - optBiasns.Get_BiasAcc()).norm() > 1e-6 )
                std::cerr<<"acc bias change in optimization?"<<std::endl;
        }
        for (auto id_local_lm_pair : local_lms) {
            auto pMP = id_local_lm_pair.second;
            ::g2o::VertexSBAPointXYZ* vPoint = static_cast<::g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->id_+maxKFid+1));
            pMP->set_pos_in_world(vPoint->estimate());
            pMP->update_normal_and_depth();
        }

        /*
        for (auto id_local_keyfrm_pair : local_keyfrms) {
            auto local_keyfrm = id_local_keyfrm_pair.second;

            auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(local_keyfrm);
            local_keyfrm->set_cam_pose(keyfrm_vtx->estimate());
        }

        for (auto id_local_lm_pair : local_lms) {
            auto local_lm = id_local_lm_pair.second;

            auto lm_vtx = lm_vtx_container.get_vertex(local_lm);
            local_lm->set_pos_in_world(lm_vtx->estimate());
            local_lm->update_normal_and_depth();
        }*/
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    if(mapper_)
    {
        mapper_->SetMapUpdateFlagInTracking(true);
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
}

/*
void local_bundle_adjuster::optimize(data::keyframe *curr_keyfrm, std::unordered_map<unsigned int, data::keyframe*> &local_keyfrms, bool* const force_stop_flag, data::map_database* map_db, mapping_module* mapper_) const
{
    // 1. local/fixed keyframes, local landmarksを集計する

    // correct local keyframes of the current keyframe
    //std::unordered_map<unsigned int, data::keyframe*> local_keyfrms;

    local_keyfrms[curr_keyfrm->id_] = curr_keyfrm;
    const auto curr_covisibilities = curr_keyfrm->graph_node_->get_covisibilities();
    for (auto local_keyfrm : curr_covisibilities) {
        if (!local_keyfrm) {
            continue;
        }
        if (local_keyfrm->will_be_erased()) {
            continue;
        }

        local_keyfrms[local_keyfrm->id_] = local_keyfrm;
    }

    // correct local landmarks seen in local keyframes
    std::unordered_map<unsigned int, data::landmark*> local_lms;

    for (auto local_keyfrm : local_keyfrms) {
        const auto landmarks = local_keyfrm.second->get_landmarks();
        for (auto local_lm : landmarks) {
            if (!local_lm) {
                continue;
            }
            if (local_lm->will_be_erased()) {
                continue;
            }

            // 重複を避ける
            if (local_lms.count(local_lm->id_)) {
                continue;
            }

            local_lms[local_lm->id_] = local_lm;
        }
    }

    // fixed keyframes: keyframes which observe local landmarks but which are NOT in local keyframes
    std::unordered_map<unsigned int, data::keyframe*> fixed_keyfrms;

    for (auto local_lm : local_lms) {
        const auto observations = local_lm.second->get_observations();
        for (auto& obs : observations) {
            auto fixed_keyfrm = obs.first;
            if (!fixed_keyfrm) {
                continue;
            }
            if (fixed_keyfrm->will_be_erased()) {
                continue;
            }

            // local keyframesに属しているときは追加しない
            if (local_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            // 重複を避ける
            if (fixed_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            fixed_keyfrms[fixed_keyfrm->id_] = fixed_keyfrm;
        }
    }

    // 2. optimizerを構築

    auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = ::g2o::make_unique<::g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    ::g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);

    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    // 3. keyframeをg2oのvertexに変換してoptimizerにセットする

    // shot vertexのcontainer
    g2o::se3::shot_vertex_container keyfrm_vtx_container(0, local_keyfrms.size() + fixed_keyfrms.size());
    // vertexに変換されたkeyframesを保存しておく
    std::unordered_map<unsigned int, data::keyframe*> all_keyfrms;

    // local keyframesをoptimizerにセット
    for (auto& id_local_keyfrm_pair : local_keyfrms) {
        auto local_keyfrm = id_local_keyfrm_pair.second;

        all_keyfrms.emplace(id_local_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(local_keyfrm, local_keyfrm->id_ == 0);
        optimizer.addVertex(keyfrm_vtx);
    }

    // fixed keyframesをoptimizerにセット
    for (auto& id_fixed_keyfrm_pair : fixed_keyfrms) {
        auto fixed_keyfrm = id_fixed_keyfrm_pair.second;

        all_keyfrms.emplace(id_fixed_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(fixed_keyfrm, true);
        optimizer.addVertex(keyfrm_vtx);
    }

    // 4. keyframeとlandmarkのvertexをreprojection edgeで接続する

    // landmark vertexのcontainer
    g2o::landmark_vertex_container lm_vtx_container(keyfrm_vtx_container.get_max_vertex_id() + 1, local_lms.size());

    // reprojection edgeのcontainer
    using reproj_edge_wrapper = g2o::se3::reproj_edge_wrapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> reproj_edge_wraps;
    reproj_edge_wraps.reserve(all_keyfrms.size() * local_lms.size());

    // 有意水準5%のカイ2乗値
    // 自由度n=2
    constexpr float chi_sq_2D = 5.99146;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    // 自由度n=3
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    for (auto& id_local_lm_pair : local_lms) {
        auto local_lm = id_local_lm_pair.second;

        // landmarkをg2oのvertexに変換してoptimizerにセットする
        auto lm_vtx = lm_vtx_container.create_vertex(local_lm, false);
        optimizer.addVertex(lm_vtx);

        const auto observations = local_lm->get_observations();
        for (const auto& obs : observations) {
            auto keyfrm = obs.first;
            auto idx = obs.second;
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }

            const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
            const auto& undist_keypt = keyfrm->undist_keypts_.at(idx);
            const float x_right = keyfrm->stereo_x_right_.at(idx);
            const float inv_sigma_sq = keyfrm->inv_level_sigma_sq_.at(undist_keypt.octave);
            const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                         ? sqrt_chi_sq_2D
                                         : sqrt_chi_sq_3D;
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, local_lm, lm_vtx,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
            reproj_edge_wraps.push_back(reproj_edge_wrap);
            optimizer.addEdge(reproj_edge_wrap.edge_);
        }
    }

    // 5. 1回目の最適化を実行

    if (force_stop_flag) {
        if (*force_stop_flag) {
            return;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(num_first_iter_);

    // 6. アウトライア除去をして2回目の最適化を実行

    bool run_robust_BA = true;

    if (force_stop_flag) {
        if (*force_stop_flag) {
            run_robust_BA = false;
        }
    }

    if (run_robust_BA) {
        for (auto& reproj_edge_wrap : reproj_edge_wraps) {
            auto edge = reproj_edge_wrap.edge_;

            auto local_lm = reproj_edge_wrap.lm_;
            if (local_lm->will_be_erased()) {
                continue;
            }

            if (reproj_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }

            edge->setRobustKernel(nullptr);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(num_second_iter_);
    }

    // 7. アウトライアを集計する

    std::vector<std::pair<data::keyframe*, data::landmark*>> outlier_observations;
    outlier_observations.reserve(reproj_edge_wraps.size());

    for (auto& reproj_edge_wrap : reproj_edge_wraps) {
        auto edge = reproj_edge_wrap.edge_;

        auto local_lm = reproj_edge_wrap.lm_;
        if (local_lm->will_be_erased()) {
            continue;
        }

        if (reproj_edge_wrap.is_monocular_) {
            if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
        else {
            if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
    }

    // 8. 情報を更新

    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        if (!outlier_observations.empty()) {
            for (auto& outlier_obs : outlier_observations) {
                auto keyfrm = outlier_obs.first;
                auto lm = outlier_obs.second;
                keyfrm->erase_landmark(lm);
                lm->erase_observation(keyfrm);
            }
        }

        for (auto id_local_keyfrm_pair : local_keyfrms) {
            auto local_keyfrm = id_local_keyfrm_pair.second;

            auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(local_keyfrm);
            local_keyfrm->set_cam_pose(keyfrm_vtx->estimate());
        }

        for (auto id_local_lm_pair : local_lms) {
            auto local_lm = id_local_lm_pair.second;

            auto lm_vtx = lm_vtx_container.get_vertex(local_lm);
            local_lm->set_pos_in_world(lm_vtx->estimate());
            local_lm->update_normal_and_depth();
        }
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    if(mapper_)
    {
        mapper_->SetMapUpdateFlagInTracking(true);
    }
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
}*/


} // namespace optimize
} // namespace openvslam
