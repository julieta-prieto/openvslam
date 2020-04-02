#ifndef OPENVSLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_H
#define OPENVSLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_H

#include "openvslam/mapping_module.h"

namespace openvslam {

class mapping_module;

namespace data {
class keyframe;
class map_database;
} // namespace data

namespace optimize {

class local_bundle_adjuster {
public:
    /**
     * Constructor
     * @param map_db
     * @param num_first_iter
     * @param num_second_iter
     */
    explicit local_bundle_adjuster(const unsigned int num_first_iter = 5,
                                   const unsigned int num_second_iter = 10);

    /**
     * Destructor
     */
    virtual ~local_bundle_adjuster() = default;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    // CONSTRUCTOR MODIFICADO
    /**
     * Perform optimization
     * @param curr_keyfrm
     * @param force_stop_flag
     */
    void optimize(data::keyframe* curr_keyfrm, bool* const force_stop_flag, data::map_database* map_db, mapping_module* mapper_=NULL) const;
    // Añadido
    void optimize(data::keyframe *curr_keyfrm, std::unordered_map<unsigned int, data::keyframe*> &local_keyfrms, bool* const force_stop_flag, data::map_database* map_db, mapping_module* mapper_=NULL) const;

    void LocalBundleAdjustmentNavState(data::keyframe *pKF, std::unordered_map<unsigned int, data::keyframe*> &local_keyfrms, bool* const force_stop_flag, data::map_database* map_db, cv::Mat& gw, mapping_module* mapper_=NULL) const;
    void LocalBundleAdjustmentNavState(data::keyframe *pKF, std::list<data::keyframe*> &local_keyfrms, bool* const force_stop_flag, data::map_database* map_db, cv::Mat& gw, mapping_module* mapper_=NULL) const;
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------------

private:
    //! number of iterations of first optimization
    const unsigned int num_first_iter_;
    //! number of iterations of second optimization
    const unsigned int num_second_iter_;
};

} // namespace optimize
} // namespace openvslam

#endif // OPENVSLAM_OPTIMIZE_LOCAL_BUNDLE_ADJUSTER_H
