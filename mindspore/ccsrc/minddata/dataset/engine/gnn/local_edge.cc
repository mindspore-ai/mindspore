/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "minddata/dataset/engine/gnn/local_edge.h"

#include <string>

namespace mindspore {
namespace dataset {
namespace gnn {

LocalEdge::LocalEdge(EdgeIdType id, EdgeType type, WeightType weight, NodeIdType src_id, NodeIdType dst_id)
    : Edge(id, type, weight, src_id, dst_id) {}

Status LocalEdge::GetFeatures(FeatureType feature_type, std::shared_ptr<Feature> *out_feature) {
  auto itr = std::find_if(
    features_.begin(), features_.end(),
    [feature_type](std::pair<FeatureType, std::shared_ptr<Feature>> item) { return item.first == feature_type; });
  if (itr != features_.end()) {
    *out_feature = itr->second;
    return Status::OK();
  } else {
    std::string err_msg = "Invalid feature type:" + std::to_string(feature_type);
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
}

Status LocalEdge::UpdateFeature(const std::shared_ptr<Feature> &feature) {
  auto itr = std::find_if(
    features_.begin(), features_.end(),
    [feature](std::pair<FeatureType, std::shared_ptr<Feature>> item) { return item.first == feature->type(); });
  if (itr != features_.end()) {
    RETURN_STATUS_UNEXPECTED("Feature already exists");
  } else {
    features_.emplace_back(std::make_pair(feature->type(), feature));
    return Status::OK();
  }
}

}  // namespace gnn
}  // namespace dataset
}  // namespace mindspore
