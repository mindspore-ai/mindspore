/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "fl/server/local_meta_store.h"

namespace mindspore {
namespace fl {
namespace server {
void LocalMetaStore::remove_value(const std::string &name) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (key_to_meta_.count(name) != 0) {
    (void)key_to_meta_.erase(key_to_meta_.find(name));
  }
}

bool LocalMetaStore::has_value(const std::string &name) {
  std::unique_lock<std::mutex> lock(mtx_);
  return key_to_meta_.count(name) != 0;
}

void LocalMetaStore::set_curr_iter_num(size_t num) {
  std::unique_lock<std::mutex> lock(mtx_);
  curr_iter_num_ = num;
}

const size_t LocalMetaStore::curr_iter_num() {
  std::unique_lock<std::mutex> lock(mtx_);
  return curr_iter_num_;
}

void LocalMetaStore::set_curr_instance_state(InstanceState instance_state) { instance_state_ = instance_state; }

const InstanceState LocalMetaStore::curr_instance_state() { return instance_state_; }

const void LocalMetaStore::put_aggregation_feature_map(const std::string &name, const Feature &feature) {
  if (aggregation_feature_map_.count(name) > 0) {
    MS_LOG(WARNING) << "Put feature " << name << " failed.";
    return;
  }
  aggregation_feature_map_[name] = feature;
}

std::unordered_map<std::string, Feature> &LocalMetaStore::aggregation_feature_map() { return aggregation_feature_map_; }

bool LocalMetaStore::verifyAggregationFeatureMap(const std::unordered_map<std::string, size_t> &model) {
  // feature map size in Hybrid training is not equal with upload model size
  if (model.size() > aggregation_feature_map_.size()) {
    return false;
  }

  for (const auto &weight : model) {
    std::string weight_name = weight.first;
    size_t weight_size = weight.second;
    if (aggregation_feature_map_.count(weight_name) == 0) {
      return false;
    }
    if (weight_size != aggregation_feature_map_[weight_name].weight_size) {
      return false;
    }
  }
  return true;
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
