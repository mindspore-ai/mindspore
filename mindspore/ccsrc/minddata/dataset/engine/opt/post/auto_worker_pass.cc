/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <cmath>
#include <algorithm>

#include "minddata/dataset/engine/ir/datasetops/batch_node.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/engine/opt/post/auto_worker_pass.h"

namespace mindspore {
namespace dataset {

// this will become the RootNode:DatasetNode when it is turned on
Status AutoWorkerPass::RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  uint8_t config = GlobalContext::config_manager()->get_auto_worker_config();

  OpWeightPass pass(kOpWeightConfigs[config < kOpWeightConfigs.size() ? config : 0]);

  std::string weight_str;
  for (const auto &p : pass.weight_profile_) weight_str += ("(" + p.first + "=" + std::to_string(p.second) + ")");
  int32_t num_shards = GlobalContext::config_manager()->get_num_shards_for_auto_num_workers();
  num_shards = std::min(std::max(1, num_shards), thread_cnt_);

  MS_LOG(INFO) << "AutoWorkerPass is enabled; this could override existing num_workers set in each parallel op."
               << "total number of threads on this CPU: " << thread_cnt_ << ", "
               << "min num_workers to override:" << min_num_workers_ << ", "
               << "max num_workers to override:" << max_num_workers_ << ", "
               << "adjusted num_shards (between 1 and total thread cnt): " << num_shards
               << ", weight profile:" << weight_str << ".";

  // get the maximum weight of all the ops, this value is used to ensure the ratio of num_workers between ops
  float max_weight = 0;
  for (const auto &p : pass.weight_profile_) max_weight = std::max(max_weight, p.second);
  RETURN_IF_NOT_OK(pass.Run(root_ir, modified));
  if (pass.parallel_ops_.size() > 3) {
    MS_LOG(WARNING) << "AutoNumWorker right now is only suitable for simple dataset pipelines that has at most, 1 leaf "
                    << "1 batch and 1 map. AutoNumWorker may not be optimal for usage on complex pipelines.";
  }

  for (auto &p : pass.parallel_ops_) {
    // get the num worker via the weight ratio
    int32_t num_workers = std::ceil((thread_cnt_ * p.second) / (pass.weight_sum_ * num_shards));
    // this is to ensure when thread_cnt_ is very large let's say 192, the num_worker ratio is still kept
    // e.g. the optional 2:1 ratio between minddataset and batch
    int32_t cur_node_max = std::ceil(p.second * max_num_workers_ / max_weight);
    // this will ensure that num_workers will fall with the range of [1,cur_node_max]
    int32_t cur_node_num_worker = std::max(std::min(num_workers, cur_node_max), min_num_workers_);

    // if the num_worker to set is same as original, skip setting and printing the logs
    if (cur_node_num_worker == p.first->num_workers()) continue;
    // log the change via warning msg so user can see what the num_worker is being set for which op
    MS_LOG(WARNING) << "AutoNumWorker enabled, num_workers in " << p.first->Name() << " is auto-adjusted from "
                    << std::to_string(p.first->num_workers()) + " to " + std::to_string(cur_node_num_worker);
    p.first->SetNumWorkers(cur_node_num_worker);
  }
  return Status::OK();
}

Status AutoWorkerPass::OpWeightPass::Visit(std::shared_ptr<MapNode> node, bool *const modified) {
  auto itr = weight_profile_.find(node->Name());
  CHECK_FAIL_RETURN_UNEXPECTED(itr != weight_profile_.end(), node->Name() + "'s weight doesn't exist.");
  int32_t weight = itr->second;
  weight_sum_ += weight;
  parallel_ops_.emplace_back(std::make_pair(std::static_pointer_cast<DatasetNode>(node), weight));
  return Status::OK();
}

Status AutoWorkerPass::OpWeightPass::Visit(std::shared_ptr<BatchNode> node, bool *const modified) {
  auto itr = weight_profile_.find(node->Name());
  CHECK_FAIL_RETURN_UNEXPECTED(itr != weight_profile_.end(), node->Name() + "'s weight doesn't exist.");
  int32_t weight = itr->second;
  weight_sum_ += weight;
  parallel_ops_.emplace_back(std::make_pair(std::static_pointer_cast<DatasetNode>(node), weight));
  return Status::OK();
}

Status AutoWorkerPass::OpWeightPass::Visit(std::shared_ptr<MappableSourceNode> node, bool *const modified) {
  RETURN_OK_IF_TRUE(node->Name() == kGeneratorNode);  // generator is pipeline op, skip this
  auto itr = weight_profile_.find("MappableSource");
  CHECK_FAIL_RETURN_UNEXPECTED(itr != weight_profile_.end(),
                               "LeafSourceNode::" + node->Name() + "'s weight doesn't exist.");
  int32_t weight = itr->second;
  weight_sum_ += weight;
  parallel_ops_.emplace_back(std::make_pair(std::static_pointer_cast<DatasetNode>(node), weight));
  return Status::OK();
}

Status AutoWorkerPass::OpWeightPass::Visit(std::shared_ptr<NonMappableSourceNode> node, bool *const modified) {
  auto itr = weight_profile_.find("NonMappableSource");
  CHECK_FAIL_RETURN_UNEXPECTED(itr != weight_profile_.end(),
                               "NonLeafSource::" + node->Name() + "'s weight doesn't exist.");
  int32_t weight = itr->second;
  weight_sum_ += weight;
  parallel_ops_.emplace_back(std::make_pair(std::static_pointer_cast<DatasetNode>(node), weight));
  return Status::OK();
}

Status AutoWorkerPass::OpWeightPass::Visit(std::shared_ptr<DatasetNode> node, bool *const modified) {
  weight_sum_ += GetNodeWeightFromProfile(node);
  return Status::OK();
}

float AutoWorkerPass::OpWeightPass::GetNodeWeightFromProfile(std::shared_ptr<DatasetNode> node) {
  auto itr = weight_profile_.find(node->Name());
  // returns 0 if name doesn't exist in the weight profile
  return itr == weight_profile_.end() ? 0 : itr->second;
}

}  // namespace dataset
}  // namespace mindspore
