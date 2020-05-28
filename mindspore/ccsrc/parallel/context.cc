/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "parallel/context.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <map>

#include "common/utils.h"
#include "parallel/device_manager.h"

namespace mindspore {
namespace parallel {
static std::map<std::string, std::vector<int>> param_shapes;

std::vector<std::string> PARALLEL_MODE_LIST = {STAND_ALONE, DATA_PARALLEL, HYBRID_PARALLEL, SEMI_AUTO_PARALLEL,
                                               AUTO_PARALLEL};
std::vector<std::string> STRATEGY_SEARCH_MODE_LIST = {DYNAMIC_PROGRAMMING, RECURSIVE_PROGRAMMING};

std::shared_ptr<ParallelContext> ParallelContext::inst_context_ = nullptr;

std::shared_ptr<ParallelContext> ParallelContext::GetInstance() {
  if (inst_context_ == nullptr) {
    inst_context_.reset(new (std::nothrow) ParallelContext());
  }
  return inst_context_;
}

ParallelContext::ParallelContext() { Reset(); }

void ParallelContext::Reset() {
  mirror_mean_ = false;
  cast_before_mirror_ = true;
  loss_repeated_mean_ = true;
  device_num_ = 1;
  global_rank_ = 0;
  communication_backend_ = HCCL_BACKEND;
  device_num_is_set_ = false;
  global_rank_is_set_ = false;
  parallel_mode_ = STAND_ALONE;
  parameter_broadcast_ = false;
  parameter_broadcast_is_set_ = false;
  enable_all_reduce_fusion_ = false;
  strategy_ckpt_load_file_ = "";
  strategy_ckpt_save_file_ = "";
}

void ParallelContext::set_device_num(int32_t device_num) {
  device_num_ = device_num;
  device_num_is_set_ = true;
}

void ParallelContext::set_global_rank(int32_t global_rank) {
  global_rank_ = global_rank;
  global_rank_is_set_ = true;
}

void ParallelContext::set_mirror_mean(bool mirror_mean) { mirror_mean_ = mirror_mean; }

void ParallelContext::set_cast_before_mirror(bool cast_before_mirror) { cast_before_mirror_ = cast_before_mirror; }

void ParallelContext::set_loss_repeated_mean(bool loss_repeated_mean) { loss_repeated_mean_ = loss_repeated_mean; }

void ParallelContext::set_communication_backend(const std::string &communication_backend) {
  communication_backend_ = communication_backend;
}

bool ParallelContext::set_parallel_mode(const std::string &parallel_mode) {
  auto iter = std::find(PARALLEL_MODE_LIST.begin(), PARALLEL_MODE_LIST.end(), parallel_mode);
  if (iter == PARALLEL_MODE_LIST.end()) {
    MS_LOG(INFO) << "Invalid parallel mode:" << parallel_mode;
    return false;
  }
  parallel_mode_ = parallel_mode;
  return true;
}

bool ParallelContext::set_strategy_search_mode(const std::string &strategy_search_mode) {
  auto iter = std::find(STRATEGY_SEARCH_MODE_LIST.begin(), STRATEGY_SEARCH_MODE_LIST.end(), strategy_search_mode);
  if (iter == STRATEGY_SEARCH_MODE_LIST.end()) {
    MS_LOG(INFO) << "Invalid strategy search mode mode: " << strategy_search_mode;
    return false;
  }
  strategy_search_mode_ = strategy_search_mode;
  return true;
}

void ParallelContext::set_parameter_broadcast(bool parameter_broadcast) {
  parameter_broadcast_ = parameter_broadcast;
  parameter_broadcast_is_set_ = true;
}

void ParallelContext::set_strategy_ckpt_load_file(const std::string &strategy_ckpt_load_file) {
  strategy_ckpt_load_file_ = strategy_ckpt_load_file;
}

void ParallelContext::set_strategy_ckpt_save_file(const std::string &strategy_ckpt_save_file) {
  strategy_ckpt_save_file_ = strategy_ckpt_save_file;
}

void ParallelContext::SetAllReduceFusionSplitIndices(const std::vector<uint32_t> indices, const std::string &group) {
  all_reduce_fusion_split_indices_[group] = indices;
}

const std::vector<uint32_t> ParallelContext::GetAllReduceFusionSplitIndices(const std::string &group) const {
  auto iter = all_reduce_fusion_split_indices_.find(group);
  if (iter != all_reduce_fusion_split_indices_.end()) {
    return iter->second;
  }
  return {};
}

void ParallelContext::SetAllReduceFusionSplitSizes(const std::vector<uint32_t> sizes, const std::string &group) {
  all_reduce_fusion_split_sizes_[group] = sizes;
}

const std::vector<uint32_t> ParallelContext::GetAllReduceFusionSplitSizes(const std::string &group) const {
  auto iter = all_reduce_fusion_split_sizes_.find(group);
  if (iter != all_reduce_fusion_split_sizes_.end()) {
    return iter->second;
  }
  return {};
}

// Clear param_shapes before training in auto-parallel or semi-auto-parallel mode
void ParallelParameterContextInit(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!func_graph->has_flag(AUTO_PARALLEL) || !func_graph->has_flag(TRAINING)) {
    return;
  }
  param_shapes.clear();
}

// Restore the parameters' shape for evaluation/prediction in auto-parallel or semi-auto-parallel mode
void ParallelParameterContextRestoreInNoTraining(const FuncGraphPtr &func_graph, const ParameterPtr &param_node,
                                                 AbstractBasePtr ptr) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(param_node);
  MS_EXCEPTION_IF_NULL(ptr);
  if (!func_graph->has_flag(AUTO_PARALLEL) || (func_graph->flags().count(TRAINING) == 0) ||
      func_graph->flags()[TRAINING]) {
    return;
  }

  auto iter = param_shapes.find(param_node->name());
  if (iter == param_shapes.end()) {
    MS_LOG(WARNING) << "Can not found the shape for parameter " << param_node->name();
    return;
  }
  std::vector<int> shape = iter->second;
  std::shared_ptr<abstract::BaseShape> base_shape = std::make_shared<abstract::Shape>(shape);
  ptr->set_shape(base_shape);
  MS_LOG(DEBUG) << "The parameter name is " << param_node->name() << ", the shape is " << shape;
}

// Checkpoint the parameters' shape for training in auto-parallel or semi-auto-parallel mode
void ParallelParameterContextCkptInTraining(const FuncGraphPtr &func_graph, const ParameterPtr &param_node,
                                            const AbstractBasePtr &ptr) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(param_node);
  MS_EXCEPTION_IF_NULL(ptr);
  if (!func_graph->has_flag(AUTO_PARALLEL) || !func_graph->has_flag(TRAINING)) {
    return;
  }

  std::vector<int> shape = dyn_cast<abstract::Shape>(ptr->GetShapeTrack())->shape();
  auto ret = param_shapes.try_emplace(param_node->name(), shape);
  if (!ret.second) {
    MS_LOG(EXCEPTION) << "The shape for parameter name " << param_node->name() << " is existed";
    return;
  }

  MS_LOG(DEBUG) << "The parameter name is " << param_node->name() << ", the shape is " << shape;
}
}  // namespace parallel
}  // namespace mindspore
