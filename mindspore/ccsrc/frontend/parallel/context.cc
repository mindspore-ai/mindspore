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

#include "frontend/parallel/context.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <utility>

#include "frontend/parallel/device_manager.h"

namespace mindspore {
namespace parallel {
static std::map<std::string, Shape> param_shapes;

std::vector<std::string> PARALLEL_MODE_LIST = {STAND_ALONE, DATA_PARALLEL, HYBRID_PARALLEL, SEMI_AUTO_PARALLEL,
                                               AUTO_PARALLEL};
std::vector<std::string> STRATEGY_SEARCH_MODE_LIST = {DYNAMIC_PROGRAMMING, RECURSIVE_PROGRAMMING};

std::vector<std::string> COMMUNI_PARALLEL_MODE_LIST = {ALL_GROUP_PARALLEL, SAME_SERVER_GROUP_PARALLEL,
                                                       NO_GROUP_PARALLEL};

std::shared_ptr<ParallelContext> ParallelContext::inst_context_ = nullptr;

std::shared_ptr<ParallelContext> ParallelContext::GetInstance() {
  if (inst_context_ == nullptr) {
    inst_context_.reset(new (std::nothrow) ParallelContext());
  }
  return inst_context_;
}

ParallelContext::ParallelContext() { Reset(); }

void ParallelContext::Reset() {
  gradients_mean_ = false;
  full_batch_ = false;
  gradient_fp32_sync_ = true;
  loss_repeated_mean_ = true;
  device_num_ = 1;
  global_rank_ = 0;
  device_num_is_set_ = false;
  global_rank_is_set_ = false;
  parallel_mode_ = STAND_ALONE;
  parameter_broadcast_ = false;
  parameter_broadcast_is_set_ = false;
  enable_all_reduce_fusion_ = false;
  strategy_ckpt_load_file_ = "";
  strategy_ckpt_save_file_ = "";
  enable_parallel_optimizer_ = false;
  all_reduce_fusion_split_indices_.clear();
  all_reduce_fusion_split_sizes_.clear();
  strategy_search_mode_ = DYNAMIC_PROGRAMMING;
  pipeline_stage_split_num_ = 1;
  grad_accumulation_step_ = 1;
  init_param_shape_ = false;
  communi_parallel_mode_ = ALL_GROUP_PARALLEL;
}

void ParallelContext::set_device_num(int64_t device_num) {
  device_num_ = device_num;
  device_num_is_set_ = true;
}

void ParallelContext::set_global_rank(int64_t global_rank) {
  global_rank_ = global_rank;
  global_rank_is_set_ = true;
}

void ParallelContext::set_gradients_mean(bool gradients_mean) { gradients_mean_ = gradients_mean; }

void ParallelContext::set_full_batch(bool full_batch) { full_batch_ = full_batch; }

void ParallelContext::set_grad_accumulation_step(int64_t grad_accumulation_step) {
  grad_accumulation_step_ = grad_accumulation_step;
}

void ParallelContext::set_gradient_fp32_sync(bool gradient_fp32_sync) { gradient_fp32_sync_ = gradient_fp32_sync; }

void ParallelContext::set_loss_repeated_mean(bool loss_repeated_mean) { loss_repeated_mean_ = loss_repeated_mean; }

void ParallelContext::set_pipeline_stage_split_num(const int64_t stage_num) { pipeline_stage_split_num_ = stage_num; }

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

void ParallelContext::set_group_ckpt_save_file(const std::string &group_ckpt_save_file) {
  group_ckpt_save_file_ = group_ckpt_save_file;
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

bool ParallelContext::set_communi_parallel_mode(const std::string &communi_parallel_mode) {
  auto iter = std::find(COMMUNI_PARALLEL_MODE_LIST.begin(), COMMUNI_PARALLEL_MODE_LIST.end(), communi_parallel_mode);
  if (iter == COMMUNI_PARALLEL_MODE_LIST.end()) {
    MS_LOG(INFO) << "Invalid communication parallel mode:" << communi_parallel_mode;
    return false;
  }

  communi_parallel_mode_ = communi_parallel_mode;
  return true;
}

// Clear param_shapes before training in auto-parallel or semi-auto-parallel mode
void ParallelContext::ParallelParameterContextInitShape(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!func_graph->has_flag(AUTO_PARALLEL)) {
    return;
  }

  if (!func_graph->has_flag(TRAINING)) {
    init_param_shape_ = false;
    MS_LOG(INFO) << "In parallel evaluation or prediction, may be need to restore the parameter shape";
    return;
  }

  if ((ParallelContext::GetInstance()->grad_accumulation_step() > 1) && !func_graph->has_flag(ACCUMULATION)) {
    init_param_shape_ = false;
    MS_LOG(INFO) << "In parallel grad accumulation second graph, need to restore the parameter shape";
  } else {
    param_shapes.clear();
    init_param_shape_ = true;
    MS_LOG(INFO) << "Init the parameter shape dict";
  }
}

// Restore the parameters' shape for evaluation/prediction in auto-parallel or semi-auto-parallel mode
void ParallelContext::ParallelParameterContextRestoreShape(const FuncGraphPtr &func_graph,
                                                           const ParameterPtr &param_node, AbstractBasePtr ptr) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(param_node);
  MS_EXCEPTION_IF_NULL(ptr);
  if (!func_graph->has_flag(AUTO_PARALLEL)) {
    return;
  }

  if (init_param_shape_) {
    return;
  }
  auto iter = param_shapes.find(param_node->name());
  if (iter == param_shapes.end()) {
    MS_LOG(WARNING) << "Can not found the shape for parameter " << param_node->name();
    return;
  }
  Shape shape = iter->second;
  std::shared_ptr<abstract::BaseShape> base_shape = std::make_shared<abstract::Shape>(shape);
  ptr->set_shape(base_shape);
  MS_LOG(INFO) << "The parameter name is " << param_node->name() << ", the shape is " << shape;
}

// Clear param_shapes before training in auto-parallel or semi-auto-parallel mode
// Checkpoint the parameters' shape for training in auto-parallel or semi-auto-parallel mode
void ParallelContext::ParallelParameterContextCkptShape(const FuncGraphPtr &func_graph, const ParameterPtr &param_node,
                                                        const AbstractBasePtr &ptr) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(param_node);
  MS_EXCEPTION_IF_NULL(ptr);
  if (!func_graph->has_flag(AUTO_PARALLEL)) {
    return;
  }

  if (!init_param_shape_) {
    return;
  }
  std::vector<int64_t> shape = dyn_cast<abstract::Shape>(ptr->GetShapeTrack())->shape();
  auto ret = param_shapes.try_emplace(param_node->name(), shape);
  if (!ret.second) {
    MS_LOG(EXCEPTION) << "The shape for parameter name " << param_node->name() << " is existed";
    return;
  }

  MS_LOG(DEBUG) << "The parameter name is " << param_node->name() << ", the shape is " << shape;
}
}  // namespace parallel
}  // namespace mindspore
