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

#include "parallel/costmodel_context.h"

#include <memory>

#include "parallel/allreduce_fusion/allreduce_fusion.h"
#include "parallel/auto_parallel/graph_costmodel.h"

namespace mindspore {
namespace parallel {
std::shared_ptr<CostModelContext> CostModelContext::cm_context_inst_ = nullptr;

std::shared_ptr<CostModelContext> CostModelContext::GetInstance() {
  if (cm_context_inst_ == nullptr) {
    MS_LOG(INFO) << "Create costmodel_context";
    cm_context_inst_.reset(new (std::nothrow) CostModelContext());
  }
  return cm_context_inst_;
}

CostModelContext::CostModelContext() {
  ResetCostModel();
  ResetAlgoParameters();
}

void CostModelContext::ResetCostModel() {
  device_memory_capacity_ = DEFAULT_DEVICE_MEMORY_CAPACITY;
  costmodel_alpha_ = DEFAULT_COST_MODEL_ALPHA;
  costmodel_beta_ = DEFAULT_COST_MODEL_BETA;
  costmodel_gamma_ = DEFAULT_COST_MODEL_GAMMA;
  costmodel_communi_threshold_ = DEFAULT_COST_MODEL_COMMUNI_THRESHOLD;
  costmodel_communi_const_ = DEFAULT_COST_MODEL_COMMUNI_CONST;
  costmodel_communi_bias_ = DEFAULT_COST_MODEL_COMMUNI_BIAS;
  costmodel_allreduce_fusion_algorithm_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALGORITHM;
  costmodel_allreduce_fusion_times_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TIMES;
  costmodel_allreduce_fusion_tail_percent_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TAIL_PERCENT;
  costmodel_allreduce_fusion_tail_time_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TAIL_TIME;
  costmodel_allreduce_fusion_allreduce_inherent_time_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALLREDUCE_INHERENT_TIME;
  costmodel_allreduce_fusion_allreduce_bandwidth_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALLREDUCE_BANDWIDTH;
  costmodel_allreduce_fusion_computation_time_parameter_ =
    DEFAULT_COST_MODEL_ALLREDUCE_FUSION_COMPUTATION_TIME_PARAMETER;
}

void CostModelContext::ResetAlgoParameters() {
  costmodel_simplify_cal_ = DEFAULT_COST_MODEL_SIMPLIFY_CALCULATION;
  tensor_slice_alignment_enable_ = DEFAULT_TENSOR_SLICE_ALIGNMENT_ENABLE;
  tensor_slice_alignment_size_ = DEFAULT_TENSOR_SLICE_ALIGNMENT_SIZE;
  fully_use_device_ = DEFAULT_FULLY_USE_DEVICES;
  elementwise_stra_follow_ = DEFAULT_ELEMENTWISE_OP_STRA_FOLLOW;
}

void CostModelContext::set_device_memory_capacity(double dm_capacity) { device_memory_capacity_ = dm_capacity; }

void CostModelContext::set_costmodel_alpha(double cm_alpha) { costmodel_alpha_ = cm_alpha; }

void CostModelContext::set_costmodel_beta(double cm_beta) { costmodel_beta_ = cm_beta; }

void CostModelContext::set_costmodel_gamma(double cm_gamma) { costmodel_gamma_ = cm_gamma; }

void CostModelContext::set_costmodel_simplify_cal(bool cm_simplify) { costmodel_simplify_cal_ = cm_simplify; }

void CostModelContext::set_costmodel_communi_threshold(double cm_communi_th) {
  costmodel_communi_threshold_ = cm_communi_th;
}

void CostModelContext::set_costmodel_communi_const(double cm_communi_const) {
  costmodel_communi_const_ = cm_communi_const;
}

void CostModelContext::set_costmodel_communi_bias(double cm_communi_bias) { costmodel_communi_bias_ = cm_communi_bias; }

void CostModelContext::set_costmodel_allreduce_fusion_algorithm(int32_t algorithm) {
  costmodel_allreduce_fusion_algorithm_ = algorithm;
}

void CostModelContext::set_costmodel_allreduce_fusion_times(int32_t allreduce_fusion_times) {
  costmodel_allreduce_fusion_times_ = allreduce_fusion_times;
}

void CostModelContext::set_costmodel_allreduce_fusion_tail_percent(double tail_percent) {
  costmodel_allreduce_fusion_tail_percent_ = tail_percent;
}

void CostModelContext::set_costmodel_allreduce_fusion_tail_time(double tail_time) {
  costmodel_allreduce_fusion_tail_time_ = tail_time;
}

void CostModelContext::set_costmodel_allreduce_fusion_allreduce_inherent_time(double allreduce_inherent_time) {
  costmodel_allreduce_fusion_allreduce_inherent_time_ = allreduce_inherent_time;
}

void CostModelContext::set_costmodel_allreduce_fusion_allreduce_bandwidth(double allreduce_bandwidth) {
  costmodel_allreduce_fusion_allreduce_bandwidth_ = allreduce_bandwidth;
}

void CostModelContext::set_costmodel_allreduce_fusion_computation_time_parameter(double computation_time_parameter) {
  costmodel_allreduce_fusion_computation_time_parameter_ = computation_time_parameter;
}

void CostModelContext::set_tensor_slice_alignment_enable(bool ts_align) { tensor_slice_alignment_enable_ = ts_align; }

void CostModelContext::set_tensor_slice_alignment_size(size_t ts_align_size) {
  tensor_slice_alignment_size_ = ts_align_size;
}

void CostModelContext::set_fully_use_device(bool fully_use) { fully_use_device_ = fully_use; }

void CostModelContext::set_elementwise_stra_follow(bool elementwise_follow) {
  elementwise_stra_follow_ = elementwise_follow;
}
}  // namespace parallel
}  // namespace mindspore
