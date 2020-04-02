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

#ifndef MINDSPORE_CCSRC_PARALLEL_COSTMODEL_CONTEXT_H_
#define MINDSPORE_CCSRC_PARALLEL_COSTMODEL_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>

#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
class CostModelContext {
 public:
  ~CostModelContext() = default;
  CostModelContext(const CostModelContext&) = delete;
  CostModelContext& operator=(const CostModelContext&) = delete;
  void ResetCostModel();
  void ResetAlgoParameters();

  static std::shared_ptr<CostModelContext> GetInstance();

  // DEVICE_MEMORY_CAPACITY
  void set_device_memory_capacity(double);
  double device_memory_capacity() const { return device_memory_capacity_; }

  // COST_MODEL_ALPHA
  void set_costmodel_alpha(double);
  double costmodel_alpha() const { return costmodel_alpha_; }

  // COST_MODEL_BETA
  void set_costmodel_beta(double);
  double costmodel_beta() const { return costmodel_beta_; }

  // COST_MODEL_GAMMA
  void set_costmodel_gamma(double);
  double costmodel_gamma() const { return costmodel_gamma_; }

  // COST_MODEL_SIMPLIFY_CALCULATION
  void set_costmodel_simplify_cal(bool);
  bool costmodel_simplify_cal() const { return costmodel_simplify_cal_; }

  // COST_MODEL_COMMUNI_THRESHOLD
  void set_costmodel_communi_threshold(double);
  double costmodel_communi_threshold() const { return costmodel_communi_threshold_; }

  // COST_MODEL_COMMUNI_CONST
  void set_costmodel_communi_const(double);
  double costmodel_communi_const() const { return costmodel_communi_const_; }

  // COST_MODEL_COMMUNI_BIAS
  void set_costmodel_communi_bias(double);
  double costmodel_communi_bias() const { return costmodel_communi_bias_; }

  void set_costmodel_allreduce_fusion_algorithm(int32_t);
  int32_t costmodel_allreduce_fusion_algorithm() const { return costmodel_allreduce_fusion_algorithm_; }

  void set_costmodel_allreduce_fusion_times(int32_t);
  int32_t costmodel_allreduce_fusion_times() const { return costmodel_allreduce_fusion_times_; }

  void set_costmodel_allreduce_fusion_tail_percent(double);
  double costmodel_allreduce_fusion_tail_percent() const { return costmodel_allreduce_fusion_tail_percent_; }

  void set_costmodel_allreduce_fusion_tail_time(double);
  double costmodel_allreduce_fusion_tail_time() const { return costmodel_allreduce_fusion_tail_time_; }

  void set_costmodel_allreduce_fusion_allreduce_inherent_time(double);
  double costmodel_allreduce_fusion_allreduce_inherent_time() const {
    return costmodel_allreduce_fusion_allreduce_inherent_time_;
  }

  void set_costmodel_allreduce_fusion_allreduce_bandwidth(double);
  double costmodel_allreduce_fusion_allreduce_bandwidth() const {
    return costmodel_allreduce_fusion_allreduce_bandwidth_;
  }

  void set_costmodel_allreduce_fusion_computation_time_parameter(double);
  double costmodel_allreduce_fusion_computation_time_parameter() const {
    return costmodel_allreduce_fusion_computation_time_parameter_;
  }

  // TENSOR_SLICE_ALIGNMENT_ENABLE
  void set_tensor_slice_alignment_enable(bool);
  bool tensor_slice_alignment_enable() const { return tensor_slice_alignment_enable_; }

  // TENSOR_SLICE_ALIGNMENT_SIZE
  void set_tensor_slice_alignment_size(size_t);
  size_t tensor_slice_alignment_size() const { return tensor_slice_alignment_size_; }

  // NOT_FULLY_USE_DEVICES
  void set_not_fully_use_device(bool);
  bool not_fully_use_device() const { return not_fully_use_device_; }

  // ELEMENTWISE_OP_STRA_FOLLOW
  void set_elementwise_stra_follow(bool);
  bool elementwise_stra_follow() const { return elementwise_stra_follow_; }

 private:
  CostModelContext();
  static std::shared_ptr<CostModelContext> cm_context_inst_;

  // DEVICE_MEMORY_CAPACITY
  double device_memory_capacity_;

  // COST_MODEL_ALPHA
  double costmodel_alpha_;

  // COST_MODEL_BETA
  double costmodel_beta_;

  // COST_MODEL_GAMMA
  double costmodel_gamma_;

  // COST_MODEL_SIMPLIFY_CALCULATION
  bool costmodel_simplify_cal_;

  // COST_MODEL_COMMUNI_THRESHOLD
  double costmodel_communi_threshold_;

  // COST_MODEL_COMMUNI_CONST
  double costmodel_communi_const_;

  // COST_MODEL_COMMUNI_BIAS
  double costmodel_communi_bias_;

  int32_t costmodel_allreduce_fusion_algorithm_;

  int32_t costmodel_allreduce_fusion_times_;

  double costmodel_allreduce_fusion_tail_percent_;

  double costmodel_allreduce_fusion_tail_time_;

  double costmodel_allreduce_fusion_allreduce_inherent_time_;

  double costmodel_allreduce_fusion_allreduce_bandwidth_;

  double costmodel_allreduce_fusion_computation_time_parameter_;

  // TENSOR_SLICE_ALIGNMENT_ENABLE
  bool tensor_slice_alignment_enable_;

  // TENSOR_SLICE_ALIGNMENT_SIZE
  size_t tensor_slice_alignment_size_;

  // NOT_FULLY_USE_DEVICES
  bool not_fully_use_device_;

  // ELEMENTWISE_OP_STRA_FOLLOW
  bool elementwise_stra_follow_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_COSTMODEL_CONTEXT_H_
