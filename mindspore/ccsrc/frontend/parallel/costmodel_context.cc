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

#include "frontend/parallel/costmodel_context.h"

#include <memory>

#include "frontend/parallel/allreduce_fusion/allreduce_fusion.h"
#include "utils/ms_context.h"

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
  costmodel_beta_ = DEFAULT_COST_MODEL_BETA_ASCEND;
  costmodel_gamma_ = DEFAULT_COST_MODEL_GAMMA;
  costmodel_communi_threshold_ = DEFAULT_COST_MODEL_COMMUNI_THRESHOLD;
  costmodel_communi_const_ = DEFAULT_COST_MODEL_COMMUNI_CONST;
  costmodel_communi_bias_ = DEFAULT_COST_MODEL_COMMUNI_BIAS;
  is_multi_subgraphs_ = DEFAULT_IS_MULTI_SUBGRAPHS;
  run_phase_ = TRAINING_PHASE;
  costmodel_allreduce_fusion_algorithm_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALGORITHM;
  costmodel_allreduce_fusion_times_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TIMES;
  costmodel_allreduce_fusion_tail_percent_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TAIL_PERCENT;
  costmodel_allreduce_fusion_tail_time_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_TAIL_TIME;
  costmodel_allreduce_fusion_allreduce_inherent_time_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALLREDUCE_INHERENT_TIME;
  costmodel_allreduce_fusion_allreduce_bandwidth_ = DEFAULT_COST_MODEL_ALLREDUCE_FUSION_ALLREDUCE_BANDWIDTH;
  costmodel_allreduce_fusion_computation_time_parameter_ =
    DEFAULT_COST_MODEL_ALLREDUCE_FUSION_COMPUTATION_TIME_PARAMETER;
  dp_algo_single_loop_ = DEFAULT_DP_ALGO_SINGLE_LOOP;
  rp_matmul_mem_coef_ = DEFAULT_RP_MATMUL_MEM_COEF;
}

void CostModelContext::ResetAlgoParameters() {
  costmodel_simplify_cal_ = DEFAULT_COST_MODEL_SIMPLIFY_CALCULATION;
  tensor_slice_alignment_enable_ = DEFAULT_TENSOR_SLICE_ALIGNMENT_ENABLE;
  tensor_slice_alignment_size_ = DEFAULT_TENSOR_SLICE_ALIGNMENT_SIZE;
  fully_use_device_ = DEFAULT_FULLY_USE_DEVICES;
  elementwise_stra_follow_ = DEFAULT_ELEMENTWISE_OP_STRA_FOLLOW;
  triangle_star_strategy_overwrite_ = DEFAULT_TRIANGLE_STAR_STRATEGY_OVERWRITE;
  dp_algo_enable_approxi_ = DEFAULT_DP_ALGO_ENABLE_APPROX;
  dp_algo_approxi_epsilon_ = DEFAULT_DP_ALGO_APPROX_EPSILON;
}

void CostModelContext::PrintCostModel() {
  MS_LOG(INFO) << "device_memory_capacity: " << device_memory_capacity_ << ".";
  MS_LOG(INFO) << "costmodel_alpha: " << costmodel_alpha_ << ".";
  MS_LOG(INFO) << "costmodel_beta: " << costmodel_beta_ << ".";
  MS_LOG(INFO) << "costmodel_gamma: " << costmodel_gamma_ << ".";
  MS_LOG(INFO) << "costmodel_simplify_cal: " << costmodel_simplify_cal_ << ".";
  MS_LOG(INFO) << "costmodel_communi_threshold: " << costmodel_communi_threshold_ << ".";
  MS_LOG(INFO) << "costmodel_communi_const: " << costmodel_communi_const_ << ".";
  MS_LOG(INFO) << "costmodel_communi_bias: " << costmodel_communi_bias_ << ".";
  MS_LOG(INFO) << "is_multi_subgraphs: " << is_multi_subgraphs_ << ".";
  MS_LOG(INFO) << "triangle_star_strategy_overwrite: " << triangle_star_strategy_overwrite_ << ".";
  MS_LOG(INFO) << "dp_algo_enable_approxi: " << dp_algo_enable_approxi_ << ".";
  MS_LOG(INFO) << "dp_algo_approxi_epsilon: " << dp_algo_approxi_epsilon_ << ".";
  MS_LOG(INFO) << "dp_algo_single_loop: " << dp_algo_single_loop_ << ".";
  MS_LOG(INFO) << "run_phase: " << run_phase_ << ".";
  MS_LOG(INFO) << "tensor_slice_alignment_enable: " << tensor_slice_alignment_enable_ << ".";
  MS_LOG(INFO) << "tensor_slice_align_size: " << tensor_slice_alignment_size_ << ".";
  MS_LOG(INFO) << "fully_use_device: " << fully_use_device_ << ".";
  MS_LOG(INFO) << "elementwise_stra_follow: " << elementwise_stra_follow_ << ".";
  MS_LOG(INFO) << "rp_matmul_mem_coef: " << rp_matmul_mem_coef_ << ".";
}

void CostModelContext::set_costmodel_context_for_device(const std::string &device_target) {
  if (device_target == kGPUDevice) {
    costmodel_beta_ = DEFAULT_COST_MODEL_BETA_GPU;
  }
}

void CostModelContext::set_dp_algo_approxi_epsilon(double epsilon) {
  if (epsilon <= 0 || epsilon > 1) {
    MS_LOG(EXCEPTION) << "'epsilon' must be in (0, 1]";
  }
  dp_algo_approxi_epsilon_ = epsilon;
}

void CostModelContext::set_rp_matmul_mem_coef(double coef) {
  if (coef <= 0) {
    MS_LOG(EXCEPTION) << "'coef' must be positive";
  }
  rp_matmul_mem_coef_ = coef;
}

void CostModelContext::set_dp_algo_enable_approxi(bool approxi) {
  if (approxi) {
    MS_LOG(INFO) << "dp_algo_enable_approx: true.";
  } else {
    MS_LOG(INFO) << "dp_algo_enable_approx: false.";
  }
  dp_algo_enable_approxi_ = approxi;
}

void CostModelContext::set_device_memory_capacity(double dm_capacity) {
  if (dm_capacity <= 0) {
    MS_LOG(EXCEPTION) << "'device_memory_capacity' must be positive.";
  }
  device_memory_capacity_ = dm_capacity;
}

void CostModelContext::set_costmodel_alpha(double cm_alpha) {
  if (cm_alpha <= 0) {
    MS_LOG(EXCEPTION) << "'costmodel_alpha' must be positive.";
  }
  costmodel_alpha_ = cm_alpha;
}

void CostModelContext::set_costmodel_beta(double cm_beta) {
  if (cm_beta <= 0) {
    MS_LOG(EXCEPTION) << "'costmodel_beta' must be positive.";
  }
  costmodel_beta_ = cm_beta;
}

void CostModelContext::set_costmodel_gamma(double cm_gamma) {
  if ((cm_gamma < 0) || (cm_gamma > 1)) {
    MS_LOG(EXCEPTION) << "'costmodel_gamma' must in [0, 1].";
  }
  costmodel_gamma_ = cm_gamma;
}

void CostModelContext::set_costmodel_simplify_cal(bool cm_simplify) {
  if (cm_simplify) {
    MS_LOG(INFO) << "costmodel_simplify_cal: true.";
  } else {
    MS_LOG(INFO) << "costmodel_simplify_cal: false.";
  }
  costmodel_simplify_cal_ = cm_simplify;
}

void CostModelContext::set_costmodel_communi_threshold(double cm_communi_th) {
  if (cm_communi_th < 0) {
    MS_LOG(EXCEPTION) << "'costmodel_communi_threshold' must be non-zero.";
  }
  costmodel_communi_threshold_ = cm_communi_th;
}

void CostModelContext::set_costmodel_communi_const(double cm_communi_const) {
  if (cm_communi_const < 0) {
    MS_LOG(EXCEPTION) << "'costmodel_communi_const' must be non-zero.";
  }
  costmodel_communi_const_ = cm_communi_const;
}

void CostModelContext::set_costmodel_communi_bias(double cm_communi_bias) {
  if (cm_communi_bias < 0) {
    MS_LOG(EXCEPTION) << "'costmodel_communi_bias' must be non-zero.";
  }
  costmodel_communi_bias_ = cm_communi_bias;
}

void CostModelContext::set_multi_subgraphs(bool multi_graphs) {
  if (multi_graphs) {
    MS_LOG(INFO) << "multi_subgraphs: true.";
  } else {
    MS_LOG(INFO) << "multi_subgraphs: false.";
  }
  is_multi_subgraphs_ = multi_graphs;
}
void CostModelContext::set_costmodel_allreduce_fusion_algorithm(int64_t algorithm) {
  costmodel_allreduce_fusion_algorithm_ = algorithm;
}

void CostModelContext::set_costmodel_allreduce_fusion_times(int64_t allreduce_fusion_times) {
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

void CostModelContext::set_tensor_slice_alignment_enable(bool ts_align) {
  if (ts_align) {
    MS_LOG(INFO) << "tensor_slice_align_enable: true.";
  } else {
    MS_LOG(INFO) << "tensor_slice_align_enable: false.";
  }
  tensor_slice_alignment_enable_ = ts_align;
}

void CostModelContext::set_tensor_slice_alignment_size(size_t ts_align_size) {
  if (ts_align_size == 0) {
    MS_LOG(EXCEPTION) << "'tensor_slice_align_size' must be positive.";
  }
  tensor_slice_alignment_size_ = ts_align_size;
}

void CostModelContext::set_fully_use_device(bool fully_use) {
  if (fully_use) {
    MS_LOG(INFO) << "fully_use_devices: true.";
  } else {
    MS_LOG(INFO) << "fully_use_devices: false.";
  }
  fully_use_device_ = fully_use;
}

void CostModelContext::set_elementwise_stra_follow(bool elementwise_follow) {
  if (elementwise_follow) {
    MS_LOG(INFO) << "elementwise_op_strategy_follow: true.";
  } else {
    MS_LOG(INFO) << "elementwise_op_strategy_follow: false.";
  }
  elementwise_stra_follow_ = elementwise_follow;
}

void CostModelContext::set_triangle_star_strategy_overwrite(bool overwrite) {
  if (overwrite) {
    MS_LOG(INFO) << "triangle_star_strategy_overwrite: true.";
  } else {
    MS_LOG(INFO) << "triangle_star_strategy_overwrite: false.";
  }
  triangle_star_strategy_overwrite_ = overwrite;
}

void CostModelContext::set_run_phase(int64_t phase) {
  if (phase != 0 && phase != 1) {
    MS_LOG(EXCEPTION) << "'run_phase' must be in {0, 1}";
  }
  run_phase_ = phase;
}

void CostModelContext::set_dp_algo_single_loop(bool single_loop) {
  if (single_loop) {
    MS_LOG(INFO) << "dp_algo_single_loop: true.";
  } else {
    MS_LOG(INFO) << "dp_algo_single_loop: false.";
  }
  dp_algo_single_loop_ = single_loop;
}

struct CostRegister {
  CostRegister() {
    MsContext::device_seter([](const std::string &device_target) {
      CostModelContext::GetInstance()->set_costmodel_context_for_device(device_target);
    });
  }
  ~CostRegister() = default;
} cost_regsiter;
}  // namespace parallel
}  // namespace mindspore
