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

#include "frontend/parallel/auto_parallel/operator_costmodel.h"

#include <algorithm>
#include <random>
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
void OperatorCost::set_is_parameter(const std::vector<bool> &is_parameter) { is_parameter_ = is_parameter; }

void OperatorCost::set_is_parameter_involve(const std::vector<bool> &is_parameter_inv) {
  is_parameter_involve_ = is_parameter_inv;
  is_inputs_should_in_memory_ = std::vector<bool>(is_parameter_involve_.size(), false);
}

void OperatorCost::set_output_parameter_involve(int64_t output_para) { output_parameter_involve_ = output_para; }

void OperatorCost::SetInputAndOutputTypeLength(const std::vector<size_t> &input_lengths,
                                               const std::vector<size_t> &output_lengths) {
  inputs_type_lengths_ = input_lengths;
  outputs_type_lengths_ = output_lengths;
}

void OperatorCost::set_output_critical(int64_t critical) { is_outputs_critical_ = critical; }

double OperatorCost::GetMemoryCost(const std::vector<TensorInfo> &inputs,
                                   const std::vector<TensorInfo> &outputs) const {
  return GetInputMemoryCost(inputs, outputs) + GetOutputMemoryCost(inputs, outputs);
}

double OperatorCost::GetInputMemoryCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &) const {
  double result = 0.0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_inputs_should_in_memory_[i]) {
      result += ListProduct(inputs[i].slice_shape()) * static_cast<double>(inputs_type_lengths_[i]);
    }
  }
  return result;
}

double OperatorCost::GetOutputMemoryCost(const std::vector<TensorInfo> &,
                                         const std::vector<TensorInfo> &outputs) const {
  double result = 0.0;
  if (is_output_should_in_memory_) {
    // When this operator has multiple outputs, they all contributes to the memory.
    for (size_t i = 0; i < outputs.size(); ++i) {
      result += ListProduct(outputs[i].slice_shape()) * static_cast<double>(outputs_type_lengths_[i]);
    }
  }
  return result;
}

double OperatorCost::GetMemoryCostForInference(const std::vector<TensorInfo> &,
                                               const std::vector<TensorInfo> &outputs) const {
  double result = 0.0;
  if (is_outputs_critical_ == -1) {
    MS_LOG(EXCEPTION) << "The critical flag is not set.";
  }
  if (is_outputs_critical_ == 1) {
    for (size_t i = 0; i < outputs.size(); ++i) {
      result += ListProduct(outputs[i].slice_shape()) * static_cast<double>(outputs_type_lengths_[i]);
    }
  }
  return result;
}

// return the per device communication cost in the forward phase.
double MatMulCost::GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                      int64_t) const {
  TensorInfo input0 = inputs[0];
  TensorInfo output0 = outputs[0];
  Shape input0_shape = input0.shape();
  Shape input0_slice_shape = input0.slice_shape();
  if (input0_shape[input0_shape.size() - 1] == input0_slice_shape[input0_slice_shape.size() - 1]) {
    // If the reduced dimension has not been partitioned, then there is no communication cost.
    return 0.0;
  } else {
    // Else, the communication cost is the size (number of bytes) of a slice of output tensor.
    return ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]);
  }
}

// return the per device communication cost in the forward phase.
double MatMulCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                       int64_t stage_id) const {
  // In backward phase, the communication cost is incurred only when tensor B is a Parameter and tensor B does not
  // fully utilize all devices
  double result = 0.0;
  if (is_parameter_[1]) {
    TensorInfo input1 = inputs[1];  // tensor B
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }

    if (total_device_num != LongToSize(used_device_num))
      result += ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  }

  return result;
}

// Return the per device computation cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double MatMulCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs,
                                             const std::vector<TensorInfo> &outputs, int64_t) const {
  // In forward phase, the computation cost = slice(A) + slice(B) + (0 or 1) allreduce(slice(C))
  double result = 0.0;
  TensorInfo output0 = outputs[0];
  Shape input0_slice_shape = inputs[0].slice_shape();
  Shape input1_slice_shape = inputs[1].slice_shape();
  Shape input0_shape = inputs[0].shape();
  if (input0_shape[input0_shape.size() - 1] != input0_slice_shape[input0_slice_shape.size() - 1]) {
    // If the reduced dimension has been partitioned, then there is no communication cost.
    result += ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]);
  }
  result += ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) +
            ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  return result;
}

// Return the per device computation cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double MatMulCost::GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                              int64_t stage_id) const {
  // In backward phase, the computation cost = (0 or 1) allreduce(slice(B))
  double result = 0.0;
  if (is_parameter_[1]) {
    TensorInfo input1 = inputs[1];  // tensor B
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }

    if (total_device_num != LongToSize(used_device_num))
      result += ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  }

  return result;
}

// Not taking account of output
void MatMulCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

// Taking account of input
void MatMulCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  if (is_parameter_[0]) {
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }

  if (is_parameter_[1]) {
    is_inputs_should_in_memory_[1] = true;
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  } else if (is_parameter_involve_[1]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  }
}

// Return the per device communication cost in the forward phase.
double CastCost::GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const {
  // ReLU is the element-wise operator, thus it does not need communication in the forward phase
  return 0.0;
}

// Return the per device communication cost in the backward phase.
double CastCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                     int64_t stage_id) const {
  double result = 0.0;
  if (is_parameter_[0]) {
    TensorInfo input1 = inputs[0];
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }
    if (total_device_num != LongToSize(used_device_num)) {
      result = ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
    }
  }
  return result;
}

// Return the per device computation cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double CastCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                           int64_t) const {
  TensorInfo input0 = inputs[0];
  Shape input0_slice_shape = input0.slice_shape();
  return ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
}

// Return the per device computation cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double CastCost::GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                            int64_t) const {
  return 0.0;
}

// Not taking account of output
void CastCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

// Not taking account of input
void CastCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
}

// Taking account of output
void SqrtCost::CalculateOutputInMemory() { is_output_should_in_memory_ = is_parameter_involve_[0]; }

// Taking account of input
void GeLUCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  if (is_parameter_[0]) {
    is_inputs_should_in_memory_[0] = true;
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  }
}

// Return the per device communication cost in the forward phase.
double SoftmaxCost::GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                       int64_t) const {
  // In the forward phase, the communication cost = 0
  return 0.0;
}

// Return the per device communication cost in the backward phase.
double SoftmaxCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                        int64_t stage_id) const {
  double result = 0.0;
  if (is_parameter_[0]) {
    TensorInfo input1 = inputs[0];
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }
    if (total_device_num != LongToSize(used_device_num)) {
      result = ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
    }
  }
  return result;
}

// Return the per device computation cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double SoftmaxCost::GetForwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &outputs,
                                              int64_t) const {
  if (outputs.empty() || outputs_type_lengths_.empty()) {
    MS_LOG(EXCEPTION) << "The outputs or outputs_type_length is empty";
  }

  // use output for Tile operator
  TensorInfo output_info = outputs[0];
  Shape output_slice_shape = output_info.slice_shape();
  return ListProduct(output_slice_shape) * static_cast<double>(outputs_type_lengths_[0]);
}

// Return the per device computation cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double SoftmaxCost::GetBackwardComputationCost(const std::vector<mindspore::parallel::TensorInfo> &,
                                               const std::vector<mindspore::parallel::TensorInfo> &, int64_t) const {
  return 0.0;
}

// Taking account of output
void SoftmaxCost::CalculateOutputInMemory() { is_output_should_in_memory_ = is_parameter_involve_[0]; }

// Not taking account of input
void SoftmaxCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
}

// Not taking account of output
void PackCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

// Not taking account of input
void PackCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
}

// Not taking account of output
void TileCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

// Taking account of input
void TileCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'y'
  if (is_parameter_[0]) {
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }

  if (!is_inputs_should_in_memory_[1]) {
    is_inputs_should_in_memory_[1] = is_parameter_[1];
  }
}

// Not taking account of output
void BroadcastToCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

void BroadcastToCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
}

// Taking account of input
void ReLU6Cost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  if (is_parameter_[0]) {
    is_inputs_should_in_memory_[0] = true;
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  }
}

// Taking account of input
void TransposeCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'y'
  if (is_parameter_[0]) {
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }

  if (!is_inputs_should_in_memory_[1]) {
    is_inputs_should_in_memory_[1] = is_parameter_[1];
  }
}

// return the per device communication cost in the forward phase.
double TmpIdentityCost::GetForwardCommCost(const std::vector<mindspore::parallel::TensorInfo> &,
                                           const std::vector<mindspore::parallel::TensorInfo> &, int64_t) const {
  // Identity is the element-wise operator, thus it does not need communication in the forward phase
  return 0.0;
}

// return the per device communication cost in the backward phase.
double TmpIdentityCost::GetBackwardCommCost(const std::vector<mindspore::parallel::TensorInfo> &,
                                            const std::vector<mindspore::parallel::TensorInfo> &, int64_t) const {
  // Identity is the element-wise operator, thus it does not need communication in the backward phase
  return 0.0;
}

// Return the per device computation cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double TmpIdentityCost::GetForwardComputationCost(const std::vector<mindspore::parallel::TensorInfo> &,
                                                  const std::vector<mindspore::parallel::TensorInfo> &, int64_t) const {
  return 0.0;
}

// Return the per device computation cost in the backward phase. The cost is calculated according to the bytes
// this operator uses
double TmpIdentityCost::GetBackwardComputationCost(const std::vector<mindspore::parallel::TensorInfo> &,
                                                   const std::vector<mindspore::parallel::TensorInfo> &,
                                                   int64_t) const {
  return 0.0;
}

// Not taking account of output
void TmpIdentityCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

// Not taking account of input
void TmpIdentityCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
}

double BatchParallelCost::GetForwardComputationCost(const std::vector<mindspore::parallel::TensorInfo> &inputs,
                                                    const std::vector<mindspore::parallel::TensorInfo> &,
                                                    int64_t) const {
  double cost = 0.0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    cost += ListProduct(inputs[i].slice_shape()) * static_cast<double>(inputs_type_lengths_[i]);
  }
  return cost;
}

double BatchParallelCost::GetBackwardComputationCost(const std::vector<mindspore::parallel::TensorInfo> &,
                                                     const std::vector<mindspore::parallel::TensorInfo> &,
                                                     int64_t) const {
  return 0.0;
}

double BatchParallelCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                              int64_t stage_id) const {
  double result = 0.0;
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

  for (size_t j = 0; j < inputs.size(); ++j) {
    if (!is_parameter_[j]) {
      continue;
    }
    TensorInfo input_a_tensor_info = inputs[j];
    Shape input_a_shape = input_a_tensor_info.shape();
    Shape input_a_slice_shape = input_a_tensor_info.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input_a_shape.size(); ++i) {
      used_device_num *= input_a_shape[i] / input_a_slice_shape[i];
    }
    if (total_device_num != LongToSize(used_device_num)) {
      result += ListProduct(input_a_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
    }
  }

  return result;
}

void BatchParallelCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

void BatchParallelCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  if (is_parameter_[0]) {
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }

  if (is_parameter_[1]) {
    is_inputs_should_in_memory_[1] = true;
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  } else if (is_parameter_involve_[1]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  }
}

void SparseSoftmaxCrossEntropyWithLogitsCost::CalculateOutputInMemory() {
  is_output_should_in_memory_ = is_parameter_involve_[0];
}

void SparseSoftmaxCrossEntropyWithLogitsCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
  is_inputs_should_in_memory_[1] = is_parameter_[1];
}
// return the per device communication cost in the forward phase.
double PReLUCost::GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const {
  // prelu does not need communication in the forward phase
  return 0.0;
}

// return the per device communication cost in the backward phase.
double PReLUCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                      int64_t stage_id) const {
  double result = 0.0;
  if (is_parameter_[1]) {
    TensorInfo input1 = inputs[1];
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }
    if (total_device_num != LongToSize(used_device_num)) {
      result = ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
    }
  }
  return result;
}

// Return the per device computation cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double PReLUCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                            int64_t) const {
  // In forward phase, the computation cost = slice(A) + slice(B)
  Shape input0_slice_shape = inputs[0].slice_shape();
  Shape input1_slice_shape = inputs[1].slice_shape();
  double result = ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) +
                  ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  return result;
}

// Return the per device computation cost in the backward phase. The cost is calculated according to the bytes
// this operator uses
double PReLUCost::GetBackwardComputationCost(const std::vector<mindspore::parallel::TensorInfo> &inputs,
                                             const std::vector<mindspore::parallel::TensorInfo> &,
                                             int64_t stage_id) const {
  // In backward phase, the computation cost = (0 or 1) allreduce(slice(B))
  double result = 0.0;
  if (is_parameter_[1]) {
    TensorInfo input1 = inputs[1];  // tensor B
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }

    if (total_device_num != LongToSize(used_device_num)) {
      result += ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
    }
  }
  return result;
}

void PReLUCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

void PReLUCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of both 'x' and 'y';
  // when calculating 'dy', taking account of both 'x' and 'y'
  if (is_parameter_involve_[0] || is_parameter_involve_[1]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }
}

// return the per device communication cost in the forward phase.
double OneHotCost::GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const {
  // onehot does not need communication in the forward phase
  return 0.0;
}

// return the per device communication cost in the backward phase.
double OneHotCost::GetBackwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                       int64_t) const {
  // onehot does not need communication in the backward phase
  return 0.0;
}

// Return the per device computation cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double OneHotCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                             int64_t) const {
  // In onehot's forward phase, the computation cost = slice(A)
  Shape input0_slice_shape = inputs[0].slice_shape();
  return ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
}

// Return the per  device computation cost in the backward phase. The cost is calculated according to the bytes
// this operator uses
double OneHotCost::GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                              int64_t) const {
  return 0.0;
}

// Not taking account of output
void OneHotCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

// Not taking account of input
void OneHotCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
  is_inputs_should_in_memory_[1] = is_parameter_[1];
  is_inputs_should_in_memory_[ONEHOT_INPUTS_SIZE - 2] = is_parameter_[ONEHOT_INPUTS_SIZE - 2];
  is_inputs_should_in_memory_[ONEHOT_INPUTS_SIZE - 1] = is_parameter_[ONEHOT_INPUTS_SIZE - 1];
}

// return the per device communication cost in the forward phase.
double SoftmaxCrossEntropyWithLogitsCost::GetForwardCommCost(const std::vector<TensorInfo> &,
                                                             const std::vector<TensorInfo> &, int64_t) const {
  // SoftmaxCrossEntropyWithLogitsCost does not need communication in the forward phase
  return 0.0;
}

// return the per device communication cost in the backward phase.
double SoftmaxCrossEntropyWithLogitsCost::GetBackwardCommCost(const std::vector<TensorInfo> &,
                                                              const std::vector<TensorInfo> &, int64_t) const {
  // SoftmaxCrossEntropyWithLogitsCost does not need communication in the backward phase
  return 0.0;
}

// Return the per device computation cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double SoftmaxCrossEntropyWithLogitsCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs,
                                                                    const std::vector<TensorInfo> &, int64_t) const {
  // In forward phase, the computation cost = slice(A) + slice(B)
  Shape input0_slice_shape = inputs[0].slice_shape();
  Shape input1_slice_shape = inputs[1].slice_shape();
  double result = ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) +
                  ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  return result;
}

// Return the per device computation cost in the backward phase. The cost is calculated according to the bytes
// this operator uses
double SoftmaxCrossEntropyWithLogitsCost::GetBackwardComputationCost(const std::vector<TensorInfo> &,
                                                                     const std::vector<TensorInfo> &, int64_t) const {
  return 0.0;
}

// Taking account of output
void SoftmaxCrossEntropyWithLogitsCost::CalculateOutputInMemory() {
  is_output_should_in_memory_ = is_parameter_involve_[0];
}

void SoftmaxCrossEntropyWithLogitsCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
  is_inputs_should_in_memory_[1] = is_parameter_[1];
}

// return the per device communication cost in the forward phase.
double ReshapeCost::GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                       int64_t stage_id) const {
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  RankList dev_list = g_device_manager->GetDeviceListByStageId(stage_id);
  TensorRedistribution tensor_redistribution(false, true);
  if (tensor_redistribution.Init(inputs[0].tensor_layout(), outputs[0].tensor_layout(), dev_list) == FAILED) {
    MS_LOG(EXCEPTION) << "Failure: tensor_redistribution init failed.";
  }
  if (tensor_redistribution.ComputeCost() == FAILED) {
    MS_LOG(EXCEPTION) << "Failure: tensor_redistribution ComputeCost failed.";
  }
  return (inputs_type_lengths_[0] * tensor_redistribution.comm_cost());
}

// return the per device communication cost in the backward phase.
double ReshapeCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                        int64_t stage_id) const {
  double result = 0.0;
  if (is_parameter_[0]) {
    TensorInfo input1 = inputs[0];
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }
    if (total_device_num != LongToSize(used_device_num)) {
      result = ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
    }
  }
  return result;
}

// Return the per device computation cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double ReshapeCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs,
                                              const std::vector<TensorInfo> &outputs, int64_t stage_id) const {
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  RankList dev_list = g_device_manager->GetDeviceListByStageId(stage_id);
  TensorRedistribution tensor_redistribution(false, true);
  if (tensor_redistribution.Init(inputs[0].tensor_layout(), outputs[0].tensor_layout(), dev_list) == FAILED) {
    MS_LOG(EXCEPTION) << "Failure: tensor_redistribution init failed.";
  }
  if (tensor_redistribution.ComputeCost() == FAILED) {
    MS_LOG(EXCEPTION) << "Failure: tensor_redistribution ComputeCost failed.";
  }
  return (inputs_type_lengths_[0] * tensor_redistribution.computation_cost());
}

// Return the per device computation cost in the backward phase. The cost is calculated according to the bytes
// this operator uses
double ReshapeCost::GetBackwardComputationCost(const std::vector<mindspore::parallel::TensorInfo> &,
                                               const std::vector<mindspore::parallel::TensorInfo> &, int64_t) const {
  return 0.0;
}

void ReshapeCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

void ReshapeCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
  is_inputs_should_in_memory_[1] = is_parameter_[1];
}

double SubCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                          int64_t) const {
  double result = ListProduct(inputs[0].slice_shape()) * static_cast<double>(inputs_type_lengths_[0]) +
                  ListProduct(inputs[1].slice_shape()) * static_cast<double>(inputs_type_lengths_[1]);
  return result;
}

double SubCost::GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                           int64_t stage_id) const {
  double result = 0.0;
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

  if (is_parameter_[0]) {
    TensorInfo input_a_tensor_info = inputs[0];
    Shape input_a_shape = input_a_tensor_info.shape();
    Shape input_a_slice_shape = input_a_tensor_info.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input_a_shape.size(); ++i) {
      used_device_num *= input_a_shape[i] / input_a_slice_shape[i];
    }

    if (total_device_num != LongToSize(used_device_num))
      result += ListProduct(input_a_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
  }

  if (is_parameter_[1]) {
    TensorInfo input_b_tensor_info = inputs[1];
    Shape input_b_shape = input_b_tensor_info.shape();
    Shape input_b_slice_shape = input_b_tensor_info.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input_b_shape.size(); ++i) {
      used_device_num *= input_b_shape[i] / input_b_slice_shape[i];
    }

    if (total_device_num != LongToSize(used_device_num))
      result += ListProduct(input_b_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  }
  return result;
}

double SubCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                    int64_t stage_id) const {
  double result = 0.0;
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

  if (is_parameter_[0]) {
    TensorInfo input_a_tensor_info = inputs[0];
    Shape input_a_shape = input_a_tensor_info.shape();
    Shape input_a_slice_shape = input_a_tensor_info.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input_a_shape.size(); ++i) {
      used_device_num *= input_a_shape[i] / input_a_slice_shape[i];
    }

    if (total_device_num != LongToSize(used_device_num))
      result += ListProduct(input_a_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
  }

  if (is_parameter_[1]) {
    TensorInfo input_b_tensor_info = inputs[1];
    Shape input_b_shape = input_b_tensor_info.shape();
    Shape input_b_slice_shape = input_b_tensor_info.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input_b_shape.size(); ++i) {
      used_device_num *= input_b_shape[i] / input_b_slice_shape[i];
    }

    if (total_device_num != LongToSize(used_device_num))
      result += ListProduct(input_b_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  }

  return result;
}

// Not taking account of output
void SubCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

// Not taking account of input
void SubCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
  is_inputs_should_in_memory_[1] = is_parameter_[1];
}

// Taking account of input
void MulCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  if (is_parameter_[0]) {
    // 'x' is parameter, so it should be in memory.
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      // In this case, if 'y' is not be calculated by the previous operator, then 'y' should be included here.
      is_inputs_should_in_memory_[1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }

  if (is_parameter_[1]) {
    is_inputs_should_in_memory_[1] = true;
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  } else if (is_parameter_involve_[1]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  }
}

// Taking account of output
void DivCost::CalculateOutputInMemory() { is_output_should_in_memory_ = is_parameter_involve_[1]; }

// Taking account of input
void DivCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'y'
  if (is_parameter_[0]) {
    // 'x' is parameter, so it should be in memory.
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      // In this case, if 'y' is not be calculated by the previous operator, then 'y' should be included here.
      is_inputs_should_in_memory_[1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }

  // When calculating 'dy', taking account of 'y'
  if (is_parameter_involve_[1]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }
}

// Taking account of input
void ModCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', not taking account of 'x' and 'y'
  is_inputs_should_in_memory_[0] = is_parameter_[0];
  // When calculating 'dy', taking account of 'x' and 'y'
  if (is_parameter_involve_[1]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }
}

void PowCost::CalculateOutputInMemory() { is_output_should_in_memory_ = is_parameter_involve_[1]; }

void PowCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of both 'x' and 'power'
  if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }
  // When calculating 'dpower', taking account of 'x'
  if (is_parameter_[1]) {
    is_inputs_should_in_memory_[1] = true;
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  } else if (is_parameter_involve_[1]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  }
}

void AssignCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'x'
  if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  }
  // When calculating 'dy', not taking account of 'x' and 'y'
  is_inputs_should_in_memory_[1] = is_parameter_[1];
}

void SigmoidCrossEntropyWithLogitsCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of both 'x' and 'y'
  if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }
  // When calculating 'dy', not taking account of 'x' and 'y'
  if (!is_inputs_should_in_memory_[1]) {
    is_inputs_should_in_memory_[1] = is_parameter_[1];
  }
}

void Atan2Cost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of both 'x' and 'y'; when calculating 'dy', taking account of both 'x' and
  // 'y'
  if (is_parameter_involve_[0] || is_parameter_involve_[1]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }
}

void DivNoNanCost::CalculateOutputInMemory() { is_output_should_in_memory_ = is_parameter_involve_[1]; }

void DivNoNanCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'y'
  if (is_parameter_[0]) {
    // 'x' is parameter, so it should be in memory.
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      // In this case, if 'y' is not be calculated by the previous operator, then 'y' should be included here.
      is_inputs_should_in_memory_[1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }

  // When calculating 'dy', taking account of 'y'
  if (is_parameter_involve_[1]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }
}

void MaximumCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of both 'x' and 'y';
  // when calculating 'dy', taking account of both 'x' and 'y'
  if (is_parameter_involve_[0] || is_parameter_involve_[1]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }
}

void SliceCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'y' and 'z'
  if (is_parameter_[0]) {
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
    if ((prev_output_in_mem.find(SLICE_INPUTS_SIZE - 1) == prev_output_in_mem.end()) ||
        (!prev_output_in_mem.at(SLICE_INPUTS_SIZE - 1))) {
      is_inputs_should_in_memory_[SLICE_INPUTS_SIZE - 1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
    if ((prev_output_in_mem.find(SLICE_INPUTS_SIZE - 1) == prev_output_in_mem.end()) ||
        (!prev_output_in_mem.at(SLICE_INPUTS_SIZE - 1))) {
      is_inputs_should_in_memory_[SLICE_INPUTS_SIZE - 1] = true;
    }
  }

  if (!is_inputs_should_in_memory_[1]) {
    is_inputs_should_in_memory_[1] = is_parameter_[1];
  }
  if (!is_inputs_should_in_memory_[SLICE_INPUTS_SIZE - 1]) {
    is_inputs_should_in_memory_[SLICE_INPUTS_SIZE - 1] = is_parameter_[SLICE_INPUTS_SIZE - 1];
  }
}

void StridedSliceCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'y', 'z' and 'w'
  if (is_parameter_[0]) {
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
    if ((prev_output_in_mem.find(STRIDED_SLICE_INPUTS_SIZE - 2) == prev_output_in_mem.end()) ||
        (!prev_output_in_mem.at(STRIDED_SLICE_INPUTS_SIZE - 2))) {
      is_inputs_should_in_memory_[STRIDED_SLICE_INPUTS_SIZE - 2] = true;
    }
    if ((prev_output_in_mem.find(STRIDED_SLICE_INPUTS_SIZE - 1) == prev_output_in_mem.end()) ||
        (!prev_output_in_mem.at(STRIDED_SLICE_INPUTS_SIZE - 1))) {
      is_inputs_should_in_memory_[STRIDED_SLICE_INPUTS_SIZE - 1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
    if ((prev_output_in_mem.find(STRIDED_SLICE_INPUTS_SIZE - 2) == prev_output_in_mem.end()) ||
        (!prev_output_in_mem.at(STRIDED_SLICE_INPUTS_SIZE - 2))) {
      is_inputs_should_in_memory_[STRIDED_SLICE_INPUTS_SIZE - 2] = true;
    }
    if ((prev_output_in_mem.find(STRIDED_SLICE_INPUTS_SIZE - 1) == prev_output_in_mem.end()) ||
        (!prev_output_in_mem.at(STRIDED_SLICE_INPUTS_SIZE - 1))) {
      is_inputs_should_in_memory_[STRIDED_SLICE_INPUTS_SIZE - 1] = true;
    }
  }

  if (!is_inputs_should_in_memory_[1]) {
    is_inputs_should_in_memory_[1] = is_parameter_[1];
  }
  if (!is_inputs_should_in_memory_[STRIDED_SLICE_INPUTS_SIZE - 2]) {
    is_inputs_should_in_memory_[STRIDED_SLICE_INPUTS_SIZE - 2] = is_parameter_[STRIDED_SLICE_INPUTS_SIZE - 2];
  }
  if (!is_inputs_should_in_memory_[STRIDED_SLICE_INPUTS_SIZE - 1]) {
    is_inputs_should_in_memory_[STRIDED_SLICE_INPUTS_SIZE - 1] = is_parameter_[STRIDED_SLICE_INPUTS_SIZE - 1];
  }
}

void DropOutDoMaskCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

void DropOutDoMaskCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'y'
  if (is_parameter_[0]) {
    // 'x' is parameter, so it should be in memory.
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      // In this case, if 'y' is not be calculated by the previous operator, then 'y' should be included here.
      is_inputs_should_in_memory_[1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }

  if (!is_inputs_should_in_memory_[1]) {
    is_inputs_should_in_memory_[1] = is_parameter_[1];
  }
  is_inputs_should_in_memory_[DROPOUTDOMASK_INPUTS_SIZE - 1] = is_parameter_[DROPOUTDOMASK_INPUTS_SIZE - 1];
}

bool IsDataParallel(const Shape &shape, const Shape &slice_shape, int64_t stage_id) {
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
  auto strategy0 = shape[0] / slice_shape[0];

  return (total_device_num == LongToSize(strategy0));
}

double ReduceSumCost::GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                         int64_t stage_id) const {
  double result = 0.0;
  TensorInfo input0 = inputs[0];
  TensorInfo output0 = outputs[0];
  Shape input0_shape = input0.shape();
  Shape input0_slice_shape = input0.slice_shape();
  if (cross_batch_ && IsDataParallel(input0_shape, input0_slice_shape, stage_id)) {
    return result;
  }
  std::vector<int64_t> dim_list = input0.reduce_dim();
  auto pos = std::find_if(dim_list.begin(), dim_list.end(), [input0_shape, input0_slice_shape](int64_t index) {
    return input0_shape[LongToSize(index)] != input0_slice_shape[LongToSize(index)];
  });
  if (pos != dim_list.end()) {
    result += ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]);
  }

  return result;
}

double ReduceSumCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                          int64_t stage_id) const {
  double result = 0.0;
  if (is_parameter_[0]) {
    TensorInfo input_tensor_info = inputs[0];
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

    Shape input_shape = input_tensor_info.shape();
    Shape input_slice_shape = input_tensor_info.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      used_device_num *= input_shape[i] / input_slice_shape[i];
    }

    if (total_device_num != LongToSize(used_device_num))
      result += ListProduct(input_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
  }

  return result;
}

double ReduceSumCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs,
                                                const std::vector<TensorInfo> &outputs, int64_t stage_id) const {
  double result = 0.0;
  TensorInfo input0 = inputs[0];
  TensorInfo output0 = outputs[0];
  std::vector<int64_t> dim_list = input0.reduce_dim();
  Shape input0_slice_shape = input0.slice_shape();
  Shape input0_shape = input0.shape();
  if (!cross_batch_ || !IsDataParallel(input0_shape, input0_slice_shape, stage_id)) {
    auto pos = std::find_if(dim_list.begin(), dim_list.end(), [input0_shape, input0_slice_shape](int64_t index) {
      return input0_shape[LongToSize(index)] != input0_slice_shape[LongToSize(index)];
    });
    if (pos != dim_list.end()) {
      result += ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]);
    }
  }
  result += ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);

  return result;
}

// Not taking account of output
void ReduceSumCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

void ReduceSumCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'y'
  if (is_parameter_[0]) {
    // 'x' is parameter, so it should be in memory.
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      // In this case, if 'y' is not be calculated by the previous operator, then 'y' should be included here.
      is_inputs_should_in_memory_[1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }

  // Not taking account of 'y'
  if (!is_inputs_should_in_memory_[1]) {
    is_inputs_should_in_memory_[1] = is_parameter_[1];
  }
}

double ReduceMeanCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs,
                                                 const std::vector<TensorInfo> &outputs, int64_t stage_id) const {
  double result = 0.0;
  TensorInfo input0 = inputs[0];
  TensorInfo output0 = outputs[0];
  std::vector<int64_t> dim_list = input0.reduce_dim();
  Shape input0_slice_shape = input0.slice_shape();
  Shape input0_shape = input0.shape();
  if (!cross_batch_ || !IsDataParallel(input0_shape, input0_slice_shape, stage_id)) {
    auto pos = std::find_if(dim_list.begin(), dim_list.end(), [input0_shape, input0_slice_shape](int64_t index) {
      return input0_shape[LongToSize(index)] != input0_slice_shape[LongToSize(index)];
    });
    if (pos != dim_list.end()) {
      result += ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]) * 2.0;
    }
  }
  result += ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);

  return result;
}

void ReduceMinCost::CalculateOutputInMemory() { is_output_should_in_memory_ = is_parameter_involve_[0]; }

void ReduceMinCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'y'
  if (is_parameter_[0]) {
    // 'x' is parameter, so it should be in memory.
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      // In this case, if 'y' is not be calculated by the previous operator, then 'y' should be included here.
      is_inputs_should_in_memory_[1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }

  // Not taking account of 'y'
  if (!is_inputs_should_in_memory_[1]) {
    is_inputs_should_in_memory_[1] = is_parameter_[1];
  }
}

void ArgMaxWithValueCost::CalculateOutputInMemory() { is_output_should_in_memory_ = is_parameter_involve_[0]; }

void ArgMaxWithValueCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'x'
  if (is_parameter_[0]) {
    is_inputs_should_in_memory_[0] = true;
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
  }
}

double DropOutCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                              int64_t) const {
  if (inputs.empty()) {
    return 0.0;
  }
  TensorInfo input0 = inputs[0];
  Shape input0_slice_shape = input0.slice_shape();
  return ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) * DROPOUT_COST_RATE;
}

// return the per device communication cost in the forward phase.
double GatherV2Cost::GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                        int64_t) const {
  // GatherV2Cost does not need communication in the forward phase
  return 0.0;
}

// return the per device communication cost in the backward phase.
double GatherV2Cost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                         int64_t stage_id) const {
  double result = 0.0;
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

  for (size_t j = 0; j < inputs.size(); ++j) {
    if (!is_parameter_[j]) {
      continue;
    }
    TensorInfo input_a_tensor_info = inputs[j];
    Shape input_a_shape = input_a_tensor_info.shape();
    Shape input_a_slice_shape = input_a_tensor_info.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input_a_shape.size(); ++i) {
      used_device_num *= input_a_shape[i] / input_a_slice_shape[i];
    }
    if (total_device_num != LongToSize(used_device_num)) {
      result += ListProduct(input_a_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
    }
  }

  return result;
}

double GatherV2Cost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                               int64_t) const {
  // In forward phase, the computation cost = slice(A) + slice(B)
  Shape input0_slice_shape = inputs[0].slice_shape();
  Shape input1_slice_shape = inputs[1].slice_shape();
  double result = ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) +
                  ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  return result;
}

double GatherV2Cost::GetBackwardComputationCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &,
                                                int64_t) const {
  return 0.0;
}

// Not taking account of output
void GatherV2Cost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

void GatherV2Cost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'y' and 'z'
  if (is_parameter_[0]) {
    // 'x' is parameter, so it should be in memory.
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
    if ((prev_output_in_mem.find(GATHERV2_INPUTS_SIZE - 1) == prev_output_in_mem.end()) ||
        (!prev_output_in_mem.at(GATHERV2_INPUTS_SIZE - 1))) {
      is_inputs_should_in_memory_[GATHERV2_INPUTS_SIZE - 1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
    if ((prev_output_in_mem.find(GATHERV2_INPUTS_SIZE - 1) == prev_output_in_mem.end()) ||
        (!prev_output_in_mem.at(GATHERV2_INPUTS_SIZE - 1))) {
      is_inputs_should_in_memory_[GATHERV2_INPUTS_SIZE - 1] = true;
    }
  }

  if (!is_inputs_should_in_memory_[1]) {
    is_inputs_should_in_memory_[1] = is_parameter_[1];
  }
  if (!is_inputs_should_in_memory_[GATHERV2_INPUTS_SIZE - 1]) {
    is_inputs_should_in_memory_[GATHERV2_INPUTS_SIZE - 1] = is_parameter_[GATHERV2_INPUTS_SIZE - 1];
  }
}

void GetNextCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

void GetNextCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  if (is_inputs_should_in_memory_.size() == 0) {
    return;
  }
  is_inputs_should_in_memory_[0] = is_parameter_[0];
}

double DSDMatmulCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                                int64_t) const {
  double result = 0.0;
  if (inputs_type_lengths_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid inputs type size " << inputs_type_lengths_.size() << " for layer norm cost";
  }

  for (size_t index = 0; index < inputs.size(); ++index) {
    TensorInfo tensor_info = inputs[index];
    Shape slice_shape = tensor_info.slice_shape();
    result += ListProduct(slice_shape) * static_cast<double>(inputs_type_lengths_[index]);
  }
  return result;
}

void DSDMatmulCost::CalculateOutputInMemory() {
  is_output_should_in_memory_ =
    (std::find(is_parameter_involve_.begin(), is_parameter_involve_.end(), true) != is_parameter_involve_.end());
}

void DSDMatmulCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  bool keep_mem =
    (std::find(is_parameter_.begin(), is_parameter_.end(), true) != is_parameter_.end()) ||
    (std::find(is_parameter_involve_.begin(), is_parameter_involve_.end(), true) != is_parameter_involve_.end());
  std::fill(is_inputs_should_in_memory_.begin(), is_inputs_should_in_memory_.end(), keep_mem);
}

void UniqueCost::CalculateOutputInMemory() { is_output_should_in_memory_ = is_parameter_involve_[0]; }

void UniqueCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
}

double LayerNormCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                          int64_t stage_id) const {
  double result = 0.0;
  if (is_parameter_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid parameter size " << is_parameter_.size() << " for layer norm cost";
  }
  if (inputs_type_lengths_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid inputs type size " << inputs_type_lengths_.size() << " for layer norm cost";
  }

  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

  for (size_t index = 0; index < inputs.size(); ++index) {
    if (is_parameter_[index]) {
      TensorInfo tensor_info = inputs[index];
      Shape shape = tensor_info.shape();
      Shape slice_shape = tensor_info.slice_shape();
      int64_t used_device_num = 1;
      for (size_t i = 0; i < shape.size(); ++i) {
        if (slice_shape[i] == 0) {
          MS_LOG(EXCEPTION) << "Invalid slice shape " << ShapeToString(slice_shape);
        }
        used_device_num *= shape[i] / slice_shape[i];
      }
      if (total_device_num != LongToSize(used_device_num)) {
        result += ListProduct(slice_shape) * static_cast<double>(inputs_type_lengths_[index]);
      }
    }
  }
  return result;
}

double LayerNormCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                                int64_t) const {
  double result = 0.0;
  if (inputs_type_lengths_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid inputs type size " << inputs_type_lengths_.size() << " for layer norm cost";
  }

  for (size_t index = 0; index < inputs.size(); ++index) {
    TensorInfo tensor_info = inputs[index];
    Shape slice_shape = tensor_info.slice_shape();
    result += ListProduct(slice_shape) * static_cast<double>(inputs_type_lengths_[index]);
  }
  return result;
}

void LayerNormCost::CalculateOutputInMemory() {
  is_output_should_in_memory_ =
    is_parameter_involve_[0] || is_parameter_involve_[1] || is_parameter_involve_[LAYERNORM_INPUTS_SIZE - 1];
}

void LayerNormCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of both 'x' and 'y'
  // When calculating 'dy', taking account of both 'x' and 'y'
  if (is_parameter_involve_[0] || is_parameter_involve_[1]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }
  is_inputs_should_in_memory_[LAYERNORM_INPUTS_SIZE - 1] = is_parameter_[LAYERNORM_INPUTS_SIZE - 1];
}

double UniqueCost::GetForwardCommCost(const std::vector<TensorInfo> &, const std::vector<TensorInfo> &, int64_t) const {
  return 0.0;
}
double UniqueCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                       int64_t stage_id) const {
  double result = 0.0;
  if (is_parameter_[0]) {
    TensorInfo input = inputs[0];
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
    Shape input_shape = input.shape();
    Shape input_slice_shape = input.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      used_device_num *= input_shape[i] / input_slice_shape[i];
    }
    if (total_device_num != LongToSize(used_device_num)) {
      result = ListProduct(input_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
    }
  }
  return result;
}
double UniqueCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                             int64_t) const {
  // In forward phase, the computation cost = slice(A) + slice(B)
  Shape input_slice_shape = inputs[0].slice_shape();
  double result = ListProduct(input_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
  return result;
}
double UniqueCost::GetBackwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                              int64_t stage_id) const {
  // In backward phase, the computation cost = (0 or 1) allreduce(slice(B))
  double result = 0.0;
  if (is_parameter_[0]) {
    TensorInfo input = inputs[0];  // tensor B
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

    Shape input_shape = input.shape();
    Shape input_slice_shape = input.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      used_device_num *= input_shape[i] / input_slice_shape[i];
    }

    if (total_device_num != LongToSize(used_device_num)) {
      result += ListProduct(input_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
    }
  }
  return result;
}

double GatherV2PCost::GetForwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &outputs,
                                         int64_t) const {
  double result = 0.0;
  if (outputs_type_lengths_.size() != outputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid inputs type size " << inputs_type_lengths_.size() << " for gatherv2 cost";
  }
  // don't split axis
  if (strategy_.at(LongToSize(axis_)) == 1) {
    return result;
  }

  // split axis
  auto param_shape = inputs[0].slice_shape();
  auto index_shape = inputs[1].slice_shape();
  Shape reducescatter_shape = index_shape;
  if (param_shape.size() == 2) {
    reducescatter_shape.push_back(param_shape.at(LongToSize(1 - axis_)));
  }
  result += ListProduct(reducescatter_shape) * static_cast<double>(outputs_type_lengths_[0]);
  return result;
}

double GatherV2PCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                          int64_t stage_id) const {
  double result = 0.0;
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

  for (size_t j = 0; j < inputs.size(); ++j) {
    if (!is_parameter_[j]) {
      continue;
    }
    TensorInfo input_a_tensor_info = inputs[j];
    Shape input_a_shape = input_a_tensor_info.shape();
    Shape input_a_slice_shape = input_a_tensor_info.slice_shape();
    int64_t used_device_num = 1;
    for (size_t i = 0; i < input_a_shape.size(); ++i) {
      used_device_num *= input_a_shape[i] / input_a_slice_shape[i];
    }
    if (total_device_num != LongToSize(used_device_num)) {
      result += ListProduct(input_a_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
    }
  }
  return result;
}

double UniformCandidateSamplerCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs,
                                                              const std::vector<TensorInfo> &, int64_t) const {
  Shape input0_slice_shape = inputs[0].slice_shape();
  if (inputs_type_lengths_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid inputs type size " << inputs_type_lengths_.size()
                      << " for UniformCandidateSampler cost";
  }

  double result = ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);

  return result;
}

void UniformCandidateSamplerCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

void UniformCandidateSamplerCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  is_inputs_should_in_memory_[0] = is_parameter_[0];
}

double GatherV2PCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                                int64_t) const {
  double result = 0.0;
  Shape input0_slice_shape = inputs[0].slice_shape();
  Shape input1_slice_shape = inputs[1].slice_shape();
  if (inputs_type_lengths_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid inputs type size " << inputs_type_lengths_.size() << " for gatherv2 cost";
  }
  // don't split axis
  if (strategy_.at(LongToSize(axis_)) == 1) {
    result += ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) +
              ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  } else {
    // split axis
    result += ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) * GATHERV2_COST_WEIGHT0 +
              ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]) * GATHERV2_COST_WEIGHT1;
  }

  return result;
}

double GatherV2PCost::GetBackwardComputationCost(const std::vector<TensorInfo> &inputs,
                                                 const std::vector<TensorInfo> &outputs, int64_t) const {
  double result = 0.0;
  Shape input1_slice_shape = inputs[1].slice_shape();
  Shape output0_slice_shape = outputs[0].slice_shape();
  // don't split axis
  if (strategy_.at(LongToSize(axis_)) == 1) {
    result += ListProduct(output0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
  } else {
    // split axis
    result += ListProduct(output0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) * GATHERV2_COST_WEIGHT2 +
              ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]) * GATHERV2_COST_WEIGHT3;
  }

  return result;
}

// The forward communication is determined by whether the slice is column split or row split
// The number of segments is actually the shape[0] of the output, which is the cost of the AllReduce
double UnsortedSegmentSumCost::GetForwardCommCost(const std::vector<TensorInfo> &inputs,
                                                  const std::vector<TensorInfo> &outputs, int64_t) const {
  TensorInfo input0 = inputs[0];
  TensorInfo input1 = inputs[1];
  TensorInfo output0 = outputs[0];
  Shape input0_shape = input0.shape();
  Shape input0_slice_shape = inputs[0].slice_shape();
  double result = 0.0;
  if (inputs_type_lengths_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid inputs type size " << inputs_type_lengths_.size() << " for UnsortedSegmentSum cost";
  }
  // If the shape b is not the same as the shape a, we regard it as column slice
  for (size_t i = 0; i < input1.shape().size(); ++i) {
    if (input0_shape[i] != input0_slice_shape[i]) {
      result = ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]);
      return result;
    }
  }
  return result;
}

double UnsortedSegmentSumCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs,
                                                   const std::vector<TensorInfo> &outputs, int64_t) const {
  TensorInfo input0 = inputs[0];
  TensorInfo input1 = inputs[1];
  TensorInfo output0 = outputs[0];
  Shape input0_shape = input0.shape();
  Shape input0_slice_shape = inputs[0].slice_shape();
  double result = 0.0;
  if (inputs_type_lengths_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid inputs type size " << inputs_type_lengths_.size() << " for UnsortedSegmentSum cost";
  }
  if (is_parameter_[0]) {
    // If the forward process has a AllReduce, then the backward also needs one.
    for (size_t i = 0; i < input1.shape().size(); ++i) {
      if (input0_shape[i] != input0_slice_shape[i]) {
        result = ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]);
        return result;
      }
    }
  }
  return result;
}
double UnsortedSegmentSumCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs,
                                                         const std::vector<TensorInfo> &outputs, int64_t) const {
  // In forward phase, the computation cost = slice(A) + slice(B)
  Shape input0_slice_shape = inputs[0].slice_shape();
  Shape input1_slice_shape = inputs[1].slice_shape();
  Shape output_slice_shape = outputs[0].slice_shape();
  double result = ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) +
                  ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]) +
                  ListProduct(output_slice_shape) * static_cast<double>(outputs_type_lengths_[0]);
  return result;
}

// Not taking account of output
void UnsortedSegmentSumCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

// Taking account of input
void UnsortedSegmentSumCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'y'
  if (is_parameter_[0]) {
    is_inputs_should_in_memory_[0] = true;
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  } else if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
  }

  if (!is_inputs_should_in_memory_[1]) {
    is_inputs_should_in_memory_[1] = is_parameter_[1];
  }
  is_inputs_should_in_memory_[UNSORTEDSEGMENTSUM_INPUTS_SIZE - 1] = is_parameter_[UNSORTEDSEGMENTSUM_INPUTS_SIZE - 1];
}

double UnsortedSegmentMinCost::GetForwardCommCost(const std::vector<TensorInfo> &inputs,
                                                  const std::vector<TensorInfo> &outputs, int64_t) const {
  TensorInfo input0 = inputs[0];
  TensorInfo input1 = inputs[1];
  TensorInfo output0 = outputs[0];
  Shape input0_shape = input0.shape();
  Shape input0_slice_shape = inputs[0].slice_shape();
  double result = 0.0;
  if (inputs_type_lengths_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid inputs type size " << inputs_type_lengths_.size()
                      << " for UnsortedSegmentMinCost cost";
  }
  // If the shape b is not the same as the shape a, we regard it as column slice
  // The cost is a AllGather operation, the shape is the same as the output of UnsortedSegmentMin.
  for (size_t i = 0; i < input1.shape().size(); ++i) {
    if (input0_shape[i] != input0_slice_shape[i]) {
      result = ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]);
      return result;
    }
  }
  return result;
}

double UnsortedSegmentMinCost::GetBackwardCommCost(const std::vector<TensorInfo> &inputs,
                                                   const std::vector<TensorInfo> &outputs, int64_t) const {
  TensorInfo input0 = inputs[0];
  TensorInfo input1 = inputs[1];
  TensorInfo output0 = outputs[0];
  Shape input0_shape = input0.shape();
  Shape input0_slice_shape = inputs[0].slice_shape();
  double result = 0.0;
  if (inputs_type_lengths_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid inputs type size " << inputs_type_lengths_.size()
                      << " for UnsortedSegmentMinCost cost";
  }
  if (is_parameter_[0]) {
    // If the forward process has a AllGather, then the backward also needs one ReduceScatter.
    for (size_t i = 0; i < input1.shape().size(); ++i) {
      if (input0_shape[i] != input0_slice_shape[i]) {
        result = ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]);
        return result;
      }
    }
  }
  return result;
}
double UnsortedSegmentMinCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs,
                                                         const std::vector<TensorInfo> &outputs, int64_t) const {
  // In forward phase, the computation cost = slice(A) + slice(B)
  Shape input0_slice_shape = inputs[0].slice_shape();
  Shape input1_slice_shape = inputs[1].slice_shape();
  Shape output_slice_shape = outputs[0].slice_shape();
  // The forward operation is UnsortedSegmentMin + ReudceMin
  double result = ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) +
                  ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]) +
                  ListProduct(output_slice_shape) * static_cast<double>(outputs_type_lengths_[0]) +
                  ListProduct(output_slice_shape) * static_cast<double>(outputs_type_lengths_[0]);  // ReduceMin
  return result;
}

// Taking account of output
void UnsortedSegmentMinCost::CalculateOutputInMemory() { is_output_should_in_memory_ = is_parameter_involve_[0]; }

// Taking account of input
void UnsortedSegmentMinCost::CalculateInputsInMemory(const std::map<size_t, bool> &prev_output_in_mem) {
  // When calculating 'dx', taking account of 'x', 'y' and 'z'
  if (is_parameter_involve_[0]) {
    if ((prev_output_in_mem.find(0) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(0))) {
      is_inputs_should_in_memory_[0] = true;
    }
    if ((prev_output_in_mem.find(1) == prev_output_in_mem.end()) || (!prev_output_in_mem.at(1))) {
      is_inputs_should_in_memory_[1] = true;
    }
    if ((prev_output_in_mem.find(UNSORTEDSEGMENTSUM_INPUTS_SIZE - 1) == prev_output_in_mem.end()) ||
        (!prev_output_in_mem.at(UNSORTEDSEGMENTSUM_INPUTS_SIZE - 1))) {
      is_inputs_should_in_memory_[UNSORTEDSEGMENTSUM_INPUTS_SIZE - 1] = true;
    }
  }
  if (!is_inputs_should_in_memory_[1]) {
    is_inputs_should_in_memory_[1] = is_parameter_[1];
  }
  if (!is_inputs_should_in_memory_[UNSORTEDSEGMENTSUM_INPUTS_SIZE - 1]) {
    is_inputs_should_in_memory_[UNSORTEDSEGMENTSUM_INPUTS_SIZE - 1] = is_parameter_[UNSORTEDSEGMENTSUM_INPUTS_SIZE - 1];
  }
}

// Not taking account of output
void VirtualDatasetCost::CalculateOutputInMemory() { is_output_should_in_memory_ = false; }

// Not taking account of input
void VirtualDatasetCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  for (size_t i = 0; i < is_inputs_should_in_memory_.size(); ++i) {
    is_inputs_should_in_memory_[i] = is_parameter_[i];
  }
}

double MatmulDDSCost::GetForwardComputationCost(const std::vector<TensorInfo> &inputs, const std::vector<TensorInfo> &,
                                                int64_t) const {
  double result = 0.0;
  if (inputs_type_lengths_.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Invalid inputs type size " << inputs_type_lengths_.size() << " for layer norm cost";
  }

  for (size_t index = 0; index < inputs.size(); ++index) {
    TensorInfo tensor_info = inputs[index];
    Shape slice_shape = tensor_info.slice_shape();
    result += ListProduct(slice_shape) * static_cast<double>(inputs_type_lengths_[index]);
  }
  return result;
}

// Not taking account of output
void MatmulDDSCost::CalculateOutputInMemory() {
  is_output_should_in_memory_ =
    (std::find(is_parameter_involve_.begin(), is_parameter_involve_.end(), true) != is_parameter_involve_.end());
}

// Taking account of input
void MatmulDDSCost::CalculateInputsInMemory(const std::map<size_t, bool> &) {
  bool keep_mem =
    (std::find(is_parameter_.begin(), is_parameter_.end(), true) != is_parameter_.end()) ||
    (std::find(is_parameter_involve_.begin(), is_parameter_involve_.end(), true) != is_parameter_involve_.end());
  std::fill(is_inputs_should_in_memory_.begin(), is_inputs_should_in_memory_.end(), keep_mem);
}
}  // namespace parallel
}  // namespace mindspore
