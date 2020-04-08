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

#include "parallel/auto_parallel/operator_costmodel.h"

#include <random>
#include "parallel/device_matrix.h"
#include "parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
void OperatorCost::set_is_parameter(const std::vector<bool>& is_parameter) { is_parameter_ = is_parameter; }

void OperatorCost::SetInputAndOutputTypeLength(const std::vector<size_t>& input_lengths,
                                               const std::vector<size_t>& output_lengths) {
  inputs_type_lengths_ = input_lengths;
  outputs_type_lengths_ = output_lengths;
}

// return the per device communication cost in the forward phase.
double MatMulCost::GetForwardCommCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>& outputs,
                                      const int32_t&) const {
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
double MatMulCost::GetBackwardCommCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                       const int32_t& stage_id) const {
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
    int32_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }

    if (total_device_num != IntToSize(used_device_num))
      result += ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  }

  return result;
}

// Return the per device memory cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double MatMulCost::GetForwardComputationCost(const std::vector<TensorInfo>& inputs,
                                             const std::vector<TensorInfo>& outputs, const int32_t&) const {
  // In forward phase, the memory cost = slice(A) + slice(B) + (0 or 1) allreduce(slice(C))
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

// Return the per device memory cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double MatMulCost::GetBackwardComputationCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                              const int32_t& stage_id) const {
  // In backward phase, the memory cost = (0 or 1) allreduce(slice(B))
  double result = 0.0;
  if (is_parameter_[1]) {
    TensorInfo input1 = inputs[1];  // tensor B
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int32_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }

    if (total_device_num != IntToSize(used_device_num))
      result += ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  }

  return result;
}

// Return the per device communication cost in the forward phase.
double ActivationCost::GetForwardCommCost(const std::vector<TensorInfo>&, const std::vector<TensorInfo>&,
                                          const int32_t&) const {
  // ReLU is the element-wise operator, thus it does not need communication in the forward phase
  return 0.0;
}

// Return the per device communication cost in the backward phase.
double ActivationCost::GetBackwardCommCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                           const int32_t& stage_id) const {
  double result = 0.0;
  if (is_parameter_[0]) {
    TensorInfo input1 = inputs[0];
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int32_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }
    if (total_device_num != IntToSize(used_device_num)) {
      result = ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
    }
  }
  return result;
}

// Return the per memory cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double ActivationCost::GetForwardComputationCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                                 const int32_t&) const {
  TensorInfo input0_info = inputs[0];
  Shape input0_slice_shape = input0_info.slice_shape();
  return ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
}

// Return the per memory cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double ActivationCost::GetBackwardComputationCost(const std::vector<TensorInfo>&, const std::vector<TensorInfo>&,
                                                  const int32_t&) const {
  return 0.0;
}

// Return the per device communication cost in the forward phase.
double SoftmaxCost::GetForwardCommCost(const std::vector<TensorInfo>&, const std::vector<TensorInfo>&,
                                       const int32_t&) const {
  // In the forward phase, the communication cost = 0
  return 0.0;
}

// Return the per device communication cost in the backward phase.
double SoftmaxCost::GetBackwardCommCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                        const int32_t& stage_id) const {
  double result = 0.0;
  if (is_parameter_[0]) {
    TensorInfo input1 = inputs[0];
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int32_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }
    if (total_device_num != IntToSize(used_device_num)) {
      result = ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
    }
  }
  return result;
}

// Return the per memory cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double SoftmaxCost::GetForwardComputationCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                              const int32_t&) const {
  // In the forward phase, the memory cost = slice(A)
  TensorInfo input0 = inputs[0];
  Shape input0_slice_shape = input0.slice_shape();
  return ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
}

// Return the per memory cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double SoftmaxCost::GetBackwardComputationCost(const std::vector<mindspore::parallel::TensorInfo>&,
                                               const std::vector<mindspore::parallel::TensorInfo>&,
                                               const int32_t&) const {
  return 0.0;
}

// return the per device communication cost in the forward phase.
double TmpIdentityCost::GetForwardCommCost(const std::vector<mindspore::parallel::TensorInfo>&,
                                           const std::vector<mindspore::parallel::TensorInfo>&, const int32_t&) const {
  // Identity is the element-wise operator, thus it does not need communication in the forward phase
  return 0.0;
}

// return the per device communication cost in the backward phase.
double TmpIdentityCost::GetBackwardCommCost(const std::vector<mindspore::parallel::TensorInfo>&,
                                            const std::vector<mindspore::parallel::TensorInfo>&, const int32_t&) const {
  // Identity is the element-wise operator, thus it does not need communication in the backward phase
  return 0.0;
}

// Return the per memory cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double TmpIdentityCost::GetForwardComputationCost(const std::vector<mindspore::parallel::TensorInfo>& inputs,
                                                  const std::vector<mindspore::parallel::TensorInfo>&,
                                                  const int32_t&) const {
  TensorInfo input0_info = inputs[0];
  Shape input0_slice_shape = input0_info.slice_shape();
  return ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
}

// Return the per memory cost in the backward phase. The cost is calculated according to the bytes
// this operator uses
double TmpIdentityCost::GetBackwardComputationCost(const std::vector<mindspore::parallel::TensorInfo>&,
                                                   const std::vector<mindspore::parallel::TensorInfo>&,
                                                   const int32_t&) const {
  return 0.0;
}

double BatchParallelCost::GetForwardComputationCost(const std::vector<mindspore::parallel::TensorInfo>& inputs,
                                                    const std::vector<mindspore::parallel::TensorInfo>&,
                                                    const int32_t&) const {
  double cost = 0.0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    cost += ListProduct(inputs[i].slice_shape()) * static_cast<double>(inputs_type_lengths_[i]);
  }
  return cost;
}

double BatchParallelCost::GetBackwardComputationCost(const std::vector<mindspore::parallel::TensorInfo>&,
                                                     const std::vector<mindspore::parallel::TensorInfo>&,
                                                     const int32_t&) const {
  return 0.0;
}

// return the per device communication cost in the forward phase.
double PReLUCost::GetForwardCommCost(const std::vector<TensorInfo>&, const std::vector<TensorInfo>&,
                                     const int32_t&) const {
  // prelu does not need communication in the forward phase
  return 0.0;
}

// return the per device communication cost in the backward phase.
double PReLUCost::GetBackwardCommCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                      const int32_t& stage_id) const {
  double result = 0.0;
  if (is_parameter_[1]) {
    TensorInfo input1 = inputs[1];
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int32_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }
    if (total_device_num != IntToSize(used_device_num)) {
      result = ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
    }
  }
  return result;
}

// Return the per memory cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double PReLUCost::GetForwardComputationCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                            const int32_t&) const {
  // In forward phase, the memory cost = slice(A) + slice(B)
  Shape input0_slice_shape = inputs[0].slice_shape();
  Shape input1_slice_shape = inputs[1].slice_shape();
  double result = ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) +
                  ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  return result;
}

// Return the per memory cost in the backward phase. The cost is calculated according to the bytes
// this operator uses
double PReLUCost::GetBackwardComputationCost(const std::vector<mindspore::parallel::TensorInfo>& inputs,
                                             const std::vector<mindspore::parallel::TensorInfo>&,
                                             const int32_t& stage_id) const {
  // In backward phase, the memory cost = (0 or 1) allreduce(slice(B))
  double result = 0.0;
  if (is_parameter_[1]) {
    TensorInfo input1 = inputs[1];  // tensor B
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

    Shape input1_shape = input1.shape();
    Shape input1_slice_shape = input1.slice_shape();
    int32_t used_device_num = 1;
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      used_device_num *= input1_shape[i] / input1_slice_shape[i];
    }

    if (total_device_num != IntToSize(used_device_num)) {
      result += ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
    }
  }
  return result;
}

// return the per device communication cost in the forward phase.
double OneHotCost::GetForwardCommCost(const std::vector<TensorInfo>&, const std::vector<TensorInfo>&,
                                      const int32_t&) const {
  // onehot does not need communication in the forward phase
  return 0.0;
}

// return the per device communication cost in the backward phase.
double OneHotCost::GetBackwardCommCost(const std::vector<TensorInfo>&, const std::vector<TensorInfo>&,
                                       const int32_t&) const {
  // onehot does not need communication in the backward phase
  return 0.0;
}

// Return the per memory cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double OneHotCost::GetForwardComputationCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                             const int32_t&) const {
  // In onehot's forward phase, the memory cost = slice(A)
  Shape input0_slice_shape = inputs[0].slice_shape();
  return ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
}

// Return the per memory cost in the backward phase. The cost is calculated according to the bytes
// this operator uses
double OneHotCost::GetBackwardComputationCost(const std::vector<TensorInfo>&, const std::vector<TensorInfo>&,
                                              const int32_t&) const {
  return 0.0;
}

// return the per device communication cost in the forward phase.
double SoftmaxCrossEntropyWithLogitsCost::GetForwardCommCost(const std::vector<TensorInfo>&,
                                                             const std::vector<TensorInfo>&, const int32_t&) const {
  // SoftmaxCrossEntropyWithLogitsCost does not need communication in the forward phase
  return 0.0;
}

// return the per device communication cost in the backward phase.
double SoftmaxCrossEntropyWithLogitsCost::GetBackwardCommCost(const std::vector<TensorInfo>&,
                                                              const std::vector<TensorInfo>&, const int32_t&) const {
  // SoftmaxCrossEntropyWithLogitsCost does not need communication in the backward phase
  return 0.0;
}

// Return the per memory cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double SoftmaxCrossEntropyWithLogitsCost::GetForwardComputationCost(const std::vector<TensorInfo>& inputs,
                                                                    const std::vector<TensorInfo>&,
                                                                    const int32_t&) const {
  // In forward phase, the memory cost = slice(A) + slice(B)
  Shape input0_slice_shape = inputs[0].slice_shape();
  Shape input1_slice_shape = inputs[1].slice_shape();
  double result = ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) +
                  ListProduct(input1_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  return result;
}

// Return the per memory cost in the backward phase. The cost is calculated according to the bytes
// this operator uses
double SoftmaxCrossEntropyWithLogitsCost::GetBackwardComputationCost(const std::vector<TensorInfo>&,
                                                                     const std::vector<TensorInfo>&,
                                                                     const int32_t&) const {
  return 0.0;
}

// return the per device communication cost in the forward phase.
double ReshapeCost::GetForwardCommCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>& outputs,
                                       const int32_t& stage_id) const {
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  RankList dev_list = g_device_manager->GetDeviceListByStageId(stage_id);
  TensorRedistribution tensor_redistribution;
  if (tensor_redistribution.Init(inputs[0].tensor_layout(), outputs[0].tensor_layout(), dev_list) == FAILED) {
    MS_LOG(EXCEPTION) << "Failure: tensor_redistribution init failed.";
  }
  if (tensor_redistribution.ComputeCost() == FAILED) {
    MS_LOG(EXCEPTION) << "Failure: tensor_redistribution ComputeCost failed.";
  }
  return (inputs_type_lengths_[0] * tensor_redistribution.comm_cost());
}

// return the per device communication cost in the backward phase.
double ReshapeCost::GetBackwardCommCost(const std::vector<TensorInfo>&, const std::vector<TensorInfo>&,
                                        const int32_t&) const {
  return 0.0;
}

// Return the per memory cost in the forward phase. The cost is calculated according to the bytes
// this operator uses
double ReshapeCost::GetForwardComputationCost(const std::vector<TensorInfo>& inputs,
                                              const std::vector<TensorInfo>& outputs, const int32_t& stage_id) const {
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  RankList dev_list = g_device_manager->GetDeviceListByStageId(stage_id);
  TensorRedistribution tensor_redistribution;
  if (tensor_redistribution.Init(inputs[0].tensor_layout(), outputs[0].tensor_layout(), dev_list) == FAILED) {
    MS_LOG(EXCEPTION) << "Failure: tensor_redistribution init failed.";
  }
  if (tensor_redistribution.ComputeCost() == FAILED) {
    MS_LOG(EXCEPTION) << "Failure: tensor_redistribution ComputeCost failed.";
  }
  return (inputs_type_lengths_[0] * tensor_redistribution.computation_cost());
}

// Return the per memory cost in the backward phase. The cost is calculated according to the bytes
// this operator uses
double ReshapeCost::GetBackwardComputationCost(const std::vector<mindspore::parallel::TensorInfo>&,
                                               const std::vector<mindspore::parallel::TensorInfo>&,
                                               const int32_t&) const {
  return 0.0;
}

double ArithmeticCost::GetForwardComputationCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                                 const int32_t&) const {
  double result;
  result = ListProduct(inputs[0].slice_shape()) * static_cast<double>(inputs_type_lengths_[0]) +
           ListProduct(inputs[1].slice_shape()) * static_cast<double>(inputs_type_lengths_[1]);
  return result;
}

double ArithmeticCost::GetBackwardComputationCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                                  const int32_t& stage_id) const {
  double result = 0.0;
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

  if (is_parameter_[0]) {
    TensorInfo input_a_tensor_info = inputs[0];
    Shape input_a_shape = input_a_tensor_info.shape();
    Shape input_a_slice_shape = input_a_tensor_info.slice_shape();
    int32_t used_device_num = 1;
    for (size_t i = 0; i < input_a_shape.size(); ++i) {
      used_device_num *= input_a_shape[i] / input_a_slice_shape[i];
    }

    if (total_device_num != IntToSize(used_device_num))
      result += ListProduct(input_a_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
  }

  if (is_parameter_[1]) {
    TensorInfo input_b_tensor_info = inputs[1];
    Shape input_b_shape = input_b_tensor_info.shape();
    Shape input_b_slice_shape = input_b_tensor_info.slice_shape();
    int32_t used_device_num = 1;
    for (size_t i = 0; i < input_b_shape.size(); ++i) {
      used_device_num *= input_b_shape[i] / input_b_slice_shape[i];
    }

    if (total_device_num != IntToSize(used_device_num))
      result += ListProduct(input_b_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  }
  return result;
}

double ArithmeticCost::GetBackwardCommCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                           const int32_t& stage_id) const {
  double result = 0.0;
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

  if (is_parameter_[0]) {
    TensorInfo input_a_tensor_info = inputs[0];
    Shape input_a_shape = input_a_tensor_info.shape();
    Shape input_a_slice_shape = input_a_tensor_info.slice_shape();
    int32_t used_device_num = 1;
    for (size_t i = 0; i < input_a_shape.size(); ++i) {
      used_device_num *= input_a_shape[i] / input_a_slice_shape[i];
    }

    if (total_device_num != IntToSize(used_device_num))
      result += ListProduct(input_a_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
  }

  if (is_parameter_[1]) {
    TensorInfo input_b_tensor_info = inputs[1];
    Shape input_b_shape = input_b_tensor_info.shape();
    Shape input_b_slice_shape = input_b_tensor_info.slice_shape();
    int32_t used_device_num = 1;
    for (size_t i = 0; i < input_b_shape.size(); ++i) {
      used_device_num *= input_b_shape[i] / input_b_slice_shape[i];
    }

    if (total_device_num != IntToSize(used_device_num))
      result += ListProduct(input_b_slice_shape) * static_cast<double>(inputs_type_lengths_[1]);
  }

  return result;
}

bool IsDataParallel(const Shape& shape, const Shape& slice_shape, const int32_t& stage_id) {
  CheckGlobalDeviceManager();
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
  auto strategy0 = shape[0] / slice_shape[0];

  return (total_device_num == IntToSize(strategy0));
}

double ReduceMethodCost::GetForwardCommCost(const std::vector<TensorInfo>& inputs,
                                            const std::vector<TensorInfo>& outputs, const int32_t& stage_id) const {
  double result = 0.0;
  TensorInfo input0 = inputs[0];
  TensorInfo output0 = outputs[0];
  Shape input0_shape = input0.shape();
  Shape input0_slice_shape = input0.slice_shape();
  if (cross_batch_ && IsDataParallel(input0_shape, input0_slice_shape, stage_id)) {
    return result;
  }
  std::vector<int32_t> dim_list = input0.reduce_dim();
  std::vector<int>::iterator pos;
  pos = std::find_if(dim_list.begin(), dim_list.end(), [input0_shape, input0_slice_shape](const int32_t& index) {
    return input0_shape[IntToSize(index)] != input0_slice_shape[IntToSize(index)];
  });
  if (pos != dim_list.end()) {
    result += ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]);
  }

  return result;
}

double ReduceMethodCost::GetBackwardCommCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                             const int32_t& stage_id) const {
  double result = 0.0;
  if (is_parameter_[0]) {
    TensorInfo input_tensor_info = inputs[0];
    CheckGlobalDeviceManager();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(stage_id).size();

    Shape input_shape = input_tensor_info.shape();
    Shape input_slice_shape = input_tensor_info.slice_shape();
    int32_t used_device_num = 1;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      used_device_num *= input_shape[i] / input_slice_shape[i];
    }

    if (total_device_num != IntToSize(used_device_num))
      result += ListProduct(input_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);
  }

  return result;
}

double ReduceMethodCost::GetForwardComputationCost(const std::vector<TensorInfo>& inputs,
                                                   const std::vector<TensorInfo>& outputs,
                                                   const int32_t& stage_id) const {
  double result = 0.0;
  TensorInfo input0 = inputs[0];
  TensorInfo output0 = outputs[0];
  std::vector<int32_t> dim_list = input0.reduce_dim();
  Shape input0_slice_shape = input0.slice_shape();
  Shape input0_shape = input0.shape();
  if (!cross_batch_ || !IsDataParallel(input0_shape, input0_slice_shape, stage_id)) {
    std::vector<int>::iterator pos;
    pos = std::find_if(dim_list.begin(), dim_list.end(), [input0_shape, input0_slice_shape](const int32_t& index) {
      return input0_shape[IntToSize(index)] != input0_slice_shape[IntToSize(index)];
    });
    if (pos != dim_list.end()) {
      result += ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]);
    }
  }
  result += ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);

  return result;
}

double ReduceMeanCost::GetForwardComputationCost(const std::vector<TensorInfo>& inputs,
                                                 const std::vector<TensorInfo>& outputs,
                                                 const int32_t& stage_id) const {
  double result = 0.0;
  TensorInfo input0 = inputs[0];
  TensorInfo output0 = outputs[0];
  std::vector<int32_t> dim_list = input0.reduce_dim();
  Shape input0_slice_shape = input0.slice_shape();
  Shape input0_shape = input0.shape();
  if (!cross_batch_ || !IsDataParallel(input0_shape, input0_slice_shape, stage_id)) {
    std::vector<int>::iterator pos;
    pos = std::find_if(dim_list.begin(), dim_list.end(), [input0_shape, input0_slice_shape](const int32_t& index) {
      return input0_shape[IntToSize(index)] != input0_slice_shape[IntToSize(index)];
    });
    if (pos != dim_list.end()) {
      result += ListProduct(output0.slice_shape()) * static_cast<double>(outputs_type_lengths_[0]) * 2.0;
    }
  }
  result += ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]);

  return result;
}

double DropOutCost::GetForwardComputationCost(const std::vector<TensorInfo>& inputs, const std::vector<TensorInfo>&,
                                              const int32_t&) const {
  if (inputs.empty()) {
    return 0.0;
  }
  TensorInfo input0 = inputs[0];
  Shape input0_slice_shape = input0.slice_shape();
  return ListProduct(input0_slice_shape) * static_cast<double>(inputs_type_lengths_[0]) * DROPOUT_COST_RATE;
}
}  // namespace parallel
}  // namespace mindspore
