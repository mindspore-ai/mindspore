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

#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include <functional>
#include <numeric>
#include <memory>
#include <utility>
#include <string>
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/shape_util.h"

namespace mindspore {
namespace parallel {
Status TensorRedistribution::Init(const TensorLayout &from, const TensorLayout &to, const RankList &dev_list) {
  from_origin_ = from;
  to_origin_ = to;
  if (from_origin_.tensor_shape().size() != to_origin_.tensor_shape().size()) {
    MS_LOG(ERROR) << "from shape size must be equal to to shape size!";
    MS_LOG(ERROR) << "reshape from_origin_ " << from_origin_.ToString();
    MS_LOG(ERROR) << "reshape to_origin_ " << to_origin_.ToString();
    return Status::FAILED;
  }

  dev_list_ = dev_list;
  from_ = from_origin_.SqueezeShape();
  to_ = to_origin_.SqueezeShape();
  return Status::SUCCESS;
}

RedistributionOpListPtr TensorRedistribution::InferTensorRedistributionOperatorListUnExpand(bool is_cost_model) {
  TensorLayout from_repeat = from_origin_.TransferRepeatLayout();
  TensorLayout to_repeat = to_origin_.TransferRepeatLayout();
  MS_LOG(DEBUG) << "reshape from_repeat " << from_repeat.ToString();
  MS_LOG(DEBUG) << "reshape to_repeat " << to_repeat.ToString();
  MS_LOG(DEBUG) << "reshape from_origin_ " << from_origin_.ToString();
  MS_LOG(DEBUG) << "reshape to_origin_ " << to_origin_.ToString();
  MS_LOG(DEBUG) << "reshape from_ " << from_.ToString();
  MS_LOG(DEBUG) << "reshape to_ " << to_.ToString();
  OperatorVector operator_vector;
  OutPutInfoVector output_info_vector;
  if (InferRedistribution(from_origin_, from_repeat, &operator_vector, &output_info_vector, is_cost_model) ==
      Status::FAILED) {
    return nullptr;
  }
  if (from_repeat.slice_shape().array() != to_repeat.slice_shape().array()) {
    reshape_flag_ = true;
    ConstructOperator constructor;
    constructor.UpdateTensorShape(from_repeat.slice_shape().array());
    Arrangement shape = to_repeat.slice_shape();
    MS_LOG(DEBUG) << "reshape " << shape.ToString();
    if (constructor.ReshapeOP(shape.array()) == Status::FAILED) {
      return nullptr;
    } else {
      operator_vector.push_back(constructor.GetOperator());
      output_info_vector.push_back(std::make_pair(false, 0));
    }
  }
  if (InferRedistribution(to_repeat, to_origin_, &operator_vector, &output_info_vector, is_cost_model) ==
      Status::FAILED) {
    return nullptr;
  }
  return std::make_shared<std::pair<OperatorVector, OutPutInfoVector>>(
    std::make_pair(operator_vector, output_info_vector));
}

RedistributionOpListPtr TensorRedistribution::InferTensorRedistributionOperatorList(bool is_cost_model) {
  // Step 1: Match device arrangement between from_ and to_
  RedistributionLayoutTransfer layout_transfer;
  Status status = layout_transfer.Init(from_, to_);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  TensorLayout from_layout;
  TensorLayout to_layout;
  if (layout_transfer.IsDynamicShape()) {
    from_layout = layout_transfer.from_in();
    to_layout = layout_transfer.to_in();
  } else {
    std::shared_ptr<ReshapeLayoutTransfer> ptr = layout_transfer.UnifyDeviceArrangementAndTensorShape();
    if (ptr == nullptr) {
      MS_LOG(ERROR) << "Infer tensor layout return nullptr!";
      return nullptr;
    }
    if (!ptr->ExpandAble()) {
      expand_able_ = false;
      return InferTensorRedistributionOperatorListUnExpand(is_cost_model);
    }
    from_layout = ptr->from_in();
    to_layout = ptr->to_in();
  }
  MS_LOG(DEBUG) << "reshape from_layout " << from_layout.ToString();
  MS_LOG(DEBUG) << "reshape to_layout " << to_layout.ToString();
  MS_LOG(DEBUG) << "reshape from_origin_ " << from_origin_.ToString();
  MS_LOG(DEBUG) << "reshape to_origin_ " << to_origin_.ToString();
  MS_LOG(DEBUG) << "reshape from_ " << from_.ToString();
  MS_LOG(DEBUG) << "reshape to_ " << to_.ToString();
  // Step 2: Infer redistribution and insert operators
  OperatorVector operator_vector;
  OutPutInfoVector output_info_vector;
  if (InferRedistribution(from_layout, to_layout, &operator_vector, &output_info_vector, is_cost_model) !=
      Status::SUCCESS) {
    return nullptr;
  }
  // Step 3: Infer reshape and insert operators
  if (InferReshape(from_layout, to_layout, &operator_vector, &output_info_vector) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Construct Reshape operator failed!";
    return nullptr;
  }
  return std::make_shared<std::pair<OperatorVector, OutPutInfoVector>>(
    std::make_pair(operator_vector, output_info_vector));
}

Status TensorRedistribution::InferReshape(const TensorLayout &from_layout, const TensorLayout &to_layout,
                                          OperatorVector *const operator_vector,
                                          OutPutInfoVector *const output_info_vector) {
  MS_EXCEPTION_IF_NULL(operator_vector);
  MS_EXCEPTION_IF_NULL(output_info_vector);
  ConstructOperator constructor;
  if (operator_list_.empty()) {
    if (from_origin_.slice_shape().array() != to_origin_.slice_shape().array() || keep_reshape_) {
      reshape_flag_ = true;
      constructor.UpdateTensorShape(from_origin_.slice_shape().array());
      Arrangement shape = to_origin_.slice_shape();
      MS_LOG(DEBUG) << "reshape " << shape.ToString();
      if (constructor.ReshapeOP(shape.array()) == Status::FAILED) {
        return Status::FAILED;
      } else {
        (void)operator_vector->insert(operator_vector->cbegin(), constructor.GetOperator());
        (void)output_info_vector->insert(output_info_vector->cbegin(), std::make_pair(false, 0));
      }
    }
    return Status::SUCCESS;
  }

  if (from_origin_.slice_shape().array() != from_layout.slice_shape().array()) {
    reshape_flag_ = true;
    constructor.UpdateTensorShape(from_origin_.slice_shape().array());
    Arrangement shape = from_layout.slice_shape();
    MS_LOG(DEBUG) << "reshape " << shape.ToString();
    if (constructor.ReshapeOP(shape.array()) == Status::FAILED) {
      return Status::FAILED;
    } else {
      (void)operator_vector->insert(operator_vector->cbegin(), constructor.GetOperator());
      (void)output_info_vector->insert(output_info_vector->cbegin(), std::make_pair(false, 0));
    }
  }

  if (from_origin_.base_slice_shape().array() != from_origin_.slice_shape().array()) {
    reshape_flag_ = true;
    constructor.UpdateTensorShape(from_origin_.base_slice_shape().array());
    Arrangement shape = from_origin_.slice_shape();
    MS_LOG(DEBUG) << "reshape " << shape.ToString();
    if (constructor.ReshapeOP(shape.array()) == Status::FAILED) {
      return Status::FAILED;
    } else {
      (void)operator_vector->insert(operator_vector->cbegin(), constructor.GetOperator());
      (void)output_info_vector->insert(output_info_vector->cbegin(), std::make_pair(false, 0));
    }
  }

  if (to_origin_.slice_shape().array() != to_layout.slice_shape().array()) {
    reshape_flag_ = true;
    constructor.UpdateTensorShape(to_layout.slice_shape().array());
    Arrangement shape = to_origin_.slice_shape();
    MS_LOG(DEBUG) << "step_parallel to reshape " << shape.ToString();
    if (constructor.ReshapeOP(shape.array()) == Status::FAILED) {
      return Status::FAILED;
    } else {
      (void)operator_vector->insert(operator_vector->cend(), constructor.GetOperator());
      (void)output_info_vector->insert(output_info_vector->cend(), std::make_pair(false, 0));
    }
  }

  if (to_origin_.slice_shape().array() != to_origin_.base_slice_shape().array()) {
    reshape_flag_ = true;
    constructor.UpdateTensorShape(to_origin_.slice_shape().array());
    Arrangement shape = to_origin_.base_slice_shape();
    MS_LOG(DEBUG) << "step_parallel to reshape " << shape.ToString();
    if (constructor.ReshapeOP(shape.array()) == Status::FAILED) {
      return Status::FAILED;
    } else {
      (void)operator_vector->insert(operator_vector->cend(), constructor.GetOperator());
      (void)output_info_vector->insert(output_info_vector->cend(), std::make_pair(false, 0));
    }
  }
  return Status::SUCCESS;
}

Status TensorRedistribution::InferRedistribution(const TensorLayout &from_layout, const TensorLayout &to_layout,
                                                 OperatorVector *const operator_vector,
                                                 OutPutInfoVector *const output_info_vector, bool is_cost_model) {
  MS_EXCEPTION_IF_NULL(operator_vector);
  MS_EXCEPTION_IF_NULL(output_info_vector);
  RedistributionOperatorInfer operator_infer(construct_op_flag_);
  if (operator_infer.Init(from_layout, to_layout.tensor_map(), dev_list_, is_cost_model) == Status::FAILED) {
    MS_LOG(ERROR) << "Init operatorInfer failed";
    return Status::FAILED;
  }
  if (operator_infer.InferRedistributionOperator() != Status::SUCCESS) {
    MS_LOG(ERROR) << "Infer redistribution failed";
    return Status::FAILED;
  } else {
    for (auto op : operator_infer.operator_vector()) {
      (void)operator_vector->insert(operator_vector->cend(), op);
    }
    for (auto info : operator_infer.output_info_vector()) {
      (void)output_info_vector->insert(output_info_vector->cend(), info);
    }
    for (auto opc : operator_infer.operator_list()) {
      (void)operator_list_.insert(operator_list_.cend(), opc);
    }
  }
  return Status::SUCCESS;
}

Status TensorRedistribution::ComputeCost() {
  RedistributionOpListPtr redistribution_oplist_ptr = InferTensorRedistributionOperatorList(true);
  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG(ERROR) << "Failure: InferTensorRedistribution failed";
    return Status::FAILED;
  }
  // Compute redistribution communication cost and computation cost
  for (auto &op_cost : operator_list_) {
    OperatorR op = op_cost.first;
    Shape slice_shape = op_cost.second;
    double prod =
      std::accumulate(slice_shape.begin(), slice_shape.end(), static_cast<double>(1.0), std::multiplies<double>());
    std::string str = op.first;
    if (str == PERMUTE_BY_AXIS && ComputePermuteCost(prod, op.second) != Status::SUCCESS) {
      return Status::FAILED;
    } else if (str == CONCAT_BY_AXIS && ComputeConcatCost(prod, op.second) != Status::SUCCESS) {
      return Status::FAILED;
    } else {
      // There is only computation cost in SplitByAxis.
      // computation cost = before_slice_shape
      computation_cost_ += prod;
      // This addition may be erroneous
      memory_cost_ += prod;
    }
  }
  if (reshape_flag()) {
    Shape prev_shape;
    if (expand_able_) {
      prev_shape = from_.slice_shape().array();
    } else {
      prev_shape = from_.tensor_shape().array();
    }
    double prev_prod =
      std::accumulate(prev_shape.begin(), prev_shape.end(), static_cast<double>(1.0), std::multiplies<double>());
    computation_cost_ += COST_FACTOR * prev_prod;
    memory_cost_ += COST_FACTOR * prev_prod;
  }
  return Status::SUCCESS;
}

Status TensorRedistribution::ComputePermuteCost(double input_size, const Shape &attrs) {
  // Since AlltoAll is a virtual operator, the expanded operators are used here to compute cost.
  // communication cost = all_gather + reduce_scatter = before_slice_shape + after_slice_shape
  if (attrs.size() < TRANSFER_PERMUTE_ARGS_SIZE) {
    MS_LOG(ERROR) << "attrs size should not be less than 5!";
    return Status::FAILED;
  }
  forward_comm_cost_ += input_size * ALLTOALL_SCALE_FACTOR;
  backward_comm_cost_ += input_size * ALLTOALL_SCALE_FACTOR;
  comm_cost_ += COST_FACTOR * input_size * ALLTOALL_SCALE_FACTOR;
  int64_t concat_dim = attrs[TRANSFER_PERMUTE_CONCAT_DIM_INDEX];
  if (concat_dim == 0) {
    // memory cost = all_gather
    computation_cost_ += input_size;
    memory_cost_ += input_size;
  } else {
    // memory cost = all_gather + split + concat
    int64_t dev_num = attrs[TRANSFER_PERMUTE_DEV_NUM_INDEX];
    computation_cost_ += (input_size + input_size * dev_num + input_size * dev_num);
    memory_cost_ += (input_size * dev_num + input_size * dev_num + input_size);
  }
  return Status::SUCCESS;
}

Status TensorRedistribution::ComputeConcatCost(double input_size, const Shape &attrs) {
  // communication cost = all_gather + reduce_scatter = before_slice_shape + after_slice_shape
  // computation cost = before_slice_shape
  if (attrs.size() < TRANSFER_CONCAT_ARGS_SIZE) {
    MS_LOG(ERROR) << "op.second size should not be less than 3!";
    return Status::FAILED;
  }
  double dev_num = attrs[TRANSFER_CONCAT_SPLIT_COUNT_INDEX];
  // here, communication cost = all_gather + reduce_scatter
  forward_comm_cost_ += input_size * dev_num * ALLGATHER_REDUCESCATTER_SCALE_FACTOR;
  backward_comm_cost_ += input_size * ALLGATHER_REDUCESCATTER_SCALE_FACTOR;
  comm_cost_ += input_size * (dev_num + 1.0) * ALLGATHER_REDUCESCATTER_SCALE_FACTOR;
  int64_t concat_dim = attrs[TRANSFER_CONCAT_TENSOR_DIM_INDEX];
  if (concat_dim == 0) {
    // computation cost = all_gather
    computation_cost_ += input_size;
    memory_cost_ += input_size * dev_num;
  } else {
    // computation cost = all_gather + split + concat
    computation_cost_ += (input_size + input_size * dev_num + input_size * dev_num);
    memory_cost_ += (input_size * dev_num + input_size * dev_num + input_size);
  }
  return Status::SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
