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

#include "parallel/tensor_layout/tensor_redistribution.h"
#include <cfloat>
#include <functional>
#include <numeric>
#include "parallel/status.h"
#include "parallel/tensor_layout/shape_util.h"
#include "common/utils.h"

namespace mindspore {
namespace parallel {

Status TensorRedistribution::Init(const TensorLayout& from, const TensorLayout& to, const RankList& dev_list) {
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

RedistributionOpListPtr TensorRedistribution::InferTensorRedistributionOperatorList() {
  // Step 1: Match device arrangement between from_ and to_
  RedistributionLayoutTransfer layout_transfer;
  Status status = layout_transfer.Init(from_, to_);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  std::shared_ptr<ReshapeLayoutTransfer> ptr = layout_transfer.UnifyDeviceArrangementAndTensorShape();
  if (ptr == nullptr) {
    MS_LOG(ERROR) << "Infer tensor layout return nullptr!";
    return nullptr;
  }
  TensorLayout from_layout = ptr->from_in();
  TensorLayout to_layout = ptr->to_in();
  MS_LOG(DEBUG) << "reshape from_layout " << from_layout.ToString();
  MS_LOG(DEBUG) << "reshape to_layout " << to_layout.ToString();
  MS_LOG(DEBUG) << "reshape from_origin_ " << from_origin_.ToString();
  MS_LOG(DEBUG) << "reshape to_origin_ " << to_origin_.ToString();
  MS_LOG(DEBUG) << "reshape from_ " << from_.ToString();
  MS_LOG(DEBUG) << "reshape to_ " << to_.ToString();
  // Step 2: Infer redistribution and insert operators
  RedistributionOperatorInfer operator_infer(construct_op_flag_);
  if (operator_infer.Init(from_layout, to_layout.tensor_map(), dev_list_) == Status::FAILED) {
    MS_LOG(ERROR) << "Init operatorInfer failed!";
    return nullptr;
  }
  OperatorVector operator_vector;
  OutPutInfoVector output_info_vector;
  if (operator_infer.InferRedistributionOperator() != Status::SUCCESS) {
    MS_LOG(ERROR) << "Infer redistribution failed!";
    return nullptr;
  } else {
    operator_vector = operator_infer.operator_vector();
    output_info_vector = operator_infer.output_info_vector();
    operator_list_ = operator_infer.operator_list();
  }

  // Step 3: Infer reshape and insert operators
  if (InferReshape(from_layout, to_layout, &operator_vector, &output_info_vector) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Construct Reshape operator failed!";
    return nullptr;
  }

  return std::make_shared<std::pair<OperatorVector, OutPutInfoVector>>(
    std::make_pair(operator_vector, output_info_vector));
}

Status TensorRedistribution::InferReshape(const TensorLayout& from_layout, const TensorLayout& to_layout,
                                          OperatorVector* const operator_vector,
                                          OutPutInfoVector* const output_info_vector) {
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
        (void)operator_vector->insert(operator_vector->begin(), constructor.GetOperator());
        (void)output_info_vector->insert(output_info_vector->begin(), std::make_pair(false, 0));
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
      (void)operator_vector->insert(operator_vector->begin(), constructor.GetOperator());
      (void)output_info_vector->insert(output_info_vector->begin(), std::make_pair(false, 0));
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
      (void)operator_vector->insert(operator_vector->end(), constructor.GetOperator());
      (void)output_info_vector->insert(output_info_vector->end(), std::make_pair(false, 0));
    }
  }
  return Status::SUCCESS;
}

Status TensorRedistribution::ComputeCost() {
  RedistributionOpListPtr redistribution_oplist_ptr = InferTensorRedistributionOperatorList();
  if (redistribution_oplist_ptr == nullptr) {
    MS_LOG(ERROR) << "Failure: InferTensorRedistribution failed";
    return Status::FAILED;
  }
  // Compute redistribution communication cost and memory cost
  for (auto& op_cost : operator_list_) {
    OperatorR op = op_cost.first;
    Shape slice_shape = op_cost.second;
    double prod =
      std::accumulate(slice_shape.begin(), slice_shape.end(), static_cast<double>(1.0), std::multiplies<double>());
    std::string str = op.first;
    if (str == PERMUTE_BY_AXIS) {
      // The shape does not change after PermuteByAxis operation.
      // communication cost = all_to_all + all_to_all = 2 * slice_shape
      // memory cost = slice_shape
      forward_comm_cost_ += prod;
      backward_comm_cost_ += prod;
      comm_cost_ += 2.0 * prod;
      mem_cost_ += prod;
    } else if (str == CONCAT_BY_AXIS) {
      // communication cost = all_gather + reduce_scatter = before_slice_shape + after_slice_shape
      // memory cost = before_slice_shape
      if (op.second.size() < 3) {
        MS_LOG(ERROR) << "op.second size should not be less than 3!";
        return Status::FAILED;
      }
      double dev_num = op.second[2];
      // here, communication cost = all_gather + reduce_scatter
      forward_comm_cost_ += prod * dev_num;
      backward_comm_cost_ += prod;
      comm_cost_ += prod * (dev_num + 1.0);
      int32_t concat_dim = op.second[0];
      if (concat_dim == 0) {
        // memory cost = all_gather
        mem_cost_ += prod;
      } else {
        // memory cost = all_gather + split + concat
        mem_cost_ += (prod + prod * dev_num + prod * dev_num);
      }
    } else {
      // There is only memory cost in SplitByAxis.
      // memory cost = before_slice_shape
      mem_cost_ += prod;
    }
  }
  if (reshape_flag()) {
    Shape prev_slice_shape = from_.slice_shape().array();
    double prev_prod = std::accumulate(prev_slice_shape.begin(), prev_slice_shape.end(), 1, std::multiplies<int>());
    mem_cost_ += 2.0 * prev_prod;
  }
  return Status::SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
