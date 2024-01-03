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
constexpr int64_t DYNAMIC_DIM_VAL = -1;

Status TensorRedistribution::Init(const TensorLayout &from, const TensorLayout &to, const RankList &dev_list) {
  from_origin_ = from;
  to_origin_ = to;
  auto func = [](const Shape &shape) -> bool {
    return std::find(shape.begin(), shape.end(), DYNAMIC_DIM_VAL) != shape.end();
  };
  if (!func(from_origin_.tensor_shape().array()) && !func(to_origin_.tensor_shape().array()) &&
      from_origin_.tensor_shape().size() != to_origin_.tensor_shape().size()) {
    MS_LOG(ERROR) << "from shape size must be equal to to shape size! from shape size is "
                  << from_origin_.tensor_shape().size() << ", to shape size is " << to_origin_.tensor_shape().size();
    MS_LOG(ERROR) << "reshape from_origin_ " << from_origin_.ToString();
    MS_LOG(ERROR) << "reshape to_origin_ " << to_origin_.ToString();
    return Status::FAILED;
  }
  dev_list_ = dev_list;
  from_ = from_origin_.SqueezeShape();
  to_ = to_origin_.SqueezeShape();
  this->is_inited_ = true;
  return Status::SUCCESS;
}

void TensorRedistribution::CreateAssembledDynamicMapping(RedistributionOpListPtr *redistribution_oplist_ptr,
                                                         const FuncGraphPtr &func_graph, const CNodePtr &pre_cnode) {
  MS_LOG(DEBUG) << "Start to create assembled dynamic shape mapping." << std::endl;
  this->dynamic_dim_mapping_.clear();
  ReplacementMemo from_layout_memo = this->layout_transfer_.FromLayoutDimsReplacementMemo();
  for (auto &redistribution_op : (*redistribution_oplist_ptr)->first) {
    // Create instance_name
    std::string op_name = redistribution_op.first;
    if (op_name == RESHAPE) {
      // Pattern: PrimFunc_Shape->TupleGetItem->MakeTuple->PrimFunc_Reshape
      // 1. New shape and set pre_cnode to its inputs.
      auto prim = std::make_shared<Primitive>(SHAPE);
      MS_EXCEPTION_IF_NULL(prim);
      prim->set_instance_name("Dynamic-shape-op");
      AnfNodePtrList shape_node_inputs(2);  // 1 for primitive value node.
      shape_node_inputs[0] = NewValueNode(prim);
      shape_node_inputs[1] = pre_cnode;
      auto shape_cnode = func_graph->NewCNode(shape_node_inputs);
      // 2. Create TupleGetItem node to get dim value and insert to mapping.
      for (const auto &iter : from_layout_memo) {
        int64_t dim = iter.first;
        int64_t replacement = iter.second;
        auto prim_tuple_get_item = std::make_shared<Primitive>(TUPLE_GETITEM);
        AnfNodePtrList inputs{NewValueNode(prim_tuple_get_item), shape_cnode, NewValueNode(MakeValue(dim))};
        auto tuple_get_item_cnode = func_graph->NewCNode(inputs);
        this->dynamic_dim_mapping_.insert({replacement, tuple_get_item_cnode});
        MS_LOG(DEBUG) << "Create TupleGetItem for dim=" << dim << " to replace value=" << replacement;
      }
      continue;
    }
    if (op_name == STRIDEDSLICE) {
      MS_LOG(WARNING) << "StridedSlice is not supported yet.";
    }
  }
}

void AppendOperatorVecStr(const OperatorVector &vec, std::string *res) {
  for (size_t i = 0; i < vec.size(); ++i) {
    res->append(vec.at(i).first);
    if (i != vec.size() - 1) {
      res->append(", ");
    }
  }
}

RedistributionOpListPtr TensorRedistribution::InferTensorRedistributionOperatorListUnExpand(bool is_cost_model) {
  MS_LOG(DEBUG) << "Start to infer tensor redistribution with unexpanded.";
  TensorLayout from_origin = this->IsAssembledStaticShape() ? this->assembled_from_layout() : from_origin_;
  TensorLayout to_origin = this->IsAssembledStaticShape() ? this->assembled_to_layout() : to_origin_;
  TensorLayout from_layout = this->from_;
  TensorLayout to_layout = this->to_;
  TensorLayout from_repeat = from_origin.TransferRepeatLayout();
  TensorLayout to_repeat = to_origin.TransferRepeatLayout();

  MS_LOG(DEBUG) << "reshape from_origin_ " << from_origin.ToString();
  MS_LOG(DEBUG) << "reshape to_origin_ " << to_origin.ToString();
  MS_LOG(DEBUG) << "reshape from_ " << from_layout.ToString();
  MS_LOG(DEBUG) << "reshape to_ " << to_layout.ToString();

  OperatorVector operator_vector;
  OutPutInfoVector output_info_vector;
  if (InferRedistribution(from_origin, from_repeat, &operator_vector, &output_info_vector, is_cost_model) ==
      Status::FAILED) {
    return nullptr;
  }
#ifdef DEBUG
  std::string operator_vec_str;
  AppendOperatorVecStr(operator_vector, &operator_vec_str);
  MS_LOG(DEBUG) << "After InferRedistribution line112, operator_vector size: " << operator_vector.size()
                << ", operator_vector: " << operator_vec_str;
#endif
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
      output_info_vector.emplace_back(std::make_pair(false, 0));
    }
  }
  if (InferRedistribution(to_repeat, to_origin, &operator_vector, &output_info_vector, is_cost_model) ==
      Status::FAILED) {
    return nullptr;
  }
#ifdef DEBUG
  operator_vec_str.clear();
  AppendOperatorVecStr(operator_vector, &operator_vec_str);
  MS_LOG(DEBUG) << "After InferRedistribution line130, operator_vector size: " << operator_vector.size()
                << ", operator_vector: " << operator_vec_str;
#endif
  return std::make_shared<std::pair<OperatorVector, OutPutInfoVector>>(
    std::make_pair(operator_vector, output_info_vector));
}

RedistributionOpListPtr TensorRedistribution::InferTensorRedistributionOperatorList(bool is_cost_model) {
  MS_LOG(DEBUG) << "Start to infer tensor redistribution.";
  // Step 1: Match device arrangement between from_ and to_
  // RedistributionLayoutTransfer layout_transfer;
  // Step 0: Do dynamic shape to static shape conversion.
  // TensorRedistribution::Init() only save from and to tensor layout, and squeezed from and to layout.
  // We can change from_ and to_ in RedistributionLayoutTransfer object directly.
  // RedistributionLayoutTransfer::Init() will check whether is dynamic shape,
  // if the static shape cannot be created, reuse early process.
  Status status = this->layout_transfer_.Init(from_, to_);
  if (status != Status::SUCCESS) {
    return nullptr;
  }
  TensorLayout from_layout;
  TensorLayout to_layout;
  if (this->layout_transfer_.IsDynamicShape() && !this->layout_transfer_.IsAssembledStaticShape()) {
    from_layout = this->layout_transfer_.from_in();
    to_layout = this->layout_transfer_.to_in();
  } else {
    // init a new layout_transfer
    std::shared_ptr<ReshapeLayoutTransfer> ptr = this->layout_transfer_.UnifyDeviceArrangementAndTensorShape();
    if (ptr == nullptr) {
      MS_LOG(ERROR) << "Infer tensor layout return nullptr!";
      return nullptr;
    }
    this->layout_transfer_.Init(ptr->from_in(), ptr->to_in(), true);
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
  if (this->layout_transfer_.IsAssembledStaticShape()) {
    // Change assemble static shape to dynamic.
    if (this->layout_transfer_.RollbackToDynamicShape() != Status::SUCCESS) {
      MS_LOG(ERROR) << "Rollback assembled static shape to dynamic shape failed. from: " << from_layout.ToString()
                    << ", to: " << to_layout.ToString();
      return nullptr;
    }
    from_layout = this->layout_transfer_.from_in();
    to_layout = this->layout_transfer_.to_in();
  }
#ifdef DEBUG
  std::string operator_vec_str;
  AppendOperatorVecStr(operator_vector, &operator_vec_str);
  MS_LOG(DEBUG) << "After InferRedistribution, operator_vector size: " << operator_vector.size()
                << ", operator_vector: " << operator_vec_str;
#endif
  if (InferReshape(from_layout, to_layout, &operator_vector, &output_info_vector) != Status::SUCCESS) {
    MS_LOG(ERROR) << "Construct Reshape operator failed!";
    return nullptr;
  }

  return std::make_shared<std::pair<OperatorVector, OutPutInfoVector>>(
    std::make_pair(operator_vector, output_info_vector));
}

bool IsSameShape(const Shape &src, const Shape &tgt) {
  if (src.size() != tgt.size()) {
    return false;
  }
  for (size_t i = 0; i < src.size(); ++i) {
    if (src[i] == -1 || tgt[i] == -1) {
      continue;
    }
    if (src[i] != tgt[i]) {
      return false;
    }
  }
  return true;
}

Status TensorRedistribution::InferReshape(const TensorLayout &from_layout, const TensorLayout &to_layout,
                                          OperatorVector *const operator_vector,
                                          OutPutInfoVector *const output_info_vector) {
  MS_EXCEPTION_IF_NULL(operator_vector);
  MS_EXCEPTION_IF_NULL(output_info_vector);
  MS_LOG(DEBUG) << "Start to infer reshape.";
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

  // 1. 需要知道哪个轴是动态的，哪个轴是常量，只比较常量轴，但是是否能保证from_origin_和from_layout的rank是一样的？
  // from_origin_是静态，那from_layout也一定是静态，如果from_origin_是动态，那from_layout也一定是动态
  // 先支持from_origin_和from_layout的rank一样的场景
  if (!IsSameShape(from_origin_.slice_shape().array(), from_layout.slice_shape().array())) {
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

  if (!IsSameShape(to_origin_.slice_shape().array(), to_layout.slice_shape().array())) {
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
  MS_LOG(DEBUG) << "Start to infer redistribution.";
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
