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

#include "parallel/ops_info/matmul_info.h"

#include <algorithm>
#include <functional>
#include <vector>
#include <utility>
#include <memory>
#include <string>

#include "ir/value.h"
#include "parallel/device_matrix.h"
#include "parallel/tensor_layout/tensor_redistribution.h"
#include "parallel/device_manager.h"
#include "parallel/auto_parallel/graph_costmodel.h"

namespace mindspore {
namespace parallel {
void SetDevMatrixShape(const Dimensions& mat_a_strategy, const Dimensions& mat_b_strategy, bool transpose_b,
                       Shape* dev_matrix_shape) {
  MS_EXCEPTION_IF_NULL(dev_matrix_shape);
  size_t mat_a_size = mat_a_strategy.size();
  size_t mat_b_size = mat_b_strategy.size();
  if (mat_a_size >= mat_b_size) {
    // for example: mat_a_strategy:[2,4,8,16], mat_b_strategy:[4,16,32]
    // dev_matrix_shape:[2,4,8,16,32] (transpose_b is false)

    // [2],[4] in the example above
    for (size_t i = 0; i < SECOND_FROM_END(mat_a_size); ++i) {
      dev_matrix_shape->push_back(mat_a_strategy.at(i));
    }
  } else {
    // for example: mat_a_strategy:[8,16], mat_b_strategy:[2,4,16,32]
    // dev_matrix_shape:[2,4,8,16,32] (transpose_b is false)

    // [2],[4] in the example above
    for (size_t i = 0; i < SECOND_FROM_END(mat_b_size); ++i) {
      dev_matrix_shape->push_back(mat_b_strategy.at(i));
    }
  }

  // [8],[16] in the example above
  dev_matrix_shape->push_back(mat_a_strategy.at(SECOND_FROM_END(mat_a_size)));
  dev_matrix_shape->push_back(mat_a_strategy.back());

  // [32] in the example above
  if (!transpose_b) {
    dev_matrix_shape->push_back(mat_b_strategy.back());
  } else {
    dev_matrix_shape->push_back(mat_b_strategy.at(SECOND_FROM_END(mat_b_size)));
  }
}

Status MatMulBase::GetAttrs() {
  if (attrs_.size() < MATMUL_ATTRS_SIZE) {
    MS_LOG(ERROR) << name_ << " : The size of attrs small than 2.";
    return FAILED;
  }

  auto transpose_a_iter = attrs_.find(TRANSPOSE_A);
  if (transpose_a_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(transpose_a_iter->second);
    if (transpose_a_iter->second->isa<BoolImm>()) {
      transpose_a_ = transpose_a_iter->second->cast<BoolImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << " : The value of transpose_a is not bool.";
      return FAILED;
    }
  }

  auto transpose_b_iter = attrs_.find(TRANSPOSE_B);
  if (transpose_b_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(transpose_b_iter->second);
    if (transpose_b_iter->second->isa<BoolImm>()) {
      transpose_b_ = transpose_b_iter->second->cast<BoolImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << " : The value of transpose_a is not bool.";
      return FAILED;
    }
  }

  // infer inputs dimension size
  if ((inputs_shape_.size() != MATMUL_INPUTS_SIZE) || (outputs_shape_.size() != MATMUL_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size or outputs shape size is wrong.";
    return FAILED;
  }
  mat_a_dimension_ = inputs_shape_.at(0).size();
  mat_b_dimension_ = inputs_shape_.at(1).size();

  return SUCCESS;
}

Status CheckRelevantDimension(const Dimensions& long_strategy, const Dimensions& short_strategy) {
  size_t long_size = long_strategy.size();
  size_t short_size = short_strategy.size();
  if (long_size < short_size) {
    MS_LOG(ERROR) << "Size error, the size of long strategy is " << long_size << ", the size of short strategy is "
                  << short_size;
    return FAILED;
  }

  size_t len_diff = long_size - short_size;
  for (size_t j = 0; j < SECOND_FROM_END(short_size); ++j) {
    if (long_strategy.at(len_diff + j) != short_strategy.at(j)) {
      MS_LOG(ERROR) << "Strategies of relevant dimensions are not equal, long strategy is "
                    << ShapeToString(long_strategy) << ", short strategy is " << ShapeToString(short_strategy);
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MatMul::CheckStrategy(const StrategyPtr& strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    }
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions mat_a_strategy = stra.at(0);
  Dimensions mat_b_strategy = stra.at(1);

  size_t mat_a_size = mat_a_strategy.size();
  size_t mat_b_size = mat_b_strategy.size();
  if ((mat_a_size != mat_a_dimension_) || (mat_b_size != mat_b_dimension_)) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : The dimensions of mat_a or mat_b's strategy is wrong.";
    } else {
      MS_LOG(ERROR) << name_ << " : The dimensions of mat_a or mat_b's strategy is wrong.";
    }
    return FAILED;
  }

  // for example: mat_a_strategy:[2,4,8,16], mat_b_strategy:[4,16,32]
  // dev_matrix_shape:[2,4,8,16,32] (transpose_b is false)
  // [16] in the example above
  if (!transpose_b_ && (mat_a_strategy.back() != mat_b_strategy.at(SECOND_FROM_END(mat_b_size)))) {
    MS_LOG(ERROR) << name_ << " : Strategies of relevant dimensions are not equal.";
    return FAILED;
  } else if (transpose_b_ && (mat_a_strategy.back() != mat_b_strategy.back())) {
    MS_LOG(ERROR) << name_ << " : Strategies of relevant dimensions are not equal.";
    return FAILED;
  }

  if (mat_a_size >= mat_b_size) {
    if (CheckRelevantDimension(mat_a_strategy, mat_b_strategy) != SUCCESS) {
      MS_LOG(ERROR) << name_ << " : Strategies of relevant dimensions are not equal.";
      return FAILED;
    }
  } else {
    if (CheckRelevantDimension(mat_b_strategy, mat_a_strategy) != SUCCESS) {
      MS_LOG(ERROR) << name_ << " : Strategies of relevant dimensions are not equal.";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status MatMulBase::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  Dimensions mat_a_strategy = stra.at(0);
  Dimensions mat_b_strategy = stra.at(1);

  SetDevMatrixShape(mat_a_strategy, mat_b_strategy, transpose_b_, &dev_matrix_shape_);
  return SUCCESS;
}

// all-reduce weight's grad
Status MatMulBase::InferMirrorOps() {
  mirror_ops_.clear();

  Shape mat_b_tensor_map = inputs_tensor_map_[1];
  std::vector<Group> mat_b_group;
  if (CreateGroupByTensorMap(mat_b_tensor_map, &mat_b_group) != SUCCESS) {
    return FAILED;
  }

  OperatorVector op_for_inputs;  // op_for_inputs is empty
  OperatorVector op_for_weight;

  if (mat_b_group.empty()) {
    MS_LOG(INFO) << name_ << " : The mirror ops is empty.";
    return SUCCESS;
  } else {
    op_for_weight = CreateMirrorOps(mat_b_group[0].name(), mat_b_group[0].GetDevNum());
    mirror_ops_.push_back(op_for_inputs);
    mirror_ops_.push_back(op_for_weight);
    MS_LOG(INFO) << name_ << " : Create the mirror ops for weight success, group is " << mat_b_group[0].name();
  }

  return SUCCESS;
}

Status MatMulBase::InferForwardCommunication() {
  forward_op_.clear();
  size_t dimension = dev_matrix_shape_.size();
  size_t relevant_dimension_index = SECOND_FROM_END(dimension);
  // Relevant dimension is not split and all reduce is not required
  if (dev_matrix_shape_.at(relevant_dimension_index) == MIN_SLICE_NUM) {
    MS_LOG(INFO) << name_ << " : Forward all reduce is not required.";
    return SUCCESS;
  }

  std::vector<Group> group_list;
  if (CreateGroupByDim(relevant_dimension_index, &group_list) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Infer forward communication, create group failed.";
    return FAILED;
  } else if (group_list.empty()) {
    MS_LOG(INFO) << name_ << " : Forward all reduce is not required.";
    return SUCCESS;
  }

  Operator op = CreateAllReduceOp(REDUCE_OP_SUM, group_list[0].name());
  forward_op_.push_back(op);

  MS_LOG(INFO) << name_ << " : The group name of forward communication is " << group_list[0].name();
  return SUCCESS;
}

// dev_matrix_shape: [a, b, c, d, e], then output strategy: [a, b, c, e];
Dimensions GetOutputStrategy(const Shape& dev_matrix_shape, int32_t repeated_calculation_num) {
  Dimensions output_strategy = dev_matrix_shape;
  if (repeated_calculation_num > 1) {
    // move the first dimension(repeated_calc_num_)
    (void)output_strategy.erase(output_strategy.begin());
  }

  // delete the second-to-last element
  (void)output_strategy.erase(output_strategy.begin() +
                              static_cast<different_type>(SECOND_FROM_END(output_strategy.size())));
  return output_strategy;
}

Status MatMulBase::InferTensorMap() {
  size_t size = dev_matrix_shape_.size();
  if (repeated_calc_num_ > 1) {
    // move the first dimension(repeated_calc_num_), just for the convenience of tensor-map's calculation
    size = dev_matrix_shape_.size() - 1;
  }

  std::vector<int32_t> tensor_map_index;
  // such as 5: tensor_map_index [4,3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back((int32_t)(LAST_INDEX(size) - i));
  }

  // infer output tensor map: [4,3,2,0], delete the second-from-end element
  TensorMap output_tensor_map = tensor_map_index;
  (void)output_tensor_map.erase(output_tensor_map.begin() + static_cast<different_type>(SECOND_FROM_END(size)));

  // infer mat_a tensor map
  // for example: mat_a_dimension is 4, mat_a tensor map:[4,3,2,1]
  TensorMap mat_a_tensor_map = tensor_map_index;
  // delete last one element
  mat_a_tensor_map.pop_back();
  // delete the first (dev_matrix_size - 1 - mat_a_dimension) elements
  (void)mat_a_tensor_map.erase(
    mat_a_tensor_map.begin(),
    mat_a_tensor_map.begin() + static_cast<different_type>(LAST_INDEX(size) - mat_a_dimension_));

  // infer mat_b tensor map
  TensorMap mat_b_tensor_map = tensor_map_index;
  // delete the third-to-last element
  (void)mat_b_tensor_map.erase(mat_b_tensor_map.begin() + static_cast<different_type>(THIRD_FROM_END(size)));
  // delete the first (dev_matrix_size - 1 - mat_b_dimension) elements
  (void)mat_b_tensor_map.erase(
    mat_b_tensor_map.begin(),
    mat_b_tensor_map.begin() + static_cast<different_type>(LAST_INDEX(size) - mat_b_dimension_));
  if (transpose_b_) {
    // swap the last two elements
    int32_t last_value = mat_b_tensor_map.back();
    mat_b_tensor_map.pop_back();
    (void)mat_b_tensor_map.insert(
      mat_b_tensor_map.begin() + static_cast<different_type>(LAST_INDEX(mat_b_tensor_map.size())), last_value);
  }

  inputs_tensor_map_.push_back(mat_a_tensor_map);
  inputs_tensor_map_.push_back(mat_b_tensor_map);
  outputs_tensor_map_.push_back(output_tensor_map);
  return SUCCESS;
}

Status MatMulBase::InferTensorLayout(TensorLayouts* inputs_layout, TensorLayouts* outputs_layout) {
  TensorLayout mat_a_layout, mat_b_layout, output_layout;
  if ((mat_a_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], inputs_shape_[0]) != SUCCESS) ||
      (mat_b_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[1], inputs_shape_[1]) != SUCCESS) ||
      (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], outputs_shape_[0]) != SUCCESS)) {
    return FAILED;
  }

  inputs_layout->push_back(mat_a_layout);
  inputs_layout->push_back(mat_b_layout);
  outputs_layout->push_back(output_layout);
  return SUCCESS;
}

Status MatMulBase::InferTensorInfo() {
  // infer tensor shape
  Shape mat_a_shape = inputs_shape_.at(0);
  Shape mat_b_shape = inputs_shape_.at(1);
  Shape output_shape = outputs_shape_.at(0);

  // infer slice shape
  Shapes inputs_slice_shape, outputs_slice_shape;
  Dimensions output_strategy = GetOutputStrategy(dev_matrix_shape_, repeated_calc_num_);

  Strategys inputs_strategy = strategy_->GetInputDim();
  Strategys outputs_strategy = {output_strategy};
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    return FAILED;
  }
  Shape mat_a_slice_shape = inputs_slice_shape.at(0);
  Shape mat_b_slice_shape = inputs_slice_shape.at(1);
  Shape output_slice_shape = outputs_slice_shape.at(0);

  // infer tensor layout
  TensorLayouts inputs_layout, outputs_layout;
  if (InferTensorLayout(&inputs_layout, &outputs_layout) != SUCCESS) {
    return FAILED;
  }

  TensorLayout mat_a_layout = inputs_layout.at(0);
  TensorLayout mat_b_layout = inputs_layout.at(1);
  TensorLayout output_layout = outputs_layout.at(0);
  TensorInfo mat_a_tensor_info(mat_a_layout, mat_a_shape, mat_a_slice_shape);
  TensorInfo mat_b_tensor_info(mat_b_layout, mat_b_shape, mat_b_slice_shape);
  TensorInfo output_tensor_info(output_layout, output_shape, output_slice_shape);

  inputs_tensor_info_.push_back(mat_a_tensor_info);
  inputs_tensor_info_.push_back(mat_b_tensor_info);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status MatMulBase::Init(const StrategyPtr& strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status MatMulBase::InitForCostModel(const StrategyPtr& strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Init for cost model failed.";
    } else {
      MS_LOG(ERROR) << name_ << " : Init for cost model failed.";
    }
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init for cost model success.";
  return SUCCESS;
}

Status MatMulBase::SwapLastTwoElements(mindspore::parallel::Shape* const input) {
  if (input->size() < 2) {
    MS_LOG(ERROR) << name_ << " : The size of inputs small than 2.";
    return FAILED;
  }
  auto last_1st_value = input->at(input->size() - 1);
  auto last_2nd_value = input->at(input->size() - 2);
  input->pop_back();
  input->pop_back();
  input->push_back(last_1st_value);
  input->push_back(last_2nd_value);
  return SUCCESS;
}

Status MatMulBase::GenerateStrategies(int32_t stage_id) {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : GetAttrs failed.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  std::vector<int32_t> dev_list = g_device_manager->GetDeviceListByStageId(stage_id);
  size_t dev_num = dev_list.size();
  Shape input0_shape = inputs_shape_[0], input1_shape = inputs_shape_[1];
  if (transpose_a_) {
    if (SwapLastTwoElements(&input0_shape) == FAILED) {
      MS_LOG(ERROR) << name_ << " : Swap last two elements failed.";
    }
  }
  if (transpose_b_) {
    if (SwapLastTwoElements(&input1_shape) == FAILED) {
      MS_LOG(ERROR) << name_ << " : Swap last two elements failed.";
    }
  }
  // The shape of input0 (input1)
  // E.g., input0 = [100, 200, 300], input1 = [300, 400]

  // Combining the input0_shape and input1_shape
  // E.g., combined_shape = [100, 200, 300, 400]
  is_auto_parallel_ = true;
  size_t input1_shape_size = input1_shape.size(), input0_shape_size = input0_shape.size();
  Dimensions combined_partitions;
  Shape combined_shape;
  // In SwapLastTwoElements(), it is guaranteed that input0_shape.size() and input1_shape.size() are both larger than 2
  if (input0_shape.size() >= input1_shape.size()) {
    combined_shape = input0_shape;
    combined_shape.push_back(input1_shape[input1_shape.size() - 1]);
  } else {
    combined_shape = input1_shape;
    combined_shape.push_back(input0_shape[input0_shape.size() - 2]);
  }
  std::function<void(uint32_t, size_t)> recursive = [&stage_id, &dev_num, &combined_partitions, &combined_shape,
                                                     &input1_shape_size, &recursive, &input0_shape_size,
                                                     this](uint32_t current_index, size_t n) {
    // Finishing the recursive steps, if the strategy is valid, then calculate the cost
    // for this operator under the strategy.
    if (current_index == combined_shape.size()) {
      StrategyPtr sp;
      if (this->PrepareStrategy(stage_id, dev_num, combined_partitions, input0_shape_size, input1_shape_size, &sp) ==
          FAILED) {
        return;
      }
      if (this->SetCostUnderStrategy(sp) == FAILED) {
        MS_LOG(WARNING) << name_ << " : Calculating cost for strategy failed.";
        return;
      }
    } else {
      MS_LOG(DEBUG) << name_ << " : The value input0_shape_size: " << input0_shape_size
                    << ", input1_shape_size: " << input1_shape_size;
      for (uint32_t i = 1; i <= n; i *= 2) {
        if (n % i == 0 && IntToSize(combined_shape[current_index]) % i == 0) {
          combined_partitions.push_back(i);
          recursive(current_index + 1, n / i);
          combined_partitions.pop_back();
        }
      }
    }
  };
  recursive(0, dev_num);
  if (strategy_cost_.empty()) {
    MS_LOG(EXCEPTION) << name_ << " : No available strategy.";
  }
  return Status::SUCCESS;
}

Status MatMulBase::PrepareStrategy(int32_t stage_id, size_t dev_num,
                                   mindspore::parallel::Dimensions combined_partitions, size_t input0_shape_size,
                                   size_t input1_shape_size, mindspore::parallel::StrategyPtr* const sp) {
  int32_t product = std::accumulate(combined_partitions.begin(), combined_partitions.end(), 1, std::multiplies<int>());
  if (NOT_FULLY_USE_DEVICES) {
    if (IntToSize(product) > dev_num) {
      return FAILED;
    }
  } else {
    if (IntToSize(product) != dev_num) {
      return FAILED;
    }
  }
  Dimensions input0_partitions, input1_partitions;
  if (input0_shape_size >= input1_shape_size) {
    for (size_t i = 0; i < input0_shape_size; ++i) {
      input0_partitions.push_back(combined_partitions[i]);
    }
    if (input1_shape_size == 2) {
      input1_partitions.push_back(combined_partitions[combined_partitions.size() - 2]);
      input1_partitions.push_back(combined_partitions[combined_partitions.size() - 1]);
    } else {
      // input1_shape.size() > 2
      for (size_t j = combined_partitions.size() - input1_shape_size - 1; j < combined_partitions.size(); ++j) {
        if (j == combined_partitions.size() - 3) {
          continue;
        }
        input1_partitions.push_back(combined_partitions[j]);
      }
    }
  } else {
    for (size_t i = 0; i < input1_shape_size; ++i) {
      input1_partitions.push_back(combined_partitions[i]);
    }
    for (size_t j = combined_partitions.size() - input0_shape_size - 1; j < combined_partitions.size() - 3; ++j) {
      input0_partitions.push_back(combined_partitions[j]);
    }
    input0_partitions.push_back(combined_partitions[combined_partitions.size() - 1]);
    input0_partitions.push_back(combined_partitions[combined_partitions.size() - 3]);
  }
  if (transpose_a_) {
    if (SwapLastTwoElements(&input0_partitions) == FAILED) {
      MS_LOG(ERROR) << name_ << " : Swap last two elements failed.";
    }
  }
  if (transpose_b_) {
    if (SwapLastTwoElements(&input1_partitions) == FAILED) {
      MS_LOG(ERROR) << name_ << " : Swap last two elements failed.";
    }
  }
  std::vector<Dimensions> stras;
  stras.push_back(input0_partitions);
  stras.push_back(input1_partitions);
  (*sp) = std::make_shared<Strategy>(stage_id, stras);

  return SUCCESS;
}

void MatMulBase::InitTensorInfoForCost(std::vector<TensorInfo>* relica_inputs_tensor_vector) {
  TensorLayout tly;
  if (transpose_a_) {
    Shape replica_input0_shape(inputs_tensor_info_[0].shape());
    Shape replica_input0_slice_shape(inputs_tensor_info_[0].slice_shape());
    if (SwapLastTwoElements(&replica_input0_shape) == FAILED) {
      MS_LOG(ERROR) << name_ << " : Swap last two elements failed.";
    }
    if (SwapLastTwoElements(&replica_input0_slice_shape) == FAILED) {
      MS_LOG(ERROR) << name_ << " : Swap last two elements failed.";
    }

    TensorInfo replica_input0_info(tly, replica_input0_shape, replica_input0_slice_shape);
    relica_inputs_tensor_vector->push_back(replica_input0_info);
  } else {
    relica_inputs_tensor_vector->push_back(inputs_tensor_info_[0]);
  }
  if (transpose_b_) {
    Shape replica_input1_shape(inputs_tensor_info_[1].shape());
    Shape replica_input1_slice_shape(inputs_tensor_info_[1].slice_shape());
    if (SwapLastTwoElements(&replica_input1_shape) == FAILED) {
      MS_LOG(ERROR) << name_ << " : Swap last two elements failed.";
    }
    if (SwapLastTwoElements(&replica_input1_slice_shape) == FAILED) {
      MS_LOG(ERROR) << name_ << " : Swap last two elements failed.";
    }

    TensorInfo replica_input1_info(tly, replica_input1_shape, replica_input1_slice_shape);
    relica_inputs_tensor_vector->push_back(replica_input1_info);
  } else {
    relica_inputs_tensor_vector->push_back(inputs_tensor_info_[1]);
  }
}

Status MatMulBase::CheckForTensorSliceValid() const {
  if (!TENSOR_SLICE_ALIGNMENT_ENABLE) {
    return SUCCESS;
  }
  if (inputs_tensor_info_.empty()) {
    return FAILED;
  }
  for (auto& one_input_tensor : inputs_tensor_info_) {
    auto slice_shape = one_input_tensor.slice_shape();
    if ((IntToSize(slice_shape[LAST_INDEX(slice_shape.size())]) % TENSOR_SLICE_ALIGNMENT_SIZE != 0) ||
        (IntToSize(slice_shape[SECOND_FROM_END(slice_shape.size())]) % TENSOR_SLICE_ALIGNMENT_SIZE != 0)) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status MatMulBase::SetCostUnderStrategy(const mindspore::parallel::StrategyPtr& strategy) {
  if (InitForCostModel(strategy) == FAILED) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Initialization under the strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << " : Initialization under the strategy failed.";
    }
    return FAILED;
  }
  PrintStrategy(strategy);
  // Check whether the tensor slice of input_tensor_info is valid or not
  if (CheckForTensorSliceValid() != SUCCESS) {
    MS_LOG(INFO) << name_ << " : The tensor slice is not valid under this strategy.";
    return FAILED;
  }
  // Here, a replicated inputs_ is constructed for the transposed TensorInfo.
  std::vector<TensorInfo> relica_inputs_tensor_vector;
  InitTensorInfoForCost(&relica_inputs_tensor_vector);

  int32_t stage_id = strategy->GetInputStage();
  // Here, we use the origin outputs_, because we only use the slice size of the output tensor.
  // It does not matter whether the output tensor is transposed or not.
  double memory_cost =
    matmulcost_ptr->GetForwardMemoryCost(relica_inputs_tensor_vector, outputs_tensor_info_, stage_id);
  double communication_cost = matmulcost_ptr->GetCommCost(relica_inputs_tensor_vector, outputs_tensor_info_, stage_id);
  std::shared_ptr<Cost> result = std::make_shared<Cost>(memory_cost, communication_cost);
  result->communication_without_parameter_ =
    matmulcost_ptr->GetForwardCommCost(relica_inputs_tensor_vector, outputs_tensor_info_, stage_id);
  result->communication_with_partial_para_ =
    result->communication_without_parameter_ +
    COST_MODEL_GAMMA * (communication_cost - result->communication_without_parameter_);

  // Breaking ties for preferring data parallelization
  BreakingTiesForPerferringDataParallel(strategy, result);
  MS_LOG(DEBUG) << name_ << " : memory_cost: " << result->memory_cost_
                << ", communication_cost: " << result->communication_cost_
                << ", communication_without_parameter_: " << result->communication_without_parameter_
                << ", communication_with_partial_para_: " << result->communication_with_partial_para_;
  // refine communication cost calculation for practice
  RefineForPracticalCost(result, false);

  std::shared_ptr<StrategyWithCost> swc =
    std::make_shared<StrategyWithCost>(strategy, inputs_tensor_info_, outputs_tensor_info_);
  swc->cost_list.push_back(result);
  strategy_cost_.emplace_back(swc);

  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
