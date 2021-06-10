/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/tensordot_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
static std::string AxesToString(const std::vector<int32_t> &shape) {
  std::string str = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    str += std::to_string(shape[i]);
    if (i < shape.size() - 1) {
      str += ", ";
    }
  }
  return str + "]";
}

void TensorDotInfo::ShowAxes() {
  if (axes_tuple_.size()) {
    MS_LOG(INFO) << name_ << ": The axes tuple is " << AxesToString(axes_tuple_);
  } else if (axes_tuple_tuple_.size()) {
    MS_LOG(INFO) << name_ << ": The axes tuple tuple is " << AxesToString(axes_tuple_tuple_[0]) << " and "
                 << AxesToString(axes_tuple_tuple_[1]);
  }
}

Status TensorDotInfo::GetAttrs() {
  auto axes_iter = attrs_.find(AXES);
  if (axes_iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find the axes attr";
    return FAILED;
  }

  MS_EXCEPTION_IF_NULL(axes_iter->second);
  if (axes_iter->second->isa<Int32Imm>()) {
    axes_int_ = axes_iter->second->cast<Int32ImmPtr>()->value();
    if ((axes_int_ < 0) || (IntToSize(axes_int_) > inputs_shape_[0].size()) ||
        (IntToSize(axes_int_) > inputs_shape_[1].size())) {
      MS_LOG(ERROR) << name_ << ": The value of axes int (" << axes_int_ << ") is out of range";
      return FAILED;
    }
    axes_type_ = INT_TYPE;
  } else if (axes_iter->second->isa<ValueTuple>() || axes_iter->second->isa<ValueList>()) {
    std::vector<ValuePtr> var_tuple = GetValueSequeue(axes_iter->second);
    if (var_tuple.size() != 2) {
      MS_LOG(ERROR) << name_ << ": The length of axes tuple must be 2, bug got " << var_tuple.size();
      return FAILED;
    }

    for (size_t i = 0; i < var_tuple.size(); ++i) {
      if (var_tuple[i]->isa<Int32Imm>()) {
        int32_t ele_var = var_tuple[i]->cast<Int32ImmPtr>()->value();
        if (ele_var < 0) {
          ele_var += SizeToInt(inputs_shape_[i].size());
        }
        axes_tuple_.push_back(ele_var);
      } else {
        std::vector<int32_t> var_ele = GetValue<std::vector<int32_t>>(var_tuple[i]);
        for (auto &ele : var_ele) {
          if (ele < 0) {
            MS_LOG(DEBUG) << name_ << ": The element of axes is " << ele;
            ele += SizeToInt(inputs_shape_[i].size());
          }
        }
        axes_tuple_tuple_.push_back(var_ele);
      }
    }

    if (!axes_tuple_.empty()) {
      axes_type_ = TUPLE_TYPE;
      MS_LOG(ERROR) << name_ << ": Now do not support axes type is TUPLE_TYPE";
      return FAILED;
    } else if (!axes_tuple_tuple_.empty()) {
      axes_type_ = TUPLE_TUPLE_TYPE;
    }
  } else {
    MS_LOG(ERROR) << name_ << ": The axes is not int or tuple or list";
    return FAILED;
  }

  ShowAxes();
  return SUCCESS;
}

Status TensorDotInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  Strategys stra = strategy->GetInputDim();
  if (stra.size() != 2) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy size " << stra.size();
    return FAILED;
  }
  Dimensions input_a_strategy = stra[0];
  Dimensions input_b_strategy = stra[1];

  if (axes_type_ == INT_TYPE) {  // for example: axes = 3, [a, b, c, d] and [b, c, d, e]
    for (size_t i = 0; i < IntToSize(axes_int_); ++i) {
      if (input_a_strategy[input_a_strategy.size() - IntToSize(axes_int_) + i] != input_b_strategy[i]) {
        MS_LOG(ERROR) << name_ << ": The strategies of relevant dimensions are no equal";
        return FAILED;
      }
    }
  } else if (axes_type_ == TUPLE_TUPLE_TYPE) {
    for (size_t i = 0; i < axes_tuple_tuple_[0].size(); ++i) {
      if (input_a_strategy[IntToSize(axes_tuple_tuple_[0][i])] !=
          input_b_strategy[IntToSize(axes_tuple_tuple_[1][i])]) {
        MS_LOG(ERROR) << name_ << ": The strategies of relevant dimensions are no equal";
        return FAILED;
      }
    }
  } else {
    MS_LOG(ERROR) << name_ << ": Now do not support axes type is TUPLE_TYPE";
    return FAILED;
  }
  return SUCCESS;
}

Status TensorDotInfo::InferDevMatrixShape() {
  Strategys stra = strategy_->GetInputDim();
  Dimensions input_a_strategy = stra.at(0);
  Dimensions input_b_strategy = stra.at(1);

  if (axes_type_ == INT_TYPE) {
    dev_matrix_shape_ = input_a_strategy;
    for (size_t i = IntToSize(axes_int_); i < input_b_strategy.size(); i++) {
      dev_matrix_shape_.push_back(input_b_strategy[i]);
    }
  } else if (axes_type_ == TUPLE_TUPLE_TYPE) {
    dev_matrix_shape_ = input_a_strategy;
    for (size_t i = 0; i < input_b_strategy.size(); ++i) {
      bool found = false;
      for (auto &ele : axes_tuple_tuple_[1]) {
        if (i == IntToSize(ele)) {
          found = true;
          break;
        }
      }

      if (!found) {
        dev_matrix_shape_.push_back(input_b_strategy[i]);
      }
    }
  } else {
    MS_LOG(ERROR) << name_ << ": Now do not support axes type is TUPLE_TYPE";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": The dev matrix is " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status TensorDotInfo::InferForwardCommunication() {
  forward_op_.clear();
  Shape forward_group_map = outputs_tensor_map_[0];
  // handle the repeat calculation, the forward communication's group can not include the dimension of repeat
  // calculation
  if (repeated_calc_num_ > 1) {
    if (repeated_num_in_dev_matrix_right_) {
      forward_group_map.push_back(0);
    } else {
      forward_group_map.push_back(dev_matrix_shape_.size() - 1);
    }
  }

  std::vector<Group> forward_group;
  if (CreateGroupByTensorMap(forward_group_map, &forward_group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create group by tensor map failed";
    return FAILED;
  }

  if (forward_group.empty()) {
    MS_LOG(INFO) << name_ << ": No need to create forward op";
    return SUCCESS;
  }

  Operator op = CreateAllReduceOp(REDUCE_OP_SUM, forward_group[0].name());
  forward_op_.push_back(op);
  MS_LOG(INFO) << name_ << ": The group name of forward communication is " << forward_group[0].name();
  return SUCCESS;
}

void TensorDotInfo::InferTensorMapAxesInt(const TensorMap &tensor_map_index) {
  // infer input_b tensor map
  // for example: the dimension of input_b is 4, the tensor map is [3, 2, 1, 0]
  TensorMap input_b_tensor_map, output_tensor_map;
  for (size_t i = 0; i < inputs_shape_[1].size(); i++) {
    input_b_tensor_map.push_back((int64_t)(inputs_shape_[1].size() - i - 1));
  }

  // infer output tensor map
  output_tensor_map = tensor_map_index;
  (void)output_tensor_map.erase(
    output_tensor_map.begin() + static_cast<different_type>(inputs_shape_[0].size() - IntToSize(axes_int_)),
    output_tensor_map.begin() + static_cast<different_type>(inputs_shape_[0].size()));

  inputs_tensor_map_.push_back(input_b_tensor_map);
  outputs_tensor_map_.push_back(output_tensor_map);
}

void TensorDotInfo::InferTensorMapAxesTuple(size_t size, const TensorMap &input_a_tensor_map,
                                            const TensorMap &tensor_map_index) {
  // for example: [a, b, c, d] + [e, f, b, c, d] -> [a, e, f], axes is ((1, 2, 3), (2, 3, 4))
  // the tensor map of inputs:[5, 4, 3, 2], [1, 0, 4, 3, 2], and the tensor map of output: [5, 1, 0]
  // infer input_b tensor map
  TensorMap input_b_tensor_map, output_tensor_map, tmp_b_map_index;
  for (size_t i = 0; i < size - inputs_shape_[0].size(); ++i) {
    tmp_b_map_index.push_back((int64_t)(size - inputs_shape_[0].size() - i - 1));  // [1, 0]
  }
  for (size_t i = 0; i < inputs_shape_[1].size(); ++i) {
    bool found = false;
    size_t relevant_a_index = 0;
    for (size_t j = 0; j < axes_tuple_tuple_[1].size(); ++j) {
      if (i == IntToSize(axes_tuple_tuple_[1][j])) {
        found = true;
        relevant_a_index = IntToSize(axes_tuple_tuple_[0][j]);
        break;
      }
    }

    if (!found) {
      input_b_tensor_map.push_back(tmp_b_map_index.front());
      tmp_b_map_index.erase(tmp_b_map_index.begin());
    } else {
      input_b_tensor_map.push_back(input_a_tensor_map[relevant_a_index]);
    }
  }

  // infer output tensor map
  for (size_t i = 0; i < size; ++i) {
    bool found = false;
    for (size_t j = 0; j < axes_tuple_tuple_[0].size(); ++j) {
      if (i == IntToSize(axes_tuple_tuple_[0][j])) {
        found = true;
        break;
      }
    }
    if (!found) {
      output_tensor_map.push_back(tensor_map_index[i]);
    }
  }
  inputs_tensor_map_.push_back(input_b_tensor_map);
  outputs_tensor_map_.push_back(output_tensor_map);
}

Status TensorDotInfo::InferTensorMap() {
  size_t size = dev_matrix_shape_.size();
  if (repeated_calc_num_ > 1) {
    // move the repeat calculation dimension, just for the convenience of tensor-map's calculation
    size = dev_matrix_shape_.size() - 1;
  }

  TensorMap tensor_map_index, input_a_tensor_map;
  // such as 5: tensor_map_index [4, 3, 2, 1, 0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back((int64_t)(LAST_INDEX(size) - i));
  }

  // infer input_a tensor map
  // for example: the dimension of input_a is 4, the tensor map is [4, 3, 2, 1]
  for (size_t i = 0; i < inputs_shape_[0].size(); i++) {
    input_a_tensor_map.push_back(tensor_map_index[i]);
  }
  inputs_tensor_map_.push_back(input_a_tensor_map);

  if (axes_type_ == INT_TYPE) {
    InferTensorMapAxesInt(tensor_map_index);
  } else if (axes_type_ == TUPLE_TUPLE_TYPE) {
    InferTensorMapAxesTuple(size, input_a_tensor_map, tensor_map_index);
  } else {
    MS_LOG(ERROR) << name_ << ": Now do not support axes type is TUPLE_TYPE";
    return FAILED;
  }

  return SUCCESS;
}

Status TensorDotInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init success";
  return SUCCESS;
}

Status TensorDotInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success";
  return SUCCESS;
}

std::shared_ptr<Strategys> TensorDotInfo::GenerateBatchStrategies() {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Get attr failed";
  }
  Dimensions input_a_strategy(inputs_shape_[0].size(), 1);
  Dimensions input_b_strategy(inputs_shape_[1].size(), 1);

  input_a_strategy[0] = stage_device_size_;

  if (axes_type_ == INT_TYPE) {
    if (IntToSize(axes_int_) == inputs_shape_[0].size()) {
      input_b_strategy[0] = stage_device_size_;  // find the relevant dimension for input_b
    }
  } else if (axes_type_ == TUPLE_TUPLE_TYPE) {
    // if the input_a's axes contain 0, the input_b has the relevant dimension with batch dimension
    bool found = false;
    size_t relevant_index = 0;
    for (size_t i = 0; i < axes_tuple_tuple_[0].size(); ++i) {
      if (axes_tuple_tuple_[0][i] == 0) {
        found = true;
        relevant_index = i;
        break;
      }
    }

    if (found) {
      // find the relevant
      input_b_strategy[IntToSize(axes_tuple_tuple_[1][relevant_index])] = stage_device_size_;
    }
  } else {
    MS_LOG(EXCEPTION) << name_ << ": Now do not support TUPLE_TYPE";
  }

  Strategys strategy = {input_a_strategy, input_b_strategy};
  return std::make_shared<Strategys>(strategy);
}

std::vector<StrategyPtr> TensorDotInfo::GenerateOpStrategies(int64_t) {
  std::vector<StrategyPtr> sp_vector;
  return sp_vector;
}

Status TensorDotInfo::SetCostUnderStrategy(const mindspore::parallel::StrategyPtr &) { return SUCCESS; }
}  // namespace parallel
}  // namespace mindspore
