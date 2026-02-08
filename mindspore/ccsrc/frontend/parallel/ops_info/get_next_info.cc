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

#include "frontend/parallel/ops_info/get_next_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status GetNextInfo::InferTensorMap() {
  auto slice_dim_iter = std::find(dev_matrix_shape_origin_.begin(), dev_matrix_shape_origin_.end(), shard_num_);
  if (slice_dim_iter == dev_matrix_shape_origin_.end()) {
    MS_LOG(ERROR) << name_ << ": The dataset shard strategy only support shard in one dim.";
    return FAILED;
  }
  size_t slice_dim = size_t(slice_dim_iter - dev_matrix_shape_origin_.begin());
  for (size_t i = 0; i < dataset_strategy_.size(); i++) {
    Shape tensor_map_index;
    for (auto dim : dataset_strategy_[i]) {
      if (dim == 1) {
        tensor_map_index.push_back(MAP_NONE);
      } else if (dim == shard_num_) {
        if (repeated_num_in_dev_matrix_right_ && dev_matrix_shape_origin_.size() != dev_matrix_shape_.size() &&
            is_auto_parallel_) {
          tensor_map_index.push_back(dev_matrix_shape_origin_.size() - slice_dim);
        } else {
          tensor_map_index.push_back(dev_matrix_shape_origin_.size() - 1 - slice_dim);
        }
      } else {
        MS_LOG(ERROR) << name_ << ": The dataset shard strategy only support fully shard in one dim.";
        return FAILED;
      }
    }
    outputs_tensor_map_.push_back(tensor_map_index);
  }
  return SUCCESS;
}

Status GetNextInfo::InferTensorLayout(TensorLayouts *outputs_layout) {
  if (outputs_layout == nullptr) {
    MS_LOG(ERROR) << name_ << " : The layout is null.";
    return FAILED;
  }
  for (size_t i = 0; i < outputs_shape_.size(); ++i) {
    TensorLayout output_layout;
    if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[i], outputs_shape_[i]) != SUCCESS) {
      return FAILED;
    }
    outputs_layout->push_back(output_layout);
  }
  return SUCCESS;
}

Status GetNextInfo::InferTensorInfo() {
  TensorLayouts outputs_layout;
  if (InferTensorLayout(&outputs_layout) != SUCCESS) {
    return FAILED;
  }
  for (size_t i = 0; i < outputs_shape_.size(); ++i) {
    TensorInfo output_tensor_info(outputs_layout[i]);
    outputs_tensor_info_.push_back(output_tensor_info);
  }
  return SUCCESS;
}

Status GetNextInfo::InferDevMatrixShape() {
  if (dataset_strategy_.empty()) {
    MS_LOG(ERROR) << "The dataset strategy is empty";
    return FAILED;
  }
  auto dev_matrix_iter =
    std::max_element(dataset_strategy_.begin(), dataset_strategy_.end(),
                     [](const Dimensions &stra1, const Dimensions &stra2) { return stra1.size() < stra2.size(); });
  if (dev_matrix_iter != dataset_strategy_.end()) {
    dev_matrix_shape_ = *dev_matrix_iter;
  }
  auto shard_num_iter = std::max_element(dev_matrix_shape_.begin(), dev_matrix_shape_.end());
  if (shard_num_iter != dev_matrix_shape_.end()) {
    shard_num_ = *shard_num_iter;
  }
  dev_matrix_shape_origin_ = dev_matrix_shape_;
  return SUCCESS;
}

Status GetNextInfo::CheckStrategy(const StrategyPtr &strategy) {
  Strategies stras = strategy->GetInputDim();
  for (Dimensions stra : stras) {
    if (stra.size() != 0) {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
      return FAILED;
    }
  }
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  if (!ParallelContext::GetInstance()->dataset_strategy().empty()) {
    dataset_strategy_ = ParallelContext::GetInstance()->dataset_strategy();
  } else {
    bool full_batch = ParallelContext::GetInstance()->full_batch();
    int64_t dev_num = full_batch ? 1 : g_device_manager->stage_device_num();
    for (size_t i = 0; i < outputs_shape_.size(); i++) {
      Dimensions input_strategy;
      for (size_t j = 0; j < outputs_shape_[i].size(); j++) {
        input_strategy.push_back(1);
      }
      dataset_strategy_.push_back(input_strategy);
    }
    for (auto &stra : dataset_strategy_) {
      if (!stra.empty()) {
        stra[0] = dev_num;
      }
    }
  }
  return SUCCESS;
}

Status GetNextInfo::GetAttrTypes() {
  auto iter = attrs_.find(TYPES);
  if (iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<ValueList>()) {
      auto iter_cast = iter->second->cast<ValueListPtr>();
      MS_EXCEPTION_IF_NULL(iter_cast);
      auto types = iter_cast->value();
      for (auto &type : types) {
        MS_EXCEPTION_IF_NULL(type);
        types_.push_back(type->ToString());
      }
    } else if (iter->second->isa<ValueTuple>()) {
      auto iter_tuple = iter->second->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(iter_tuple);
      auto tuple_types = iter_tuple->value();
      for (auto &ele : tuple_types) {
        MS_EXCEPTION_IF_NULL(ele);
        types_.push_back(ele->ToString());
      }
    } else {
      MS_LOG(ERROR) << name_ << " : The value of types is not list.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GetNextInfo::GetAttrShapes() {
  shapes_ = outputs_shape_;
  if (shapes_.size() == 0) {
    MS_LOG(ERROR) << name_ << " : Shape is None.";
    return FAILED;
  }
  return SUCCESS;
}

Status GetNextInfo::GetAttrOutPutNum() {
  auto iter = attrs_.find(GETNEXT_NUM);
  if (iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<Int64Imm>()) {
      output_num_ = iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << " : The value of output_num is not int64_t.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GetNextInfo::GetAttrs() {
  repeated_num_in_dev_matrix_right_ = false;
  if (ParallelContext::GetInstance()->dataset_repeat_dim_right()) {
    repeated_num_in_dev_matrix_right_ = true;
  }

  if (GetAttrTypes() == FAILED || GetAttrShapes() == FAILED || GetAttrOutPutNum() == FAILED) {
    return FAILED;
  }
  if (types_.size() != LongToSize(output_num_) || shapes_.size() != LongToSize(output_num_) || output_num_ == 0) {
    MS_LOG(ERROR) << name_ << " : The output_num is not equal to shapes size.";
    return FAILED;
  }
  return SUCCESS;
}

void GetNextInfo::InferReplaceOps() {
  Shapes out_shapes;
  (void)std::transform(outputs_tensor_info_.begin(), outputs_tensor_info_.end(), std::back_inserter(out_shapes),
                       [](auto tensor_info) { return tensor_info.slice_shape(); });
  ValuePtr new_shapes = MakeValue(out_shapes);
  Attr attr_types = std::make_pair(TYPES, attrs_[TYPES]);
  Attr attr_shapes = std::make_pair(SHAPES, new_shapes);
  Attr attr_num = std::make_pair(GETNEXT_NUM, attrs_[GETNEXT_NUM]);
  Attr attr_shared_name = std::make_pair(SHARED_NAME, attrs_[SHARED_NAME]);
  OperatorAttrs attrs = {attr_types, attr_shapes, attr_num, attr_shared_name};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  replace_op_ = {std::make_pair(GET_NEXT, args)};
}

Status GetNextInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> GetNextInfo::GenerateOpStrategies(int64_t stage_id) {
  Strategies stra;
  StrategyPtr sp = std::make_shared<Strategy>(stage_id, stra);
  std::vector<StrategyPtr> sp_vector;
  sp_vector.push_back(sp);
  return sp_vector;
}

Status GetNextInfo::CheckOutputLayout() {
  if (outputs_shape_.size() != outputs_tensor_info_.size()) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "Output size is " << outputs_shape_.size() << ", output layout size is "
                   << outputs_tensor_info_.size() << ", they are not equal";
    } else {
      MS_LOG(ERROR) << "Output size is " << outputs_shape_.size() << ", output layout size is "
                    << outputs_tensor_info_.size() << ", they are not equal";
      return FAILED;
    }
    return FAILED;
  }
  return SUCCESS;
}

REGISTER(GetNextInfo);
}  // namespace parallel
}  // namespace mindspore
