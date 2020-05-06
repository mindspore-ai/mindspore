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

#include "parallel/ops_info/get_next_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "parallel/device_matrix.h"
#include "parallel/strategy.h"
#include "parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status GetNextInfo::InferTensorMap() {
  for (auto shp : shapes_) {
    TensorMap out_tensor_map;
    for (size_t i = 0; i < shp.size(); ++i) {
      out_tensor_map.push_back(SizeToInt(dev_matrix_shape_.size() - i - 1));
    }
    outputs_tensor_map_.push_back(out_tensor_map);
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

Strategys GetNextInfo::GetOutputStrategy() {
  Strategys outputs_strategy;
  for (auto shp : shapes_) {
    Dimensions out_strategy;
    out_strategy.push_back(dev_num_);
    for (size_t i = 1; i < shp.size(); ++i) {
      out_strategy.push_back(1);
    }
    outputs_strategy.push_back(out_strategy);
  }
  return outputs_strategy;
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
  size_t max_shape_length = 0;
  for (auto shp : shapes_) {
    if (max_shape_length < shp.size()) {
      max_shape_length = shp.size();
    }
  }
  if (max_shape_length == 0) {
    MS_LOG(ERROR) << name_ << " : shape is 0";
  }
  dev_matrix_shape_.push_back(dev_num_);
  for (size_t i = 1; i < max_shape_length; ++i) {
    dev_matrix_shape_.push_back(1);
  }
  return SUCCESS;
}

Status GetNextInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed";
    return FAILED;
  }
  if (InferReplaceOps(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Infer replace Ops failed";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << " : Init success";
  return SUCCESS;
}

Status GetNextInfo::CheckStrategy(const StrategyPtr &strategy) {
  std::vector<Dimensions> stras = strategy->GetInputDim();
  for (Dimensions stra : stras) {
    if (stra.size() != 0) {
      if (is_auto_parallel_) {
        MS_LOG(DEBUG) << name_ << " : Invalid strategy.";
      } else {
        MS_LOG(ERROR) << name_ << " : Invalid strategy.";
      }
      return FAILED;
    }
  }
  int32_t stage = strategy->GetInputStage();
  int32_t dev_num = SizeToInt(g_device_manager->GetDeviceListByStageId(stage).size());
  dev_num_ = dev_num;
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
      auto iter_cast = iter->second->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(iter_cast);
      auto types = iter_cast->value();
      for (auto &type : types) {
        MS_EXCEPTION_IF_NULL(type);
        types_.push_back(type->ToString());
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
    if (iter->second->isa<Int32Imm>()) {
      output_num_ = iter->second->cast<Int32ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << " : The value of output_num is not int.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GetNextInfo::GetAttrs() {
  if (GetAttrTypes() == FAILED || GetAttrShapes() == FAILED || GetAttrOutPutNum() == FAILED) {
    return FAILED;
  }
  if (types_.size() != IntToSize(output_num_) || shapes_.size() != IntToSize(output_num_) || output_num_ == 0) {
    MS_LOG(ERROR) << name_ << " : The output_num is not equal to shapes size.";
    return FAILED;
  }
  return SUCCESS;
}

Status GetNextInfo::InferReplaceOps(const StrategyPtr &) {
  Shapes out_shapes = outputs_shape_;
  for (size_t i = 0; i < out_shapes.size(); ++i) {
    if (dev_num_ <= 0) {
      MS_LOG(ERROR) << name_ << " : The dev num is 0.";
      return FAILED;
    }
    if (out_shapes[i][0] % dev_num_ != 0) {
      MS_LOG(ERROR) << name_ << " : batch num cannot floor div dev num.";
      return FAILED;
    }
    out_shapes[i][0] = out_shapes[i][0] / dev_num_;
  }
  ValuePtr new_shapes = MakeValue(out_shapes);
  Attr attr_types = std::make_pair(TYPES, attrs_[TYPES]);
  Attr attr_shapes = std::make_pair(SHAPES, new_shapes);
  Attr attr_num = std::make_pair(GETNEXT_NUM, attrs_[GETNEXT_NUM]);
  Attr attr_shared_name = std::make_pair(SHARED_NAME, attrs_[SHARED_NAME]);
  OperatorAttrs attrs = {attr_types, attr_shapes, attr_num, attr_shared_name};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  replace_op_ = {std::make_pair(GET_NEXT, args)};
  return SUCCESS;
}

Status GetNextInfo::InitForCostModel(const StrategyPtr &strategy) {
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

Status GetNextInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Set cost under strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << " : Set cost under strategy failed.";
    }
    return FAILED;
  }
  return SUCCESS;
}

Status GetNextInfo::GenerateStrategies(int32_t stage_id) {
  is_auto_parallel_ = true;
  std::vector<Dimensions> stra;
  StrategyPtr sp = std::make_shared<Strategy>(stage_id, stra);
  if (SetCostUnderStrategy(sp) == SUCCESS) {
    MS_LOG(INFO) << name_ << " : Successfully generated strategy.";
    PrintStrategy(sp);
  } else {
    MS_LOG(ERROR) << name_ << " : Generating strategy failed.";
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
