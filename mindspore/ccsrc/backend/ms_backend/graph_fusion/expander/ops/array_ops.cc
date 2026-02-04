/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/graph_fusion/expander/base/ir_builder.h"
#include "backend/ms_backend/graph_fusion/expander/base/utils.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::graphkernel::expander {
namespace {
bool GetShapeValue(const NodePtr &node, ShapeVector *shape) {
  if (node == nullptr || shape == nullptr) {
    return false;
  }
  auto value = node->GetValue();
  if (value == nullptr || !IsValueKnown(value)) {
    MS_LOG(DEBUG) << "input is not const value";
    return false;
  }
  if (value->isa<ValueSequence>()) {
    auto seq = value->cast<ValueSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq);
    for (const auto &v : seq->value()) {
      if (v == nullptr) {
        return false;
      }
      if (v->isa<Int64Imm>()) {
        shape->push_back(GetValue<int64_t>(v));
      } else {
        MS_LOG(DEBUG) << "ValueSequence element is not int64";
        return false;
      }
    }
  } else if (value->isa<tensor::Tensor>()) {
    *shape = CheckAndConvertUtils::CheckTensorIntValue("input", value, "GetShapeValue");
  } else {
    MS_LOG(DEBUG) << "input is not ValueSequence or Tensor";
    return false;
  }
  return true;
}
}  // namespace

REG_EXPANDER_FUNC("Identity").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(0);
  auto x_shape = input_x->GetShape();
  if (IsDynamicRank(x_shape) || std::count_if(x_shape.begin(), x_shape.end(), [](int64_t sh) { return sh < 0; }) > 1) {
    MS_LOG(DEBUG) << "Skip dynamic shape case";
    return {};
  }
  auto result = ib->Reshape(input_x, ib->Tensor(x_shape));
  return {result};
});

REG_EXPANDER_FUNC("ZerosLike").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex0);
  auto x_shape = input_x->GetShape();
  if (IsDynamic(x_shape)) {
    MS_LOG(DEBUG) << "Skip dynamic shape case";
    return {};
  }
  if (IsShapeEmpty(x_shape)) {
    MS_LOG(DEBUG) << "Skip empty shape case";
    return {};
  }
  auto shape = ib->Value(x_shape);
  auto const_zero = ib->Tensor(0, input_x->GetDtype());
  auto result = ib->BroadcastTo(const_zero, shape);
  return {result};
});

REG_EXPANDER_FUNC("ZerosLikeExt").SetBody(BODYFUNC(ib) {
  const auto &input = ib->input(kIndex0);
  auto input_type = input->GetDtype();
  auto dtype = ib->input(kIndex1);
  auto input_shape = input->GetShape();
  auto out_type = dtype->GetDtype() != TypeIdToType(kMetaTypeNone)
                    ? TypeIdToType(static_cast<TypeId>(GetValue<int64_t>(dtype->GetValue())))
                    : input_type;
  if (IsDynamic(input_shape)) {
    MS_LOG(DEBUG) << "ZerosLikeExt Skip dynamic shape case";
    return {};
  }
  if (IsShapeEmpty(input_shape)) {
    MS_LOG(DEBUG) << "ZerosLikeExt Skip empty shape case";
    return {};
  }
  auto const_zero = ib->Tensor(0, out_type);
  auto result = ib->BroadcastTo(const_zero, ib->Value(input_shape));
  return {result};
});

REG_EXPANDER_FUNC("FillV2").SetBody(BODYFUNC(ib) {
  const auto &shape = ib->input(kIndex0);
  ShapeVector output_shape;
  if (!GetShapeValue(shape, &output_shape)) {
    MS_LOG(DEBUG) << "Fail to get shape value";
    return {};
  }
  const auto &val = ib->input(kIndex1);
  auto value_ptr = val->GetValue();
  if (value_ptr == nullptr || !IsValueKnown(value_ptr)) {
    MS_LOG(DEBUG) << "value is not const value";
    return {};
  }
  auto value_type = val->GetDtype();
  MS_EXCEPTION_IF_NULL(value_type);
  auto value_type_id = value_type->type_id();
  if (value_type_id != kNumberTypeFloat16 && value_type_id != kNumberTypeFloat32 &&
      value_type_id != kNumberTypeBFloat16 && value_type_id != kNumberTypeInt32) {
    MS_LOG(DEBUG) << "value data type is not supported: " << TypeIdToString(value_type_id);
    return {};
  }
  auto result = ib->BroadcastTo(val, ib->Value(output_shape));
  return {result};
});

REG_EXPANDER_FUNC("RepeatInterleaveInt").SetBody(BODYFUNC(ib) {
  auto input = ib->input(kIndex0);
  auto shape = input->GetShape();
  if (IsDynamicRank(shape)) {
    MS_LOG(DEBUG) << "Input is dynamic rank";
    return {};
  }
  auto repeat = ib->input(kIndex1);
  auto repeat_value_ptr = repeat->GetValue();
  if (repeat_value_ptr == nullptr || !IsValueKnown(repeat_value_ptr)) {
    MS_LOG(DEBUG) << "repeat is not const value";
    return {};
  }
  auto repeat_value = GetValue<int64_t>(repeat_value_ptr);
  const auto &dim = ib->input(kIndex2);
  auto dim_value_ptr = dim->GetValue();
  if (dim_value_ptr == nullptr || !IsValueKnown(dim_value_ptr)) {
    MS_LOG(DEBUG) << "dim is not const value";
    return {};
  }
  auto dim_value = GetValue<int64_t>(dim_value_ptr);
  if (dim_value < 0) {
    dim_value = shape.size() + dim_value;
  }
  if (shape[dim_value] == 1) {
    shape[dim_value] = repeat_value;
    return {ib->BroadcastTo(input, ib->Tensor(shape))};
  }
  shape.insert(shape.begin() + dim_value + 1, 1);
  auto expand = ib->Reshape(input, ib->Tensor(shape));
  shape[dim_value + 1] = repeat_value;
  auto broadcast = ib->BroadcastTo(expand, ib->Tensor(shape));
  auto res_shape = input->GetShape();
  res_shape[dim_value] = res_shape[dim_value] * repeat_value;
  auto result = ib->Reshape(broadcast, ib->Tensor(res_shape));
  return {result};
});
}  // namespace mindspore::graphkernel::expander
