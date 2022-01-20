/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/grad/strided_slice_grad.h"
#include <string>
#include <memory>
#include <set>
#include <bitset>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
void CheckSliceType(const AbstractBasePtr &input_arg, const std::string &arg_name, const std::string &prim_name) {
  if (input_arg->isa<abstract::AbstractTuple>()) {
    auto temp_value = input_arg->BuildValue();
    (void)CheckAndConvertUtils::CheckTupleInt(arg_name, temp_value, prim_name);
    return;
  } else if (input_arg->isa<abstract::AbstractTensor>()) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid(arg_name, input_arg->BuildType(), {kInt64}, prim_name);
    return;
  }
  MS_EXCEPTION(TypeError) << "For StridedSlice, begin, end and stride must be tuple or Tensor.";
}

abstract::ShapePtr StridedSliceGradInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  const size_t shape_index = 1;
  auto prim_name = primitive->name();
  auto shapex = input_args[shape_index];
  CheckSliceType(shapex, "shapex", prim_name);
  auto shape_value = shapex->BuildValue();
  MS_EXCEPTION_IF_NULL(shape_value);

  ShapeVector out_shape;
  abstract::ShapePtr ret_shape;

  if (shapex->isa<abstract::AbstractTuple>()) {
    out_shape = CheckAndConvertUtils::CheckTupleInt("input[shapex]", shape_value, prim_name);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  if (!shapex->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For StridedSliceGrad, shapex must be tuple or Tensor.";
  }

  if (shape_value->isa<tensor::Tensor>()) {
    out_shape = CheckAndConvertUtils::CheckTensorIntValue("shapex", shape_value, prim_name);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  // shape_value is AnyValue
  auto shapex_shape = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, shape_index);
  if (shapex_shape->shape().size() != 1) {
    MS_EXCEPTION(ValueError) << "For StridedSliceGrad, shapex must be 1-D.";
  }
  auto shapex_len = LongToSize(shapex_shape->shape()[0]);
  auto abstract_tensor = shapex->cast<abstract::AbstractTensorPtr>();
  auto shape_min_value = abstract_tensor->get_min_value();
  auto shape_max_value = abstract_tensor->get_max_value();
  if (shape_min_value == nullptr || shape_max_value == nullptr) {
    MS_LOG(EXCEPTION) << "Max_value or min value of shapex can not be empty when shapex is not a constant.";
  }

  auto shape_max = GetValue<std::vector<int64_t>>(shape_max_value);
  auto shape_min = GetValue<std::vector<int64_t>>(shape_min_value);
  if (shape_max.size() != shapex_len || shape_min.size() != shapex_len) {
    MS_LOG(EXCEPTION) << "For " << prim_name << ", shapex's min value size: " << shape_min.size()
                      << ", or max value size: " << shape_max.size() << ", not match with shapex size: " << shapex_len;
  }
  for (size_t i = 0; i < shapex_len; i++) {
    if (shape_min[i] == shape_max[i]) {
      out_shape.push_back(shape_min[i]);
    } else {
      out_shape.push_back(abstract::Shape::SHP_ANY);
    }
  }
  return std::make_shared<abstract::Shape>(out_shape, shape_min, shape_max);
}

TypePtr StridedSliceGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const size_t dy_index = 0;
  const size_t begin_index = 2;
  const size_t end_index = 3;
  const size_t stride_index = 4;
  auto valid_types = common_valid_types;
  valid_types.insert(kComplex128);
  valid_types.insert(kComplex64);
  valid_types.insert(kBool);

  CheckSliceType(input_args[begin_index], "begin", prim_name);
  CheckSliceType(input_args[end_index], "end", prim_name);
  CheckSliceType(input_args[stride_index], "stride", prim_name);
  auto dy_type = input_args[dy_index]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("dy", dy_type, valid_types, prim_name);
  return dy_type;
}
}  // namespace

AbstractBasePtr StridedSliceGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto res = abstract::MakeAbstract(StridedSliceGradInferShape(primitive, input_args),
                                    StridedSliceGradInferType(primitive, input_args));
  return res;
}

void StridedSliceGrad::set_begin_mask(int64_t begin_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kBeginMask, begin_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kBeginMask, MakeValue(begin_mask));
}
int64_t StridedSliceGrad::get_begin_mask() const {
  auto value_ptr = GetAttr(kBeginMask);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceGrad::set_end_mask(int64_t end_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kEndMask, end_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kEndMask, MakeValue(end_mask));
}
int64_t StridedSliceGrad::get_end_mask() const {
  auto value_ptr = GetAttr(kEndMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceGrad::set_ellipsis_mask(int64_t ellipsis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kEllipsisMask, ellipsis_mask, kGreaterEqual, 0, this->name());
  std::bitset<sizeof(int64_t) * 8> bs(ellipsis_mask);
  std::ostringstream buffer;
  if (bs.count() > 1) {
    buffer << "For" << this->name() << ", only support one ellipsis in the index, but got " << this->get_end_mask();
    MS_EXCEPTION(ValueError) << buffer.str();
  }
  (void)this->AddAttr(kEllipsisMask, MakeValue(ellipsis_mask));
}
int64_t StridedSliceGrad::get_ellipsis_mask() const {
  auto value_ptr = GetAttr(kEllipsisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceGrad::set_new_axis_mask(int64_t new_axis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kNewAxisMask, new_axis_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kNewAxisMask, MakeValue(new_axis_mask));
}
int64_t StridedSliceGrad::get_new_axis_mask() const {
  auto value_ptr = GetAttr(kNewAxisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceGrad::set_shrink_axis_mask(int64_t shrink_axis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kShrinkAxisMask, shrink_axis_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kShrinkAxisMask, MakeValue(shrink_axis_mask));
}
int64_t StridedSliceGrad::get_shrink_axis_mask() const {
  auto value_ptr = GetAttr(kShrinkAxisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceGrad::Init(int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask,
                            int64_t shrink_axis_mask) {
  this->set_begin_mask(begin_mask);
  this->set_end_mask(end_mask);
  this->set_ellipsis_mask(ellipsis_mask);
  this->set_new_axis_mask(new_axis_mask);
  this->set_shrink_axis_mask(shrink_axis_mask);
}
REGISTER_PRIMITIVE_EVAL_IMPL(StridedSliceGrad, prim::kPrimStridedSliceGrad, StridedSliceGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
