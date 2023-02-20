/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/strided_slice_v2_grad.h"

#include <string>
#include <memory>
#include <set>
#include <bitset>
#include <vector>
#include <ostream>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t num_eight = 8;
void CheckSliceV2Type(const AbstractBasePtr &input_arg, const std::string &arg_name, const std::string &prim_name) {
  if (input_arg->isa<abstract::AbstractTuple>()) {
    auto temp_value = input_arg->BuildValue();
    (void)CheckAndConvertUtils::CheckTupleInt(arg_name, temp_value, prim_name);
    return;
  } else if (input_arg->isa<abstract::AbstractTensor>()) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid(arg_name, input_arg->BuildType(), {kInt64, kInt32}, prim_name);
    return;
  }
  MS_EXCEPTION(TypeError) << "For 'StridedSlice',  'begin', 'end' and 'stride' must be a tuple or Tensor.";
}

abstract::ShapePtr StridedSliceV2GradInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  const size_t shape_index = 0;
  auto prim_name = primitive->name();
  auto shapex = input_args[shape_index];
  CheckSliceV2Type(shapex, "shapex", prim_name);

  auto out_shape = GetShapeValue(primitive, shapex);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr StridedSliceV2GradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const size_t dy_index = 4;
  const size_t begin_index = 1;
  const size_t end_index = 2;
  const size_t stride_index = 3;
  auto valid_types = common_valid_types;
  valid_types.insert(kComplex128);
  valid_types.insert(kComplex64);
  valid_types.insert(kBool);

  CheckSliceV2Type(input_args[begin_index], "begin", prim_name);
  CheckSliceV2Type(input_args[end_index], "end", prim_name);
  CheckSliceV2Type(input_args[stride_index], "stride", prim_name);
  auto dy_type = input_args[dy_index]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("dy", dy_type, valid_types, prim_name);
  return dy_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(StridedSliceV2Grad, BaseOperator);
AbstractBasePtr StridedSliceV2GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto res = abstract::MakeAbstract(StridedSliceV2GradInferShape(primitive, input_args),
                                    StridedSliceV2GradInferType(primitive, input_args));
  return res;
}

void StridedSliceV2Grad::set_begin_mask(int64_t begin_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kBeginMask, begin_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kBeginMask, api::MakeValue(begin_mask));
}
int64_t StridedSliceV2Grad::get_begin_mask() const {
  auto value_ptr = GetAttr(kBeginMask);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceV2Grad::set_end_mask(int64_t end_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kEndMask, end_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kEndMask, api::MakeValue(end_mask));
}
int64_t StridedSliceV2Grad::get_end_mask() const {
  auto value_ptr = GetAttr(kEndMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceV2Grad::set_ellipsis_mask(int64_t ellipsis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kEllipsisMask, ellipsis_mask, kGreaterEqual, 0, this->name());
  std::bitset<sizeof(int64_t) * num_eight> bs(ellipsis_mask);
  std::ostringstream buffer;
  if (bs.count() > 1) {
    buffer << "For" << this->name() << ", only support one ellipsis in the index, but got " << this->get_end_mask()
           << ".";
    MS_EXCEPTION(ValueError) << buffer.str();
  }
  (void)this->AddAttr(kEllipsisMask, api::MakeValue(ellipsis_mask));
}
int64_t StridedSliceV2Grad::get_ellipsis_mask() const {
  auto value_ptr = GetAttr(kEllipsisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceV2Grad::set_new_axis_mask(int64_t new_axis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kNewAxisMask, new_axis_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kNewAxisMask, api::MakeValue(new_axis_mask));
}
int64_t StridedSliceV2Grad::get_new_axis_mask() const {
  auto value_ptr = GetAttr(kNewAxisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceV2Grad::set_shrink_axis_mask(int64_t shrink_axis_mask) {
  (void)CheckAndConvertUtils::CheckInteger(kShrinkAxisMask, shrink_axis_mask, kGreaterEqual, 0, this->name());
  (void)this->AddAttr(kShrinkAxisMask, api::MakeValue(shrink_axis_mask));
}
int64_t StridedSliceV2Grad::get_shrink_axis_mask() const {
  auto value_ptr = GetAttr(kShrinkAxisMask);
  return GetValue<int64_t>(value_ptr);
}
void StridedSliceV2Grad::Init(int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask,
                              int64_t shrink_axis_mask) {
  this->set_begin_mask(begin_mask);
  this->set_end_mask(end_mask);
  this->set_ellipsis_mask(ellipsis_mask);
  this->set_new_axis_mask(new_axis_mask);
  this->set_shrink_axis_mask(shrink_axis_mask);
}

// AG means auto generated
class MIND_API AGStridedSliceV2GradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return StridedSliceV2GradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return StridedSliceV2GradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return StridedSliceV2GradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(StridedSliceV2Grad, prim::kPrimStridedSliceV2Grad, AGStridedSliceV2GradInfer, false);
}  // namespace ops
}  // namespace mindspore
