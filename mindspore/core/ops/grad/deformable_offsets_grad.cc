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

#include <vector>
#include <string>
#include <memory>
#include <set>

#include "ops/grad/deformable_offsets_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kDeformableOffsetsGradInputDoutIndex = 0;
constexpr size_t kDeformableOffsetsGradInputInputIndex = 1;
constexpr size_t kDeformableOffsetsGradInputOffsetsIndex = 2;
constexpr int64_t kDeformableOffsetsGradInputSize = 3;

std::vector<abstract::BaseShapePtr> DeformableOffsetsGradInferShape(const PrimitivePtr &,
                                                                    const std::vector<AbstractBasePtr> &input_args) {
  auto dx_shape = input_args[kDeformableOffsetsGradInputInputIndex]->BuildShape();
  auto d_offset_mask_shape = input_args[kDeformableOffsetsGradInputOffsetsIndex]->BuildShape();
  return {dx_shape, d_offset_mask_shape};
}

std::vector<TypePtr> DeformableOffsetsGradInferType(const PrimitivePtr &prim,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();

  auto dout_type = input_args[kDeformableOffsetsGradInputDoutIndex]->BuildType();
  auto x_type = input_args[kDeformableOffsetsGradInputInputIndex]->BuildType();
  auto offsets_type = input_args[kDeformableOffsetsGradInputOffsetsIndex]->BuildType();

  std::set<TypePtr> valid_type = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTypeValid("dout", dout_type, valid_type, prim_name);
  auto x_elem_type = CheckAndConvertUtils::CheckTypeValid("x", x_type, valid_type, prim_name);
  auto offset_elem_type = CheckAndConvertUtils::CheckTypeValid("offsets_type", offsets_type, valid_type, prim_name);

  return {x_elem_type, offset_elem_type};
}
}  // namespace

MIND_API_OPERATOR_IMPL(DeformableOffsetsGrad, BaseOperator);
AbstractBasePtr DeformableOffsetsGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // check inputs num.
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kDeformableOffsetsGradInputSize, prim_name);
  auto out_grad_types = DeformableOffsetsGradInferType(primitive, input_args);
  auto out_grad_shapes = DeformableOffsetsGradInferShape(primitive, input_args);
  std::vector<abstract::AbstractBasePtr> out_grads_abs;
  for (size_t i = 0; i < out_grad_shapes.size(); ++i) {
    auto grad_i_abs = std::make_shared<abstract::AbstractTensor>(out_grad_types[i], out_grad_shapes[i]);
    (void)out_grads_abs.emplace_back(grad_i_abs);
  }
  return std::make_shared<abstract::AbstractTuple>(out_grads_abs);
}

void DeformableOffsetsGrad::Init(const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                                 const std::vector<int64_t> &ksize, const std::vector<int64_t> &dilations,
                                 const std::string &data_format, int64_t deformable_groups, bool modulated) {
  set_strides(strides);
  set_pads(pads);
  set_kernel_size(ksize);
  set_dilations(dilations);
  set_format(data_format);
  set_deformable_groups(deformable_groups);
  set_modulated(modulated);
}

void DeformableOffsetsGrad::set_strides(const std::vector<int64_t> &stride) {
  (void)AddAttr(kStrides, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStrides, stride, name())));
}

void DeformableOffsetsGrad::set_pads(const std::vector<int64_t> &pad) {
  const int64_t pad_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("pad_size_size", SizeToLong(pad.size()), kEqual, pad_size, name());
  (void)AddAttr(kPads, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPads, pad, name())));
}

void DeformableOffsetsGrad::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)AddAttr(kKSize, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKSize, kernel_size, name())));
}

void DeformableOffsetsGrad::set_dilations(const std::vector<int64_t> &dilation) {
  (void)AddAttr(kDilations, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kDilations, dilation, name())));
}

void DeformableOffsetsGrad::set_format(const std::string &format) { (void)AddAttr(kFormat, api::MakeValue(format)); }

void DeformableOffsetsGrad::set_deformable_groups(int64_t group) {
  (void)AddAttr(kDeformableGroups,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kDeformableGroups, group, kGreaterThan, 0, name())));
}

void DeformableOffsetsGrad::set_modulated(bool modulated) { (void)AddAttr(kModulated, api::MakeValue(modulated)); }

std::vector<int64_t> DeformableOffsetsGrad::get_strides() const {
  auto value_ptr = GetAttr(kStrides);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> DeformableOffsetsGrad::get_pads() const {
  auto value_ptr = GetAttr(kPads);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> DeformableOffsetsGrad::get_kernel_size() const {
  auto value_ptr = GetAttr(kKSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> DeformableOffsetsGrad::get_dilations() const {
  auto value_ptr = GetAttr(kDilations);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::string DeformableOffsetsGrad::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::string>(value_ptr);
}

int64_t DeformableOffsetsGrad::get_deformable_groups() const {
  auto value_ptr = GetAttr(kDeformableGroups);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<int64_t>(value_ptr);
}

bool DeformableOffsetsGrad::get_modulated() const {
  auto value_ptr = GetAttr(kModulated);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

REGISTER_PRIMITIVE_EVAL_IMPL(DeformableOffsetsGrad, prim::kPrimDeformableOffsetsGrad, DeformableOffsetsGradInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
