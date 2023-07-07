/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/deformable_conv2d.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "ir/dtype/number.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
void DeformableConv2dPadFunction(std::vector<int64_t> *output_hw, const std::vector<int64_t> &kernel_size,
                                 const std::vector<int64_t> &strides, const std::vector<int64_t> &dilations,
                                 const std::vector<int64_t> &pads, int64_t x_h, int64_t x_w, uint64_t h_axis,
                                 uint64_t w_axis) {
  int64_t out_h = -1;
  int64_t out_w = -1;
  constexpr size_t top_index = 0;
  constexpr size_t bottom_index = 1;
  constexpr size_t left_index = 2;
  constexpr size_t right_index = 3;
  if (x_h != abstract::Shape::kShapeDimAny) {
    out_h = static_cast<int64_t>(std::floor(1 + ((x_h * 1.0) + pads[top_index] + pads[bottom_index] - kernel_size[0] -
                                                 LongToFloat((kernel_size[0] - 1) * (dilations[h_axis] - 1))) /
                                                  strides[h_axis]));
  }
  if (x_w != abstract::Shape::kShapeDimAny) {
    out_w = static_cast<int64_t>(std::floor(1 + ((x_w * 1.0) + pads[left_index] + pads[right_index] - kernel_size[1] -
                                                 LongToFloat((kernel_size[1] - 1) * (dilations[w_axis] - 1))) /
                                                  strides[w_axis]));
  }
  output_hw->push_back(out_h);
  output_hw->push_back(out_w);
}

void CheckDeformableConv2dOutputHeightAndWight(const std::string &prim_name, const std::vector<int64_t> &output_hw,
                                               const std::vector<int64_t> &offset_shape) {
  if ((output_hw[kIndex0] != abstract::Shape::kShapeDimAny && offset_shape[kIndex2] != abstract::Shape::kShapeDimAny &&
       output_hw[kIndex0] != offset_shape[kIndex2]) ||
      (output_hw[kIndex1] != abstract::Shape::kShapeDimAny && offset_shape[kIndex3] != abstract::Shape::kShapeDimAny &&
       output_hw[kIndex1] != offset_shape[kIndex3])) {
    MS_EXCEPTION(ValueError)
      << "For '" << prim_name
      << ", the H and W dims of offsets input should be equal to the computed H and W dims of the "
         "output of deformable_conv2d. But got H and W dims of offsets input: ("
      << offset_shape[kIndex2] << ", " << offset_shape[kIndex3]
      << "), computed H and W dims of the output of deformable_conv2d: (" << output_hw[kIndex0] << ", "
      << output_hw[kIndex1] << ").";
  }
}

abstract::ShapePtr DeformableConv2dInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  constexpr int64_t input_size = 3;
  (void)CheckAndConvertUtils::CheckInteger("input_args size", static_cast<int64_t>(input_args.size()), kGreaterEqual,
                                           input_size, prim_name);
  for (const auto &arg : input_args) {
    MS_EXCEPTION_IF_NULL(arg);
  }

  constexpr uint64_t n_axis = 0;
  constexpr uint64_t c_axis = 1;
  constexpr uint64_t h_axis = 2;
  constexpr uint64_t w_axis = 3;
  constexpr size_t strides_num = 4;
  constexpr size_t dilations_num = 4;

  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto filter_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto offsets_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape());

  auto x_shape = x_shape_map[kShape];
  auto offsets_shape = offsets_shape_map[kShape];
  if (IsDynamicRank(x_shape) || IsDynamicRank(offsets_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  constexpr int64_t shape_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("x shape size", SizeToLong(x_shape.size()), kEqual, shape_size, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("offsets shape size", SizeToLong(offsets_shape.size()), kEqual, shape_size,
                                           prim_name);

  auto filter_shape = filter_shape_map[kShape];
  (void)CheckAndConvertUtils::CheckInteger("filter size", SizeToLong(filter_shape.size()), kEqual, shape_size,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("filter height", filter_shape[h_axis], kGreaterThan, 0, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("filter width", filter_shape[w_axis], kGreaterThan, 0, prim_name);

  std::vector<int64_t> strides = CheckAndConvertUtils::CheckAttrTuple(primitive, kAttrStrides, strides_num);
  if (strides[n_axis] != 1 || strides[c_axis] != 1) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', the values of 'strides' according to N and C dimensions must be set to 1, but got N: " << strides[n_axis]
      << ", C: " << strides[c_axis] << ".";
  }

  std::vector<int64_t> dilations = CheckAndConvertUtils::CheckAttrTuple(primitive, kAttrDilations, dilations_num);
  if (dilations[n_axis] != 1 || dilations[c_axis] != 1) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', the values of 'strides' according to N and C dimensions must be set to 1, but got N: " << dilations[n_axis]
      << ", C: " << dilations[c_axis] << ".";
  }

  int64_t deformable_groups =
    CheckAndConvertUtils::CheckAttrInt64Positive(prim_name, primitive->GetAttr(kAttrDfmGroup), kAttrDfmGroup);
  (void)CheckAndConvertUtils::CheckInteger("deformable_groups", deformable_groups, kGreaterThan, 0, prim_name);
  if (x_shape[c_axis] != abstract::Shape::kShapeDimAny && x_shape[c_axis] % deformable_groups != 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', 'C_in' of input 'x' shape must be divisible by 'deformable_groups'"
                             << ", but got 'C_in' of input 'x' shape: " << x_shape[c_axis]
                             << ", and 'deformable_groups': " << deformable_groups << ".";
  }

  constexpr int64_t offsets_channel_factor = 3;
  std::vector<int64_t> kernel_size{filter_shape[h_axis], filter_shape[w_axis]};
  if (offsets_shape[c_axis] != abstract::Shape::kShapeDimAny &&
      offsets_shape[c_axis] != deformable_groups * offsets_channel_factor * kernel_size[0] * kernel_size[1]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', 'C_in' of input 'offsets' shape must be equal to "
                             << offsets_channel_factor << " * kernel_height * kernel_width * deformable_groups"
                             << ", but got 'C_in' of input 'offsets' shape: " << offsets_shape[c_axis]
                             << ", kernel_height: " << kernel_size[0] << ", kernel_width: " << kernel_size[1]
                             << ", deformable_groups: " << deformable_groups << ".";
  }

  if (!GetValue<bool>(primitive->GetAttr(kAttrModulated))) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the value of 'modulated' only support to be set to True.";
  }

  constexpr size_t pads_num = 4;
  std::vector<int64_t> pads = CheckAndConvertUtils::CheckAttrTuple(primitive, kPads, pads_num);
  std::vector<int64_t> output_hw;
  DeformableConv2dPadFunction(&output_hw, kernel_size, strides, dilations, pads, x_shape[h_axis], x_shape[w_axis],
                              h_axis, w_axis);
  CheckDeformableConv2dOutputHeightAndWight(prim_name, output_hw, offsets_shape);

  ShapeVector output_shape{x_shape[n_axis], x_shape[c_axis], offsets_shape[h_axis], offsets_shape[w_axis]};
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr DeformableConv2dInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(DeformableConv2d, BaseOperator);
void DeformableConv2d::Init(const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                            const std::vector<int64_t> &dilations, int64_t groups, const std::string &data_format,
                            int64_t deformable_groups, bool modulated) {
  set_strides(strides);
  set_pads(pads);
  set_dilations(dilations);
  set_data_format(data_format);
  set_groups(groups);
  set_deformable_groups(deformable_groups);
  set_modulated(modulated);
}

void DeformableConv2d::set_groups(int64_t groups) {
  (void)AddAttr(kAttrGroups,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kAttrGroups, groups, kGreaterThan, 0, name())));
}

int64_t DeformableConv2d::get_groups() const { return GetValue<int64_t>(GetAttr(kAttrGroups)); }

void DeformableConv2d::set_strides(const std::vector<int64_t> &strides) {
  constexpr int64_t strides_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("strides_size", SizeToLong(strides.size()), kEqual, strides_num, name());
  (void)AddAttr(kStrides, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStrides, strides, name())));
}

std::vector<int64_t> DeformableConv2d::get_strides() const { return GetValue<std::vector<int64_t>>(GetAttr(kStrides)); }

void DeformableConv2d::set_pads(const std::vector<int64_t> &pads) {
  constexpr int64_t pads_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("pads_size", SizeToLong(pads.size()), kEqual, pads_num, name());
  (void)AddAttr(kPads, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPads, pads, name())));
}

std::vector<int64_t> DeformableConv2d::get_pads() const { return GetValue<std::vector<int64_t>>(GetAttr(kPads)); }

void DeformableConv2d::set_dilations(const std::vector<int64_t> &dilations) {
  constexpr int64_t dilations_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("dilations_size", SizeToLong(dilations.size()), kEqual, dilations_size,
                                           name());
  (void)AddAttr(kAttrDilations,
                api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kAttrDilations, dilations, name())));
}

std::vector<int64_t> DeformableConv2d::get_dilations() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kAttrDilations));
}

void DeformableConv2d::set_data_format(const std::string &data_format) {
  (void)AddAttr(kFormat, api::MakeValue(data_format));
}

std::string DeformableConv2d::get_data_format() const { return GetValue<std::string>(GetAttr(kFormat)); }

void DeformableConv2d::set_deformable_groups(int64_t deformable_groups) {
  (void)AddAttr(kAttrDfmGroup, api::MakeValue(CheckAndConvertUtils::CheckInteger(kAttrDfmGroup, deformable_groups,
                                                                                 kGreaterThan, 0, name())));
}

int64_t DeformableConv2d::get_deformable_groups() const { return GetValue<int64_t>(GetAttr(kAttrDfmGroup)); }

void DeformableConv2d::set_modulated(bool modulated) { (void)AddAttr(kAttrModulated, api::MakeValue(modulated)); }

bool DeformableConv2d::get_modulated() const { return GetValue<bool>(GetAttr(kAttrModulated)); }

AbstractBasePtr DeformableConv2dInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("DeformableConv2d infer", SizeToLong(input_args.size()), kGreaterEqual,
                                           input_num, primitive->name());
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("offsets", input_args[1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return abstract::MakeAbstract(DeformableConv2dInferShape(primitive, input_args),
                                DeformableConv2dInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGDeformableConv2dInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DeformableConv2dInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DeformableConv2dInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DeformableConv2dInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DeformableConv2d, prim::kPrimDeformableConv2d, AGDeformableConv2dInfer, false);
}  // namespace ops
}  // namespace mindspore
