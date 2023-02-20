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

#include "ops/deformable_offsets.h"

#include <map>
#include <set>
#include <algorithm>
#include <memory>
#include <cmath>
#include <iterator>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
int64_t CheckAttrInt64Positive(const std::string &op, const ValuePtr &attr, const std::string &attr_name) {
  MS_EXCEPTION_IF_NULL(attr);
  int64_t attr_val = attr->cast<Int64ImmPtr>()->value();
  if (attr_val <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << op << "', the '" << attr_name
                             << "' should be greater than 0, but got: " << attr_val << ".";
  }
  return attr_val;
}

std::vector<int64_t> CheckAttrTuple(const PrimitivePtr &prim, const std::string &attr_name, size_t num_element) {
  auto attr = prim->GetAttr(attr_name);
  MS_EXCEPTION_IF_NULL(attr);
  std::vector<int64_t> result;
  if (!attr->isa<ValueTuple>()) {
    MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', the '" << attr_name
                             << "' should be a tuple[int64], but got: " << attr->ToString() << ".";
  }
  std::vector<ValuePtr> attr_vec = attr->cast<ValueTuplePtr>()->value();
  if (attr_vec.size() != num_element) {
    MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', the '" << attr_name
                             << "' should be a tuple[int64] with size " << num_element << ", but its size is "
                             << attr_vec.size() << ".";
  }
  (void)std::transform(attr_vec.begin(), attr_vec.end(), std::back_inserter(result),
                       [&prim, &attr_name](const ValuePtr &e) -> int64_t {
                         auto value = GetValue<int64_t>(e);
                         if (value < 0) {
                           MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', the element of '" << attr_name
                                                    << "' should not be negative number, but got " << value << ".";
                         }
                         return value;
                       });
  return result;
}

std::vector<int64_t> CheckAttrTupleAndNCDimensions(const PrimitivePtr &primitive, const std::string &attr_name,
                                                   size_t num, uint64_t n_axis, uint64_t c_axis) {
  std::vector<int64_t> tuple = CheckAttrTuple(primitive, attr_name, num);
  if (tuple[n_axis] != 1 || tuple[c_axis] != 1) {
    MS_EXCEPTION(ValueError)
      << "For '" << primitive->name()
      << "', the values of 'strides' according to N and C dimensions must be set to 1, but got N: " << tuple[n_axis]
      << ", C: " << tuple[c_axis] << ".";
  }
  return tuple;
}

void DeformableOffsetsPadFunction(std::vector<int64_t> *output_hw, const std::vector<int64_t> &kernel_size,
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

void CheckOutputHeightAndWight(const std::string &prim_name, const std::vector<int64_t> &output_hw,
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

abstract::ShapePtr DeformableOffsetsInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &arg : input_args) {
    MS_EXCEPTION_IF_NULL(arg);
  }
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto offsets_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  auto offsets_shape = offsets_shape_map[kShape];
  if (IsDynamicRank(x_shape) || IsDynamicRank(offsets_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  constexpr int64_t shape_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("x shape size", SizeToLong(x_shape.size()), kEqual, shape_size, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("offsets shape size", SizeToLong(offsets_shape.size()), kEqual, shape_size,
                                           prim_name);

  constexpr uint64_t n_axis = 0;
  constexpr uint64_t c_axis = 1;
  constexpr uint64_t h_axis = 2;
  constexpr uint64_t w_axis = 3;
  constexpr size_t strides_num = 4;
  auto strides = CheckAttrTupleAndNCDimensions(primitive, kAttrStrides, strides_num, n_axis, c_axis);

  constexpr size_t dilations_num = 4;
  auto dilations = CheckAttrTupleAndNCDimensions(primitive, kAttrDilations, dilations_num, n_axis, c_axis);

  int64_t deformable_groups = CheckAttrInt64Positive(prim_name, primitive->GetAttr(kAttrDfmGroup), kAttrDfmGroup);
  if (x_shape[c_axis] != abstract::Shape::kShapeDimAny && x_shape[c_axis] % deformable_groups != 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', 'C_in' of input 'x' shape must be divisible by 'deformable_groups'"
                             << ", but got 'C_in' of input 'x' shape: " << x_shape[c_axis]
                             << ", and 'deformable_groups': " << deformable_groups << ".";
  }

  constexpr int64_t offsets_channel_factor = 3;
  constexpr size_t kernel_size_num = 2;
  std::vector<int64_t> kernel_size = CheckAttrTuple(primitive, kAttrKsize, kernel_size_num);
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
  std::vector<int64_t> pads = CheckAttrTuple(primitive, kPads, pads_num);
  std::vector<int64_t> output_hw;
  DeformableOffsetsPadFunction(&output_hw, kernel_size, strides, dilations, pads, x_shape[h_axis], x_shape[w_axis],
                               h_axis, w_axis);
  CheckOutputHeightAndWight(prim_name, output_hw, offsets_shape);

  ShapeVector output_shape{x_shape[n_axis], x_shape[c_axis], output_hw[0] * kernel_size[0],
                           output_hw[1] * kernel_size[1]};
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr DeformableOffsetsInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(DeformableOffsets, BaseOperator);
void DeformableOffsets::Init(const std::vector<int64_t> &strides, const std::vector<int64_t> &pads,
                             const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &dilations,
                             const std::string &data_format, int64_t deformable_groups, bool modulated) {
  set_strides(strides);
  set_pads(pads);
  set_kernel_size(kernel_size);
  set_dilations(dilations);
  set_data_format(data_format);
  set_deformable_groups(deformable_groups);
  set_modulated(modulated);
}

void DeformableOffsets::set_strides(const std::vector<int64_t> &strides) {
  constexpr int64_t strides_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("strides_size", SizeToLong(strides.size()), kEqual, strides_num, name());
  (void)AddAttr(kStrides, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStrides, strides, name())));
}

std::vector<int64_t> DeformableOffsets::get_strides() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kStrides));
}

void DeformableOffsets::set_pads(const std::vector<int64_t> &pads) {
  constexpr int64_t pads_num = 4;
  (void)CheckAndConvertUtils::CheckInteger("pads_size", SizeToLong(pads.size()), kEqual, pads_num, name());
  (void)AddAttr(kPads, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPads, pads, name())));
}

std::vector<int64_t> DeformableOffsets::get_pads() const { return GetValue<std::vector<int64_t>>(GetAttr(kPads)); }

void DeformableOffsets::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  constexpr int64_t kernel_size_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("kernel_size_size", SizeToLong(kernel_size.size()), kEqual, kernel_size_num,
                                           name());
  (void)AddAttr(kAttrKsize, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kAttrKsize, kernel_size, name())));
}

std::vector<int64_t> DeformableOffsets::get_kernel_size() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kAttrKsize));
}

void DeformableOffsets::set_dilations(const std::vector<int64_t> &dilations) {
  constexpr int64_t dilations_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("dilations_size", SizeToLong(dilations.size()), kEqual, dilations_size,
                                           name());
  (void)AddAttr(kAttrDilations,
                api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kAttrDilations, dilations, name())));
}

std::vector<int64_t> DeformableOffsets::get_dilations() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kAttrDilations));
}

void DeformableOffsets::set_data_format(const std::string &data_format) {
  (void)AddAttr(kFormat, api::MakeValue(data_format));
}

std::string DeformableOffsets::get_data_format() const { return GetValue<std::string>(GetAttr(kFormat)); }

void DeformableOffsets::set_deformable_groups(int64_t deformable_groups) {
  (void)AddAttr(kAttrDfmGroup, api::MakeValue(CheckAndConvertUtils::CheckInteger(kAttrDfmGroup, deformable_groups,
                                                                                 kGreaterThan, 0, name())));
}

int64_t DeformableOffsets::get_deformable_groups() const { return GetValue<int64_t>(GetAttr(kAttrDfmGroup)); }

void DeformableOffsets::set_modulated(bool modulated) { (void)AddAttr(kAttrModulated, api::MakeValue(modulated)); }

bool DeformableOffsets::get_modulated() const { return GetValue<bool>(GetAttr(kAttrModulated)); }

AbstractBasePtr DeformableOffsetsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("DeformableOffsets infer", SizeToLong(input_args.size()), kGreaterEqual,
                                           input_num, primitive->name());
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("offsets", input_args[1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return abstract::MakeAbstract(DeformableOffsetsInferShape(primitive, input_args),
                                DeformableOffsetsInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGDeformableOffsetsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DeformableOffsetsInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DeformableOffsetsInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return DeformableOffsetsInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DeformableOffsets, prim::kPrimDeformableOffsets, AGDeformableOffsetsInfer, false);
}  // namespace ops
}  // namespace mindspore
