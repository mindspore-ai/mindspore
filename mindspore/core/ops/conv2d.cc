/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/conv2d.h"

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <cmath>
#include <iterator>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

using mindspore::abstract::Shape;
namespace mindspore {
namespace ops {
namespace {
// check functions
constexpr size_t kernel_size_num = 2;
constexpr size_t stride_num = 2;
constexpr size_t dilation_num = 2;
constexpr size_t padding_num = 4;
constexpr size_t start_index = 2;
constexpr size_t top_padding = 0;
constexpr size_t bottom_padding = 1;
constexpr size_t left_padding = 2;
constexpr size_t right_padding = 3;

void CheckShapeAnyAndPositive(const std::string &op, const ShapeVector &shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if ((shape[i] < 0) && (shape[i] != abstract::Shape::kShapeDimAny)) {
      MS_EXCEPTION(ValueError) << "For '" << op << "',  shape element [" << i
                               << "] must be positive integer or -1, but got: " << shape[i] << ".";
    }
  }
}

int64_t CheckAttrPositiveInt64(const std::string &op, const ValuePtr &attr, const std::string &attr_name) {
  MS_EXCEPTION_IF_NULL(attr);
  auto attr_value = attr->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(attr_value);
  int64_t attr_val = attr_value->value();
  if (attr_val <= 0) {
    MS_LOG(EXCEPTION) << "For '" << op << "', '" << attr_name << "' should be greater than 0, but got: " << attr_val
                      << ".";
  }
  return attr_val;
}

std::vector<int64_t> CheckAttrIntOrTuple(const ValuePtr &attr, const size_t start_idx, const size_t num_element) {
  std::vector<int64_t> result;
  MS_EXCEPTION_IF_NULL(attr);
  if (attr->isa<ValueTuple>()) {
    std::vector<ValuePtr> attr_vec = attr->cast<ValueTuplePtr>()->value();
    if (attr_vec.size() < start_idx + num_element) {
      MS_LOG(EXCEPTION) << "ValueTuple size verify failed. ValueTuple size is " << attr_vec.size()
                        << ", start index is " << start_idx << ", element number is " << num_element;
    }
    auto it_start = attr_vec.begin() + SizeToLong(start_idx);
    (void)std::transform(it_start, it_start + SizeToLong(num_element), std::back_inserter(result),
                         [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  } else {
    int64_t attr_val = attr->cast<Int64ImmPtr>()->value();
    (void)result.insert(result.begin(), num_element, attr_val);
  }
  return result;
}

void Conv2DPadFunction(std::vector<int64_t> *output_hw, std::vector<int64_t> *pad_list, const int64_t x_h,
                       const int64_t x_w, const std::vector<int64_t> &kernel, const std::vector<int64_t> &stride,
                       const std::vector<int64_t> &dilation, const int64_t &pad_mode,
                       const std::vector<int64_t> &padding, const bool is_min_shape = false) {
  MS_EXCEPTION_IF_NULL(pad_list);
  if (pad_mode == PadMode::VALID) {
    int64_t out_h = -1;
    int64_t out_w = -1;
    if (x_h != abstract::Shape::kShapeDimAny) {
      out_h =
        static_cast<int64_t>(std::ceil(((x_h * 1.0) - static_cast<float>(dilation[0] * (kernel[0] - 1))) / stride[0]));
      if (is_min_shape && out_h < 1) {
        out_h = 1L;
      }
    }
    if (x_w != abstract::Shape::kShapeDimAny) {
      out_w =
        static_cast<int64_t>(std::ceil(((x_w * 1.0) - static_cast<float>(dilation[1] * (kernel[1] - 1))) / stride[1]));
      if (is_min_shape && out_w < 1) {
        out_w = 1L;
      }
    }
    output_hw->push_back(out_h);
    output_hw->push_back(out_w);
    constexpr size_t pad_size = 4;
    (void)pad_list->insert(pad_list->begin(), pad_size, 0);
  } else if (pad_mode == PadMode::SAME) {
    if (x_h == abstract::Shape::kShapeDimAny) {
      output_hw->push_back(abstract::Shape::kShapeDimAny);
      pad_list->push_back(abstract::Shape::kShapeDimAny);
      pad_list->push_back(abstract::Shape::kShapeDimAny);
    } else {
      output_hw->push_back(static_cast<int64_t>(std::ceil((x_h * 1.0) / stride[0])));
      int64_t pad_needed_h = (output_hw->at(0) - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 - x_h;
      pad_needed_h = std::max(int64_t(0), pad_needed_h);
      pad_list->push_back(static_cast<int64_t>(std::floor(pad_needed_h / 2)));
      pad_list->push_back(pad_needed_h - pad_list->at(0));
    }

    if (x_w == abstract::Shape::kShapeDimAny) {
      output_hw->push_back(abstract::Shape::kShapeDimAny);
      pad_list->push_back(abstract::Shape::kShapeDimAny);
      pad_list->push_back(abstract::Shape::kShapeDimAny);
    } else {
      output_hw->push_back(static_cast<int64_t>(std::ceil((x_w * 1.0) / stride[1])));
      int64_t pad_needed_w = (output_hw->at(1) - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 - x_w;
      pad_needed_w = std::max(int64_t(0), pad_needed_w);
      pad_list->push_back(static_cast<int64_t>(std::floor(pad_needed_w / 2)));
      pad_list->push_back(pad_needed_w - pad_list->at(kInputIndex2));
    }
  } else if (pad_mode == PadMode::PAD) {
    (void)pad_list->insert(pad_list->begin(), padding.begin(), padding.end());
    int64_t out_h = -1;
    int64_t out_w = -1;
    if (x_h != abstract::Shape::kShapeDimAny) {
      out_h = static_cast<int64_t>(std::floor(1 + ((x_h * 1.0) + pad_list->at(0) + pad_list->at(1) - kernel[0] -
                                                   static_cast<float>((kernel[0] - 1) * (dilation[0] - 1))) /
                                                    stride[0]));
      if (is_min_shape && out_h < 1) {
        out_h = 1L;
      }
    }
    if (x_w != abstract::Shape::kShapeDimAny) {
      out_w =
        static_cast<int64_t>(std::floor(1 + ((x_w * 1.0) + pad_list->at(kInputIndex2) + pad_list->at(kInputIndex3) -
                                             kernel[1] - static_cast<float>((kernel[1] - 1) * (dilation[1] - 1))) /
                                              stride[1]));
      if (is_min_shape && out_w < 1) {
        out_w = 1L;
      }
    }
    output_hw->push_back(out_h);
    output_hw->push_back(out_w);
  }
}

bool CheckConv2dShape(const std::string &prim_name, const std::vector<AbstractBasePtr> &input_args,
                      const std::vector<int64_t> &x_shape, const std::vector<int64_t> &w_shape,
                      const std::vector<int64_t> &padding, int64_t pad_mode, uint64_t w_axis, uint64_t h_axis) {
  auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  auto w_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 1);
  if (x_shape_ptr->IsDynamic() || w_shape_ptr->IsDynamic()) {
    return true;
  }
  if (w_shape[w_axis] != abstract::Shape::kShapeDimAny && pad_mode != PadMode::SAME) {
    int64_t input_height = x_shape[h_axis];
    int64_t input_width = x_shape[w_axis];
    if (pad_mode == PadMode::PAD) {
      input_width += padding[left_padding] + padding[right_padding];
      input_height += padding[top_padding] + padding[bottom_padding];
    }
    if (input_height < w_shape[h_axis] || input_width < w_shape[w_axis]) {
      return false;
    }
  }
  return true;
}

abstract::ShapePtr Conv2dInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto w_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  auto w_shape = w_shape_map[kShape];

  ShapeVector output_shape;
  if (IsDynamicRank(x_shape) || IsDynamicRank(w_shape)) {
    std::vector<ValuePtr> pad_list_val = {MakeValue(0), MakeValue(0), MakeValue(0), MakeValue(0)};
    primitive->set_attr("pad_list", MakeValue(pad_list_val));
    output_shape = {abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(output_shape);
  }

  const int64_t shape_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("x shape size", SizeToLong(x_shape.size()), kEqual, shape_size, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("w shape size", SizeToLong(w_shape.size()), kEqual, shape_size, prim_name);
  CheckShapeAnyAndPositive(prim_name + " x_shape", x_shape);
  CheckShapeAnyAndPositive(prim_name + " w_shape", w_shape);
  const uint64_t n_axis = 0;
  uint64_t c_axis = 1;
  uint64_t h_axis = 2;
  uint64_t w_axis = 3;
  int64_t data_format = CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr("format"));
  if (data_format == static_cast<int64_t>(Format::NHWC)) {
    c_axis = 3;
    h_axis = 1;
    w_axis = 2;
  }
  int64_t group = CheckAttrPositiveInt64(prim_name, primitive->GetAttr("group"), "group");
  if ((x_shape[c_axis] != abstract::Shape::kShapeDimAny) && (w_shape[c_axis] != abstract::Shape::kShapeDimAny) &&
      ((x_shape[c_axis] / group) != w_shape[c_axis])) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', 'C_in' of input 'x' shape divide by parameter 'group' must be "
                         "equal to 'C_in' of input 'weight' shape: "
                      << w_shape[c_axis] << ", but got 'C_in' of input 'x' shape: " << x_shape[c_axis]
                      << ", and 'group': " << group << ".";
  }
  int64_t out_channel = CheckAttrPositiveInt64(prim_name, primitive->GetAttr("out_channel"), "out_channel");
  if (w_shape[n_axis] == abstract::Shape::kShapeDimAny) {
    out_channel = w_shape[n_axis];
  } else {
    if (w_shape[n_axis] != out_channel) {
      MS_LOG(EXCEPTION) << "For '" << prim_name << "', 'w_shape[" << n_axis
                        << "]' must be equal to 'out_channel', but got 'w_shape[" << n_axis << "]': " << w_shape[n_axis]
                        << ", 'out_channel': " << out_channel << ".";
    }
  }
  std::vector<int64_t> kernel_size = CheckAttrIntOrTuple(primitive->GetAttr("kernel_size"), 0, kernel_size_num);
  if ((w_shape[h_axis] != abstract::Shape::kShapeDimAny) && (w_shape[h_axis] != kernel_size[0])) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', 'w_shape[" << h_axis
                      << "]' must be equal to 'kernel_size[0]', but got 'w_shape[" << h_axis
                      << "]': " << w_shape[h_axis] << ", 'kernel_size[0]': " << kernel_size[0] << ".";
  }
  if ((w_shape[w_axis] != abstract::Shape::kShapeDimAny) && (w_shape[w_axis] != kernel_size[1])) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', 'w_shape[" << w_axis
                      << "]' must be equal to 'kernel_size[1]', but got 'w_shape[" << w_axis
                      << "]': " << w_shape[w_axis] << ", 'kernel_size[1]': " << kernel_size[1] << ".";
  }
  std::vector<int64_t> stride = CheckAttrIntOrTuple(primitive->GetAttr("stride"), start_index, stride_num);
  std::vector<int64_t> dilation = CheckAttrIntOrTuple(primitive->GetAttr("dilation"), start_index, dilation_num);
  std::vector<int64_t> padding = CheckAttrIntOrTuple(primitive->GetAttr("pad"), 0, padding_num);
  int64_t pad_mode;
  CheckAndConvertUtils::GetPadModEnumValue(primitive->GetAttr("pad_mode"), &pad_mode);
  if (!CheckConv2dShape(prim_name, input_args, x_shape, w_shape, padding, pad_mode, w_axis, h_axis)) {
    MS_EXCEPTION(ValueError) << "For 'Conv2d', input shape's h and w after padding must be greater than or equal to "
                                "kernel_size's h and w respectively.";
  }
  std::vector<int64_t> output_hw;
  std::vector<int64_t> pad_list;
  Conv2DPadFunction(&output_hw, &pad_list, x_shape[h_axis], x_shape[w_axis], kernel_size, stride, dilation, pad_mode,
                    padding);
  std::vector<ValuePtr> pad_list_val = {MakeValue(pad_list[0]), MakeValue(pad_list[1]), MakeValue(pad_list[2]),
                                        MakeValue(pad_list[3])};
  primitive->set_attr("pad_list", MakeValue(pad_list_val));

  output_shape = (data_format == static_cast<int64_t>(Format::NHWC))
                   ? ShapeVector{x_shape[n_axis], output_hw[0], output_hw[1], out_channel}
                   : ShapeVector{x_shape[n_axis], out_channel, output_hw[0], output_hw[1]};
  CheckShapeAnyAndPositive(prim_name + " output_shape", output_shape);
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr Conv2dInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kInt8, kInt32, kInt64, kFloat16, kFloat32};
  auto out_type = CheckAndConvertUtils::CheckTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
  if (out_type->type_id() == TypeId::kNumberTypeInt8) {
    out_type = kInt32;
  }
  return out_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Conv2D, BaseOperator);
void Conv2D::Init(int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode, const PadMode &pad_mode,
                  const std::vector<int64_t> &pad, const std::vector<int64_t> &stride,
                  const std::vector<int64_t> &dilation, int64_t group, const Format &format) {
  set_kernel_size(kernel_size);
  set_stride(stride);
  set_dilation(dilation);
  set_pad(pad);
  set_pad_mode(pad_mode);
  set_mode(mode);
  set_out_channel(out_channel);
  set_group(group);
  set_format(format);
}

void Conv2D::set_out_channel(int64_t out_channel) {
  (void)AddAttr(kOutChannel,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

void Conv2D::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)AddAttr(kKernelSize,
                api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, name())));
}

void Conv2D::set_stride(const std::vector<int64_t> &stride) {
  (void)AddAttr(kStride, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStride, stride, name())));
}

void Conv2D::set_dilation(const std::vector<int64_t> &dilation) {
  (void)AddAttr(kDilation, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kDilation, dilation, name())));
}

void Conv2D::set_pad_mode(const PadMode &pad_mode) {
  std::vector<int64_t> pad = get_pad();
  if (pad_mode == PAD) {
    for (auto item : pad) {
      CheckAndConvertUtils::Check(kPadItem, item, kGreaterEqual, 0, name());
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, {0, 0, 0, 0}, name());
  }
  int64_t swi = pad_mode;
  (void)AddAttr(kPadMode, api::MakeValue(swi));
}

void Conv2D::set_pad(const std::vector<int64_t> &pad) {
  const int64_t pad_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("pad_size", SizeToLong(pad.size()), kEqual, pad_size, name());
  (void)AddAttr(kPad, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name())));
}

void Conv2D::set_mode(int64_t mode) {
  (void)AddAttr(kMode, api::MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

void Conv2D::set_group(int64_t group) {
  (void)AddAttr(kGroup, api::MakeValue(CheckAndConvertUtils::CheckInteger(kGroup, group, kGreaterThan, 0, name())));
}

void Conv2D::set_format(const Format &format) {
  int64_t f = format;
  (void)AddAttr(kFormat, api::MakeValue(f));
}

int64_t Conv2D::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

std::vector<int64_t> Conv2D::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2D::get_stride() const {
  auto value_ptr = GetAttr(kStride);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv2D::get_dilation() const {
  auto value_ptr = GetAttr(kDilation);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

PadMode Conv2D::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv2D::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Conv2D::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv2D::get_group() const {
  auto value_ptr = GetAttr(kGroup);
  return GetValue<int64_t>(value_ptr);
}

Format Conv2D::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

AbstractBasePtr Conv2dInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("Conv2d infer", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           primitive->name());
  const std::set<TypePtr> valid_types = {kInt8, kInt32, kInt64, kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[0]->BuildType());
  (void)types.emplace("w", input_args[1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return abstract::MakeAbstract(Conv2dInferShape(primitive, input_args), Conv2dInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGConv2dInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return Conv2dInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return Conv2dInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return Conv2dInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Conv2D, prim::kPrimConv2D, AGConv2dInfer, false);
}  // namespace ops
}  // namespace mindspore
