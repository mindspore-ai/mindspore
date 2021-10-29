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
#include "ir/dtype/tensor_type.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

using mindspore::abstract::Shape;
namespace mindspore {
namespace ops {
namespace {
// check functions
void CheckShapeAnyAndPositive(const std::string &op, const ShapeVector &shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if ((shape[i] < 0) && (shape[i] != Shape::SHP_ANY)) {
      MS_EXCEPTION(ValueError) << op << " shape element [" << i << "] must be positive integer or SHP_ANY, but got "
                               << shape[i];
    }
  }
}

void CheckShapeAllPositive(const std::string &op, const ShapeVector &shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) {
      MS_LOG(EXCEPTION) << op << " shape element [" << i << "] must be positive integer, but got " << shape[i];
    }
  }
}

int64_t CheckAttrPositiveInt64(const std::string &op, const ValuePtr &attr, const std::string &attr_name) {
  int64_t attr_val = attr->cast<Int64ImmPtr>()->value();
  if (attr_val <= 0) {
    MS_LOG(EXCEPTION) << op << " invalid " << attr_name << " value: " << attr_val << ", should be greater then 0";
  }
  return attr_val;
}

std::vector<int64_t> CheckAttrIntOrTuple(const ValuePtr &attr, const size_t start_idx, const size_t num_element) {
  std::vector<int64_t> result;
  MS_EXCEPTION_IF_NULL(attr);
  if (attr->isa<ValueTuple>()) {
    std::vector<ValuePtr> attr_vec = attr->cast<ValueTuplePtr>()->value();
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
  if (pad_mode == PadMode::VALID) {
    int64_t out_h = -1;
    int64_t out_w = -1;
    if (x_h != Shape::SHP_ANY) {
      out_h =
        static_cast<int64_t>(std::ceil(((x_h * 1.0) - static_cast<double>(dilation[0] * (kernel[0] - 1))) / stride[0]));
      if (is_min_shape && out_h < 1) {
        out_h = 1L;
      }
    }
    if (x_w != Shape::SHP_ANY) {
      out_w =
        static_cast<int64_t>(std::ceil(((x_w * 1.0) - static_cast<double>(dilation[1] * (kernel[1] - 1))) / stride[1]));
      if (is_min_shape && out_w < 1) {
        out_w = 1L;
      }
    }
    output_hw->push_back(out_h);
    output_hw->push_back(out_w);
    constexpr size_t pad_size = 4;
    (void)pad_list->insert(pad_list->begin(), pad_size, 0);
  } else if (pad_mode == PadMode::SAME) {
    if (x_h == Shape::SHP_ANY) {
      output_hw->push_back(Shape::SHP_ANY);
      pad_list->push_back(Shape::SHP_ANY);
      pad_list->push_back(Shape::SHP_ANY);
    } else {
      output_hw->push_back(static_cast<int64_t>(std::ceil((x_h * 1.0) / stride[0])));
      int64_t pad_needed_h = (output_hw->at(0) - 1) * stride[0] + dilation[0] * (kernel[0] - 1) + 1 - x_h;
      pad_needed_h = std::max((int64_t)0, pad_needed_h);
      pad_list->push_back(static_cast<int64_t>(std::floor(pad_needed_h / 2)));
      pad_list->push_back(pad_needed_h - pad_list->at(0));
    }

    if (x_w == Shape::SHP_ANY) {
      output_hw->push_back(Shape::SHP_ANY);
      pad_list->push_back(Shape::SHP_ANY);
      pad_list->push_back(Shape::SHP_ANY);
    } else {
      output_hw->push_back(static_cast<int64_t>(std::ceil((x_w * 1.0) / stride[1])));
      int64_t pad_needed_w = (output_hw->at(1) - 1) * stride[1] + dilation[1] * (kernel[1] - 1) + 1 - x_w;
      pad_needed_w = std::max((int64_t)0, pad_needed_w);
      pad_list->push_back(static_cast<int64_t>(std::floor(pad_needed_w / 2)));
      pad_list->push_back(pad_needed_w - pad_list->at(kInputIndex2));
    }
  } else if (pad_mode == PadMode::PAD) {
    (void)pad_list->insert(pad_list->begin(), padding.begin(), padding.end());
    int64_t out_h = -1;
    int64_t out_w = -1;
    if (x_h != Shape::SHP_ANY) {
      out_h = static_cast<int64_t>(std::floor(
        1 + ((x_h * 1) + pad_list->at(0) + pad_list->at(1) - kernel[0] - ((kernel[0] - 1) * (dilation[0] - 1))) /
              stride[0]));
      if (is_min_shape && out_h < 1) {
        out_h = 1L;
      }
    }
    if (x_w != Shape::SHP_ANY) {
      out_w = static_cast<int64_t>(std::floor(1 + ((x_w * 1) + pad_list->at(kInputIndex2) + pad_list->at(kInputIndex3) -
                                                   kernel[1] - ((kernel[1] - 1) * (dilation[1] - 1))) /
                                                    stride[1]));
      if (is_min_shape && out_w < 1) {
        out_w = 1L;
      }
    }
    output_hw->push_back(out_h);
    output_hw->push_back(out_w);
  }
}

abstract::ShapePtr Conv2dInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto w_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  auto w_shape = w_shape_map[kShape];
  const int64_t shape_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("x shape size", SizeToLong(x_shape.size()), kEqual, shape_size, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("w shape size", SizeToLong(w_shape.size()), kEqual, shape_size, prim_name);
  auto x_min_shape = x_shape_map[kMinShape];
  auto x_max_shape = x_shape_map[kMaxShape];
  auto w_min_shape = w_shape_map[kMinShape];
  auto w_max_shape = w_shape_map[kMaxShape];
  CheckAndConvertUtils::CheckMinMaxShape(x_shape, &x_min_shape, &x_max_shape);
  CheckAndConvertUtils::CheckMinMaxShape(w_shape, &w_min_shape, &w_max_shape);
  CheckShapeAnyAndPositive(prim_name + " x_shape", x_shape);
  CheckShapeAnyAndPositive(prim_name + " w_shape", w_shape);
  CheckShapeAllPositive(prim_name + " x_min_shape", x_min_shape);
  CheckShapeAllPositive(prim_name + " x_max_shape", x_max_shape);
  CheckShapeAllPositive(prim_name + " w_min_shape", w_min_shape);
  CheckShapeAllPositive(prim_name + " w_max_shape", w_max_shape);
  const uint64_t n_axis = 0;
  uint64_t c_axis = 1;
  uint64_t h_axis = 2;
  uint64_t w_axis = 3;
  int64_t data_format = CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr("format"));
  if (data_format == Format::NHWC) {
    c_axis = 3;
    h_axis = 1;
    w_axis = 2;
  }
  int64_t group = CheckAttrPositiveInt64(prim_name, primitive->GetAttr("group"), "group");
  if ((x_shape[c_axis] != Shape::SHP_ANY) && (w_shape[c_axis] != Shape::SHP_ANY) &&
      ((x_shape[c_axis] / group) != w_shape[c_axis])) {
    MS_LOG(EXCEPTION) << "x_shape[C_in] / group must equal to w_shape[C_in] = " << w_shape[c_axis] << ", but got "
                      << (x_shape[c_axis] / group);
  }
  int64_t out_channel = CheckAttrPositiveInt64(prim_name, primitive->GetAttr("out_channel"), "out_channel");
  if ((w_shape[n_axis] != Shape::SHP_ANY) && (w_shape[n_axis] != out_channel)) {
    MS_LOG(EXCEPTION) << "w_shape[" << n_axis << "] = " << w_shape[n_axis] << " must equal to = " << out_channel;
  }
  constexpr size_t kernel_size_num = 2;
  constexpr size_t stride_num = 2;
  constexpr size_t dilation_num = 2;
  constexpr size_t padding_num = 4;
  constexpr size_t start_index = 2;
  std::vector<int64_t> kernel_size = CheckAttrIntOrTuple(primitive->GetAttr("kernel_size"), 0, kernel_size_num);
  if ((w_shape[h_axis] != Shape::SHP_ANY) && (w_shape[h_axis] != kernel_size[0])) {
    MS_LOG(EXCEPTION) << "weight height = " << w_shape[h_axis] << ", must equal to = " << kernel_size[0];
  }
  if ((w_shape[w_axis] != Shape::SHP_ANY) && (w_shape[w_axis] != kernel_size[1])) {
    MS_LOG(EXCEPTION) << "weight width = " << w_shape[w_axis] << ", must equal to = " << kernel_size[1];
  }
  std::vector<int64_t> stride = CheckAttrIntOrTuple(primitive->GetAttr("stride"), start_index, stride_num);
  std::vector<int64_t> dilation = CheckAttrIntOrTuple(primitive->GetAttr("dilation"), start_index, dilation_num);
  std::vector<int64_t> padding = CheckAttrIntOrTuple(primitive->GetAttr("pad"), 0, padding_num);
  int64_t pad_mode;
  CheckAndConvertUtils::GetPadModEnumValue(primitive->GetAttr("pad_mode"), &pad_mode);
  std::vector<int64_t> output_hw;
  std::vector<int64_t> pad_list;
  std::vector<int64_t> output_hw_min;
  std::vector<int64_t> pad_list_min;
  std::vector<int64_t> output_hw_max;
  std::vector<int64_t> pad_list_max;
  Conv2DPadFunction(&output_hw, &pad_list, x_shape[h_axis], x_shape[w_axis], kernel_size, stride, dilation, pad_mode,
                    padding);
  Conv2DPadFunction(&output_hw_min, &pad_list_min, x_min_shape[h_axis], x_min_shape[w_axis], kernel_size, stride,
                    dilation, pad_mode, padding, true);
  Conv2DPadFunction(&output_hw_max, &pad_list_max, x_max_shape[h_axis], x_max_shape[w_axis], kernel_size, stride,
                    dilation, pad_mode, padding);
  std::vector<ValuePtr> pad_list_val = {MakeValue(pad_list[0]), MakeValue(pad_list[1]), MakeValue(pad_list[2]),
                                        MakeValue(pad_list[3])};
  primitive->set_attr("pad_list", MakeValue(pad_list_val));
  ShapeVector output_shape;
  ShapeVector output_shape_min;
  ShapeVector output_shape_max;
  if (data_format == Format::NHWC) {
    output_shape = {x_shape[n_axis], output_hw[0], output_hw[1], out_channel};
    output_shape_min = {x_min_shape[n_axis], output_hw_min[0], output_hw_min[1], out_channel};
    output_shape_max = {x_max_shape[n_axis], output_hw_max[0], output_hw_max[1], out_channel};
  } else {
    output_shape = {x_shape[n_axis], out_channel, output_hw[0], output_hw[1]};
    output_shape_min = {x_min_shape[n_axis], out_channel, output_hw_min[0], output_hw_min[1]};
    output_shape_max = {x_max_shape[n_axis], out_channel, output_hw_max[0], output_hw_max[1]};
  }
  CheckShapeAnyAndPositive(prim_name + " output_shape", output_shape);
  CheckShapeAllPositive(prim_name + " output_shape_min", output_shape_min);
  CheckShapeAllPositive(prim_name + " output_shape_max", output_shape_max);
  return std::make_shared<abstract::Shape>(output_shape, output_shape_min, output_shape_max);
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
                MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

void Conv2D::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)AddAttr(kKernelSize, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, name())));
}

void Conv2D::set_stride(const std::vector<int64_t> &stride) {
  (void)AddAttr(kStride, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStride, stride, name())));
}

void Conv2D::set_dilation(const std::vector<int64_t> &dilation) {
  (void)AddAttr(kDilation, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kDilation, dilation, name())));
}

void Conv2D::set_pad_mode(const PadMode &pad_mode) {
  std::vector<int64_t> pad = get_pad();
  if (pad_mode == PAD) {
    for (auto item : pad) {
      CheckAndConvertUtils::Check(kPadItem, item, kGreaterEqual, "zeros_list", 0, name());
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, "zeros_list", {0, 0, 0, 0}, name());
  }
  int64_t swi = pad_mode;
  (void)AddAttr(kPadMode, MakeValue(swi));
}

void Conv2D::set_pad(const std::vector<int64_t> &pad) {
  const int64_t pad_size = 4;
  (void)CheckAndConvertUtils::CheckInteger("pad_size", SizeToLong(pad.size()), kEqual, pad_size, name());
  (void)AddAttr(kPad, MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name())));
}

void Conv2D::set_mode(int64_t mode) {
  (void)AddAttr(kMode, MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

void Conv2D::set_group(int64_t group) {
  (void)AddAttr(kGroup, MakeValue(CheckAndConvertUtils::CheckInteger(kGroup, group, kGreaterThan, 0, name())));
}

void Conv2D::set_format(const Format &format) {
  int64_t f = format;
  (void)AddAttr(kFormat, MakeValue(f));
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
REGISTER_PRIMITIVE_EVAL_IMPL(Conv2D, prim::kPrimConv2D, Conv2dInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
