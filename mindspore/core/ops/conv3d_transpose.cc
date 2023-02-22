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

#include "ops/conv3d_transpose.h"

#include <set>
#include <map>
#include <string>
#include <algorithm>
#include <iterator>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/dshape.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
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
void Conv3DTranspose::Init(int64_t in_channel, int64_t out_channel, const std::vector<int64_t> &kernel_size,
                           int64_t mode, const PadMode &pad_mode, const std::vector<int64_t> &pad,
                           const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, int64_t group,
                           const std::vector<int64_t> &output_padding, const Format &format) {
  set_in_channel(in_channel);
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_mode(mode);
  set_pad_mode(pad_mode);
  set_pad(pad);
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_output_padding(output_padding);
  set_data_format(format);
}

void Conv3DTranspose::set_in_channel(int64_t in_channel) {
  (void)AddAttr(kInChannel,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kInChannel, in_channel, kGreaterThan, 0, name())));
}

void Conv3DTranspose::set_out_channel(int64_t out_channel) {
  (void)AddAttr(kOutChannel,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

void Conv3DTranspose::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  const int64_t kernel_len = 3;
  (void)CheckAndConvertUtils::CheckInteger(kKernelSize, SizeToLong(kernel_size.size()), kEqual, kernel_len, name());
  for (int64_t item : kernel_size) {
    (void)CheckAndConvertUtils::CheckInteger(kKernelSize, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

void Conv3DTranspose::set_stride(const std::vector<int64_t> &stride) {
  const int64_t stride_size = 5;
  (void)CheckAndConvertUtils::CheckInteger(kStrides, SizeToLong(stride.size()), kEqual, stride_size, name());
  for (int64_t item : stride) {
    (void)CheckAndConvertUtils::CheckInteger(kStrides, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kStride, api::MakeValue(stride));
}

void Conv3DTranspose::set_dilation(const std::vector<int64_t> &dilation) {
  const int64_t dilation_size = 5;
  (void)CheckAndConvertUtils::CheckInteger(kDilations, SizeToLong(dilation.size()), kGreaterEqual, dilation_size,
                                           name());
  (void)AddAttr(kDilations, api::MakeValue(dilation));
}

void Conv3DTranspose::set_pad_mode(const PadMode &pad_mode) {
  std::vector<int64_t> pad = get_pad();
  if (pad_mode == PAD) {
    for (auto item : pad) {
      CheckAndConvertUtils::Check(kPadItem, item, kGreaterEqual, 0, name());
    }
  } else if (pad_mode == VALID) {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, {0, 0, 0, 0, 0, 0}, name());
    std::vector<int64_t> output_padding = get_output_padding();
    CheckAndConvertUtils::Check(kOutputPaddings, output_padding, kEqual, {0, 0, 0, 0, 0, 0}, name());
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, {0, 0, 0, 0, 0, 0}, name());
  }
  int64_t swi = pad_mode;
  (void)AddAttr(kPadMode, api::MakeValue(swi));
}

void Conv3DTranspose::set_pad(const std::vector<int64_t> &pad) {
  const int64_t pad_size = 6;
  (void)CheckAndConvertUtils::CheckInteger("pad_size", SizeToLong(pad.size()), kEqual, pad_size, name());
  (void)AddAttr(kPad, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name())));
}

void Conv3DTranspose::set_mode(int64_t mode) {
  (void)AddAttr(kMode, api::MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

void Conv3DTranspose::set_group(int64_t group) {
  (void)AddAttr(kGroups, api::MakeValue(CheckAndConvertUtils::CheckInteger(kGroups, group, kGreaterThan, 0, name())));
}

void Conv3DTranspose::set_data_format(const Format &format) {
  int64_t f = format;
  (void)AddAttr(kDataFormat, api::MakeValue(f));
}

void Conv3DTranspose::set_output_padding(const std::vector<int64_t> &output_padding) {
  (void)CheckAndConvertUtils::CheckInteger(kOutputPaddings, SizeToLong(output_padding.size()), kGreaterEqual, 1,
                                           name());
  for (int64_t item : output_padding) {
    (void)CheckAndConvertUtils::CheckInteger(kOutputPaddings, item, kGreaterEqual, 0, name());
  }
  (void)AddAttr(kOutputPaddings, api::MakeValue(output_padding));
}

int64_t Conv3DTranspose::get_in_channel() const {
  auto value_ptr = GetAttr(kInChannel);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv3DTranspose::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

std::vector<int64_t> Conv3DTranspose::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv3DTranspose::get_stride() const {
  auto value_ptr = GetAttr(kStrides);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv3DTranspose::get_dilation() const {
  auto value_ptr = GetAttr(kDilations);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

PadMode Conv3DTranspose::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv3DTranspose::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Conv3DTranspose::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv3DTranspose::get_group() const {
  auto value_ptr = GetAttr(kGroups);
  return GetValue<int64_t>(value_ptr);
}

Format Conv3DTranspose::get_data_format() const {
  auto value_ptr = GetAttr(kDataFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv3DTranspose::get_output_padding() const {
  auto value_ptr = GetAttr(kOutputPaddings);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

namespace {
constexpr size_t kAxis0 = 0;
constexpr size_t kAxis1 = 1;
constexpr size_t kAxis2 = 2;
constexpr size_t kAxis3 = 3;
constexpr size_t kAxis4 = 4;
constexpr size_t kAxis5 = 5;

inline std::vector<int64_t> CheckTuple(const std::string &prim_name, const std::string &attr_name,
                                       const ValuePtr &attr) {
  MS_EXCEPTION_IF_NULL(attr);
  if (!attr->isa<ValueTuple>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', '" << attr_name << "' should be tuple.";
  }

  std::vector<int64_t> result;
  std::vector<ValuePtr> attr_vec = attr->cast<ValueTuplePtr>()->value();
  (void)std::transform(attr_vec.begin(), attr_vec.end(), std::back_inserter(result),
                       [](const ValuePtr &e) -> int64_t { return GetValue<int64_t>(e); });
  return result;
}

inline bool CheckShape(const std::string &op, const ShapeVector &shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    // should be positive integer or -1, or -2
    if (shape[i] == abstract::Shape::kShapeRankAny) {
      return false;
    }
    if ((shape[i] < 0) && (shape[i] != abstract::Shape::kShapeDimAny)) {
      MS_EXCEPTION(ValueError) << "For '" << op << "',  shape element [" << i
                               << "] must be positive integer or -1 or -2, but got: " << shape[i] << ".";
    }
  }
  return true;
}
}  // namespace

class Conv3DTransposeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();

    std::map<std::string, TypePtr> types;
    (void)types.emplace("x", input_args[0]->BuildType());
    (void)types.emplace("w", input_args[1]->BuildType());
    std::set<TypePtr> check_list = {kFloat16, kFloat32};
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, check_list, prim_name);

    auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
    auto w_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
    auto x_shape = x_shape_map[kShape];
    auto w_shape = w_shape_map[kShape];
    if (IsDynamicRank(x_shape) || IsDynamicRank(w_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }
    const int64_t shape_size = 5;
    (void)CheckAndConvertUtils::CheckInteger("x shape size", SizeToLong(x_shape.size()), kEqual, shape_size, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("w shape size", SizeToLong(w_shape.size()), kEqual, shape_size, prim_name);
    if (w_shape[0] != abstract::Shape::kShapeDimAny && x_shape[1] != abstract::Shape::kShapeDimAny) {
      (void)CheckAndConvertUtils::CheckInteger("filter's batch, input x's channel", w_shape[0], kEqual, x_shape[1],
                                               prim_name);
    }
    if (!CheckShape(prim_name + " x_shape", x_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }
    if (!CheckShape(prim_name + " w_shape", w_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }

    std::vector<int64_t> kernel_size = CheckTuple(prim_name, kKernelSize, primitive->GetAttr(kKernelSize));
    std::vector<int64_t> stride = CheckTuple(prim_name, kStrides, primitive->GetAttr(kStrides));
    std::vector<int64_t> dilation = CheckTuple(prim_name, kDilations, primitive->GetAttr(kDilations));
    std::vector<int64_t> pad_list = CheckTuple(prim_name, kPadList, primitive->GetAttr(kPadList));
    std::vector<int64_t> output_padding = CheckTuple(prim_name, kOutputPadding, primitive->GetAttr(kOutputPadding));
    int64_t pad_mode;
    CheckAndConvertUtils::GetPadModEnumValue(primitive->GetAttr(kPadMode), &pad_mode);

    if ((w_shape[kAxis2] != abstract::Shape::kShapeDimAny && w_shape[kAxis2] != kernel_size[kAxis0]) ||
        (w_shape[kAxis3] != abstract::Shape::kShapeDimAny && w_shape[kAxis3] != kernel_size[kAxis1]) ||
        (w_shape[kAxis4] != abstract::Shape::kShapeDimAny && w_shape[kAxis4] != kernel_size[kAxis2])) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the dimension 'DHW' of input 'weight' must be "
                               << " equal to the shape of 'kernel size', but got 'DHW' of input 'weight': ("
                               << w_shape[kAxis2] << ", " << w_shape[kAxis3] << ", " << w_shape[kAxis4]
                               << "), and 'kernel size': (" << kernel_size[kAxis0] << ", " << kernel_size[kAxis1]
                               << ", " << kernel_size[kAxis2] << ").";
    }

    int64_t d_out = abstract::Shape::kShapeDimAny;
    int64_t w_out = abstract::Shape::kShapeDimAny;
    int64_t h_out = abstract::Shape::kShapeDimAny;

    int64_t group = primitive->GetAttr(kGroup)->cast<Int64ImmPtr>()->value();
    CaculateShape(x_shape, kernel_size, stride, dilation, pad_mode, &pad_list, &output_padding, &d_out, &h_out, &w_out);
    primitive->set_attr(kPadList, MakeValue(pad_list));
    primitive->set_attr(kOutputPadding, MakeValue(output_padding));
    ShapeVector output_shape{x_shape[0], w_shape[1] * group, d_out, h_out, w_out};
    primitive->set_attr(kInput_size, MakeValue(output_shape));
    if (!CheckShape(prim_name + " output_shape", output_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }

    return std::make_shared<abstract::Shape>(output_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
    auto x_dtype = input_args[0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, primitive->name());
    return x_dtype;
  }

 private:
  void CaculateShape(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &kernel_size,
                     const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, const int64_t &pad_mode,
                     std::vector<int64_t> *pad_list, std::vector<int64_t> *output_padding, int64_t *d_out,
                     int64_t *h_out, int64_t *w_out) const {
    int64_t kernel_d = kernel_size[kAxis0];
    int64_t kernel_h = kernel_size[kAxis1];
    int64_t kernel_w = kernel_size[kAxis2];
    int64_t stride_d = stride[kAxis2];
    int64_t stride_h = stride[kAxis3];
    int64_t stride_w = stride[kAxis4];
    int64_t dilation_d = dilation[kAxis2];
    int64_t dilation_h = dilation[kAxis3];
    int64_t dilation_w = dilation[kAxis4];
    int64_t x_d = x_shape[kAxis2];
    int64_t x_h = x_shape[kAxis3];
    int64_t x_w = x_shape[kAxis4];

    if (pad_mode == PadMode::VALID) {
      if (x_d != abstract::Shape::kShapeDimAny) {
        *d_out = DeconvOutputLength(x_d, kernel_d, stride_d, dilation_d);
      }
      if (x_h != abstract::Shape::kShapeDimAny) {
        *h_out = DeconvOutputLength(x_h, kernel_h, stride_h, dilation_h);
      }
      if (x_w != abstract::Shape::kShapeDimAny) {
        *w_out = DeconvOutputLength(x_w, kernel_w, stride_w, dilation_w);
      }

      constexpr size_t pad_list_size = 6;
      constexpr size_t output_padding_size = 5;
      pad_list->clear();
      output_padding->clear();
      (void)pad_list->insert(pad_list->begin(), pad_list_size, 0);
      (void)output_padding->insert(output_padding->begin(), output_padding_size, 0);
    } else if (pad_mode == PadMode::SAME) {
      int64_t pad_head = abstract::Shape::kShapeDimAny;
      int64_t pad_tail = abstract::Shape::kShapeDimAny;
      int64_t pad_top = abstract::Shape::kShapeDimAny;
      int64_t pad_bottom = abstract::Shape::kShapeDimAny;
      int64_t pad_left = abstract::Shape::kShapeDimAny;
      int64_t pad_right = abstract::Shape::kShapeDimAny;

      const int64_t kNumber2 = 2;
      if (x_d != abstract::Shape::kShapeDimAny) {
        *d_out = x_d * stride_d;
        int64_t pad_needed_d = std::max(SizeToLong(0), (x_d - 1) * stride_d + dilation_d * (kernel_d - 1) + 1 - *d_out);
        pad_head = pad_needed_d / kNumber2;
        pad_tail = pad_needed_d - pad_head;
      }
      if (x_h != abstract::Shape::kShapeDimAny) {
        *h_out = x_h * stride_h;
        int64_t pad_needed_h = std::max(SizeToLong(0), (x_h - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - *h_out);
        pad_top = pad_needed_h / kNumber2;
        pad_bottom = pad_needed_h - pad_top;
      }
      if (x_w != abstract::Shape::kShapeDimAny) {
        *w_out = x_w * stride_w;
        int64_t pad_needed_w = std::max(SizeToLong(0), (x_w - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - *w_out);
        pad_left = pad_needed_w / kNumber2;
        pad_right = pad_needed_w - pad_left;
      }

      pad_list->clear();
      output_padding->clear();
      constexpr size_t pad_size = 5;
      (void)output_padding->insert(output_padding->begin(), pad_size, 0);
      (void)pad_list->emplace_back(pad_head);
      (void)pad_list->emplace_back(pad_tail);
      (void)pad_list->emplace_back(pad_top);
      (void)pad_list->emplace_back(pad_bottom);
      (void)pad_list->emplace_back(pad_left);
      (void)pad_list->emplace_back(pad_right);
    } else if (pad_mode == PadMode::PAD) {
      int64_t pad_head = pad_list->at(kAxis0);
      int64_t pad_tail = pad_list->at(kAxis1);
      int64_t pad_top = pad_list->at(kAxis2);
      int64_t pad_bottom = pad_list->at(kAxis3);
      int64_t pad_left = pad_list->at(kAxis4);
      int64_t pad_right = pad_list->at(kAxis5);

      if (x_d != abstract::Shape::kShapeDimAny) {
        *d_out =
          (x_d - 1) * stride_d - (pad_head + pad_tail) + dilation_d * (kernel_d - 1) + output_padding->at(kAxis2) + 1;
      }
      if (x_h != abstract::Shape::kShapeDimAny) {
        *h_out =
          (x_h - 1) * stride_h - (pad_top + pad_bottom) + dilation_h * (kernel_h - 1) + output_padding->at(kAxis3) + 1;
      }
      if (x_w != abstract::Shape::kShapeDimAny) {
        *w_out =
          (x_w - 1) * stride_w - (pad_left + pad_right) + dilation_w * (kernel_w - 1) + output_padding->at(kAxis4) + 1;
      }
    }
  }

  int64_t DeconvOutputLength(const int64_t input_length, const int64_t kernel_size, const int64_t stride_size,
                             const int64_t dilation_size) const {
    int64_t filter_size = kernel_size + (kernel_size - 1) * (dilation_size - 1);
    int64_t length;
    if (filter_size - stride_size > 0) {
      length = input_length * stride_size + filter_size - stride_size;
    } else {
      length = input_length * stride_size;
    }
    return length;
  }
};

MIND_API_OPERATOR_IMPL(Conv3DTranspose, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(Conv3DTranspose, prim::kPrimConv3DTranspose, Conv3DTransposeInfer, false);
}  // namespace ops
}  // namespace mindspore
