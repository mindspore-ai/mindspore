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

#include "ops/conv3d.h"

#include <set>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/dshape.h"
#include "include/common/utils/utils.h"
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
void Conv3D::Init(int64_t out_channel, const std::vector<int64_t> &kernel_size, int64_t mode, const PadMode &pad_mode,
                  const std::vector<int64_t> &pad, const std::vector<int64_t> &stride,
                  const std::vector<int64_t> &dilation, int64_t group, const Format &format) {
  set_out_channel(out_channel);
  set_kernel_size(kernel_size);
  set_mode(mode);
  set_pad_mode(pad_mode);
  set_pad(pad);
  set_stride(stride);
  set_dilation(dilation);
  set_group(group);
  set_data_format(format);
}

void Conv3D::set_out_channel(int64_t out_channel) {
  (void)AddAttr(kOutChannel,
                api::MakeValue(CheckAndConvertUtils::CheckInteger(kOutChannel, out_channel, kGreaterThan, 0, name())));
}

void Conv3D::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  const int64_t kernel_len = 3;
  (void)CheckAndConvertUtils::CheckInteger(kKernelSize, SizeToLong(kernel_size.size()), kEqual, kernel_len, name());
  for (int64_t item : kernel_size) {
    (void)CheckAndConvertUtils::CheckInteger(kKernelSize, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

void Conv3D::set_stride(const std::vector<int64_t> &stride) {
  const int64_t stride_size = 5;
  (void)CheckAndConvertUtils::CheckInteger(kStrides, SizeToLong(stride.size()), kEqual, stride_size, name());
  for (int64_t item : stride) {
    (void)CheckAndConvertUtils::CheckInteger(kStrides, item, kGreaterEqual, 1, name());
  }
  (void)AddAttr(kStride, api::MakeValue(stride));
}

void Conv3D::set_dilation(const std::vector<int64_t> &dilation) {
  const int64_t dilation_size = 5;
  (void)CheckAndConvertUtils::CheckInteger(kDilations, SizeToLong(dilation.size()), kGreaterEqual, dilation_size,
                                           name());
  (void)AddAttr(kDilations, api::MakeValue(dilation));
}

void Conv3D::set_pad_mode(const PadMode &pad_mode) {
  std::vector<int64_t> pad = get_pad();
  if (pad_mode == PAD) {
    for (auto item : pad) {
      CheckAndConvertUtils::Check(kPadItem, item, kGreaterEqual, 0, name());
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, {0, 0, 0, 0, 0, 0}, name());
  }
  int64_t swi = pad_mode;
  (void)AddAttr(kPadMode, api::MakeValue(swi));
}

void Conv3D::set_pad(const std::vector<int64_t> &pad) {
  const int64_t pad_size = 6;
  (void)CheckAndConvertUtils::CheckInteger("pad_size", SizeToLong(pad.size()), kEqual, pad_size, name());
  (void)AddAttr(kPad, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, name())));
}

void Conv3D::set_mode(int64_t mode) {
  (void)AddAttr(kMode, api::MakeValue(CheckAndConvertUtils::CheckInteger(kMode, mode, kEqual, 1, name())));
}

void Conv3D::set_group(int64_t group) {
  (void)AddAttr(kGroups, api::MakeValue(CheckAndConvertUtils::CheckInteger(kGroups, group, kGreaterThan, 0, name())));
}

void Conv3D::set_data_format(const Format &format) {
  int64_t f = format;
  (void)AddAttr(kDataFormat, api::MakeValue(f));
}

int64_t Conv3D::get_out_channel() const {
  auto value_ptr = GetAttr(kOutChannel);
  return GetValue<int64_t>(value_ptr);
}

std::vector<int64_t> Conv3D::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv3D::get_stride() const {
  auto value_ptr = GetAttr(kStrides);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> Conv3D::get_dilation() const {
  auto value_ptr = GetAttr(kDilations);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

PadMode Conv3D::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  return PadMode(GetValue<int64_t>(value_ptr));
}

std::vector<int64_t> Conv3D::get_pad() const {
  auto value_ptr = GetAttr(kPad);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

int64_t Conv3D::get_mode() const {
  auto value_ptr = GetAttr(kMode);
  return GetValue<int64_t>(value_ptr);
}

int64_t Conv3D::get_group() const {
  auto value_ptr = GetAttr(kGroups);
  return GetValue<int64_t>(value_ptr);
}

Format Conv3D::get_data_format() const {
  auto value_ptr = GetAttr(kDataFormat);
  return Format(GetValue<int64_t>(value_ptr));
}

namespace {
constexpr size_t kConv3DKernelSizeNum = 3;
constexpr size_t kConv3DstrideNum = 3;
constexpr size_t kConv3DDilationNum = 3;
constexpr size_t kConv3DStartIndex = 2;
constexpr size_t kConv3DPaddingNum = 6;

inline std::vector<int64_t> Conv3DCheckAttrIntOrTuple(const ValuePtr &attr, const size_t start_idx,
                                                      const size_t num_element) {
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

inline bool Conv3DCheckShape(const std::string &op, const ShapeVector &shape) {
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

class Conv3DInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex0]->BuildShape());
    auto w_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex1]->BuildShape());
    auto x_shape = x_shape_map[kShape];
    auto w_shape = w_shape_map[kShape];
    if (IsDynamicRank(x_shape) || IsDynamicRank(w_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }
    const int64_t shape_size = 5;
    (void)CheckAndConvertUtils::CheckInteger("x shape size", SizeToLong(x_shape.size()), kEqual, shape_size, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("w shape size", SizeToLong(w_shape.size()), kEqual, shape_size, prim_name);
    if (!Conv3DCheckShape(prim_name + " x_shape", x_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }
    if (!Conv3DCheckShape(prim_name + " w_shape", w_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }
    std::vector<int64_t> kernel_size =
      Conv3DCheckAttrIntOrTuple(primitive->GetAttr(kKernelSize), 0, kConv3DKernelSizeNum);
    std::vector<int64_t> stride =
      Conv3DCheckAttrIntOrTuple(primitive->GetAttr(kStrides), kConv3DStartIndex, kConv3DstrideNum);
    std::vector<int64_t> dilation =
      Conv3DCheckAttrIntOrTuple(primitive->GetAttr(kDilations), kConv3DStartIndex, kConv3DDilationNum);
    std::vector<int64_t> pad_list = Conv3DCheckAttrIntOrTuple(primitive->GetAttr(kPad), 0, kConv3DPaddingNum);
    int64_t pad_mode;
    CheckAndConvertUtils::GetPadModEnumValue(primitive->GetAttr(kPadMode), &pad_mode);

    // only support format NCDHW now
    const uint64_t n_axis = 0;
    uint64_t c_axis = 1;
    uint64_t d_axis = 2;
    uint64_t h_axis = 3;
    uint64_t w_axis = 4;
    int64_t group = primitive->GetAttr(kGroup)->cast<Int64ImmPtr>()->value();
    if ((x_shape[c_axis] != abstract::Shape::kShapeDimAny) && (w_shape[c_axis] != abstract::Shape::kShapeDimAny) &&
        ((x_shape[c_axis] / group) != w_shape[c_axis])) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', 'C_in' of input 'x' shape divide by parameter 'group' must be "
                                  "equal to 'C_in' of input 'weight' shape: "
                               << w_shape[c_axis] << ", but got 'C_in' of input 'x' shape: " << x_shape[c_axis]
                               << ", and 'group': " << group << ".";
    }
    int64_t out_channel = primitive->GetAttr(kOutChannel)->cast<Int64ImmPtr>()->value();
    if ((w_shape[n_axis] != abstract::Shape::kShapeDimAny) && (w_shape[n_axis] != out_channel)) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', 'w_shape[" << n_axis
                               << "]' must be equal to 'out_channel', but got 'w_shape[" << n_axis
                               << "]': " << w_shape[n_axis] << ", 'out_channel': " << out_channel << ".";
    }
    if ((w_shape[d_axis] != abstract::Shape::kShapeDimAny) && (w_shape[d_axis] != kernel_size[kIndex0])) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', 'w_shape[" << d_axis
                               << "]' must be equal to 'kernel_size[0]', but got 'w_shape[" << d_axis
                               << "]': " << w_shape[d_axis] << ", 'kernel_size[0]': " << kernel_size[kIndex0] << ".";
    }
    if ((w_shape[h_axis] != abstract::Shape::kShapeDimAny) && (w_shape[h_axis] != kernel_size[kIndex1])) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', 'w_shape[" << h_axis
                               << "]' must be equal to 'kernel_size[1]', but got 'w_shape[" << h_axis
                               << "]': " << w_shape[h_axis] << ", 'kernel_size[1]': " << kernel_size[kIndex1] << ".";
    }
    if ((w_shape[w_axis] != abstract::Shape::kShapeDimAny) && (w_shape[w_axis] != kernel_size[kIndex2])) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', 'w_shape[" << w_axis
                               << "]' must be equal to 'kernel_size[2]', but got 'w_shape[" << w_axis
                               << "]': " << w_shape[w_axis] << ", 'kernel_size[2]': " << kernel_size[kIndex2] << ".";
    }

    int64_t d_out = abstract::Shape::kShapeDimAny;
    int64_t h_out = abstract::Shape::kShapeDimAny;
    int64_t w_out = abstract::Shape::kShapeDimAny;
    CaculateShape(x_shape, kernel_size, stride, dilation, pad_mode, &pad_list, &d_out, &h_out, &w_out);
    primitive->set_attr(kPadList, MakeValue(pad_list));
    ShapeVector output_shape{x_shape[kIndex0], out_channel, d_out, h_out, w_out};
    if (!Conv3DCheckShape(prim_name + " output_shape", output_shape)) {
      return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    }

    return std::make_shared<abstract::Shape>(output_shape);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto prim_name = primitive->name();
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
    auto x_dtype = input_args[0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, primitive->name());

    std::map<std::string, TypePtr> types;
    (void)types.emplace("x", input_args[kIndex0]->BuildType());
    (void)types.emplace("w", input_args[kIndex1]->BuildType());
    std::set<TypePtr> check_list = {kFloat16, kFloat32};
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, check_list, prim_name);
    return x_dtype;
  }

 private:
  void CaculateShape(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &kernel_size,
                     const std::vector<int64_t> &stride, const std::vector<int64_t> &dilation, const int64_t &pad_mode,
                     std::vector<int64_t> *pad_list, int64_t *d_out, int64_t *h_out, int64_t *w_out) const {
    int64_t kernel_d = kernel_size[kIndex0];
    int64_t kernel_h = kernel_size[kIndex1];
    int64_t kernel_w = kernel_size[kIndex2];
    int64_t stride_d = stride[kIndex0];
    int64_t stride_h = stride[kIndex1];
    int64_t stride_w = stride[kIndex2];
    int64_t dilation_d = dilation[kIndex0];
    int64_t dilation_h = dilation[kIndex1];
    int64_t dilation_w = dilation[kIndex2];
    int64_t x_d = x_shape[kIndex2];
    int64_t x_h = x_shape[kIndex3];
    int64_t x_w = x_shape[kIndex4];

    if (pad_mode == PadMode::VALID) {
      if (x_d != abstract::Shape::kShapeDimAny) {
        *d_out = ConvOutputLength(x_d, kernel_d, stride_d, dilation_d);
      }
      if (x_h != abstract::Shape::kShapeDimAny) {
        *h_out = ConvOutputLength(x_h, kernel_h, stride_h, dilation_h);
      }
      if (x_w != abstract::Shape::kShapeDimAny) {
        *w_out = ConvOutputLength(x_w, kernel_w, stride_w, dilation_w);
      }
      constexpr size_t pad_list_size = 6;
      pad_list->clear();
      (void)pad_list->insert(pad_list->begin(), pad_list_size, 0);
    } else if (pad_mode == PadMode::SAME) {
      int64_t pad_head = 0;
      int64_t pad_tail = 0;
      int64_t pad_top = 0;
      int64_t pad_bottom = 0;
      int64_t pad_left = 0;
      int64_t pad_right = 0;

      const int64_t kNumber2 = 2;
      if (x_d != abstract::Shape::kShapeDimAny) {
        *d_out = CeilCompute(x_d, stride_d);
        int64_t pad_needed_d = std::max(SizeToLong(0), (*d_out - 1) * stride_d + dilation_d * (kernel_d - 1) + 1 - x_d);
        pad_head = pad_needed_d / kNumber2;
        pad_tail = pad_needed_d - pad_head;
      }
      if (x_h != abstract::Shape::kShapeDimAny) {
        *h_out = CeilCompute(x_h, stride_h);
        int64_t pad_needed_h = std::max(SizeToLong(0), (*h_out - 1) * stride_h + dilation_h * (kernel_h - 1) + 1 - x_h);
        pad_top = pad_needed_h / kNumber2;
        pad_bottom = pad_needed_h - pad_top;
      }
      if (x_w != abstract::Shape::kShapeDimAny) {
        *w_out = CeilCompute(x_w, stride_w);
        int64_t pad_needed_w = std::max(SizeToLong(0), (*w_out - 1) * stride_w + dilation_w * (kernel_w - 1) + 1 - x_w);
        pad_left = pad_needed_w / kNumber2;
        pad_right = pad_needed_w - pad_left;
      }

      pad_list->clear();
      pad_list->emplace_back(pad_head);
      pad_list->emplace_back(pad_tail);
      pad_list->emplace_back(pad_top);
      pad_list->emplace_back(pad_bottom);
      pad_list->emplace_back(pad_left);
      pad_list->emplace_back(pad_right);
    } else if (pad_mode == PadMode::PAD) {
      int64_t pad_head = pad_list->at(kIndex0);
      int64_t pad_tail = pad_list->at(kIndex1);
      int64_t pad_top = pad_list->at(kIndex2);
      int64_t pad_bottom = pad_list->at(kIndex3);
      int64_t pad_left = pad_list->at(kIndex4);
      int64_t pad_right = pad_list->at(kIndex5);

      if (x_d != abstract::Shape::kShapeDimAny) {
        *d_out = 1 + (x_d + pad_head + pad_tail - kernel_d - (kernel_d - 1) * (dilation_d - 1)) / stride_d;
      }
      if (x_h != abstract::Shape::kShapeDimAny) {
        *h_out = 1 + (x_h + pad_top + pad_bottom - kernel_h - (kernel_h - 1) * (dilation_h - 1)) / stride_h;
      }
      if (x_w != abstract::Shape::kShapeDimAny) {
        *w_out = 1 + (x_w + pad_left + pad_right - kernel_w - (kernel_w - 1) * (dilation_w - 1)) / stride_w;
      }
    }

    CheckPadList(kernel_size, dilation, pad_list);
  }

  void CheckPadList(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &dilation,
                    std::vector<int64_t> *pad_list) const {
    int64_t kernel_d = kernel_size[kIndex0];
    int64_t kernel_h = kernel_size[kIndex1];
    int64_t kernel_w = kernel_size[kIndex2];
    int64_t dilation_d = dilation[kIndex0];
    int64_t dilation_h = dilation[kIndex1];
    int64_t dilation_w = dilation[kIndex2];
    int64_t pad_head = pad_list->at(kIndex0);
    int64_t pad_tail = pad_list->at(kIndex1);
    int64_t pad_top = pad_list->at(kIndex2);
    int64_t pad_bottom = pad_list->at(kIndex3);
    int64_t pad_left = pad_list->at(kIndex4);
    int64_t pad_right = pad_list->at(kIndex5);
    int64_t filter_d = (kernel_d - 1) * dilation_d + 1;
    int64_t filter_h = (kernel_h - 1) * dilation_h + 1;
    int64_t filter_w = (kernel_w - 1) * dilation_w + 1;
    if (pad_head < 0 || pad_head >= filter_d) {
      MS_EXCEPTION(ValueError) << "For 'Conv3D', pad head must be in range [0, " << filter_d << "), but got "
                               << pad_head;
    }
    if (pad_tail < 0 || pad_tail >= filter_d) {
      MS_EXCEPTION(ValueError) << "For 'Conv3D', pad tail must be in range [0, " << filter_d << "), but got "
                               << pad_tail;
    }
    if (pad_top < 0 || pad_top >= filter_h) {
      MS_EXCEPTION(ValueError) << "For 'Conv3D', pad top must be in range [0, " << filter_h << "), but got " << pad_top;
    }
    if (pad_bottom < 0 || pad_bottom >= filter_h) {
      MS_EXCEPTION(ValueError) << "For 'Conv3D', pad bottom must be in range [0, " << filter_h << "), but got "
                               << pad_bottom;
    }
    if (pad_left < 0 || pad_left >= filter_w) {
      MS_EXCEPTION(ValueError) << "For 'Conv3D', pad left must be in range [0, " << filter_w << "), but got "
                               << pad_left;
    }
    if (pad_right < 0 || pad_right >= filter_w) {
      MS_EXCEPTION(ValueError) << "For 'Conv3D', pad right must be in range [0, " << filter_w << "), but got "
                               << pad_right;
    }
  }

  int64_t ConvOutputLength(const int64_t input_length, const int64_t kernel_size, const int64_t stride_size,
                           const int64_t dilation_size) const {
    double temp_size = static_cast<double>(input_length - dilation_size * (kernel_size - 1));
    return static_cast<int64_t>(std::ceil(temp_size / static_cast<double>(stride_size)));
  }

  int64_t CeilCompute(const int64_t shape, const int64_t stride) const {
    return static_cast<int64_t>(std::ceil(static_cast<double>(shape) / static_cast<double>(stride)));
  }
};

MIND_API_OPERATOR_IMPL(Conv3D, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(Conv3D, prim::kPrimConv3D, Conv3DInfer, false);
}  // namespace ops
}  // namespace mindspore
