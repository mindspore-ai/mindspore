/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/avg_pool_3d.h"

#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <cmath>

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
#include "ir/value.h"
#include "mindapi/base/types.h"
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
constexpr size_t kAvgPool3DPadDims = 6;

void GetAttrs(const PrimitivePtr &primitive, std::vector<int64_t> *kernel_size, std::vector<int64_t> *strides,
              int64_t *pad_mode, std::vector<int64_t> *pad_list, bool *ceil_mode, bool *count_include_pad) {
  constexpr size_t kKernelDims = 5;
  constexpr size_t kStridesDims = 5;
  MS_EXCEPTION_IF_NULL(primitive);
  // attr kernel size
  *kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));
  if (kernel_size->size() != kKernelDims) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', 'kernel_size' must be 5, but got " << kernel_size->size()
                      << ".";
  }
  // attr strides
  *strides = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStrides));
  if (strides->size() != kStridesDims) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', 'strides' must be 5, but got " << strides->size() << ".";
  }
  if (std::any_of(strides->begin(), strides->end(), [](int64_t stride) { return stride <= 0; })) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', 'strides' must be all positive, but got 'strides': " << strides << ".";
  }
  // attr pad_list
  *pad_list = GetValue<std::vector<int64_t>>(primitive->GetAttr(kPadList));
  // attr count include pad
  *count_include_pad = GetValue<bool>(primitive->GetAttr(kCountIncludePad));
  // attr pad_mode
  CheckAndConvertUtils::GetPadModEnumValue(primitive->GetAttr(kPadMode), pad_mode, true);
  // attr ceil mode
  *ceil_mode = GetValue<bool>(primitive->GetAttr(kCeilMode));
}

std::vector<int64_t> GetOutputShape(const PrimitivePtr &primitive, const std::vector<int64_t> &in_shape,
                                    int64_t kernel_d, int64_t kernel_h, int64_t kernel_w, int64_t stride_d,
                                    int64_t stride_h, int64_t stride_w, const std::vector<int64_t> &pad_list,
                                    bool ceil_mode) {
  auto in_d = in_shape[2];
  auto in_h = in_shape[3];
  auto in_w = in_shape[4];
  int64_t out_d = 0;
  int64_t out_h = 0;
  int64_t out_w = 0;
  if (stride_d == 0 || stride_h == 0 || stride_w == 0) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', stride_d or stride_h or stride_w must be non-zero, but got stride_d: " << stride_d
                      << ", stride_h: " << stride_h << ", stride_w: " << stride_w << ".";
  }
  if (ceil_mode) {
    out_d = DoubleToLong(std::floor((in_d + pad_list[0] + pad_list[1] - kernel_d + stride_d - 1) / stride_d + 1));
    out_h = DoubleToLong(
      std::floor((in_h + pad_list[kInputIndex2] + pad_list[kInputIndex3] - kernel_h + stride_h - 1) / stride_h + 1));
    out_w = DoubleToLong(std::floor((in_w + pad_list[4] + pad_list[5] - kernel_w + stride_w - 1) / stride_w + 1));
    if ((out_d - 1) * stride_d >= in_d + pad_list[0]) {
      out_d--;
    }
    if ((out_h - 1) * stride_h >= in_h + pad_list[2]) {
      out_h--;
    }
    if ((out_w - 1) * stride_w >= in_w + pad_list[4]) {
      out_w--;
    }
  } else {
    out_d = DoubleToLong(std::floor((in_d + pad_list[0] + pad_list[1] - kernel_d) / stride_d + 1));
    out_h =
      DoubleToLong(std::floor((in_h + pad_list[kInputIndex2] + pad_list[kInputIndex3] - kernel_h) / stride_h + 1));
    out_w =
      DoubleToLong(std::floor((in_w + pad_list[kInputIndex4] + pad_list[kInputIndex5] - kernel_w) / stride_w + 1));
  }
  if (IsDynamic(in_shape)) {
    out_d = in_d == abstract::Shape::kShapeDimAny ? abstract::Shape::kShapeDimAny : out_d;
    out_h = in_h == abstract::Shape::kShapeDimAny ? abstract::Shape::kShapeDimAny : out_h;
    out_w = in_w == abstract::Shape::kShapeDimAny ? abstract::Shape::kShapeDimAny : out_w;
  }
  std::vector<int64_t> output_shape = {in_shape[0], in_shape[1], out_d, out_h, out_w};
  return output_shape;
}

void GetPadsByPadding(const PrimitivePtr &primitive, int64_t in_d, int64_t in_h, int64_t in_w, int64_t kernel_d,
                      int64_t kernel_h, int64_t kernel_w, int64_t stride_d, int64_t stride_h, int64_t stride_w,
                      const int64_t &pad_mode, const std::vector<int64_t> &padding, std::vector<int64_t> *pad_list) {
  MS_EXCEPTION_IF_NULL(pad_list);
  if (pad_mode == PadMode::VALID) {
    (void)pad_list->insert(pad_list->begin(), kAvgPool3DPadDims, 0);
  } else if (pad_mode == PadMode::SAME) {
    if (stride_d == 0 || stride_h == 0 || stride_w == 0) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', stride_d or stride_h or stride_w must be non-zero, but got stride_d: " << stride_d
                        << ", stride_h: " << stride_h << ", stride_w: " << stride_w << ".";
    }
    int64_t tail_d = in_d % stride_d;
    int64_t tail_h = in_h % stride_h;
    int64_t tail_w = in_w % stride_w;
    int64_t pad_d = std::max((tail_d > 0 ? kernel_d - tail_d : kernel_d - stride_d), (int64_t)0);
    int64_t pad_h = std::max((tail_h > 0 ? kernel_h - tail_h : kernel_h - stride_h), (int64_t)0);
    int64_t pad_w = std::max((tail_w > 0 ? kernel_w - tail_w : kernel_w - stride_w), (int64_t)0);
    constexpr int twice = 2;
    pad_list->push_back(static_cast<int64_t>(std::floor(pad_d / twice)));
    pad_list->push_back(pad_d - pad_list->at(0));
    pad_list->push_back(static_cast<int64_t>(std::floor(pad_h / twice)));
    pad_list->push_back(pad_h - pad_list->at(kInputIndex2));
    pad_list->push_back(static_cast<int64_t>(std::floor(pad_w / twice)));
    pad_list->push_back(pad_w - pad_list->at(kInputIndex4));
  } else if (pad_mode == PadMode::PAD) {
    pad_list->assign(padding.begin(), padding.end());
  }
}

abstract::ShapePtr AvgPool3DInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t k5DInputDims = 5;
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input size", int64_t(input_args.size()), kEqual, 1, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  // ToSupport Dynamic rank
  constexpr int64_t k5DOuputDims = 5;
  if (IsDynamicRank(in_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>(k5DOuputDims, abstract::Shape::kShapeDimAny));
  }
  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(in_shape.size()), kEqual, k5DInputDims, op_name);

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> strides;
  std::vector<int64_t> pad_list;
  int64_t pad_mode = 0;
  bool ceil_mode = false;
  bool count_include_pad = true;
  GetAttrs(primitive, &kernel_size, &strides, &pad_mode, &pad_list, &ceil_mode, &count_include_pad);
  auto in_d = in_shape[2];
  auto in_h = in_shape[3];
  auto in_w = in_shape[4];
  auto kernel_d = kernel_size[2];
  auto kernel_h = kernel_size[3];
  auto kernel_w = kernel_size[4];
  auto stride_d = strides[2];
  auto stride_h = strides[3];
  auto stride_w = strides[4];
  std::vector<int64_t> new_pad_list;
  GetPadsByPadding(primitive, in_d, in_h, in_w, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, pad_mode,
                   pad_list, &new_pad_list);
  if (new_pad_list.size() != kAvgPool3DPadDims) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', pad_list size must be 6, but got " << new_pad_list.size()
                      << ".";
  }
  primitive->set_attr(kPadList, MakeValue(new_pad_list));

  std::vector<int64_t> out_shape = GetOutputShape(primitive, in_shape, kernel_d, kernel_h, kernel_w, stride_d, stride_h,
                                                  stride_w, new_pad_list, ceil_mode);
  if (!IsDynamic(in_shape) &&
      std::any_of(out_shape.begin(), out_shape.end(), [](int64_t shp_v) { return shp_v <= 0; })) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', output shape's all elements must be positive, but got shape: " << out_shape << ".";
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr AvgPool3DInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input size", int64_t(input_args.size()), kEqual, 1, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_dtype = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, op_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(AvgPool3D, BaseOperator);
AbstractBasePtr AvgPool3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(AvgPool3DInferShape(primitive, input_args), AvgPool3DInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGAvgPool3DInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPool3DInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPool3DInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPool3DInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AvgPool3D, prim::kPrimAvgPool3D, AGAvgPool3DInfer, false);
}  // namespace ops
}  // namespace mindspore
