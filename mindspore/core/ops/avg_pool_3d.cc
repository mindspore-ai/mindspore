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

#include "ops/avg_pool_3d.h"
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t k5DInputDims = 5;
constexpr size_t kKernelDims = 3;
constexpr size_t kStridesDims = 3;
constexpr size_t kPadDims = 6;

void GetAttrs(const PrimitivePtr &primitive, std::vector<int64_t> *kernel_size, std::vector<int64_t> *strides,
              int64_t *pad_mode, std::vector<int64_t> *pad_list, bool *ceil_mode) {
  MS_EXCEPTION_IF_NULL(primitive);
  // attr kernel size
  *kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));
  if (kernel_size->size() != kKernelDims) {
    MS_LOG(EXCEPTION) << "kernel_size of AvgPool3D must be 3.";
  }
  // attr strides
  *strides = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStrides));
  if (strides->size() != kStridesDims) {
    MS_LOG(EXCEPTION) << "strides of AvgPool3D must be 3.";
  }
  if (std::any_of(strides->begin(), strides->end(), [](int64_t stride) { return stride <= 0; })) {
    MS_EXCEPTION(ValueError) << "invalid strides, strides must be all positive.";
  }
  // sttr pad_list
  *pad_list = GetValue<std::vector<int64_t>>(primitive->GetAttr(kPadList));
  // attr pad_mode
  CheckAndConvertUtils::GetPadModEnumValue(primitive->GetAttr(kPadMode), pad_mode, true);
  // attr ceil mode
  *ceil_mode = GetValue<bool>(primitive->GetAttr(kCeilMode));
}

std::vector<int64_t> GetOutputShape(const std::vector<int64_t> &in_shape, int64_t kernel_d, int64_t kernel_h,
                                    int64_t kernel_w, int64_t stride_d, int64_t stride_h, int64_t stride_w,
                                    const std::vector<int64_t> &pad_list, bool ceil_mode) {
  auto in_d = in_shape[2];
  auto in_h = in_shape[3];
  auto in_w = in_shape[4];
  int64_t out_d = 0;
  int64_t out_h = 0;
  int64_t out_w = 0;
  if (ceil_mode) {
    out_d =
      static_cast<int64_t>(std::floor((in_d + pad_list[0] + pad_list[1] - kernel_d + stride_d - 1) / stride_d + 1));
    out_h =
      static_cast<int64_t>(std::floor((in_h + pad_list[2] + pad_list[3] - kernel_h + stride_h - 1) / stride_h + 1));
    out_w =
      static_cast<int64_t>(std::floor((in_w + pad_list[4] + pad_list[5] - kernel_w + stride_w - 1) / stride_w + 1));
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
    out_d = static_cast<int64_t>(std::floor((in_d + pad_list[0] + pad_list[1] - kernel_d) / stride_d + 1));
    out_h = static_cast<int64_t>(std::floor((in_h + pad_list[2] + pad_list[3] - kernel_h) / stride_h + 1));
    out_w = static_cast<int64_t>(std::floor((in_w + pad_list[4] + pad_list[5] - kernel_w) / stride_w + 1));
  }
  std::vector<int64_t> output_shape = {in_shape[0], in_shape[1], out_d, out_h, out_w};
  return output_shape;
}

void GetPadsByPadding(int64_t in_d, int64_t in_h, int64_t in_w, int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
                      int64_t stride_d, int64_t stride_h, int64_t stride_w, const int64_t &pad_mode,
                      const std::vector<int64_t> &padding, std::vector<int64_t> *pad_list) {
  if (pad_mode == PadMode::VALID) {
    (void)pad_list->insert(pad_list->begin(), kPadDims, 0);
  } else if (pad_mode == PadMode::SAME) {
    MS_EXCEPTION_IF_ZERO("stride_d", stride_d);
    MS_EXCEPTION_IF_ZERO("stride_h", stride_h);
    MS_EXCEPTION_IF_ZERO("stride_w", stride_w);
    int64_t tail_d = in_d % stride_d;
    int64_t tail_h = in_h % stride_h;
    int64_t tail_w = in_w % stride_w;
    int64_t pad_d = std::max((tail_d > 0 ? kernel_d - tail_d : kernel_d - stride_d), (int64_t)0);
    int64_t pad_h = std::max((tail_h > 0 ? kernel_h - tail_h : kernel_h - stride_h), (int64_t)0);
    int64_t pad_w = std::max((tail_w > 0 ? kernel_w - tail_w : kernel_w - stride_w), (int64_t)0);
    pad_list->push_back(static_cast<int64_t>(std::floor(pad_d / 2)));
    pad_list->push_back(pad_d - pad_list->at(0));
    pad_list->push_back(static_cast<int64_t>(std::floor(pad_h / 2)));
    pad_list->push_back(pad_h - pad_list->at(2));
    pad_list->push_back(static_cast<int64_t>(std::floor(pad_w / 2)));
    pad_list->push_back(pad_w - pad_list->at(4));
  } else if (pad_mode == PadMode::PAD) {
    pad_list->assign(padding.begin(), padding.end());
  }
}

abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input size", int64_t(input_args.size()), kEqual, 1, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(in_shape.size()), kEqual, k5DInputDims, op_name);

  std::vector<int64_t> kernel_size;
  std::vector<int64_t> strides;
  std::vector<int64_t> pad_list;
  int64_t pad_mode = 0;
  bool ceil_mode = false;
  GetAttrs(primitive, &kernel_size, &strides, &pad_mode, &pad_list, &ceil_mode);
  auto in_d = in_shape[2];
  auto in_h = in_shape[3];
  auto in_w = in_shape[4];
  auto kernel_d = kernel_size[0];
  auto kernel_h = kernel_size[1];
  auto kernel_w = kernel_size[2];
  auto stride_d = strides[0];
  auto stride_h = strides[1];
  auto stride_w = strides[2];
  std::vector<int64_t> new_pad_list;
  GetPadsByPadding(in_d, in_h, in_w, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, pad_mode, pad_list,
                   &new_pad_list);
  if (new_pad_list.size() != kPadDims) {
    MS_LOG(EXCEPTION) << "pad_list size must be 6.";
  }
  primitive->set_attr(kPadList, MakeValue(new_pad_list));

  std::vector<int64_t> out_shape =
    GetOutputShape(in_shape, kernel_d, kernel_h, kernel_w, stride_d, stride_h, stride_w, new_pad_list, ceil_mode);
  if (std::any_of(out_shape.begin(), out_shape.end(), [](int64_t shp_v) { return shp_v <= 0; })) {
    MS_LOG(EXCEPTION) << "output size is not valid.";
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input size", int64_t(input_args.size()), kEqual, 1, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_dtype = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, op_name);
}
}  // namespace

AbstractBasePtr AvgPool3DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(InferShape(primitive, input_args), InferType(primitive, input_args));
}

REGISTER_PRIMITIVE_EVAL_IMPL(AvgPool3D, prim::kPrimAvgPool3D, AvgPool3DInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
