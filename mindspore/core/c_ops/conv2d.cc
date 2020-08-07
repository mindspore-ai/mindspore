/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "c_ops/conv2d.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace {
using PrimConv2dPtr = std::shared_ptr<Conv2d>;
abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto conv_prim = primitive->cast<PrimConv2dPtr>();
  MS_EXCEPTION_IF_NULL(conv_prim);
  auto prim_name = conv_prim->name();
  CheckAndConvertUtils::CheckInRange("Conv2d Infer", input_args.size(), kIncludeLeft, {2, 3}, prim_name);
  auto w_shape = CheckAndConvertUtils::ConvertShapePtrToShape("w_shape", input_args[0]->GetShapeTrack(), prim_name);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[1]->GetShapeTrack(), prim_name);

  CheckAndConvertUtils::CheckInteger("weight rank", w_shape.size(), kEqual, 4, prim_name);
  CheckAndConvertUtils::CheckInteger("x rank", x_shape.size(), kEqual, 4, prim_name);
  CheckAndConvertUtils::Check("x_shape[1] / group", x_shape[1] / conv_prim->GetGroup(), kEqual, "w_shape[1]",
                              w_shape[1], conv_prim->name());
  auto out_channel = conv_prim->GetOutputChannel();
  CheckAndConvertUtils::Check("out_channel", out_channel, kEqual, "w_shape[0]", w_shape[0], conv_prim->name());
  std::vector<int> temp_w;
  std::copy(w_shape.begin() + 2, w_shape.end(), std::back_inserter(temp_w));
  CheckAndConvertUtils::Check("kernel_size", conv_prim->GetKernelSize(), kEqual, "w_shape[2:4]", temp_w,
                              conv_prim->name());

  auto kernel_size_h = w_shape[2];
  auto kernel_size_w = w_shape[3];
  auto stride = conv_prim->GetStride();
  auto dilation = conv_prim->GetDilation();
  auto stride_h = stride[2];
  auto stride_w = stride[3];
  auto dilation_h = dilation[2];
  auto dilation_w = dilation[3];
  int h_out = -1;
  int w_out = -1;
  std::vector<int> pad_list(4, 0);
  auto pad_mode = conv_prim->GetPadMode();
  if (pad_mode == "valid") {
    h_out = ceil((x_shape[2] - dilation_h * (kernel_size_h - 1)) / stride_h);
    w_out = ceil((x_shape[3] - dilation_w * (kernel_size_w - 1)) / stride_w);
  } else if (pad_mode == "same") {
    h_out = ceil(x_shape[2] / stride_h);
    w_out = ceil(x_shape[3] / stride_w);

    auto pad_needed_h = std::max(0, (h_out - 1) * stride_h + dilation_h * (kernel_size_h - 1) + 1 - x_shape[2]);
    pad_list.emplace_back(floor(pad_needed_h / 2));
    pad_list.emplace_back(pad_needed_h / 2);
    auto pad_needed_w = std::max(0, (w_out - 1) * stride_w + dilation_w * (kernel_size_w - 1) + 1 - x_shape[3]);
    auto pad_left = floor(pad_needed_w / 2);
    pad_list.emplace_back(pad_left);
    pad_list.emplace_back(pad_needed_h - pad_left);
  } else if (pad_mode == "pad") {
    std::copy(conv_prim->GetPad().begin(), conv_prim->GetPad().end(), std::back_inserter(pad_list));
    auto pad_top = conv_prim->GetPad()[0];
    auto pad_bottom = conv_prim->GetPad()[1];
    auto pad_right = conv_prim->GetPad()[2];
    auto pad_left = conv_prim->GetPad()[3];

    h_out = 1 + (x_shape[2] + pad_top + pad_bottom - kernel_size_h - (kernel_size_h - 1) * (dilation_h - 1)) / stride_h;
    w_out = 1 + (x_shape[3] + pad_left + pad_right - kernel_size_w - (kernel_size_w - 1) * (dilation_w - 1)) / stride_w;
    h_out = floor(h_out);
    w_out = floor(w_out);
  }
  conv_prim->SetPadList(pad_list);
  std::vector<int> out_shape = {x_shape[0], out_channel, h_out, w_out};
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  CheckAndConvertUtils::CheckInRange("", input_args.size(), kIncludeLeft, {2, 3}, prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_type = CheckAndConvertUtils::ConvertTypePtrToTypeId("x_dtype", input_args[0]->GetTypeTrack(), prim->name());
  const std::set<TypeId> valid_types = {kNumberTypeInt8, kNumberTypeInt32, kNumberTypeFloat16, kNumberTypeFloat32};
  std::map<std::string, TypePtr> types;
  types.emplace("x", input_args[0]->GetTypeTrack());
  types.emplace("w", input_args[1]->GetTypeTrack());
  CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  if (x_type == kNumberTypeInt8) {
    return std::make_shared<TensorType>(TypeIdToType(kNumberTypeInt32));
  }
  return std::make_shared<TensorType>(TypeIdToType(x_type));
}
}  // namespace
void Conv2d::Init(int out_channel, const std::vector<int> &kernel_size, int mode, const std::string &pad_mode,
                  const std::vector<int> &pad, const std::vector<int> &stride, const std::vector<int> &dilation,
                  int group) {
  auto prim_name = this->name();
  this->AddAttr("data_format", MakeValue("NCHW"));
  this->AddAttr("offset_a", MakeValue(0));
  this->SetKernelSize(CheckAndConvertUtils::CheckPositiveVector(kKernelSize, kernel_size, prim_name));
  this->SetStride(CheckAndConvertUtils::CheckPositiveVector(kStride, stride, this->name(), true, true));
  this->SetDilation(CheckAndConvertUtils::CheckPositiveVector(kDilation, dilation, this->name(), true, true));
  this->SetPadMode(CheckAndConvertUtils::CheckString(kPadMode, pad_mode, {"valid", "same", "pad"}, prim_name));
  CheckAndConvertUtils::CheckInteger("pad size", pad.size(), kEqual, 4, prim_name);
  if (pad_mode == "pad") {
    for (auto item : pad) {
      CheckAndConvertUtils::Check("pad item", item, kGreaterEqual, "zeros list", 0, prim_name);
    }
  } else {
    CheckAndConvertUtils::Check(kPad, pad, kEqual, "zeros list", {0, 0, 0, 0}, prim_name);
  }
  this->SetPad(CheckAndConvertUtils::CheckPositiveVector(kPad, pad, this->name(), true, true));
  this->SetMode(CheckAndConvertUtils::CheckInteger("mode", mode, kEqual, 1, prim_name));
  this->SetOutChannel(CheckAndConvertUtils::CheckInteger("out_channel", out_channel, kGreaterThan, 0, prim_name));
  this->SetGroup(CheckAndConvertUtils::CheckInteger("group", group, kGreaterThan, 0, prim_name));
}

AbstractBasePtr Conv2dInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
}  // namespace mindspore
