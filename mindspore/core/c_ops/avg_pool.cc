/**
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
#include "c_ops/avg_pool.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "c_ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
void AvgPool::set_padding(const std::string &pad) { this->AddAttr("padding", MakeValue(pad)); }

std::string AvgPool::get_padding() const {
  auto value_ptr = GetAttr("padding");
  return GetValue<std::string>(value_ptr);
}
void AvgPool::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  this->AddAttr("k_size", MakeValue(kernel_size));
}

std::vector<int64_t> AvgPool::get_kernel_size() const {
  auto value_ptr = GetAttr("k_size");
  return GetValue<std::vector<int64_t>>(value_ptr);
}
void AvgPool::set_strides(const std::vector<int64_t> &strides) { this->AddAttr("strides", MakeValue(strides)); }

std::vector<int64_t> AvgPool::get_strides() const {
  auto value_ptr = GetAttr("strides");
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void AvgPool::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &stride,
                   const std::string &padding) {
  auto prim_name = this->name();
  this->AddAttr("data_format", MakeValue("NCHW"));
  this->set_padding(CheckAndConvertUtils::CheckString("padding", padding, {"valid", "same"}, prim_name));
  this->set_kernel_size(CheckAndConvertUtils::CheckPositiveVector("k_size", kernel_size, prim_name, false, true));
  this->set_strides(CheckAndConvertUtils::CheckPositiveVector("strides", stride, this->name(), false, true));
}

abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto pool_prim = primitive->cast<PrimAvgPoolPtr>();
  MS_EXCEPTION_IF_NULL(pool_prim);
  auto op_name = pool_prim->name();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->GetShapeTrack(), op_name);
  CheckAndConvertUtils::CheckInteger("x_rank", in_shape.size(), kEqual, 4, op_name);
  auto kernel_size = pool_prim->get_kernel_size();
  auto pad_mode = pool_prim->get_padding();
  auto batch = in_shape[0];
  auto channel = in_shape[1];
  auto in_h = in_shape[2];
  auto in_w = in_shape[3];

  auto strides = pool_prim->get_strides();
  auto kernel_h = kernel_size[2];
  auto kernel_w = kernel_size[3];
  auto stride_h = strides[2];
  auto stride_w = strides[3];
  int64_t out_h = -1;
  int64_t out_w = -1;
  if (pad_mode == "valid") {
    out_h = ceil((in_h - (kernel_h - 1)) / stride_h);
    out_w = ceil((in_w - (kernel_w - 1)) / stride_w);
  } else if (pad_mode == "same") {
    out_h = ceil(in_h / stride_h);
    out_w = ceil(in_w / stride_w);
  }
  std::vector<int64_t> out_shape = {batch, channel, out_h, out_w};
  if (std::any_of(out_shape.begin(), out_shape.end(), [](int a) { return a <= 0; })) {
    MS_LOG(EXCEPTION) << "Kernel size is not valid.";
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](AbstractBasePtr a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  return input_args[0]->BuildType();
}

AbstractBasePtr AvgPoolInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_EVAL_IMPL(AvgPool, prim::kPrimAvgPool, AvgPoolInfer);
REGISTER_PRIMITIVE_C(kNameAvgPool, AvgPool);
}  // namespace mindspore
