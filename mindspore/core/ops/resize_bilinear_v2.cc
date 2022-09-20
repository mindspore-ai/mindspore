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
#include "ops/resize_bilinear_v2.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void ResizeBilinearV2::set_align_corners(const bool align_corners) {
  (void)this->AddAttr(kAlignCorners, api::MakeValue(align_corners));
}

bool ResizeBilinearV2::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  return GetValue<bool>(value_ptr);
}

void ResizeBilinearV2::set_half_pixel_centers(const bool half_pixel_centers) {
  (void)this->AddAttr(kHalfPixelCenters, api::MakeValue(half_pixel_centers));
}

bool ResizeBilinearV2::get_half_pixel_centers() const {
  auto value_ptr = GetAttr(kHalfPixelCenters);
  return GetValue<bool>(value_ptr);
}

namespace {
void GetSizeValue(const abstract::AbstractBasePtr &size, std::vector<int64_t> *size_value,
                  std::vector<int64_t> *min_size, std::vector<int64_t> *max_size) {
  MS_EXCEPTION_IF_NULL(size);
  auto size_v = size->BuildValue();
  MS_EXCEPTION_IF_NULL(size_v);
  const int64_t size_size = 2;
  auto prim_name = kNameResizeBilinearV2;
  if (size->isa<abstract::AbstractTensor>()) {
    if (size_v->isa<tensor::Tensor>()) {
      *size_value = CheckAndConvertUtils::CheckTensorIntValue("size", size_v, prim_name);
      (void)CheckAndConvertUtils::CheckPositiveVector("size", *size_value, prim_name);
    } else {
      size_value->push_back(-1);
      size_value->push_back(-1);
      auto min_value = size->cast<abstract::AbstractTensorPtr>()->get_min_value();
      auto max_value = size->cast<abstract::AbstractTensorPtr>()->get_max_value();
      if (!min_value || !max_value) {
        MS_LOG(INFO) << "inputs['size'] min or max value of " << prim_name << " is empty.";
        return;
      }
      *min_size = GetValue<std::vector<int64_t>>(min_value);
      *max_size = GetValue<std::vector<int64_t>>(max_value);
      if (min_size->size() != size_size || max_size->size() != size_size) {
        MS_EXCEPTION(ValueError) << "For " << prim_name
                                 << ", inputs['size'] min and max value size must be 2, but got min: "
                                 << min_size->size() << ",  max: " << max_size->size() << ".";
      }
    }
  } else if (size->isa<abstract::AbstractTuple>() || size->isa<abstract::AbstractList>()) {
    *size_value = CheckAndConvertUtils::CheckIntOrTupleInt("size", size_v, prim_name);
    (void)CheckAndConvertUtils::CheckPositiveVector("size", *size_value, prim_name);
  } else {
    MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the "
                            << "size"
                            << " must be a Tensor or a tuple/list with all Int elements, but got " << size->ToString();
  }
}

abstract::ShapePtr ResizeBilinearV2InferShape(const PrimitivePtr &primitive,
                                              const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  auto x_shape = x_shape_ptr->shape();
  const int64_t shape_size = 4;
  const int64_t size_size = 2;
  if (!IsDynamicRank(x_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("the dimension of input_x", SizeToLong(x_shape.size()), kEqual, shape_size,
                                             prim_name);
  }
  std::vector<int64_t> size_value;
  std::vector<int64_t> min_size;
  std::vector<int64_t> max_size;
  auto input_num = input_args.size();
  constexpr auto kInputNum = 2;
  constexpr auto kConstSizeInputNum = 1;
  if (input_num == kInputNum) {
    auto size = input_args[1];
    GetSizeValue(size, &size_value, &min_size, &max_size);
  } else if (input_num == kConstSizeInputNum) {
    // const size to attr by the convert_const_input_to_attr pass
    size_value = GetValue<std::vector<int64_t>>(primitive->GetAttr(kSize));
  } else {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the number of inputs must be "
                             << " 1 or 2, but got " << input_num;
  }
  (void)CheckAndConvertUtils::CheckInteger("the dimension of size", SizeToLong(size_value.size()), kEqual, size_size,
                                           prim_name);
  std::vector<int64_t> output_shape;
  if (IsDynamicRank(x_shape)) {
    output_shape.push_back(-1);
    output_shape.push_back(-1);
  } else {
    output_shape.push_back(x_shape[0]);
    output_shape.push_back(x_shape[1]);
  }

  output_shape.push_back(size_value[0]);
  output_shape.push_back(size_value[1]);

  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr ResizeBilinearV2InferType(const PrimitivePtr &primitive,
                                  const std::vector<abstract::AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, primitive->name());
}
}  // namespace

void ResizeBilinearV2::Init(const bool align_corners, const bool half_pixel_centers) {
  this->set_align_corners(align_corners);
  this->set_half_pixel_centers(half_pixel_centers);
}

MIND_API_OPERATOR_IMPL(ResizeBilinearV2, BaseOperator);

AbstractBasePtr ResizeBilinearV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<abstract::AbstractBasePtr> &input_args) {
  auto infer_shape = ResizeBilinearV2InferShape(primitive, input_args);
  auto infer_type = ResizeBilinearV2InferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(ResizeBilinearV2, prim::kPrimResizeBilinearV2, ResizeBilinearV2Infer, nullptr, true);

REGISTER_HOST_DEPENDS(kNameResizeBilinearV2, {1});
}  // namespace ops
}  // namespace mindspore
