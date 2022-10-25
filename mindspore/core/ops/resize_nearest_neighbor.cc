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

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "ops/resize_nearest_neighbor.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void ResizeNearestNeighbor::Init(const std::vector<int64_t> &size, const bool align_corners) {
  this->set_align_corners(align_corners);
}
void ResizeNearestNeighbor::set_align_corners(const bool align_corners) {
  (void)this->AddAttr(kAlignCorners, api::MakeValue(align_corners));
}
bool ResizeNearestNeighbor::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  return GetValue<bool>(value_ptr);
}

namespace {
abstract::ShapePtr ResizeNearestNeighborInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  auto x_shape = x_shape_ptr->shape();
  ValuePtr size_ptr;
  if (x_shape_ptr->IsDynamic() && input_args.size() > 1) {
    size_ptr = input_args[1]->BuildValue();
  } else {
    size_ptr = primitive->GetAttr(kSize);
  }
  auto size_v = CheckAndConvertUtils::CheckIntOrTupleInt("size", size_ptr, prim_name);
  (void)CheckAndConvertUtils::CheckPositiveVector("size", size_v, prim_name);
  const int64_t shape_size = 4;
  const int64_t size_size = 2;
  (void)CheckAndConvertUtils::CheckInteger("the dimension of input_x", SizeToLong(x_shape.size()), kEqual, shape_size,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("the dimension of size", SizeToLong(size_v.size()), kEqual, size_size,
                                           prim_name);
  x_shape.erase(x_shape.begin() + size_size, x_shape.end());
  x_shape.insert(x_shape.end(), size_v.begin(), size_v.end());
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr ResizeNearestNeighborInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto valid_types = common_valid_types;
  (void)valid_types.insert(kComplex128);
  (void)valid_types.insert(kComplex64);
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(ResizeNearestNeighbor, BaseOperator);
AbstractBasePtr ResizeNearestNeighborInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args)),
                                           kEqual, input_num, prim_name);
  return abstract::MakeAbstract(ResizeNearestNeighborInferShape(primitive, input_args),
                                ResizeNearestNeighborInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(ResizeNearestNeighbor, prim::kPrimResizeNearestNeighbor, ResizeNearestNeighborInfer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
