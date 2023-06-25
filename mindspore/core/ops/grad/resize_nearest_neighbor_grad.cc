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

#include <algorithm>
#include <iterator>
#include <memory>
#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/image_ops.h"
#include "ops/grad/resize_nearest_neighbor_grad.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kResizeNearestNeighborGradInputNum = 2;
constexpr auto kResizeIdx = 1;

abstract::ShapePtr ResizeNearestNeighborGradInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  if (input_args.size() != kResizeNearestNeighborGradInputNum) {
    MS_LOG(EXCEPTION) << "ResizeNearsetNeighborGrad's input num should be " << kResizeNearestNeighborGradInputNum
                      << ", but got " << input_args.size();
  }

  auto grad_shape_ptr = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  auto grad_shape = grad_shape_ptr->shape();
  std::vector<int64_t> ret_shape;
  if (IsDynamicRank(grad_shape)) {
    ret_shape.push_back(abstract::Shape::kShapeDimAny);
    ret_shape.push_back(abstract::Shape::kShapeDimAny);
  } else {
    const int64_t kVALUE_4 = 4;
    (void)CheckAndConvertUtils::CheckInteger("grads", SizeToLong(grad_shape.size()), kEqual, kVALUE_4, prim_name);
    ret_shape.push_back(grad_shape[kInputIndex0]);
    ret_shape.push_back(grad_shape[kInputIndex1]);
  }

  auto size_ptr = input_args[kResizeIdx]->BuildValue();
  MS_EXCEPTION_IF_NULL(size_ptr);
  std::vector<int64_t> size_v = GetShapeValue(primitive, input_args[kResizeIdx]);
  if (!IsDynamicRank(size_v)) {
    const int64_t kVALUE_2 = 2;
    (void)CheckAndConvertUtils::CheckInteger("size", SizeToLong(size_v.size()), kEqual, kVALUE_2, prim_name);
  }
  ret_shape.insert(ret_shape.end(), size_v.begin(), size_v.end());

  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr ResizeNearestNeighborGradInferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  return input_args[0]->BuildType();
}
}  // namespace

void ResizeNearestNeighborGrad::set_align_corners(const bool align_corners) {
  (void)AddAttr(kAlignCorners, api::MakeValue(align_corners));
  return;
}

bool ResizeNearestNeighborGrad::get_align_corners() const {
  auto value_ptr = GetAttr(kAlignCorners);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(ResizeNearestNeighborGrad, BaseOperator);
AbstractBasePtr ResizeNearestNeighborGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("infer", SizeToLong(CheckAndConvertUtils::GetRemoveMonadAbsNum(input_args)),
                                           kEqual, input_num, prim_name);
  return abstract::MakeAbstract(ResizeNearestNeighborGradInferShape(primitive, input_args),
                                ResizeNearestNeighborGradInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGResizeNearestNeighborGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeNearestNeighborGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeNearestNeighborGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ResizeNearestNeighborGradInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeNearestNeighborGrad, prim::kPrimResizeNearestNeighborGrad,
                                 AGResizeNearestNeighborGradInfer, false);
}  // namespace ops
}  // namespace mindspore
