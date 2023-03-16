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
#include <memory>
#include <vector>
#include <iterator>

#include "ops/op_utils.h"
#include "ops/grad/resize_nearest_neighbor_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

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
  auto size_ptr = input_args[kResizeIdx]->BuildValue();
  MS_EXCEPTION_IF_NULL(size_ptr);

  std::vector<int64_t> size_v;
  if (size_ptr->isa<tensor::Tensor>()) {
    auto tensor_data = CheckAndConvertUtils::CheckTensorIntValue("input[size]", size_ptr, prim_name);
    for (auto iter = tensor_data.begin(); iter != tensor_data.end(); ++iter) {
      size_v.push_back(static_cast<int64_t>(*iter));
    }
  } else if (size_ptr->isa<ValueTuple>() && IsValueKnown(size_ptr)) {
    std::vector<ValuePtr> size_vec = size_ptr->cast<ValueTuplePtr>()->value();
    (void)std::transform(size_vec.begin(), size_vec.end(), std::back_inserter(size_v),
                         [](const ValuePtr e) { return GetValue<int64_t>(e); });
  } else if (size_ptr->isa<ValueAny>()) {
    size_v.push_back(-1);
    size_v.push_back(-1);
  } else {
    size_v = GetValue<std::vector<int64_t>>(size_ptr);
  }

  std::vector<int64_t> ret_shape;
  ret_shape.push_back(grad_shape[0]);
  ret_shape.push_back(grad_shape[1]);
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
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ResizeNearestNeighborGrad, prim::kPrimResizeNearestNeighborGrad,
                                 AGResizeNearestNeighborGradInfer, false);
}  // namespace ops
}  // namespace mindspore
