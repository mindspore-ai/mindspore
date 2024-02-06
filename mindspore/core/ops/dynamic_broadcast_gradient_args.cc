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

#include "ops/dynamic_broadcast_gradient_args.h"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "ops/structure_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
int64_t CheckInputsAndGetShape(const AbstractBasePtr &input_arg, const string &prim_name) {
  MS_EXCEPTION_IF_NULL(input_arg);
  if (CheckAndConvertUtils::IsTensor(input_arg)) {
    auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_arg->GetShape())[kShape];
    auto input_size = input_shape.size();
    if (input_size != 1) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input shape must be 1-D, but got: " << input_size << "-D.";
    }
    return input_shape[0];
  } else if (CheckAndConvertUtils::IsTuple(input_arg)) {
    auto idx_shape_ptr = input_arg->GetShape();
    MS_EXCEPTION_IF_NULL(idx_shape_ptr);
    auto shape_tuple = idx_shape_ptr->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_tuple);
    size_t idx_size = shape_tuple->size();
    return SizeToLong(idx_size);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input type must be a tuple or Tensor.";
  }
}
}  // namespace

class MIND_API DynamicBroadcastGradientArgsInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = 2;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    auto x_shape0 = CheckInputsAndGetShape(input_args[0], prim_name);
    auto y_shape0 = CheckInputsAndGetShape(input_args[1], prim_name);
    ShapeVector max_shape;
    // DynamicBroadcastGradientArgs is a compute depend op
    if (x_shape0 >= 0 && y_shape0 >= 0) {
      max_shape = {x_shape0 > y_shape0 ? x_shape0 : y_shape0};
      // Currently, if the max_shape is 0, there may be some problems
      max_shape[0] = max_shape[0] != 0 ? max_shape[0] : 1;
    }

    auto out_shape = std::make_shared<abstract::Shape>(max_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape, out_shape});
  }
  TypePtr InferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &) const override {
    auto types = std::vector<TypePtr>{kInt64, kInt64};
    auto output_type = std::make_shared<Tuple>(types);
    return output_type;
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    ShapeVector shape{abstract::Shape::kShapeDimAny};
    auto elm_shape = std::make_shared<abstract::Shape>(shape);
    auto out_shape = std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{elm_shape, elm_shape});
    auto types = std::vector<TypePtr>{kInt64, kInt64};
    auto output_type = std::make_shared<Tuple>(types);
    return abstract::MakeAbstract(out_shape, output_type);
  }

  ValuePtr InferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    if (input_args.empty()) {
      MS_LOG(ERROR) << "DynamicBroadcastGradientArgs input args dose not exist.";
      return nullptr;
    }

    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }

    auto x = input_args[0]->GetValue();
    if (x->ContainsValueAny()) {
      MS_LOG(INFO) << "DynamicBroadcastGradientArgs input_0 is ValueAny, will backoff to cpu.";
      return nullptr;
    }

    auto y = input_args[1]->GetValue();
    if (y->ContainsValueAny()) {
      MS_LOG(INFO) << "DynamicBroadcastGradientArgs input_1 is ValueAny, will backoff to cpu.";
      return nullptr;
    }
    auto grad_reduce_idx = BroadcastGradientArgsInferValue(GetValue<ShapeVector>(x), GetValue<ShapeVector>(y));
    ValuePtr res = MakeValue(grad_reduce_idx);
    return res;
  }
};

ShapeArray BroadcastGradientArgsInferValue(const ShapeVector &x_shape, const ShapeVector &y_shape) {
  ShapeArray bc_axis;
  if (x_shape == y_shape) {
    (void)bc_axis.emplace_back(ShapeVector{});
    (void)bc_axis.emplace_back(ShapeVector{});
    return bc_axis;
  }
  ShapeVector grad_x_reduce_idx;
  ShapeVector grad_y_reduce_idy;
  auto x_size = x_shape.size();
  auto y_size = y_shape.size();
  auto n = std::max(x_size, y_size);
  for (size_t i = n; i >= 1; i--) {
    auto x_i = x_size < i ? 1 : x_shape[x_size - i];
    auto y_i = y_size < i ? 1 : y_shape[y_size - i];
    const int64_t reduce_idx = SizeToLong(n - i);
    if (x_i == y_i) {
      continue;
    } else if (x_i == 1) {
      grad_x_reduce_idx.push_back(reduce_idx);
    } else if (y_i == 1) {
      grad_y_reduce_idy.push_back(reduce_idx);
    } else {
      MS_LOG(EXCEPTION) << "For 'BroadcastGradientArgs', the inputs shape " << x_shape << " and " << y_shape
                        << " are not compatible";
    }
  }

  (void)bc_axis.emplace_back(std::move(grad_x_reduce_idx));
  (void)bc_axis.emplace_back(std::move(grad_y_reduce_idy));
  return bc_axis;
}

MIND_API_OPERATOR_IMPL(DynamicBroadcastGradientArgs, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(DynamicBroadcastGradientArgs, prim::kPrimDynamicBroadcastGradientArgs,
                                 DynamicBroadcastGradientArgsInfer, true);
}  // namespace ops
}  // namespace mindspore
