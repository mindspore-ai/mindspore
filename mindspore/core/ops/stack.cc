/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ops/stack.h"

#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/stack_comm.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
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
namespace {
constexpr int64_t kUnknownDim = -1;
constexpr int64_t kUnknownRank = -2;
}  // namespace
void Stack::set_axis(const int64_t axis) { (void)AddAttr(kAxis, api::MakeValue(axis)); }

int64_t Stack::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }

void Stack::Init(const int64_t axis) { this->set_axis(axis); }
namespace {
abstract::ShapePtr StackInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() < 1) {
    MS_LOG(ERROR) << "Invalid input size " << input_args.size();
  }
  const auto &prim_name = primitive->name();
  AbstractBasePtrList elements = input_args;
  if (input_args.size() == 1) {
    if (!input_args[0]->isa<abstract::AbstractSequence>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input data type must be list or tuple of tensors.";
    }
    elements = input_args[0]->cast<abstract::AbstractSequencePtr>()->elements();
  }
  (void)CheckAndConvertUtils::CheckInteger("stack element num", SizeToLong(elements.size()), kGreaterEqual, 1,
                                           primitive->name());

  bool has_rank_valid_shape = false;
  ShapeVector input_shape;
  size_t element_rank = 0;
  for (size_t i = 0; i < elements.size(); ++i) {
    MS_EXCEPTION_IF_NULL(elements[i]);
    auto input_shape_tmp = CheckAndConvertUtils::ConvertShapePtrToShapeMap(elements[i]->BuildShape())[kShape];
    if (IsDynamicRank(input_shape_tmp)) {
      continue;
    }

    if (!has_rank_valid_shape) {
      has_rank_valid_shape = true;
      input_shape = input_shape_tmp;
      element_rank = input_shape_tmp.size();
      continue;
    }
    if (input_shape_tmp.size() != input_shape.size()) {
      MS_EXCEPTION(ValueError) << "All input shape size must be the same!";
    }
    for (size_t j = 0; j < input_shape.size(); ++j) {
      if (input_shape.at(j) == kUnknownDim && input_shape_tmp.at(j) != kUnknownDim) {
        input_shape[j] = input_shape_tmp.at(j);
        continue;
      }
      if (input_shape_tmp.at(j) != input_shape.at(j)) {
        MS_EXCEPTION(ValueError) << "All input shape must be the same! " << input_shape_tmp << " And " << input_shape;
      }
    }
  }

  if (!has_rank_valid_shape) {
    return std::make_shared<abstract::Shape>(ShapeVector{kUnknownRank});
  }
  std::vector<int64_t> infer_shape = input_shape;
  auto axis_temp = GetValue<int64_t>(primitive->GetAttr(kAxis));
  CheckAndConvertUtils::CheckInRange<int64_t>("Stack axis", axis_temp, kIncludeBoth,
                                              {-SizeToLong(element_rank) - 1, SizeToLong(element_rank)},
                                              primitive->name());
  auto axis = axis_temp < 0 ? static_cast<size_t>(axis_temp) + element_rank + 1 : LongToSize(axis_temp);
  (void)infer_shape.insert(infer_shape.begin() + axis, elements.size());
  return std::make_shared<abstract::Shape>(infer_shape);
}

TypePtr StackInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto &prim_name = primitive->name();
  AbstractBasePtrList elements = input_args;
  if (input_args.size() == 1) {
    if (!input_args[0]->isa<abstract::AbstractSequence>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input data type must be list or tuple of tensors.";
    }
    elements = input_args[0]->cast<abstract::AbstractSequencePtr>()->elements();
  }
  (void)CheckAndConvertUtils::CheckInteger("stack element num", SizeToLong(elements.size()), kGreaterEqual, 1,
                                           primitive->name());
  primitive->AddAttr("num", MakeValue(SizeToLong(elements.size())));
  auto element0 = elements[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(element0);
  auto infer_type0 = element0->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type0);
  for (size_t i = 1; i < elements.size(); i++) {
    auto elementi = elements[i]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(elementi);
    auto infer_typei = elementi->BuildType();
    MS_EXCEPTION_IF_NULL(infer_typei);
    if (infer_typei == infer_type0) {
      MS_EXCEPTION(TypeError) << "All input must have the same data type!input[" << i << "] data type = " << infer_typei
                              << "infer_type0= " << infer_type0;
    }
  }
  return infer_type0;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Stack, BaseOperator);
AbstractBasePtr StackInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  auto infer_shape = StackInferShape(primitive, input_args);
  auto infer_type = StackInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

BaseShapePtr AGStackInfer::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return StackInferShape(primitive, input_args);
}

TypePtr AGStackInfer::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return StackInferType(primitive, input_args);
}
AbstractBasePtr AGStackInfer::InferShapeAndType(const abstract::AnalysisEnginePtr &engine,
                                                const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  return StackInfer(engine, primitive, input_args);
}

REGISTER_PRIMITIVE_OP_INFER_IMPL(Stack, prim::kPrimStack, AGStackInfer, false);
}  // namespace ops
}  // namespace mindspore
