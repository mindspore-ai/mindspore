/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/array_len.h"
#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ArrayLenInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  // Only called when the input of shape is dynamic shape/rank tensor.
  // infer shape
  MS_EXCEPTION_IF_NULL(primitive);
  //  auto Kscalarshape_ = 1;
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("shape infer", static_cast<int64_t>(input_args.size()), kEqual, 1, op_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto in_shape = shape_map[kShape];
  // infer type
  std::set<TypePtr> valid_params_types = {kTensorType};
  (void)CheckAndConvertUtils::CheckSubClass("shape type", input_args[0]->BuildType(), valid_params_types, op_name);
  auto seq_abs = input_args[0]->BuildShape()->cast<abstract::ShapePtr>();
  if (seq_abs->shape().empty()) {
    MS_EXCEPTION(TypeError) << "Not support len of a 0-D tensor.";
  }
  if (IsDynamicShape(in_shape)) {
    return std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
  }
  return std::make_shared<abstract::AbstractScalar>((seq_abs->shape()[0]));
}
MIND_API_OPERATOR_IMPL(array_len, BaseOperator);
class ArrayLenInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ArrayLenInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return ArrayLenInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ArrayLenInferInner(primitive, input_args);
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("shape infer", int64_t(input_args.size()), kEqual, 1, op_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    std::set<TypePtr> valid_params_types = {kTensorType};
    (void)CheckAndConvertUtils::CheckSubClass("shape type", input_args[0]->BuildType(), valid_params_types, op_name);
    auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
    if (shape_map.count(kShape) == 0) {
      MS_LOG(EXCEPTION) << "For primitive " << op_name << " the input convert shape failed.";
    }
    const auto &inshape = shape_map[kShape];
    if (IsDynamic(inshape)) {
      // If the input of shape is dynamic shape/rank tensor, value can not be directly built.
      // Run infer of shape.
      return nullptr;
    }
    return MakeValue(inshape);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(array_len, prim::kPrimArrayLen, ArrayLenInfer, false);
}  // namespace ops
}  // namespace mindspore
