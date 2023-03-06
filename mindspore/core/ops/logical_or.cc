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

#include <map>
#include <string>
#include <set>
#include <vector>
#include <memory>

#include "ops/logical_or.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LogicalOrInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  return BroadCastInferShape(op_name, input_args);
}

TypePtr LogicalOrInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto x_dtype = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_dtype);
  auto y_dtype = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(y_dtype);
  const std::basic_string<char> kBool = "Tensor[Bool]";
  std::ostringstream buffer;
  buffer << "For primitive[LogicalOr], the input argument[x, y, ] must be a type of {Tensor[Bool], }, but got ";
  if (x_dtype->ToString() != kBool) {
    MS_EXCEPTION(TypeError) << buffer.str() << x_dtype->ToString() << ".";
  }
  if (y_dtype->ToString() != kBool) {
    MS_EXCEPTION(TypeError) << buffer.str() << y_dtype->ToString() << ".";
  }
  return x_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(LogicalOr, BaseOperator);
AbstractBasePtr LogicalOrInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = LogicalOrInferType(primitive, input_args);
  auto infer_shape = LogicalOrInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGLogicalOrInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LogicalOrInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LogicalOrInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LogicalOrInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LogicalOr, prim::kPrimLogicalOr, AGLogicalOrInfer, false);
}  // namespace ops
}  // namespace mindspore
