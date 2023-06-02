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

#include "ops/buffer_append.h"

#include <memory>
#include <set>
#include <vector>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/other_ops.h"
#include "ops/op_name.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BufferAppendInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();

  AbstractBasePtrList data_shape = input_args[kInputIndex0]->cast<abstract::AbstractSequencePtr>()->elements();
  AbstractBasePtrList exp_shape = input_args[kInputIndex1]->cast<abstract::AbstractSequencePtr>()->elements();
  auto count_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];

  (void)CheckAndConvertUtils::CheckInteger("exp elements ", SizeToLong(exp_shape.size()), kEqual,
                                           SizeToLong(data_shape.size()), op_name);
  int64_t exp_batch = 1;
  if (data_shape[0]->BuildShape()->cast<abstract::ShapePtr>()->shape().size() ==
      exp_shape[0]->BuildShape()->cast<abstract::ShapePtr>()->shape().size()) {
    exp_batch = exp_shape[0]->BuildShape()->cast<abstract::ShapePtr>()->shape()[0];
    for (size_t i = 0; i < data_shape.size(); i++) {
      if (data_shape[i]->BuildShape()->cast<abstract::ShapePtr>()->shape().size() !=
          exp_shape[i]->BuildShape()->cast<abstract::ShapePtr>()->shape().size()) {
        MS_LOG(EXCEPTION) << "For " << op_name << "the dimension of " << i << "th 'exp_shape' must be equal to "
                          << "the dimension of " << i << "th 'data_shape', but got the " << i
                          << "th 'exp_shape': " << exp_shape[i]->BuildShape()->cast<abstract::ShapePtr>()->shape()
                          << ", the " << i
                          << "th 'data_shape': " << data_shape[i]->BuildShape()->cast<abstract::ShapePtr>()->shape();
      }
      if (data_shape[i]->BuildShape()->cast<abstract::ShapePtr>()->shape()[0] <
          exp_shape[i]->BuildShape()->cast<abstract::ShapePtr>()->shape()[0]) {
        MS_LOG(EXCEPTION) << "For " << op_name << "the first dimension of " << i << "th 'data_shape' "
                          << "must be greater or equal to the dimension of " << i << "th 'exp_shape', "
                          << "but got the " << i
                          << "th 'exp_shape': " << exp_shape[i]->BuildShape()->cast<abstract::ShapePtr>()->shape()
                          << ", the " << i
                          << "th 'data_shape': " << data_shape[i]->BuildShape()->cast<abstract::ShapePtr>()->shape();
      }
    }
  } else {
    for (size_t i = 0; i < data_shape.size(); i++) {
      auto d_shape = data_shape[i]->BuildShape()->cast<abstract::ShapePtr>()->shape();
      std::vector<int64_t> temp_shape(d_shape.begin() + 1, d_shape.end());
      if (temp_shape != exp_shape[i]->BuildShape()->cast<abstract::ShapePtr>()->shape()) {
        MS_LOG(EXCEPTION) << "For " << op_name << ", the " << i << "th 'exp_shape' must be equal to the " << i
                          << "th 'data_shape' which excepts the first dimension. but got the " << i
                          << "th 'exp_shape': " << exp_shape[i]->BuildShape()->cast<abstract::ShapePtr>()->shape()
                          << ", the " << i << "th 'data_shape': " << d_shape;
      }
    }
  }
  (void)primitive->AddAttr("exp_batch", MakeValue(exp_batch));
  return std::make_shared<abstract::Shape>(count_shape);
}

TypePtr BufferAppendInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();

  AbstractBasePtrList data_type = input_args[kInputIndex0]->cast<abstract::AbstractSequencePtr>()->elements();
  AbstractBasePtrList exp_type = input_args[kInputIndex1]->cast<abstract::AbstractSequencePtr>()->elements();
  auto count_type = input_args[kInputIndex2]->BuildType();
  auto head_type = input_args[kInputIndex3]->BuildType();
  for (size_t i = 0; i < data_type.size(); i++) {
    if (data_type[i]->BuildType()->type_id() != exp_type[i]->BuildType()->type_id()) {
      MS_LOG(EXCEPTION) << "For " << op_name << ", each tensor in 'exp' must has the same type with 'data',"
                        << " but got 'data_type': " << data_type[i]->BuildType()->ToString()
                        << ", 'exp_type': " << exp_type[i]->BuildType()->ToString();
    }
  }
  const std::set<TypePtr> int_types = {kInt32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("count_type", count_type);
  (void)types.emplace("head_type", head_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, int_types, op_name);

  return count_type;
}
}  // namespace

AbstractBasePtr BufferAppendInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto &input : input_args) {
    MS_EXCEPTION_IF_NULL(input);
  }
  auto types = BufferAppendInferType(primitive, input_args);
  auto shapes = BufferAppendInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

MIND_API_OPERATOR_IMPL(BufferAppend, BaseOperator);
// AG means auto generated
class MIND_API AGBufferAppendInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BufferAppendInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BufferAppendInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BufferAppendInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BufferAppend, prim::kPrimBufferAppend, AGBufferAppendInfer, false);
}  // namespace ops
}  // namespace mindspore
