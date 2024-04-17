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

std::pair<abstract::BaseShapePtrList, size_t> GetSequnceIndexShape(const PrimitivePtr &primitive, size_t index,
                                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  bool is_tuple_x = CheckAndConvertUtils::IsTuple(input_args[index]);
  bool is_list_x = CheckAndConvertUtils::IsList(input_args[index]);
  if ((!is_tuple_x) && (!is_list_x)) {
    MS_EXCEPTION(TypeError) << "For [" << op_name << "] should have ListTensor or TupleTensor input but get "
                            << input_args[index]->GetType()->ToString();
  }

  auto idx_shape_ptr = input_args[index]->GetShape();
  MS_EXCEPTION_IF_NULL(idx_shape_ptr);
  abstract::BaseShapePtrList idx_shape{};
  size_t idx_size;
  if (is_tuple_x) {
    auto shape_tuple = idx_shape_ptr->cast<abstract::TupleShapePtr>();
    idx_shape = shape_tuple->shape();
    idx_size = shape_tuple->size();
  } else {
    auto shape_list = idx_shape_ptr->cast<abstract::ListShapePtr>();
    idx_shape = shape_list->shape();
    idx_size = shape_list->size();
  }
  return std::make_pair(idx_shape, idx_size);
}

TypePtrList GetSequnceIndexType(const PrimitivePtr &primitive, size_t index,
                                const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  bool is_tuple_x = CheckAndConvertUtils::IsTuple(input_args[index]);
  bool is_list_x = CheckAndConvertUtils::IsList(input_args[index]);

  if ((!is_tuple_x) && (!is_list_x)) {
    MS_EXCEPTION(TypeError) << "For [" << op_name << "] should have ListTensor or TupleTensor input, but get "
                            << input_args[index]->GetType()->ToString();
  }

  auto idx_type_ptr = input_args[index]->GetType();
  MS_EXCEPTION_IF_NULL(idx_type_ptr);
  TypePtrList types;
  if (is_tuple_x) {
    auto types_tuple_ptr = idx_type_ptr->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(types_tuple_ptr);
    types = types_tuple_ptr->elements();
  } else {
    auto types_list_ptr = idx_type_ptr->cast<ListPtr>();
    MS_EXCEPTION_IF_NULL(types_list_ptr);
    types = types_list_ptr->elements();
  }
  return types;
}

abstract::ShapePtr BufferAppendInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();

  auto data_shape = GetSequnceIndexShape(primitive, kInputIndex0, input_args).first;
  auto data_size = GetSequnceIndexShape(primitive, kInputIndex0, input_args).second;
  auto exp_shape = GetSequnceIndexShape(primitive, kInputIndex1, input_args).first;
  auto count_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShape())[kShape];

  (void)CheckAndConvertUtils::CheckInteger("exp elements ", SizeToLong(exp_shape.size()), kEqual,
                                           SizeToLong(data_shape.size()), op_name);
  int64_t exp_batch = 1;
  if (data_shape[0]->GetShapeVector().size() == exp_shape[0]->GetShapeVector().size()) {
    exp_batch = exp_shape[0]->GetShapeVector()[0];
    for (size_t i = 0; i < data_size; i++) {
      if (data_shape[0]->GetShapeVector().size() != exp_shape[0]->GetShapeVector().size()) {
        MS_LOG(EXCEPTION) << "For " << op_name << "the dimension of " << i << "th 'exp_shape' must be equal to "
                          << "the dimension of " << i << "th 'data_shape', but got the " << i
                          << "th 'exp_shape': " << exp_shape[i]->GetShapeVector() << ", the " << i
                          << "th 'data_shape': " << data_shape[i]->GetShapeVector();
      }
      if (data_shape[i]->GetShapeVector()[0] < exp_shape[i]->GetShapeVector()[0]) {
        MS_LOG(EXCEPTION) << "For " << op_name << "the first dimension of " << i << "th 'data_shape' "
                          << "must be greater or equal to the dimension of " << i << "th 'exp_shape', "
                          << "but got the " << i << "th 'exp_shape': " << exp_shape[i]->GetShapeVector() << ", the "
                          << i << "th 'data_shape': " << data_shape[i]->GetShapeVector();
      }
    }
  } else {
    for (size_t i = 0; i < data_shape.size(); i++) {
      auto d_shape = data_shape[i]->GetShapeVector();
      std::vector<int64_t> temp_shape(d_shape.begin() + 1, d_shape.end());
      if (temp_shape != exp_shape[i]->GetShapeVector()) {
        MS_LOG(EXCEPTION) << "For " << op_name << ", the " << i << "th 'exp_shape' must be equal to the " << i
                          << "th 'data_shape' which excepts the first dimension. but got the " << i
                          << "th 'exp_shape': " << exp_shape[i]->GetShapeVector() << ", the " << i
                          << "th 'data_shape': " << d_shape;
      }
    }
  }
  (void)primitive->AddAttr("exp_batch", MakeValue(exp_batch));
  return std::make_shared<abstract::Shape>(count_shape);
}

TypePtr BufferAppendInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();

  TypePtrList data_type = GetSequnceIndexType(primitive, kInputIndex0, input_args);
  TypePtrList exp_type = GetSequnceIndexType(primitive, kInputIndex1, input_args);
  auto count_type = input_args[kInputIndex2]->GetType();
  auto head_type = input_args[kInputIndex3]->GetType();
  for (size_t i = 0; i < data_type.size(); i++) {
    auto data_type_ptr = data_type[i]->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(data_type_ptr);
    auto exp_type_ptr = exp_type[i]->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(exp_type_ptr);
    if (data_type_ptr->element()->type_id() != exp_type_ptr->element()->type_id()) {
      MS_LOG(EXCEPTION) << "For " << op_name << ", each tensor in 'exp' must has the same type with 'data',"
                        << " but got 'data_type': " << data_type_ptr->element()->ToString()
                        << ", 'exp_type': " << exp_type_ptr->element()->ToString();
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
