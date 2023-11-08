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

#include "ops/paged_attention_mask.h"
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PagedAttentionMaskInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kPagedAttentionMaskInputsNum, primitive->name());

  auto query_shape_ptr = input_args[kPagedAttentionMaskInputQueryIndex]->BuildShape();
  auto shape_element = query_shape_ptr->cast<abstract::ShapePtr>();
  return shape_element;
}

TypePtr PagedAttentionMaskInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set valid_types = {kFloat16};
  auto op_name = prim->name();
  std::map<std::string, TypePtr> types;

  (void)types.emplace("query", input_args[kPagedAttentionMaskInputQueryIndex]->BuildType());
  (void)types.emplace("key_cache", input_args[kPagedAttentionMaskInputKeyCacheIndex]->BuildType());
  (void)types.emplace("value_cache", input_args[kPagedAttentionMaskInputValueCacheIndex]->BuildType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);

  auto block_tables_type = input_args[kPagedAttentionMaskInputBlockTablesIndex]->BuildType();
  CheckAndConvertUtils::CheckTensorTypeValid("block_tables", block_tables_type, {kInt32}, op_name);
  auto context_lens_type = input_args[kPagedAttentionMaskInputContextLensIndex]->BuildType();
  CheckAndConvertUtils::CheckTensorTypeValid("context_lens", context_lens_type, {kInt32}, op_name);

  return type;  // attention_out dtype
}
}  // namespace

AbstractBasePtr PagedAttentionMaskInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kPagedAttentionMaskInputsNum, primitive->name());
  auto infer_type = PagedAttentionMaskInferType(primitive, input_args);
  auto infer_shape = PagedAttentionMaskInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(PagedAttentionMask, BaseOperator);

// AG means auto generated
class MIND_API AGPagedAttentionMaskInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PagedAttentionMaskInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PagedAttentionMaskInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PagedAttentionMaskInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PagedAttentionMask, prim::kPrimPagedAttentionMask, AGPagedAttentionMaskInfer, false);
}  // namespace ops
}  // namespace mindspore
