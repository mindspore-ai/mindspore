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
 * distributed under the License is distributed on an "ASF IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/baddbmm.h"

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>

#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
// batchmatmul
namespace {
constexpr size_t kMatSize = 3;

abstract::ShapePtr BaddbmmInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() != 5) {
    MS_LOG(EXCEPTION) << "input args size should be 5, but got " << input_args.size();
  }
  auto prim_name = primitive->name();
  auto input_shape_ptr = input_args[0]->BuildShape()->cast<abstract::ShapePtr>();
  auto batch1_shape_ptr = input_args[1]->BuildShape()->cast<abstract::ShapePtr>();
  auto batch2_shape_ptr = input_args[2]->BuildShape()->cast<abstract::ShapePtr>();

  MS_EXCEPTION_IF_NULL(input_shape_ptr);
  MS_EXCEPTION_IF_NULL(batch1_shape_ptr);
  MS_EXCEPTION_IF_NULL(batch2_shape_ptr);

  auto input_shape = input_shape_ptr->shape();
  auto batch1_shape = batch1_shape_ptr->shape();
  auto batch2_shape = batch2_shape_ptr->shape();
  if (batch1_shape.size() != kMatSize) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'batch1' must be a 3D Tensor, but got:" << batch1_shape.size();
  }

  if (batch2_shape.size() != kMatSize) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'batch2' must be a 3D Tensor, but got:" << batch2_shape.size();
  }

  if (batch1_shape[2] != batch2_shape[1]) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', first dimension of 'batch2' must be equal to 'batch1' "
                      << batch1_shape[2] << " , but got:" << batch2_shape[1];
  }

  ShapeVector ret_shape{batch1_shape[0], batch1_shape[1], batch2_shape[2]};
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr BaddbmmInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  TypePtr input_type = input_args[0]->BuildType();
  TypePtr batch1_type = input_args[1]->BuildType();
  TypePtr batch2_type = input_args[2]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  MS_EXCEPTION_IF_NULL(batch1_type);
  MS_EXCEPTION_IF_NULL(batch2_type);

  const std::set<TypePtr> valid_types = {kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", batch1_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", batch2_type, valid_types, prim_name);
  return input_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Baddbmm, BaseOperator);

AbstractBasePtr BaddbmmInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = BaddbmmInferType(primitive, input_args);
  auto infer_shape = BaddbmmInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGBaddbmmInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BaddbmmInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BaddbmmInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BaddbmmInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Baddbmm, prim::kPrimBaddbmm, AGBaddbmmInfer, false);
}  // namespace ops
}  // namespace mindspore
