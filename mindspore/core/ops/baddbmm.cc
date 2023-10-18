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

abstract::ShapePtr BaddBmmInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
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

  if (batch1_shape[0] != batch2_shape[1]) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', first dimension of 'batch2' must be equal to 'batch1' "
                      << batch1_shape[0] << " , but got:" << batch2_shape[0];
  }

  ShapeVector ret_shape{batch1_shape[0], batch1_shape[1], batch2_shape[2]};
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr BaddBmmInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  TypePtr input_type = input_args[0]->BuildType();
  TypePtr batch1_type = input_args[1]->BuildType();
  TypePtr batch2_type = input_args[2]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  MS_EXCEPTION_IF_NULL(batch1_type);
  MS_EXCEPTION_IF_NULL(batch2_type);

  if (input_type->type_id() != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', input type must be float32, but got:" << input_type->ToString();
  }

  if (batch1_type->type_id() != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', batch1 type must be float32, but got:" << batch1_type->ToString();
  }

  if (batch2_type->type_id() != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', batch2 type must be float32, but got:" << batch2_type->ToString();
  }
  return input_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(BaddBmm, BaseOperator);

AbstractBasePtr BaddBmmInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = BaddBmmInferType(primitive, input_args);
  auto infer_shape = BaddBmmInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGBaddBmmInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return BaddBmmInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return BaddBmmInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return BaddBmmInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BaddBmm, prim::kPrimBaddBmm, AGBaddBmmInfer, false);
}  // namespace ops
}  // namespace mindspore
