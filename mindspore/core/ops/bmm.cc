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
#include "ops/bmm.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr size_t kMatSize = 3;
MIND_API_OPERATOR_IMPL(Bmm, BaseOperator);
class BmmInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    if (input_args.size() != 2) {
      MS_LOG(EXCEPTION) << "input args size should be 2, but got " << input_args.size();
    }
    auto prim_name = primitive->name();
    auto input_shape_ptr = input_args[0]->BuildShape()->cast<abstract::ShapePtr>();
    auto mat2_shape_ptr = input_args[1]->BuildShape()->cast<abstract::ShapePtr>();

    MS_EXCEPTION_IF_NULL(input_shape_ptr);
    MS_EXCEPTION_IF_NULL(mat2_shape_ptr);

    auto input_shape = input_shape_ptr->shape();
    auto mat2_shape = mat2_shape_ptr->shape();
    if (input_shape.size() != kMatSize) {
      MS_LOG(EXCEPTION) << "For '" << prim_name
                        << "', input 'input' must be a 3D Tensor, but got:" << input_shape.size();
    }

    if (mat2_shape.size() != kMatSize) {
      MS_LOG(EXCEPTION) << "For '" << prim_name << "', input 'mat2' must be a 3D Tensor, but got:" << mat2_shape.size();
    }

    if (input_shape[0] != mat2_shape[0]) {
      MS_LOG(EXCEPTION) << "For '" << prim_name << "', first dimension of 'mat2' must be equal to 'input' "
                        << input_shape[0] << " , but got:" << mat2_shape[0];
    }

    if (input_shape[2] != mat2_shape[1]) {
      MS_LOG(EXCEPTION) << "For '" << prim_name << "', first dimension of 'batch2' must be equal to 'batch1' "
                        << input_shape[2] << " , but got:" << mat2_shape[1];
    }

    ShapeVector ret_shape{input_shape[0], input_shape[1], mat2_shape[2]};
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    const std::set<TypePtr> valid_types = {kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,    kUInt32,
                                           kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
    std::map<std::string, TypePtr> types;
    (void)types.emplace("x", input_args[0]->BuildType());
    (void)types.emplace("w", input_args[1]->BuildType());
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
    TypePtr x_type = input_args[0]->BuildType();
    return x_type;
  }
};
abstract::AbstractBasePtr BmmInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  BmmInfer bmm_infer;
  auto type = bmm_infer.InferType(primitive, input_args);
  auto shape = bmm_infer.InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_OP_INFER_IMPL(Bmm, prim::kPrimBmm, BmmInfer, false);
}  // namespace ops
}  // namespace mindspore
