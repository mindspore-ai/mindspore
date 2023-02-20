/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <set>
#include <vector>
#include <memory>
#include <map>
#include <string>

#include "ops/orgqr.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr OrgqrInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNoBatch = 2;
  const int64_t kInputWithBatch = 3;
  const size_t kRowIndex = 2;
  const size_t kColIndex = 1;
  const size_t kTwo = 2;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (IsDynamic(x_shape)) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  auto tau_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(x_shape);
  }
  if (x_shape.size() < kInputNoBatch) {
    MS_EXCEPTION(ValueError) << "For Orgqr, the rank of x must be greater than or equal to 2"
                             << ", while got x rank " << x_shape.size() << ".";
  }
  int64_t rank = x_shape.size();
  if (*(x_shape.end() - 1) > *(x_shape.end() - kTwo)) {
    MS_EXCEPTION(ValueError) << "For Orgqr, x.shape[-2] must be greater than or equal to x.shape[-1]"
                             << ", while x.shape[-2] is " << x_shape[rank - kRowIndex] << " and x.shape[-1] is "
                             << x_shape[rank - kColIndex] << ".";
  }
  if (tau_shape.size() < 1) {
    MS_EXCEPTION(ValueError) << "For Orgqr,  tau should have one dimension less than x"
                             << ", but tensor has no dimensions";
  }
  if (*(x_shape.end() - 1) < *(tau_shape.end() - 1)) {
    MS_EXCEPTION(ValueError) << "For Orgqr, x.shape[-1] must be greater than or equal to tau.shape[-1]"
                             << ", while x.shape[-1] is " << x_shape[rank - kColIndex] << " and "
                             << "tau.shape[-1] is " << tau_shape[rank - kColIndex] << ".";
  }
  if ((x_shape.size() - 1) != tau_shape.size()) {
    MS_EXCEPTION(ValueError) << "For Orgqr,  tau should have one dimension less than x"
                             << ", while rank of x is " << x_shape.size() << " and "
                             << "rank of tau is " << tau_shape.size() << ".";
  }
  if (rank >= kInputWithBatch) {
    for (size_t i = 0; i < rank - kRowIndex; i++) {
      if (x_shape[i] != tau_shape[i]) {
        MS_EXCEPTION(ValueError) << "For Orgqr, x and tau should share the same batch size, but x.shape[" << i
                                 << "] is " << x_shape[i] << ",and tau.shape[" << i << "] is " << tau_shape[i] << ".";
      }
    }
  }

  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr OrgqrInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  std::map<std::string, TypePtr> types;
  auto x_type = input_args[0]->BuildType();
  (void)types.emplace("x", x_type);
  (void)types.emplace("tau", input_args[kInputIndex1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Orgqr, BaseOperator);
AbstractBasePtr OrgqrInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = OrgqrInferType(primitive, input_args);
  auto infer_shape = OrgqrInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGOrgqrInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return OrgqrInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return OrgqrInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return OrgqrInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Orgqr, prim::kPrimOrgqr, AGOrgqrInfer, false);
}  // namespace ops
}  // namespace mindspore
