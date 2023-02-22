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

#include "ops/log_space.h"

#include <map>
#include <memory>
#include <set>

#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LogSpaceInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto start_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto start_shape = start_shape_map[kShape];
  if (!IsDynamicRank(start_shape) && start_shape.size() != 0) {
    MS_EXCEPTION(ValueError) << "For LogSpace, The dim of start must be 0, "
                             << "but got " << start_shape.size();
  }
  auto end_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto end_shape = end_shape_map[kShape];
  if (!IsDynamicRank(end_shape) && end_shape.size() != 0) {
    MS_EXCEPTION(ValueError) << "For LogSpace, The dim of end must be 0, "
                             << "but got " << end_shape.size();
  }
  int64_t shape_value = GetValue<int64_t>(primitive->GetAttr("steps"));
  std::vector<int64_t> state_shape = {shape_value};
  return std::make_shared<abstract::Shape>(state_shape);
}

TypePtr LogSpaceInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};

  std::map<std::string, TypePtr> types;
  (void)types.emplace("start", input_args[0]->BuildType());
  (void)types.emplace("end", input_args[1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  auto dtype_attr = prim->GetAttr("dtype");
  MS_EXCEPTION_IF_NULL(dtype_attr);
  auto infer_type = dtype_attr->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(infer_type);
  return infer_type;
}
}  // namespace
void LogSpace::Init(int64_t steps, int64_t base) {
  set_steps(steps);
  set_base(base);
}

void LogSpace::set_steps(int64_t steps) { (void)this->AddAttr(kSteps, api::MakeValue(steps)); }
void LogSpace::set_base(int64_t base) { (void)this->AddAttr(kBase, api::MakeValue(base)); }

int64_t LogSpace::get_steps() const { return GetValue<int64_t>(GetAttr(kSteps)); }

int64_t LogSpace::get_base() const { return GetValue<int64_t>(GetAttr(kBase)); }

AbstractBasePtr LogSpaceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string op_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, op_name);
  auto infer_type = LogSpaceInferType(primitive, input_args);
  auto infer_shape = LogSpaceInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(LogSpace, BaseOperator);

// AG means auto generated
class MIND_API AGLogSpaceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LogSpaceInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LogSpaceInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LogSpaceInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LogSpace, prim::kPrimLogSpace, AGLogSpaceInfer, false);
}  // namespace ops
}  // namespace mindspore
