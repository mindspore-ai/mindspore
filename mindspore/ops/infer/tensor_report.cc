/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "infer/tensor_report.h"

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "primitive/structure_ops.h"
#include "primitive/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace ops {

MIND_API_OPERATOR_IMPL(TensorReport, BaseOperator);
void TensorReport::set_side_effect_io() { (void)this->AddAttr(kSideEffectIO, api::MakeValue(true)); }

bool TensorReport::get_side_effect_io() const {
  auto value_ptr = GetAttr(kSideEffectIO);
  return GetValue<bool>(value_ptr);
}

void TensorReport::Init() { this->set_side_effect_io(); }

class OPS_API TensorReportInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    primitive->AddAttr("dyn_input_sizes", MakeValue(std::vector<int64_t>{-1, 1}));
    return std::make_shared<abstract::Shape>(ShapeVector(1));
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const auto &prim_name = primitive->name();
    const size_t input_num = 2;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                             prim_name);
    const auto name = input_args[kIndex0];
    const auto inp_x = input_args[kIndex1];
    MS_EXCEPTION_IF_NULL(name);
    MS_EXCEPTION_IF_NULL(inp_x);
    (void)CheckAndConvertUtils::CheckTypeValid("name", name->BuildType(), {kString}, primitive->name());
    (void)CheckAndConvertUtils::CheckTypeValid("inp_x", inp_x->BuildType(), {kTensorType}, primitive->name());
    return std::make_shared<TensorType>(kInt32);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    auto shape = InferShape(primitive, input_args);
    auto type = InferType(primitive, input_args);
    return abstract::MakeAbstract(shape, type);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(TensorReport, prim::kPrimTensorReport, TensorReportInfer, false);
}  // namespace ops
}  // namespace mindspore
