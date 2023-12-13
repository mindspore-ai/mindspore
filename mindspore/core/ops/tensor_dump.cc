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

#include "ops/tensor_dump.h"

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/structure_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {

namespace {
abstract::ShapePtr TensorDumpInferShape(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  return std::make_shared<abstract::Shape>(ShapeVector(1));
}
}  // namespace

MIND_API_OPERATOR_IMPL(TensorDump, BaseOperator);
void TensorDump::set_side_effect_io() { (void)this->AddAttr(kSideEffectIO, api::MakeValue(true)); }

bool TensorDump::get_side_effect_io() const {
  auto value_ptr = GetAttr(kSideEffectIO);
  return GetValue<bool>(value_ptr);
}

void TensorDump::Init() { this->set_side_effect_io(); }

class MIND_API TensorDumpInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    primitive->AddAttr("dyn_input_sizes", MakeValue(std::vector<int64_t>{-1, 1}));
    return TensorDumpInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const auto &prim_name = primitive->name();
    const size_t input_num = 2;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                             prim_name);
    const auto file = input_args[kIndex0];
    const auto input_x = input_args[kIndex1];
    MS_EXCEPTION_IF_NULL(file);
    MS_EXCEPTION_IF_NULL(input_x);
    (void)CheckAndConvertUtils::CheckTypeValid("file", file->BuildType(), {kString}, primitive->name());
    auto s = GetValue<std::string>(file->BuildValue());
    if (s.empty()) {
      MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the input argument[file]"
                               << " cannot be an empty string.";
    }
    (void)CheckAndConvertUtils::CheckTypeValid("input_x", input_x->BuildType(), {kTensorType}, primitive->name());
    return std::make_shared<TensorType>(kInt32);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(TensorDump, prim::kPrimTensorDump, TensorDumpInfer, false);
}  // namespace ops
}  // namespace mindspore
