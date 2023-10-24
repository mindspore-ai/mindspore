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

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "ops/op_utils.h"
#include "ops/ops_func_impl/mul.h"
#include "ops/sub.h"
#include "ops/ops_func_impl/real_div.h"
#include "ops/ops_func_impl/add.h"
#include "ops/ops_func_impl/sqrt.h"
#include "ops/nn_optimizer_ops.h"

namespace mindspore {
namespace ops {
namespace {
// Apply ops will have a refractor and add_infer is just a temp modify
auto AddInfer = [](const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                   const AbstractBasePtrList &input_args) {
  auto add_op = AddFuncImpl();
  return abstract::MakeAbstract(add_op.InferShape(primitive, input_args), add_op.InferType(primitive, input_args));
};

auto SqrtInfer = [](const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                    const AbstractBasePtrList &input_args) {
  auto add_op = SqrtFuncImpl();
  return abstract::MakeAbstract(add_op.InferShape(primitive, input_args), add_op.InferType(primitive, input_args));
};

abstract::AbstractBasePtr MulInfer(const PrimitivePtr &primitive, const abstract::AbstractBasePtr &x,
                                   const abstract::AbstractBasePtr &y) {
  auto mul_infer_impl = std::make_shared<ops::MulFuncImpl>();
  return MakeAbstract(mul_infer_impl->InferShape(primitive, {x, y}), mul_infer_impl->InferType(primitive, {x, y}));
}

std::vector<AbstractBasePtr> InferOutputs(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  // An object of a subclass of AbstractBase
  constexpr auto adam_input_size = 10;
  CheckArgsSize(primitive->name(), input_args, adam_input_size);
  auto input0 = input_args[0];
  auto input1 = input_args[1];
  auto input2 = input_args[2];
  auto input3 = input_args[3];
  auto input4 = input_args[4];
  auto mul0_x = input_args[5];
  auto mul1_x = input_args[6];
  auto mul2_x = input_args[7];
  auto mul3_x = input_args[8];
  auto add2_y = input_args[9];

  auto square0 = abstract::InferAbstractByFuncImpl(primitive, {input0});
  auto mul1 = ops::MulInfer(primitive, mul1_x, input0);
  auto mul0 = ops::MulInfer(primitive, mul0_x, input2);
  auto mul2 = ops::MulInfer(primitive, mul2_x, input1);
  auto mul3 = ops::MulInfer(primitive, mul3_x, square0.value());
  auto add0 = ops::AddInfer(nullptr, primitive, {mul0, mul1});
  auto add1 = ops::AddInfer(nullptr, primitive, {mul2, mul3});
  auto sqrt0 = SqrtInfer(nullptr, primitive, {add1});
  auto add2 = ops::AddInfer(nullptr, primitive, {add2_y, sqrt0});
  auto infer_impl = std::make_shared<ops::RealDivFuncImpl>();
  auto infer_shape = infer_impl->InferShape(primitive, {add0, add2});
  auto infer_type = infer_impl->InferType(primitive, {add0, add2});
  auto true_div0 = MakeAbstract(infer_shape, infer_type);
  auto mul4 = ops::MulInfer(primitive, input4, true_div0);
  auto sub0 = ops::SubInfer(nullptr, primitive, {input3, mul4});
  return {add1, add0, sub0};
}
}  // namespace

class MIND_API AdamApplyOneInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto outputs_abstract = InferOutputs(primitive, input_args);
    abstract::BaseShapePtrList out_shapes = {outputs_abstract[0]->GetShape(), outputs_abstract[1]->GetShape(),
                                             outputs_abstract[2]->GetShape()};
    return std::make_shared<abstract::TupleShape>(out_shapes);
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto outputs_abstract = InferOutputs(primitive, input_args);
    TypePtrList out_types = {outputs_abstract[0]->GetType(), outputs_abstract[1]->GetType(),
                             outputs_abstract[2]->GetType()};
    return std::make_shared<Tuple>(out_types);
  }
};

class MIND_API AdamApplyOne : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AdamApplyOne);
  /// \brief Constructor.
  AdamApplyOne() : BaseOperator("AdamApplyOne") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AdamApplyOne, prim::kPrimAdamApplyOne, AdamApplyOneInfer, false);
}  // namespace ops
}  // namespace mindspore
