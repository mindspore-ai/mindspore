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
#include "ops/array_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
class MIND_API PadAndShiftInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    // inputs: a 1-d Tensor
    const std::string op_name = primitive->name();
    auto input = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
    MS_EXCEPTION_IF_NULL(input);
    auto shape = input->GetShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->GetShapeVector().size() != 1) {
      MS_LOG(EXCEPTION) << "Rank of " << op_name << "'s input must be 1.";
    }
    // max shape equals to input0's shape.
    ShapeVector max_shape = shape->GetShapeVector();
    return std::make_shared<abstract::TensorShape>(max_shape);
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    const size_t size_expected = 3;
    CheckArgsSize(op_name, input_args, size_expected);
    auto input = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
    return input->GetType()->Clone();
  }

  // This is used for frontend infer by abstract. If MakeAbstract support make env type abstract, InferShapeAndType can
  // be deleted.
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    const size_t size_expected = 3;
    CheckArgsSize(op_name, input_args, size_expected);
    auto input = abstract::CheckArg<abstract::AbstractTensor>(op_name, input_args, 0);
    MS_EXCEPTION_IF_NULL(input);
    auto shape = input->shape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->shape().size() != 1) {
      MS_LOG(EXCEPTION) << "Rank of " << op_name << "'s input must be 1.";
    }
    ShapeVector ids_shape = {abstract::TensorShape::kShapeDimAny};
    return std::make_shared<abstract::AbstractTensor>(input->element(),
                                                      std::make_shared<abstract::TensorShape>(ids_shape));
  }
};

class MIND_API PadAndShift : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(PadAndShift);
  /// \brief Constructor.
  PadAndShift() : BaseOperator("PadAndShift") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PadAndShift, prim::kPrimPadAndShift, PadAndShiftInfer, false);
}  // namespace ops
}  // namespace mindspore
