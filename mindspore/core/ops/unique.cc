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
#include "ops/unique.h"

#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "ops/array_ops.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Unique, BaseOperator);

class MIND_API UniqueInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    // inputs: a 1-d Tensor
    auto input = input_args[0];

    auto shape = input->GetShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->GetShapeVector().size() != 1) {
      MS_LOG(EXCEPTION) << "Rank of " << primitive->name() << "'s input must be 1.";
    }

    auto ids_max_shape = shape->Clone();
    auto idx_shape = shape->Clone();
    return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList{ids_max_shape, idx_shape});
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    // inputs: a 1-d Tensor
    const std::string op_name = primitive->name();
    CheckArgsSize(op_name, input_args, 1);
    auto input = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
    return std::make_shared<Tuple>(TypePtrList{input->GetType()->Clone(), input->GetType()->Clone()});
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    // inputs: a 1-d Tensor
    const std::string op_name = primitive->name();
    CheckArgsSize(op_name, input_args, 1);
    abstract::AbstractTensorPtr input = abstract::CheckArg<abstract::AbstractTensor>(op_name, input_args, 0);

    auto shape = input->GetShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->GetShapeVector().size() != 1) {
      MS_LOG(EXCEPTION) << "Rank of " << op_name << "'s input must be 1.";
    }

    auto ids_shape = std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeDimAny});
    auto ids = std::make_shared<abstract::AbstractTensor>(input->element(), ids_shape);
    // Currently we choose the same data type as input for the idx.
    TypePtr ids_idx_type = kInt32;
    MS_EXCEPTION_IF_NULL(input->element());
    MS_EXCEPTION_IF_NULL(input->element()->GetTypeTrack());
    if (input->element()->GetTypeTrack()->type_id() == TypeId::kNumberTypeInt64) {
      ids_idx_type = kInt64;
    }
    auto idx_shape = shape->Clone();
    auto ids_idx = std::make_shared<abstract::AbstractTensor>(ids_idx_type, idx_shape);
    // outputs: ids, ids_idx
    AbstractBasePtrList elements = {ids, ids_idx};
    return std::make_shared<abstract::AbstractTuple>(elements);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Unique, prim::kPrimUnique, UniqueInfer, false);
}  // namespace ops
}  // namespace mindspore
