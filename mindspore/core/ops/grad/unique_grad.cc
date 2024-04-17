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

class MIND_API UniqueGradInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    // inputs: a 1-d Tensor
    const std::string op_name = primitive->name();
    const size_t size_expected = 2;
    CheckArgsSize(op_name, input_args, size_expected);
    auto dout = input_args[0];
    auto dout_shape = dout->GetShape();
    auto dout_tuple_shape = dout_shape->cast<abstract::TupleShapePtr>();
    auto ids_shape = (*dout_tuple_shape)[0];
    auto ids_idx_shape = (*dout_tuple_shape)[1];
    MS_EXCEPTION_IF_NULL(ids_shape);
    MS_EXCEPTION_IF_NULL(ids_idx_shape);
    if (ids_shape->GetShapeVector().size() != 1) {
      MS_LOG(EXCEPTION) << "Dims of dout[0] of " << op_name << "' input must be 1.";
    }
    if (ids_idx_shape->GetShapeVector().size() != 1) {
      MS_LOG(EXCEPTION) << "Dims of dout[1] of " << op_name << "' input must be 1.";
    }

    // outputs: dx
    return ids_idx_shape->Clone();
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    const size_t size_expected = 2;
    CheckArgsSize(op_name, input_args, size_expected);

    auto dout = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTuple);
    auto dout_type = dout->GetType();
    auto dout_tuple_type = dout_type->cast<TuplePtr>();
    auto ids_type = (*dout_tuple_type)[0];
    return ids_type->Clone();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    const size_t size_expected = 2;
    CheckArgsSize(op_name, input_args, size_expected);
    auto dout = abstract::CheckArg<abstract::AbstractTuple>(op_name, input_args, 0);
    CheckArgsSize(op_name + " dout", dout->elements(), size_expected);
    auto ids = abstract::CheckArg<abstract::AbstractTensor>(op_name, dout->elements(), 0);
    auto ids_idx = abstract::CheckArg<abstract::AbstractTensor>(op_name, dout->elements(), 1);
    auto ids_shape = ids->shape();
    auto ids_idx_shape = ids_idx->shape();
    MS_EXCEPTION_IF_NULL(ids_shape);
    MS_EXCEPTION_IF_NULL(ids_idx_shape);
    if (ids->shape()->shape().size() != 1) {
      MS_LOG(EXCEPTION) << "Dims of dout[0] of " << op_name << "' input must be 1.";
    }
    if (ids_idx->shape()->shape().size() != 1) {
      MS_LOG(EXCEPTION) << "Dims of dout[1] of " << op_name << "' input must be 1.";
    }

    // outputs: dx
    return std::make_shared<abstract::AbstractTensor>(ids->element(), ids_idx->shape());
  }
};

class MIND_API UniqueGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(UniqueGrad);
  /// \brief Constructor.
  UniqueGrad() : BaseOperator("UniqueGrad") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(UniqueGrad, prim::kPrimUniqueGrad, UniqueGradInfer, false);
}  // namespace ops
}  // namespace mindspore
