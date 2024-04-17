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
#include "ops/nn_optimizer_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
class MIND_API DynamicAssignInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    // Inputs: a tensor
    auto x = input_args[0];
    auto y = input_args[1];
    MS_EXCEPTION_IF_NULL(x);
    MS_EXCEPTION_IF_NULL(y);
    auto type = input_args[0]->GetType();
    MS_EXCEPTION_IF_NULL(type);
    if (type->type_id() == kObjectTypeRefKey) {
      return y->GetShape();
    } else {
      auto y_shape = y->GetShape()->cast<abstract::TensorShapePtr>();
      MS_EXCEPTION_IF_NULL(y_shape);
      if (!y_shape->max_shape().empty()) {
        x->set_shape(y->GetShape());
      }
      return x->GetShape();
    }
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const size_t size_expected = 2;
    CheckArgsSize(primitive->name(), input_args, size_expected);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto type = input_args[0]->GetType();
    return type;
  }

  // This is used for frontend infer by abstract.
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    // Inputs: a tensor
    const size_t size_expected = 2;
    CheckArgsSize(primitive->name(), input_args, size_expected);

    MS_LOG(INFO) << "InferImplDynamicAssign " << input_args[0];
    auto type = input_args[0]->GetType();
    MS_EXCEPTION_IF_NULL(type);
    if (type->type_id() == kObjectTypeRefKey) {
      return input_args[1]->Broaden();
    } else {
      auto x = abstract::CheckArg<abstract::AbstractTensor>(primitive->name(), input_args, 0);
      auto y = abstract::CheckArg<abstract::AbstractTensor>(primitive->name(), input_args, 1);
      MS_EXCEPTION_IF_NULL(x);
      MS_EXCEPTION_IF_NULL(y);
      auto y_shape = y->shape();
      MS_EXCEPTION_IF_NULL(y_shape);
      if (!y_shape->max_shape().empty()) {
        x->set_shape(y->shape());
      }
      return input_args[0];
    }
  }
};

class MIND_API DynamicAssign : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(DynamicAssign);
  /// \brief Constructor.
  DynamicAssign() : BaseOperator("DynamicAssign") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(DynamicAssign, prim::kPrimDynamicAssign, DynamicAssignInfer, false);
}  // namespace ops
}  // namespace mindspore
