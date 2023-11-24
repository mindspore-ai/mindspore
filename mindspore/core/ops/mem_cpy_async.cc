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
#include "ops/framework_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
class MIND_API MemCpyAsyncInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(input_args[0]);
    MS_EXCEPTION_IF_NULL(input_args[0]->GetShape());
    return std::make_shared<abstract::TensorShape>(input_args[1]->GetShape()->GetShapeVector());
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    CheckArgsSize(op_name, input_args, 1);
    auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
    return x->GetType()->Clone();
  }
};

class MIND_API MemCpyAsync : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(MemCpyAsync);
  /// \brief Constructor.
  MemCpyAsync() : BaseOperator("MemCpyAsync") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MemCpyAsync, prim::kPrimMemCpyAsync, MemCpyAsyncInfer, false);
}  // namespace ops
}  // namespace mindspore
