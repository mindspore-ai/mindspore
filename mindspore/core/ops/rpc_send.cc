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

#include "ops/rpc_send.h"

#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/framework_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(RpcSend, BaseOperator);
class MIND_API AGRpcSendInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const override {
    if (input_args.size() == static_cast<size_t>(kDim1)) {
      return input_args[kInputIndex0]->GetShape();
    } else {
      std::vector<abstract::BaseShapePtr> input_shapes;
      std::transform(input_args.cbegin(), input_args.cend(), std::back_inserter(input_shapes),
                     [](const AbstractBasePtr &input) { return input->GetShape(); });
      return std::make_shared<abstract::TupleShape>(input_shapes);
    }
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    if (input_args.empty()) {
      MS_LOG(EXCEPTION) << "The input size of RpcRecv is 0.";
    }
    if (input_args.size() == static_cast<size_t>(kDim1)) {
      return input_args[kInputIndex0]->GetType();
    } else {
      std::vector<TypePtr> input_types;
      std::transform(input_args.cbegin(), input_args.cend(), std::back_inserter(input_types),
                     [](const AbstractBasePtr &input) { return input->GetType(); });
      return std::make_shared<Tuple>(input_types);
    }
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(RpcSend, prim::kPrimRpcSend, AGRpcSendInfer, false);
}  // namespace ops
}  // namespace mindspore
