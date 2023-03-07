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

#include "ops/rpc_recv.h"

#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(RpcRecv, BaseOperator);
AbstractBasePtr RpcRecvInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &,
                             const std::vector<AbstractBasePtr> &input_args) {
  if (input_args.empty()) {
    MS_LOG(EXCEPTION) << "The input size of RpcRecv is 0.";
  }
  if (input_args.size() == static_cast<size_t>(kDim1)) {
    return input_args[kInputIndex0];
  } else {
    abstract::AbstractTuplePtr rpc_recv_abs = std::make_shared<abstract::AbstractTuple>(input_args);
    MS_EXCEPTION_IF_NULL(rpc_recv_abs);
    return rpc_recv_abs;
  }
}
REGISTER_PRIMITIVE_EVAL_IMPL(RpcRecv, prim::kPrimRpcRecv, RpcRecvInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
