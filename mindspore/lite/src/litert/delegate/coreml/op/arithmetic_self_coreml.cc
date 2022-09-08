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

#include "src/litert/delegate/coreml/op/arithmetic_self_coreml.h"
namespace mindspore::lite {
int ArithmeticSelfCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr);
  auto unary_param = op_->mutable_unary();
  switch (type_) {
    case schema::PrimitiveType_ExpFusion: {
      unary_param->set_type(CoreML::Specification::UnaryFunctionLayerParams_Operation_EXP);
      break;
    }
    case schema::PrimitiveType_Sqrt: {
      unary_param->set_type(CoreML::Specification::UnaryFunctionLayerParams_Operation_SQRT);
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported arithmetic_self type.";
      return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
