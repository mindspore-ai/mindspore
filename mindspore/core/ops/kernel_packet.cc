/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "ops/kernel_packet.h"

#include <algorithm>
#include <string>
#include <vector>

#include "ops/op_utils.h"
#include "include/common/utils/utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/base_operator.h"
#include "mindapi/src/helper.h"
#include "symbolic_shape/symbol_engine.h"

namespace mindspore::ops {
class MIND_API KernelPacketInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    auto fg = GetValue<FuncGraphPtr>(primitive->GetAttr(kAttrFuncGraph));
    auto output = fg->output();
    auto shape_mng = fg->symbol_engine();
    if (!shape_mng->Infer(input_args)) {
      return nullptr;
    }
    return shape_mng->QueryShape(output);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto fg = GetValue<FuncGraphPtr>(primitive->GetAttr(kAttrFuncGraph));
    auto output = fg->output();
    auto out_abs = output->abstract();
    if (out_abs == nullptr) {
      return nullptr;
    }
    return out_abs->BuildType();
  }
};
MIND_API_OPERATOR_IMPL(KernelPacket, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(KernelPacket, prim::kPrimKernelPacket, KernelPacketInfer, false);

}  // namespace mindspore::ops
