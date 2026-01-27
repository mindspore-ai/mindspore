/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/communication/broadcast.h"
#include <set>
#include <string>
#include <vector>
#include "mindapi/ir/value.h"
#include "primitive/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/helper.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/array_ops.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Broadcast, BaseOperator);
void CheckParallelValidTypes(const AbstractBasePtr x, const std::string &op_name) {
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x->GetType(), comm_ops_valid_types_with_bool, op_name);
}

class OPS_API BroadcastInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    CheckArgsSize(op_name, input_args, 1);
    auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTuple);
    MS_EXCEPTION_IF_NULL(x);
    MS_EXCEPTION_IF_NULL(x->GetShape());
    return x->GetShape()->Clone();
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    CheckArgsSize(op_name, input_args, 1);
    auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTuple);
    MS_EXCEPTION_IF_NULL(x);
    if (!CheckAndConvertUtils::IsTuple(x)) {
      MS_EXCEPTION(TypeError) << "For Broadcast, the 'input_x' must be a tuple, but got " << x->GetType()->ToString();
    }
    if (x->isa<abstract::AbstractTuple>()) {
      auto tuple_abs = x->cast<abstract::AbstractTuplePtr>();
      for (const auto &ele : tuple_abs->elements()) {
        CheckParallelValidTypes(ele, op_name);
      }
    }
    return x->GetType()->Clone();
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Broadcast, prim::kPrimBroadcast, BroadcastInfer, false);
}  // namespace ops
}  // namespace mindspore
