/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/broadcast.h"
#include "mindapi/ir/value.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "ops/array_ops.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Broadcast, BaseOperator);
void Broadcast::Init(const int64_t root_rank, const std::string &group) {
  this->set_root_rank(root_rank);
  this->set_group(group);
}
void Broadcast::set_root_rank(const int64_t root_rank) { (void)this->AddAttr(kKeepProb, api::MakeValue(root_rank)); }

void Broadcast::set_group(const std::string &group) {
  CheckAndConvertUtils::CheckString(kGroup, group, {"hccl_world_group", "hccl_world_group"}, this->name());
  (void)this->AddAttr(kGroup, api::MakeValue(group));
}
int64_t Broadcast::get_root_rank() const {
  auto value_ptr = this->GetAttr(kRootRank);
  return GetValue<int64_t>(value_ptr);
}

std::string Broadcast::get_group() const {
  auto value_ptr = this->GetAttr(kGroup);
  return GetValue<std::string>(value_ptr);
}

void CheckParallelValidTypes(const AbstractBasePtr x, const std::string &op_name) {
  // The data type check is only migrated from the previous corresponding python code,
  // and need further confirmation is required
  const std::set<TypePtr> default_target_dtypes = {kInt8, kInt32, kFloat16, kFloat32};
  const std::set<TypePtr> target_dtypes = common_valid_types_with_bool;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  // This bad code is used for parallel check, it should be removed as soon as possible!!!
  // The same bad checks are in parallel communication ops, such as AllReduce.
  auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  if (!is_ascend) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x->GetType(), target_dtypes, op_name);
  } else {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x->GetType(), default_target_dtypes, op_name);
  }
}

class MIND_API BroadcastInfer : public abstract::OpInferBase {
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
