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
#include "ir/anf.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "ops/framework_ops.h"
#include "ops/base_operator.h"
#include "utils/symbolic.h"

namespace mindspore {
namespace ops {
class MIND_API EnvironGetInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const override {
    auto default_value = input_args[kIndex2];
    return default_value->GetShape()->Clone();
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    // args: Three objects of a subclass of AbstractBase, env, key, default_value(default).
    CheckArgsSize(primitive->name(), input_args, kSize3);
    auto default_value = input_args[kIndex2];
    return default_value->GetType()->Clone();
  }

  // This is used for frontend infer by abstract. If MakeAbstract support make env type abstract, InferShapeAndType can
  // be deleted.
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    // args: Three objects of a subclass of AbstractBase, env, key, default_value(default).
    CheckArgsSize(primitive->name(), input_args, kSize3);
    auto key = input_args[kIndex1];
    auto default_value = input_args[kIndex2];
    TypePtr type = key->GetTypeTrack();
    MS_EXCEPTION_IF_NULL(type);
    if (type->type_id() != kObjectTypeSymbolicKeyType) {
      MS_LOG(EXCEPTION) << "EnvironGet evaluator args[1] should be a SymbolicKeyInstance but: " << key->ToString();
    }

    MS_LOG(DEBUG) << "key: " << key->ToString() << ", value: " << default_value->ToString();
    if (default_value->isa<abstract::AbstractTensor>() && abstract::EnvSetSparseResultMgr::GetInstance().Get()) {
      auto tensor_value = default_value->cast<abstract::AbstractTensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor_value);
      return std::make_shared<abstract::AbstractUndetermined>(tensor_value->element()->Clone(),
                                                              tensor_value->shape()->Clone());
    }

    if (!key->GetValueTrack()->isa<SymbolicKeyInstance>()) {
      return default_value;
    }
    ValuePtr key_value_ptr = key->GetValueTrack();
    MS_EXCEPTION_IF_NULL(key_value_ptr);
    auto key_value_track = key_value_ptr->cast<SymbolicKeyInstancePtr>();
    auto expected = key_value_track->abstract();
    MS_EXCEPTION_IF_NULL(expected);
    (void)expected->Join(default_value);
    // If expected is AbstractRef, return it's AbstractTensor as Value type other than Reference type.
    if (expected->isa<abstract::AbstractRefTensor>()) {
      const auto &abs_ref = expected->cast<abstract::AbstractRefPtr>();
      MS_EXCEPTION_IF_NULL(abs_ref);
      return abs_ref->CloneAsTensor();
    }
    return expected;
  }
};

class MIND_API EnvironGet : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(EnvironGet);
  /// \brief Constructor.
  EnvironGet() : BaseOperator("EnvironGet") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(EnvironGet, prim::kPrimEnvironGet, EnvironGetInfer, false);
}  // namespace ops
}  // namespace mindspore
