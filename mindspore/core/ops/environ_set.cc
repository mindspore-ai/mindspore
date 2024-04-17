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
class MIND_API EnvironSetInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) const override {
    return abstract::kNoShape;
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return std::make_shared<EnvType>();
  }

  // This is used for frontend infer by abstract. If MakeAbstract support make env type abstract, InferShapeAndType can
  // be deleted.
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    // args: Three objects of a subclass of AbstractBase, env, key, value.
    CheckArgsSize(primitive->name(), input_args, kSize3);

    auto key = input_args[kIndex1];
    ValuePtr key_value_ptr = key->GetValueTrack();
    MS_EXCEPTION_IF_NULL(key_value_ptr);
    auto key_value_track = key_value_ptr->cast<SymbolicKeyInstancePtr>();
    if (key_value_track == nullptr) {
      MS_LOG(EXCEPTION) << "EnvironSet evaluator args[1] expected should be able to cast to SymbolicKeyInstancePtrbut: "
                        << key_value_ptr->ToString();
    }
    auto expected = key_value_track->abstract();
    MS_EXCEPTION_IF_NULL(expected);

    auto value = input_args[kIndex2];
    MS_LOG(DEBUG) << "key: " << key->ToString() << ", value: " << value->ToString();
    if (value->isa<abstract::AbstractUndetermined>() && !value->isa<abstract::AbstractTensor>()) {
      abstract::EnvSetSparseResultMgr::GetInstance().Set(true);
    }
    return std::make_shared<abstract::AbstractScalar>(kValueAny, std::make_shared<EnvType>());
  }
};

class MIND_API EnvironSet : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(EnvironSet);
  /// \brief Constructor.
  EnvironSet() : BaseOperator("EnvironGet") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(EnvironSet, prim::kPrimEnvironSet, EnvironSetInfer, false);
}  // namespace ops
}  // namespace mindspore
