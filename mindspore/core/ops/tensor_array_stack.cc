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
#include "utils//symbolic.h"
#include "ops/other_ops.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
class MIND_API TensorArrayStackInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &) const override {
    // Infer TensorArrayStack
    const std::string op_name = primitive->name();
    auto attr_shape = primitive->GetAttr("element_shape");
    if (attr_shape == nullptr) {
      MS_LOG(EXCEPTION) << "No attribute [element_shape] in " << op_name;
    }
    auto attr_dtype = primitive->GetAttr("dtype");
    if (attr_dtype == nullptr) {
      MS_LOG(EXCEPTION) << "No attribute [dtype] in " << op_name;
    }
    auto attr_is_dynamic = primitive->GetAttr("is_dynamic_shape");
    if (attr_is_dynamic == nullptr) {
      MS_LOG(EXCEPTION) << "No attribute [is_dynamic_shape] in " << op_name;
    }
    auto attr_size = primitive->GetAttr("size");
    if (attr_size == nullptr) {
      MS_LOG(EXCEPTION) << "No attribute [size] in " << op_name;
    }
    auto size = GetValue<int64_t>(attr_size);
    auto ele_shape = GetValue<std::vector<int64_t>>(attr_shape);
    constexpr int64_t kMaxElement = 10000;
    primitive->set_attr("max_element", MakeValue(kMaxElement));
    std::shared_ptr<mindspore::abstract::AbstractTensor> output;

    auto is_dynamic = GetValue<bool>(attr_is_dynamic);
    if (is_dynamic) {
      (void)ele_shape.insert(ele_shape.cbegin(), -1);
    } else {
      (void)ele_shape.insert(ele_shape.cbegin(), size);
    }
    return std::make_shared<abstract::TensorShape>(ele_shape);
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &) const override {
    auto attr_dtype = primitive->GetAttr("dtype");
    auto type = GetValue<TypePtr>(attr_dtype);
    return type;
  }
};

class MIND_API TensorArrayStack : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(TensorArrayStack);
  /// \brief Constructor.
  TensorArrayStack() : BaseOperator("TensorArrayStack") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(TensorArrayStack, prim::kPrimTensorArrayStack, TensorArrayStackInfer, false);
}  // namespace ops
}  // namespace mindspore
