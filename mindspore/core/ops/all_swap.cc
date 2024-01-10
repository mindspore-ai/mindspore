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
#include "ops/other_ops.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
class MIND_API AllSwapInfer : public abstract::OpInferBase {
 public:
  // This is used for backend infer by kernel tensor.
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    constexpr auto all_swap_input_size = 3;
    CheckArgsSize(op_name, input_args, all_swap_input_size);
    auto tensor_in_shape = input_args[0]->GetShape()->GetShapeVector();
    // Get the content of the recv size
    auto recv_size_value_ptr = input_args[2]->GetValue();
    auto recv_size_value_opt = GetArrayValue<int64_t>(recv_size_value_ptr);
    auto recv_size_value_array = recv_size_value_opt.value().ToVector();
    int64_t infer_max_size = 0;
    for (size_t i = 0; i < recv_size_value_array.size(); ++i) {
      infer_max_size += recv_size_value_array[i];
    }
    ShapeVector max_shape = {infer_max_size / tensor_in_shape[1], tensor_in_shape[1]};
    return std::make_shared<abstract::TensorShape>(max_shape);
  }

  // This is used for backend infer by kernel tensor.
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    auto tensor_in = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
    (void)CheckAndConvertUtils::CheckArgsType(op_name, input_args, 2, kObjectTypeTensorType);
    MS_EXCEPTION_IF_NULL(tensor_in);
    return tensor_in->GetType()->Clone();
  }

  // This is used for frontend infer by abstract. If MakeAbstract support make env type abstract, InferShapeAndType can
  // be deleted.
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    const std::string op_name = primitive->name();
    constexpr auto all_swap_input_size = 3;
    CheckArgsSize(op_name, input_args, all_swap_input_size);
    auto tensor_in = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
    MS_EXCEPTION_IF_NULL(tensor_in);
    auto tensor_in_shape = tensor_in->GetShape()->GetShapeVector();
    auto send_size = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 1, kObjectTypeTensorType);
    MS_EXCEPTION_IF_NULL(send_size);
    auto recv_size = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 2, kObjectTypeTensorType);
    MS_EXCEPTION_IF_NULL(recv_size);

    // Get the content of the recv size
    auto recv_size_value_ptr = recv_size->BuildValue();
    MS_EXCEPTION_IF_NULL(recv_size_value_ptr);
    auto recv_size_tensor = recv_size_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(recv_size_tensor);
    auto data_pos = static_cast<int64_t *>(recv_size_tensor->data_c());
    MS_EXCEPTION_IF_NULL(data_pos);

    ShapeVector tensor_out_shape = {abstract::Shape::kShapeDimAny, tensor_in_shape[1]};
    return abstract::MakeAbstract(std::make_shared<abstract::TensorShape>(tensor_out_shape), tensor_in->GetType());
  }
};

class MIND_API AllSwap : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AllSwap);
  /// \brief Constructor.
  AllSwap() : BaseOperator("AllSwap") {}
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AllSwap, prim::kPrimAllSwap, AllSwapInfer, false);
}  // namespace ops
}  // namespace mindspore
