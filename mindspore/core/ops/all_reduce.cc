/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "ops/all_reduce.h"

#include <string>
#include <vector>
#include <set>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/ms_context.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(AllReduce, BaseOperator);
class AllReduceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    return input_args[0]->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t input_num = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto x_type = input_args[0]->BuildType();
    MS_EXCEPTION_IF_NULL(x_type);
    if (!x_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << x_type->ToString()
                              << ".";
    }

    // The data type check is only migrated from the previous corresponding python code,
    // and need further confirmation is required
    const std::set<TypePtr> default_target_dtypes = {kInt8, kInt32, kFloat16, kFloat32};
    const std::set<TypePtr> target_dtypes = common_valid_types_with_bool;
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
    if (!is_ascend) {
      (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, target_dtypes, prim_name);
    } else {
      (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, default_target_dtypes, prim_name);
    }

    return x_type;
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t input_num = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    auto type = InferType(primitive, input_args);
    auto shape = InferShape(primitive, input_args);
    return abstract::MakeAbstract(shape, type);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(AllReduce, prim::kPrimAllReduce, AllReduceInfer, false);
}  // namespace ops
}  // namespace mindspore
