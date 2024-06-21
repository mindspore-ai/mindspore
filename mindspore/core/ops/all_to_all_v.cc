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
#include "ops/all_to_all_v.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/other_ops.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(AlltoAllV, BaseOperator);
class AlltoAllVInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    if (!(input_args[0]->isa<abstract::AbstractTensor>())) {
      MS_LOG(EXCEPTION) << "AlltoAllV input must be tensor.";
    }

    auto recv_numel_list = GetValue<std::vector<int64_t>>(primitive->GetAttr("recv_numel_list"));
    int64_t output_numel = 0;
    for (size_t i = 0; i < recv_numel_list.size(); i++) {
      if (recv_numel_list[i] < 0) {
        output_numel = abstract::Shape::kShapeDimAny;
        break;
      }
      output_numel += recv_numel_list[i];
    }
    if (output_numel == 0) {
      return std::make_shared<abstract::TensorShape>(ShapeVector{});
    }
    return std::make_shared<abstract::TensorShape>(ShapeVector{output_numel});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    const auto prim_name = prim->name();
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);

    // flag to check different valid types on ascend
    auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);

    MS_EXCEPTION_IF_NULL(input_args[0]);
    if (!(input_args[0]->isa<abstract::AbstractTensor>())) {
      MS_LOG(EXCEPTION) << "AlltoAllV input must be tensor.";
    }

    auto x_type = input_args[0]->GetType();
    if (!is_ascend) {
      (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, common_valid_types, prim_name);
    } else {
      (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, common_valid_types_with_bool, prim_name);
    }
    return x_type->Clone();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);

    auto type = InferType(primitive, input_args);
    auto shape = InferShape(primitive, input_args);
    return abstract::MakeAbstract(shape, type);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(AlltoAllV, prim::kPrimAlltoAllV, AlltoAllVInfer, false);
}  // namespace ops
}  // namespace mindspore
