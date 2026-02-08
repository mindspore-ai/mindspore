/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "infer/ops_func_impl/communication/reduce_scatter_v.h"
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
#include "mindapi/helper.h"
#include "primitive/other_ops.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
class ReduceScatterVInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const auto prim_name = primitive->name();
    BaseShapePtr shape;
    std::vector<int64_t> input_split_sizes;
    auto rank_size_ptr = primitive->GetAttr(kRankSize);
    auto rank_size = GetValue<int64_t>(rank_size_ptr);
    auto rank_id_ptr = primitive->GetAttr(kRankId);
    auto rank_id = GetValue<int64_t>(rank_id_ptr);
    MS_LOG(DEBUG) << "For '" << prim_name << "', input rank_id : " << rank_id << ".";
    if (input_args.size() == kInputNum2) {
      (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputNum2,
                                               prim_name);
      auto value = GetArrayValue<int64_t>(input_args[kIndex1]);
      if (!value.has_value()) {
        return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeDimAny});
      }
      if (value.value().HasUnknownValue()) {
        MS_EXCEPTION(ValueError)
          << "For primitive[" << prim_name
          << "], there are unknown values in input2, please handle this case before calling this function.";
      }
      input_split_sizes = value.value().ToVector();
    } else {
      MS_LOG(EXCEPTION) << "ReduceScatterV input numbers must be 2.";
    }
    (void)CheckAndConvertUtils::CheckInteger("input_split_sizes size", static_cast<int64_t>(input_split_sizes.size()),
                                             kEqual, rank_size, prim_name);
    int64_t output_numel = 0;
    for (size_t i = 0; i < input_split_sizes.size(); i++) {
      if (input_split_sizes[i] < 0) {
        MS_LOG(EXCEPTION) << "input_split_sizes value is illegal.";
      }
      if (rank_id == static_cast<int64_t>(i)) {
        output_numel = input_split_sizes[i];
        break;
      }
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
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto x_type = input_args[0]->GetType();

    MS_EXCEPTION_IF_NULL(x_type);
    if (!x_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input0 must be a Tensor, but got: " << x_type->ToString()
                              << ".";
    }
    // flag to check different valid types on ascend
    auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);

    const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kFloat16, kFloat32, kBFloat16};
    if (!is_ascend) {
      (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, common_valid_types_with_bool, prim_name);
    } else {
      (void)CheckAndConvertUtils::CheckTypeValid("x", x_type, valid_types, prim_name);
    }
    return x_type->Clone();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    const auto prim_name = primitive->name();
    if (input_args.size() == kInputNum2) {
      (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputNum2,
                                               prim_name);
    } else {
      MS_LOG(EXCEPTION) << "ReduceScatterV input numbers must be 2.";
    }
    auto type = InferType(primitive, input_args);
    auto shape = InferShape(primitive, input_args);
    return abstract::MakeAbstract(shape, type);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};

MIND_API_OPERATOR_IMPL(ReduceScatterV, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReduceScatterV, prim::kPrimReduceScatterV, ReduceScatterVInfer, false);
}  // namespace ops
}  // namespace mindspore
