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
#include "infer/ops_func_impl/communication/all_to_all.h"

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
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(AlltoAll, BaseOperator);

class AlltoAllInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto split_count = GetValue<int64_t>(primitive->GetAttr(kAttrSplitCount));
    auto split_dim = GetValue<int64_t>(primitive->GetAttr(kAttrSplitDim));
    auto concat_dim = GetValue<int64_t>(primitive->GetAttr(kAttrConcatDim));
    auto rank_size = GetValue<int64_t>(primitive->GetAttr(kRankSize));
    auto prim_name = primitive->name();
    auto x_shape = input_args[0]->GetShape()->GetShapeVector();
    if (split_count != rank_size) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << ", the 'split_count' must be equal to 'rank_size', but got 'split_count': "
                               << split_count << ", 'rank_size': " << rank_size;
    }
    auto shape_size = SizeToLong(x_shape.size());
    if (concat_dim >= shape_size || concat_dim < -shape_size) {
      MS_EXCEPTION(IndexError) << "Invalid concat dim " << concat_dim << " is greater than shape size " << shape_size;
    }
    if (split_dim >= shape_size || split_dim < -shape_size) {
      MS_EXCEPTION(IndexError) << "Invalid split dim " << split_dim << " is greater than shape size " << shape_size;
    }
    concat_dim = concat_dim < 0 ? (concat_dim + shape_size) : (concat_dim);
    split_dim = split_dim < 0 ? (split_dim + shape_size) : (split_dim);

    if ((x_shape[split_dim] > 0) && (x_shape[split_dim] % split_count != 0)) {
      MS_EXCEPTION(ValueError)
        << "For '" << prim_name
        << ", the 'x_shape[split_dim]' must be divisible by 'split_count', but got 'x_shape[split_dim]' {x_shape["
        << split_dim << "]}, 'split_count' " << split_count << ".";
    }
    if (x_shape[concat_dim] >= 0) {
      x_shape[concat_dim] = x_shape[concat_dim] * split_count;
    }
    if (x_shape[split_dim] >= 0) {
      x_shape[split_dim] = x_shape[split_dim] / split_count;
    }
    return std::make_shared<abstract::Shape>(x_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t input_num = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto x_type = input_args[0]->GetType();
    MS_EXCEPTION_IF_NULL(x_type);
    if (!x_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << x_type->ToString()
                              << ".";
    }

    const std::set<TypePtr> default_target_dtypes = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,   kUInt16,
                                                     kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kBFloat16};
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
REGISTER_PRIMITIVE_OP_INFER_IMPL(AlltoAll, prim::kPrimAlltoAll, AlltoAllInfer, false);
}  // namespace ops
}  // namespace mindspore
