/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/layer_norm_ext.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <functional>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(LayerNormExt, BaseOperator);
class MIND_API LayerNormExtInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    constexpr int64_t input_num = 5;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
    auto input_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto norm_shape = GetValue<std::vector<int64_t>>(input_args[kInputIndex1]->BuildValue());
    const auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_shape_ptr)[kShape];
    const auto norm_dim = norm_shape.size();
    const auto input_dim = input_shape.size();
    const auto begin_axis = input_dim - norm_dim;
    if (input_dim < 2 || input_dim > 4) {
      MS_LOG(EXCEPTION) << "For '" << op_name << "', input_rank can expects 2d,3d,4d. But got: " << input_dim << "d.";
    }
    const int64_t m =
      std::accumulate(input_shape.cbegin(), input_shape.cbegin() + begin_axis, 1LL, std::multiplies<int64_t>());
    ShapeVector mean_out_shape, rstd_out_shape;
    if (m <= 0) {
      mean_out_shape = {m};
      rstd_out_shape = {m};
    } else {
      ShapeVector mean_shape;
      for (size_t i = 0; i < begin_axis; ++i) {
        (void)mean_shape.emplace_back(input_shape[i]);
      }
      for (size_t i = begin_axis; i < input_dim; ++i) {
        (void)mean_shape.emplace_back(1);
      }
      mean_out_shape = mean_shape;
      rstd_out_shape = mean_shape;
    }
    std::vector<BaseShapePtr> shapes_list = {input_shape_ptr};
    (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(mean_out_shape));
    (void)shapes_list.emplace_back(std::make_shared<abstract::Shape>(rstd_out_shape));
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    // outputs: output, mean_out, rstd_out
    MS_EXCEPTION_IF_NULL(primitive);
    const std::string op_name = primitive->name();
    constexpr int64_t input_num = 5;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);

    auto input_type = input_args[kInputIndex0]->BuildType();
    auto weight_type = input_args[kInputIndex2]->BuildType();
    auto bias_type = input_args[kInputIndex3]->BuildType();
    auto valid_types = {kFloat16, kFloat32};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input_type", input_type, valid_types, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("weight_type", weight_type, valid_types, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("bias_type", bias_type, valid_types, op_name);

    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    std::vector<TypePtr> types_list;
    types_list = {input_type, input_type, input_type};
    return std::make_shared<Tuple>(types_list);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LayerNormExt, prim::kPrimLayerNormExt, LayerNormExtInfer, false);
}  // namespace ops
}  // namespace mindspore
