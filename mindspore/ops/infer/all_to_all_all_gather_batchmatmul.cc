/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "infer/all_to_all_all_gather_batchmatmul.h"

#include <memory>
#include <vector>
#include <map>
#include <string>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "mindapi/helper.h"
#include "primitive/math_ops.h"
#include "primitive/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "utils/ms_context.h"

#include "mindspore/ops/ops_utils/op_constants.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace ops {
namespace {
enum AlltoAllAllGatherBatchMatMulInputIndex : size_t {
  kAlltoAllAllGatherBatchMatMulInputXIndex = 0,
  kAlltoAllAllGatherBatchMatMulInputWeightIndex,
  kAlltoAllAllGatherBatchMatMulInputBiasIndex,
  kAlltoAllAllGatherBatchMatMulInputNum,
};
enum AlltoAllAllGatherBatchMatMulOutputIndex : size_t {
  kAlltoAllAllGatherBatchMatMulOutputY1Index = 0,
  kAlltoAllAllGatherBatchMatMulOutputY2Index,
  kAlltoAllAllGatherBatchMatMulOutputY3Index,
  kAlltoAllAllGatherBatchMatMulOutputNum,
};
constexpr size_t kXWeightRank = 3;

abstract::TupleShapePtr AlltoAllAllGatherBatchMatMulInferShape(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = primitive->name();
  auto transpose_weight = GetValue<bool>(primitive->GetAttr("transpose_weight"));
  auto ep_world_size = GetValue<int64_t>(primitive->GetAttr("ep_world_size"));
  auto tp_world_size = GetValue<int64_t>(primitive->GetAttr("tp_world_size"));
  auto output_y2_flag = GetValue<bool>(primitive->GetAttr("output_y2_flag"));
  auto output_y3_flag = GetValue<bool>(primitive->GetAttr("output_y3_flag"));
  // in mixtral 7B, weight.shape=(E/ep, hidden, M/tp)=(1, 12288, 6144)
  auto weight_hidden_dim = transpose_weight ? 2 : 1;
  auto weight_mtp_dim = transpose_weight ? 1 : 2;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kAlltoAllAllGatherBatchMatMulInputXIndex]->BuildShape())[kShape];
  auto weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kAlltoAllAllGatherBatchMatMulInputWeightIndex]->BuildShape())[kShape];
  if (x_shape.size() != kXWeightRank || weight_shape.size() != kXWeightRank) {
    MS_LOG(EXCEPTION) << op_name << ": The rank both of x and weight must be " << kXWeightRank << ", but got "
                      << x_shape.size() << " and " << weight_shape.size();
  }
  if (ep_world_size <= 0) {
    MS_LOG(EXCEPTION) << op_name << ": The ep_world_size must be a positive integer, but got " << ep_world_size;
  }
  if (tp_world_size <= 0) {
    MS_LOG(EXCEPTION) << op_name << ": The tp_world_size must be a positive integer, but got " << tp_world_size;
  }
  if (x_shape[-1] != weight_shape[weight_hidden_dim]) {
    MS_LOG(EXCEPTION) << op_name << ": The last dim of x and the hidden dim of weight must be equal, but got "
                      << x_shape[-1] << " and " << weight_shape[weight_hidden_dim];
  }
  if (x_shape[0] != weight_shape[0] * ep_world_size) {
    MS_LOG(EXCEPTION) << op_name << ": The first dim of x and weight must be equal, but got " << x_shape[0] << " and "
                      << weight_shape[0];
  }
  if (input_args[kAlltoAllAllGatherBatchMatMulInputBiasIndex]->BuildType()->type_id() != kMetaTypeNone) {
    auto bias_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
      input_args[kAlltoAllAllGatherBatchMatMulInputBiasIndex]->BuildShape())[kShape];
    ShapeVector expect_bias_shape_2dim{weight_shape[0], weight_shape[weight_mtp_dim]};
    ShapeVector expect_bias_shape_3dim{weight_shape[0], 1, weight_shape[weight_mtp_dim]};
    if (bias_shape != expect_bias_shape_2dim && bias_shape != expect_bias_shape_3dim) {
      MS_LOG(EXCEPTION) << op_name << ": The shape of input 'bias' must be " << expect_bias_shape_2dim << " or "
                        << expect_bias_shape_3dim << ", but got " << bias_shape;
    }
  }

  abstract::BaseShapePtrList output_shape_ptr_list(kAlltoAllAllGatherBatchMatMulOutputNum);
  ShapeVector y1_shape = {weight_shape[0], x_shape[1] * ep_world_size * tp_world_size, weight_shape[weight_mtp_dim]};
  output_shape_ptr_list[kAlltoAllAllGatherBatchMatMulOutputY1Index] = std::make_shared<abstract::Shape>(y1_shape);
  if (output_y2_flag) {
    ShapeVector y2_shape = {weight_shape[0], x_shape[1] * ep_world_size * tp_world_size, x_shape[2]};
    output_shape_ptr_list[kAlltoAllAllGatherBatchMatMulOutputY2Index] = std::make_shared<abstract::Shape>(y2_shape);
  }
  if (output_y3_flag) {
    ShapeVector y3_shape = {weight_shape[0], x_shape[1] * ep_world_size * tp_world_size, weight_shape[weight_mtp_dim]};
    output_shape_ptr_list[kAlltoAllAllGatherBatchMatMulOutputY3Index] = std::make_shared<abstract::Shape>(y3_shape);
  }
  return std::make_shared<abstract::TupleShape>(output_shape_ptr_list);
}

TuplePtr AlltoAllAllGatherBatchMatMulInferType(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set valid_types = {kFloat16, kBFloat16};
  auto op_name = primitive->name();
  auto x_type = input_args[kAlltoAllAllGatherBatchMatMulInputXIndex]->GetType();
  auto weight_type = input_args[kAlltoAllAllGatherBatchMatMulInputWeightIndex]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  MS_EXCEPTION_IF_NULL(weight_type);
  if (!x_type->isa<TensorType>() || !weight_type->isa<TensorType>()) {
    MS_LOG(EXCEPTION) << op_name << ": The input x and weight must be a tensor and a tensor, but got " << x_type
                      << " and " << weight_type;
  }
  auto output_y2_flag = GetValue<bool>(primitive->GetAttr("output_y2_flag"));
  auto output_y3_flag = GetValue<bool>(primitive->GetAttr("output_y3_flag"));
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kAlltoAllAllGatherBatchMatMulInputXIndex]->BuildType());
  (void)types.emplace("weight", input_args[kAlltoAllAllGatherBatchMatMulInputWeightIndex]->BuildType());
  if (input_args[kAlltoAllAllGatherBatchMatMulInputBiasIndex]->BuildType()->type_id() != kMetaTypeNone) {
    auto bias_type = input_args[kAlltoAllAllGatherBatchMatMulInputBiasIndex]->GetType();
    if (bias_type->isa<TensorType>()) {
      MS_LOG(EXCEPTION) << op_name << ": The input bias must be a tensor, but got " << bias_type;
    }
    (void)types.emplace("bias", input_args[kAlltoAllAllGatherBatchMatMulInputBiasIndex]->BuildType());
  }
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  TypePtrList output_type_ptr_list(kAlltoAllAllGatherBatchMatMulOutputNum);
  output_type_ptr_list[kAlltoAllAllGatherBatchMatMulOutputY1Index] = type;
  if (output_y2_flag) {
    output_type_ptr_list[kAlltoAllAllGatherBatchMatMulOutputY1Index] = type;
  }
  if (output_y3_flag) {
    output_type_ptr_list[kAlltoAllAllGatherBatchMatMulOutputY1Index] = type;
  }
  return std::make_shared<Tuple>(output_type_ptr_list);
}
}  // namespace

AbstractBasePtr AlltoAllAllGatherBatchMatMulInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kAlltoAllAllGatherBatchMatMulInputNum, primitive->name());
  auto infer_type = AlltoAllAllGatherBatchMatMulInferType(primitive, input_args);
  auto infer_shape = AlltoAllAllGatherBatchMatMulInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(AlltoAllAllGatherBatchMatMul, BaseOperator);

// AG means auto generated
class OPS_API AGAlltoAllAllGatherBatchMatMulInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AlltoAllAllGatherBatchMatMulInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AlltoAllAllGatherBatchMatMulInferType(primitive, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AlltoAllAllGatherBatchMatMulInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AlltoAllAllGatherBatchMatMul, prim::kPrimAlltoAllAllGatherBatchMatMul,
                                 AGAlltoAllAllGatherBatchMatMulInfer, false);
}  // namespace ops
}  // namespace mindspore
