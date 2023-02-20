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

#include "ops/grad/fractional_max_pool3d_grad_with_fixed_ksize.h"

#include <string>
#include <memory>
#include <set>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t DIM_SIZE4 = 4;
constexpr size_t DIM_SIZE5 = 5;
constexpr int64_t kInputsSize = 3;
constexpr size_t kDim4FormatNCDHWIndexC = 0;
constexpr size_t kDim4FormatNCDHWIndexD = 1;
constexpr size_t kDim4FormatNCDHWIndexH = 2;
constexpr size_t kDim4FormatNCDHWIndexW = 3;
constexpr size_t kDim4FormatNDHWCIndexD = 0;
constexpr size_t kDim4FormatNDHWCIndexH = 1;
constexpr size_t kDim4FormatNDHWCIndexW = 2;
constexpr size_t kDim4FormatNDHWCIndexC = 3;
constexpr size_t kDim5FormatNCDHWIndexN = 0;
constexpr size_t kDim5FormatNCDHWIndexC = 1;
constexpr size_t kDim5FormatNCDHWIndexD = 2;
constexpr size_t kDim5FormatNCDHWIndexH = 3;
constexpr size_t kDim5FormatNCDHWIndexW = 4;
constexpr size_t kDim5FormatNDHWCIndexN = 0;
constexpr size_t kDim5FormatNDHWCIndexD = 1;
constexpr size_t kDim5FormatNDHWCIndexH = 2;
constexpr size_t kDim5FormatNDHWCIndexW = 3;
constexpr size_t kDim5FormatNDHWCIndexC = 4;

abstract::ShapePtr FractionalMaxPool3DGradWithFixedKsizeInferShape(const PrimitivePtr &primitive,
                                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto data_format = GetValue<std::string>(primitive->GetAttr(kFormat));
  if (data_format != "NCDHW" && data_format != "NDHWC") {
    MS_EXCEPTION(ValueError) << "For '" << op_name << "', data_format is neither NCDHW nor NDHWC." << data_format
                             << ".";
  }
  (void)CheckAndConvertUtils::CheckInteger("input_number", SizeToLong(input_args.size()), kEqual, kInputsSize, op_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex1);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex2);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto origin_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  auto out_backprop_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack())[kShape];
  auto argmax_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->GetShapeTrack())[kShape];
  if (origin_input_shape.size() != DIM_SIZE4 && origin_input_shape.size() != DIM_SIZE5) {
    MS_EXCEPTION(TypeError) << "For '" << op_name
                            << "', the dimension of 'origin_input' must be equal to 4 or 5, but got "
                            << std::to_string(origin_input_shape.size()) << ".";
  }
  if (out_backprop_shape.size() != DIM_SIZE4 && out_backprop_shape.size() != DIM_SIZE5) {
    MS_EXCEPTION(TypeError) << "For '" << op_name
                            << "', the dimension of 'out_backprop' must be equal to 4 or 5, but got "
                            << std::to_string(out_backprop_shape.size()) << ".";
  }
  if (argmax_shape.size() != DIM_SIZE4 && argmax_shape.size() != DIM_SIZE5) {
    MS_EXCEPTION(TypeError) << "For '" << op_name << "', the dimension of 'argmax' must be equal to 4 or 5, but got "
                            << std::to_string(argmax_shape.size()) << ".";
  }
  int64_t n_dim_;
  int64_t c_dim_;
  int64_t d_dim_;
  int64_t h_dim_;
  int64_t w_dim_;
  std::vector<int64_t> output_size;
  if (origin_input_shape.size() == DIM_SIZE4) {
    if (data_format == "NCDHW") {
      c_dim_ = origin_input_shape[kDim4FormatNCDHWIndexC];
      d_dim_ = origin_input_shape[kDim4FormatNCDHWIndexD];
      h_dim_ = origin_input_shape[kDim4FormatNCDHWIndexH];
      w_dim_ = origin_input_shape[kDim4FormatNCDHWIndexW];
      output_size.push_back(c_dim_);
      output_size.push_back(d_dim_);
      output_size.push_back(h_dim_);
      output_size.push_back(w_dim_);
    } else {
      d_dim_ = origin_input_shape[kDim4FormatNDHWCIndexD];
      h_dim_ = origin_input_shape[kDim4FormatNDHWCIndexH];
      w_dim_ = origin_input_shape[kDim4FormatNDHWCIndexW];
      c_dim_ = origin_input_shape[kDim4FormatNDHWCIndexC];
      output_size.push_back(d_dim_);
      output_size.push_back(h_dim_);
      output_size.push_back(w_dim_);
      output_size.push_back(c_dim_);
    }
  } else {
    if (data_format == "NCDHW") {
      n_dim_ = origin_input_shape[kDim5FormatNCDHWIndexN];
      c_dim_ = origin_input_shape[kDim5FormatNCDHWIndexC];
      d_dim_ = origin_input_shape[kDim5FormatNCDHWIndexD];
      h_dim_ = origin_input_shape[kDim5FormatNCDHWIndexH];
      w_dim_ = origin_input_shape[kDim5FormatNCDHWIndexW];
      output_size.push_back(n_dim_);
      output_size.push_back(c_dim_);
      output_size.push_back(d_dim_);
      output_size.push_back(h_dim_);
      output_size.push_back(w_dim_);
    } else {
      n_dim_ = origin_input_shape[kDim5FormatNDHWCIndexN];
      d_dim_ = origin_input_shape[kDim5FormatNDHWCIndexD];
      h_dim_ = origin_input_shape[kDim5FormatNDHWCIndexH];
      w_dim_ = origin_input_shape[kDim5FormatNDHWCIndexW];
      c_dim_ = origin_input_shape[kDim5FormatNDHWCIndexC];
      output_size.push_back(n_dim_);
      output_size.push_back(d_dim_);
      output_size.push_back(h_dim_);
      output_size.push_back(w_dim_);
      output_size.push_back(c_dim_);
    }
  }
  return std::make_shared<abstract::Shape>(output_size);
}

TypePtr FractionalMaxPool3DGradWithFixedKsizeInferType(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input_number", SizeToLong(input_args.size()), kEqual, kInputsSize, op_name);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex1);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex2);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> out_backprop_valid_types = {kFloat16, kFloat32, kFloat64, kInt32, kInt64};
  const std::set<TypePtr> argmax_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("origin_input", input_args[kInputIndex0]->BuildType(),
                                                   out_backprop_valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("argmax", input_args[kInputIndex2]->BuildType(), argmax_valid_types,
                                                   op_name);
  return CheckAndConvertUtils::CheckTensorTypeValid("out_backprop", input_args[kInputIndex1]->BuildType(),
                                                    out_backprop_valid_types, op_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(FractionalMaxPool3DGradWithFixedKsize, BaseOperator);
AbstractBasePtr FractionalMaxPool3DGradWithFixedKsizeInfer(const abstract::AnalysisEnginePtr &,
                                                           const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = FractionalMaxPool3DGradWithFixedKsizeInferType(primitive, input_args);
  auto infer_shape = FractionalMaxPool3DGradWithFixedKsizeInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape);
}

void FractionalMaxPool3DGradWithFixedKsize::Init(const std::string data_format) { set_data_format(data_format); }

void FractionalMaxPool3DGradWithFixedKsize::set_data_format(const std::string data_format) {
  (void)this->AddAttr(kFormat, api::MakeValue(data_format));
}

std::string FractionalMaxPool3DGradWithFixedKsize::get_data_format() const {
  return GetValue<std::string>(GetAttr(kFormat));
}

// AG means auto generated
class MIND_API AGFractionalMaxPool3DGradWithFixedKsizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPool3DGradWithFixedKsizeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPool3DGradWithFixedKsizeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FractionalMaxPool3DGradWithFixedKsizeInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FractionalMaxPool3DGradWithFixedKsize,
                                 prim::kPrimFractionalMaxPool3DGradWithFixedKsize,
                                 AGFractionalMaxPool3DGradWithFixedKsizeInfer, false);
}  // namespace ops
}  // namespace mindspore
