/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ops/fft_with_size.h"

#include <algorithm>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/overload.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void FFTWithSize::Init(const int64_t signal_ndim, const bool inverse, const bool real, const std::string &norm,
                       const bool onesided, const std::vector<int64_t> &signal_sizes) {
  this->set_signal_ndim(signal_ndim);
  this->set_inverse(inverse);
  this->set_real(real);
  this->set_norm(norm);
  this->set_onesided(onesided);
  this->set_signal_sizes(signal_sizes);
}
void FFTWithSize::set_signal_ndim(const int64_t signal_ndim) {
  (void)this->AddAttr("signal_ndim", api::MakeValue(signal_ndim));
}
void FFTWithSize::set_inverse(const bool inverse) { (void)this->AddAttr("inverse", api::MakeValue(inverse)); }
void FFTWithSize::set_real(const bool real) { (void)this->AddAttr("real", api::MakeValue(real)); }
void FFTWithSize::set_norm(const std::string &norm) { (void)this->AddAttr("norm", api::MakeValue(norm)); }
void FFTWithSize::set_onesided(const bool onesided) { (void)this->AddAttr("onesided", api::MakeValue(onesided)); }
void FFTWithSize::set_signal_sizes(const std::vector<int64_t> &signal_sizes) {
  (void)this->AddAttr("signal_sizes", api::MakeValue(signal_sizes));
}
int64_t FFTWithSize::get_signal_ndim() const { return GetValue<int64_t>(GetAttr("signal_ndim")); }
bool FFTWithSize::get_inverse() const { return GetValue<bool>(GetAttr("inverse")); }
bool FFTWithSize::get_real() const { return GetValue<bool>(GetAttr("real")); }
std::string FFTWithSize::get_norm() const { return GetValue<std::string>(GetAttr("norm")); }
bool FFTWithSize::get_onesided() const { return GetValue<bool>(GetAttr("onesided")); }
std::vector<int64_t> FFTWithSize::get_signal_sizes() const {
  return GetValue<std::vector<int64_t>>(GetAttr("signal_sizes"));
}

namespace {
abstract::ShapePtr FFTWithSizeInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const int64_t kDimNum = 2;
  const int64_t kSignalRankMin = 1, kSignalRankMax = 3;
  auto signal_ndim_attr = primitive->GetAttr("signal_ndim");
  auto onesided_attr = primitive->GetAttr("onesided");
  auto signal_sizes_attr = primitive->GetAttr("signal_sizes");
  auto real_attr = primitive->GetAttr("real");
  auto inverse_attr = primitive->GetAttr("inverse");
  auto norm_attr = primitive->GetAttr("norm");
  auto signal_ndim = GetValue<int64_t>(signal_ndim_attr);
  auto onesided = GetValue<bool>(onesided_attr);
  auto signal_sizes = GetValue<std::vector<int64_t>>(signal_sizes_attr);
  auto real = GetValue<bool>(real_attr);
  auto inverse = GetValue<bool>(inverse_attr);
  auto norm = GetValue<std::string>(norm_attr);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  if (std::any_of(x_shape.begin(), x_shape.end(), [](int64_t dim) { return dim < 0; })) {
    // if dynamic shape, we just return vector of -1 with x_shape.size()
    std::vector<int64_t> y_shape(x_shape.size(), -1);
    return std::make_shared<abstract::Shape>(y_shape);
  }
  auto y_shape = x_shape;
  if ((norm != "forward") && (norm != "backward") && (norm != "ortho")) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                             << "the norm must be 'forward','backward' or 'ortho'"
                             << " while norm is "
                             << "'" << norm << "'"
                             << ".";
  }
  if (signal_ndim > kSignalRankMax || signal_ndim < kSignalRankMin) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                             << "the signal_ndim must be in range of"
                             << " [" << kSignalRankMin << ", " << kSignalRankMax << "],"
                             << " while signal_ndim is " << signal_ndim << ".";
  }
  if (x_shape.size() < LongToUlong(signal_ndim)) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                             << "x's dimension must be greater than or equal to signal_ndim, "
                             << "x's dimension is " << x_shape.size() << ", signal_ndim is " << signal_ndim << ".";
  }
  if (real && onesided) {
    if (!inverse) {
      y_shape.back() = x_shape.back() / kDimNum + 1;
    } else {  // irfft, signal_sizes without batch dimension.
      if (!signal_sizes.empty() && signal_sizes.size() != LongToUlong(signal_ndim)) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                                 << "signal_sizes is expected to be empty (default)"
                                 << " or of signal_ndim=" << signal_ndim << "D, but got signal_sizes=" << signal_sizes;
      }
      if (signal_sizes.empty()) {
        y_shape.back() = (y_shape.back() - 1) * kDimNum;
      } else {
        std::vector<int64_t> valid_size_even(y_shape.end() - signal_ndim, y_shape.end());
        valid_size_even.back() = (y_shape.back() - 1) * kDimNum;
        auto valid_size_odd = valid_size_even;
        valid_size_odd.back() = valid_size_even.back() + 1;
        auto batch_rank = SizeToLong(y_shape.size()) - signal_ndim;
        for (size_t i = 0; i < LongToUlong(signal_ndim) - 1; i++) {
          if (signal_sizes[i] != y_shape[i + batch_rank]) {
            MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                                     << "got invalid signal_sizes: " << ToString(signal_sizes)
                                     << ", a valid one should be " << ToString(valid_size_even) << ", or "
                                     << ToString(valid_size_odd) << ".";
          }
        }
        if (signal_sizes.back() / kDimNum + 1 == y_shape.back()) {
          y_shape.back() = signal_sizes.back();
        } else {
          MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                                   << "got invalid signal_sizes: " << ToString(signal_sizes)
                                   << ", a valid one should be " << ToString(valid_size_even) << ", or "
                                   << ToString(valid_size_odd) << ".";
        }
      }
    }
  }
  return std::make_shared<abstract::Shape>(y_shape);
}

TypePtr FFTWithSizeInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::string prim_name = prim->name();
  CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  auto input_type = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64, kComplex64, kComplex128};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, valid_types, prim_name);
  auto real_attr = prim->GetAttr("real");
  auto inverse_attr = prim->GetAttr("inverse");
  if (inverse_attr == nullptr) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                             << "the `inverse` attr must be set.";
  }
  if (real_attr == nullptr) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                             << "the `real` attr must be set.";
  }
  auto real = GetValue<bool>(real_attr);
  auto inverse = GetValue<bool>(inverse_attr);
  auto out_type = input_type;
  if (real) {
    if (!inverse) {
      if (*(input_type->cast<TensorTypePtr>()->element()) == *(kFloat32)) {
        out_type->cast<TensorTypePtr>()->set_element(kComplex64);
      } else if (*(input_type->cast<TensorTypePtr>()->element()) == *(kFloat64)) {
        out_type->cast<TensorTypePtr>()->set_element(kComplex128);
      } else {
        MS_EXCEPTION(TypeError) << "For '" << prim_name << "', "
                                << "RFFT requires float32 or float64 inputs, but got "
                                << *(input_type->cast<TensorTypePtr>()->element()) << ".";
      }
    } else {
      if (*(input_type->cast<TensorTypePtr>()->element()) == *(kComplex64)) {
        out_type->cast<TensorTypePtr>()->set_element(kFloat32);
      } else if (*(input_type->cast<TensorTypePtr>()->element()) == *(kComplex128)) {
        out_type->cast<TensorTypePtr>()->set_element(kFloat64);
      } else {
        MS_EXCEPTION(TypeError) << "For '" << prim_name << "', "
                                << "IRFFT requires complex64 or complex128 inputs, but got "
                                << *(input_type->cast<TensorTypePtr>()->element()) << ".";
      }
    }
  } else {
    if (*(input_type->cast<TensorTypePtr>()->element()) != *(kComplex64) &&
        *(input_type->cast<TensorTypePtr>()->element()) != *(kComplex128)) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', "
                              << "FFT/IFFT requires complex64 or complex128 inputs, but got "
                              << *(input_type->cast<TensorTypePtr>()->element()) << ".";
    }
  }
  return out_type;
}
}  // namespace

AbstractBasePtr FFTWithSizeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = FFTWithSizeInferType(primitive, input_args);
  auto infer_shape = FFTWithSizeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(FFTWithSize, BaseOperator);

// AG means auto generated
class MIND_API AGFFTWithSizeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FFTWithSizeInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FFTWithSizeInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FFTWithSizeInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FFTWithSize, prim::kPrimFFTWithSize, AGFFTWithSizeInfer, false);
}  // namespace ops
}  // namespace mindspore
