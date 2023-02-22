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

#include "ops/max_pool3d_with_argmax.h"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void MaxPool3DWithArgmax::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
                               const std::vector<int64_t> &pads, const std::vector<int64_t> &dialtion, bool ceil_mode,
                               const Format &format, const TypeId &argmax_type) {
  set_kernel_size(kernel_size);
  set_strides(strides);
  set_pads(pads);
  set_dilation(dialtion);
  set_ceil_mode(ceil_mode);
  set_format(format);
  set_argmax_type(argmax_type);
}

void MaxPool3DWithArgmax::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)AddAttr(kKSize, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kKSize, kernel_size, name())));
}

void MaxPool3DWithArgmax::set_strides(const std::vector<int64_t> &strides) {
  (void)AddAttr(kStrides, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kStrides, strides, name())));
}

void MaxPool3DWithArgmax::set_pads(const std::vector<int64_t> &pads) { (void)AddAttr(kPads, api::MakeValue(pads)); }

void MaxPool3DWithArgmax::set_dilation(const std::vector<int64_t> &dilation) {
  int64_t kMinDilationSize = 3;
  int64_t size = SizeToLong(dilation.size());
  (void)CheckAndConvertUtils::CheckInteger("dilation_shape", size, kGreaterThan, kMinDilationSize, name());
  std::vector<int64_t> d;
  for (int64_t i = size - kMinDilationSize; i < size; i++) {
    d.push_back(dilation[i]);
  }
  (void)AddAttr(kDilation, api::MakeValue(CheckAndConvertUtils::CheckPositiveVector(kDilation, d, name())));
}

void MaxPool3DWithArgmax::set_ceil_mode(bool ceil_mode) { (void)AddAttr(kCeilMode, api::MakeValue(ceil_mode)); }

void MaxPool3DWithArgmax::set_format(const Format &format) {
  int64_t f = format;
  (void)AddAttr(kFormat, api::MakeValue(f));
}

void MaxPool3DWithArgmax::set_argmax_type(const TypeId &argmax_type) {
  int f = argmax_type;
  (void)AddAttr(kArgmaxType, api::MakeValue(f));
}

std::vector<int64_t> MaxPool3DWithArgmax::get_kernel_size() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kKSize));
}

std::vector<int64_t> MaxPool3DWithArgmax::get_strides() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kStrides));
}

std::vector<int64_t> MaxPool3DWithArgmax::get_pads() const { return GetValue<std::vector<int64_t>>(GetAttr(kPads)); }

std::vector<int64_t> MaxPool3DWithArgmax::get_dilation() const {
  return GetValue<std::vector<int64_t>>(GetAttr(kDilation));
}

bool MaxPool3DWithArgmax::get_ceil_mode() const { return GetValue<bool>(GetAttr(kCeilMode)); }

Format MaxPool3DWithArgmax::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!value_ptr->isa<mindspore::api::StringImm>()) {
    return Format(GetValue<int64_t>(value_ptr));
  }
  static const std::map<std::string, int64_t> valid_dataformat = {
    {"NCDHW", Format::NCDHW},
  };
  auto attr_value_str = GetValue<std::string>(value_ptr);
  (void)std::transform(attr_value_str.begin(), attr_value_str.end(), attr_value_str.begin(), toupper);
  auto iter = valid_dataformat.find(attr_value_str);
  if (iter == valid_dataformat.end()) {
    MS_LOG(EXCEPTION) << "for MaxPool3DWithArgmax, Invalid format " << attr_value_str << ", use NCDHW";
  }
  return Format(iter->second);
}

TypeId MaxPool3DWithArgmax::get_argmax_type() const {
  auto value_ptr = GetAttr(kArgmaxType);
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!value_ptr->isa<mindspore::api::StringImm>()) {
    return TypeId(GetValue<int64_t>(value_ptr));
  }
  static const std::map<std::string, int> valid_argmax_type = {
    {"int32", TypeId::kNumberTypeInt32},
    {"int64", TypeId::kNumberTypeInt64},
  };
  auto attr_value_str = GetValue<std::string>(value_ptr);
  (void)std::transform(attr_value_str.begin(), attr_value_str.end(), attr_value_str.begin(), toupper);
  auto iter = valid_argmax_type.find(attr_value_str);
  if (iter == valid_argmax_type.end()) {
    MS_LOG(EXCEPTION) << "for MaxPool3DWithArgmax, Invalid argmax type " << attr_value_str << ", use int64 or int32";
  }
  return TypeId(iter->second);
}

TuplePtr MaxPool3DWithArgmaxInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kInt8,   kInt16,  kInt32,   kInt64,   kUInt8,  kUInt16,
                                         kUInt32, kUInt64, kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_args[0]->BuildType(), valid_types, prim->name());
  auto output_dtype = input_args[0]->BuildType();
  auto Targmax = GetValue<std::string>(prim->GetAttr("argmax_type"));
  TypePtr argmax_dtype;
  if (Targmax == "int32") {
    argmax_dtype = std::make_shared<TensorType>(kInt32);
  } else if (Targmax == "int64") {
    argmax_dtype = std::make_shared<TensorType>(kInt64);
  } else {
    MS_EXCEPTION(TypeError) << "for " << prim->name() << ", The type of argmax should be int32 or int64 ";
  }
  std::vector<TypePtr> type_list = {output_dtype, argmax_dtype};
  return std::make_shared<Tuple>(type_list);
}

abstract::TupleShapePtr MaxPool3DWithArgmaxInferShape(const PrimitivePtr &prim,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  const size_t kAttrD = 0;
  const size_t kAttrH = 1;
  const size_t kAttrW = 2;
  const size_t kInputShapeSize = 5;
  const size_t kAttrsSize = 3;
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    std::vector<abstract::BaseShapePtr> shape_list = {
      std::make_shared<abstract::Shape>(std::vector<int64_t>{
        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny}),
      std::make_shared<abstract::Shape>(std::vector<int64_t>{
        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny})};
    return std::make_shared<abstract::TupleShape>(shape_list);
  }
  (void)CheckAndConvertUtils::CheckInteger("input x rank", SizeToLong(x_shape.size()), kEqual, kInputShapeSize,
                                           prim->name());
  auto ksize = GetValue<std::vector<int64_t>>(prim->GetAttr("ksize"));
  (void)CheckAndConvertUtils::CheckInteger("ksize rank", SizeToLong(ksize.size()), kEqual, kAttrsSize, prim->name());
  auto strides = GetValue<std::vector<int64_t>>(prim->GetAttr("strides"));
  (void)CheckAndConvertUtils::CheckInteger("strides rank", SizeToLong(strides.size()), kEqual, kAttrsSize,
                                           prim->name());
  auto pads = GetValue<std::vector<int64_t>>(prim->GetAttr("pads"));
  (void)CheckAndConvertUtils::CheckInteger("pads rank", SizeToLong(pads.size()), kEqual, kAttrsSize, prim->name());
  auto dilation = GetValue<std::vector<int64_t>>(prim->GetAttr("dilation"));
  (void)CheckAndConvertUtils::CheckInteger("dilation rank", SizeToLong(dilation.size()), kEqual, kAttrsSize,
                                           prim->name());
  if (IsDynamic(x_shape)) {
    std::vector<abstract::BaseShapePtr> shape_list = {std::make_shared<abstract::Shape>(std::vector<int64_t>{
                                                        x_shape[0], x_shape[1], abstract::Shape::kShapeDimAny,
                                                        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny}),
                                                      std::make_shared<abstract::Shape>(std::vector<int64_t>{
                                                        x_shape[0], x_shape[1], abstract::Shape::kShapeDimAny,
                                                        abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny})};
    return std::make_shared<abstract::TupleShape>(shape_list);
  }
  auto D_in = x_shape[kIndex2];
  auto H_in = x_shape[kIndex3];
  auto W_in = x_shape[kIndex4];
  auto D_out = 0;
  auto H_out = 0;
  auto W_out = 0;
  int64_t factor = 2;
  if (GetValue<bool>(prim->GetAttr("ceil_mode")) == false) {
    // math: out = ((input + 2 * pad - dilation * (ksize - 1) - 1) / stride) + 1;
    D_out = ((D_in + factor * pads[kAttrD] - dilation[kAttrD] * (ksize[kAttrD] - 1) - 1) / strides[kAttrD]) + 1;
    H_out = ((H_in + factor * pads[kAttrH] - dilation[kAttrH] * (ksize[kAttrH] - 1) - 1) / strides[kAttrH]) + 1;
    W_out = ((W_in + factor * pads[kAttrW] - dilation[kAttrW] * (ksize[kAttrW] - 1) - 1) / strides[kAttrW]) + 1;
  } else {
    // math: out = ((input + 2 * pad - dilation * (ksize - 1) - 1 + (stride - 1)) / stride) + 1;
    D_out = ((D_in + factor * pads[kAttrD] - dilation[kAttrD] * (ksize[kAttrD] - 1) - 1 + (strides[kAttrD] - 1)) /
             strides[kAttrD]) +
            1;
    H_out = ((H_in + factor * pads[kAttrH] - dilation[kAttrH] * (ksize[kAttrH] - 1) - 1 + (strides[kAttrH] - 1)) /
             strides[kAttrH]) +
            1;
    W_out = ((W_in + factor * pads[kAttrW] - dilation[kAttrW] * (ksize[kAttrW] - 1) - 1 + (strides[kAttrW] - 1)) /
             strides[kAttrW]) +
            1;
    // The last pooling starts inside the image.
    if ((D_out - 1) * strides[kAttrD] >= D_in + pads[kAttrD]) {
      --D_out;
    }
    if ((H_out - 1) * strides[kAttrH] >= H_in + pads[kAttrH]) {
      --H_out;
    }
    if ((W_out - 1) * strides[kAttrW] >= W_in + pads[kAttrW]) {
      --W_out;
    }
  }
  ShapeVector output_shape = {x_shape[0], x_shape[1], D_out, H_out, W_out};
  if (D_out <= 0 || H_out <= 0 || W_out <= 0) {
    MS_EXCEPTION(ValueError) << "for " << prim->name() << ", shape of out is [" << x_shape[0] << ", " << x_shape[1]
                             << ", " << D_out << ", " << H_out << ", " << W_out
                             << "]. It should be not less than zero.";
  }
  ShapeVector argmax_shape = output_shape;
  std::vector<abstract::BaseShapePtr> shape_list = {std::make_shared<abstract::Shape>(output_shape),
                                                    std::make_shared<abstract::Shape>(argmax_shape)};
  return std::make_shared<abstract::TupleShape>(shape_list);
}

AbstractBasePtr MaxPool3DWithArgmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MaxPool3DWithArgmaxInferType(primitive, input_args);
  auto infer_shape = MaxPool3DWithArgmaxInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(MaxPool3DWithArgmax, BaseOperator);

// AG means auto generated
class MIND_API AGMaxPool3DWithArgmaxInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPool3DWithArgmaxInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPool3DWithArgmaxInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MaxPool3DWithArgmaxInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MaxPool3DWithArgmax, prim::kPrimMaxPool3DWithArgmax, AGMaxPool3DWithArgmaxInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
