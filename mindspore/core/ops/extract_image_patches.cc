/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/extract_image_patches.h"

#include <map>
#include <set>

#include "ir/dtype/number.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
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
constexpr int64_t extract_image_rank_num = 4;
constexpr size_t kIdx2 = 2;
constexpr size_t kIdx3 = 3;

void ExtractImagePatches::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
                               const std::vector<int64_t> &rates, const std::string &padding) {
  set_kernel_size(kernel_size);
  set_strides(strides);
  set_rates(rates);
  set_padding(padding);
}

void ExtractImagePatches::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)CheckAndConvertUtils::CheckInteger("kernel_size_length", SizeToLong(kernel_size.size()), kEqual,
                                           extract_image_rank_num, "ExtractImagePatches");
  (void)AddAttr(kKsizes, api::MakeValue(kernel_size));
}

void ExtractImagePatches::set_strides(const std::vector<int64_t> &strides) {
  (void)CheckAndConvertUtils::CheckInteger("strides_length", SizeToLong(strides.size()), kEqual, extract_image_rank_num,
                                           "ExtractImagePatches");
  (void)AddAttr(kStrides, api::MakeValue(strides));
}

void ExtractImagePatches::set_rates(const std::vector<int64_t> &rates) {
  (void)CheckAndConvertUtils::CheckInteger("rates_length", SizeToLong(rates.size()), kEqual, extract_image_rank_num,
                                           "ExtractImagePatches");
  (void)AddAttr(kRates, api::MakeValue(rates));
}

void ExtractImagePatches::set_padding(const std::string &padding) { (void)AddAttr(kPadding, api::MakeValue(padding)); }

std::vector<int64_t> ExtractImagePatches::get_kernel_size() const {
  auto value_ptr = GetAttr(kKsizes);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> ExtractImagePatches::get_strides() const {
  auto value_ptr = GetAttr(kStrides);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::vector<int64_t> ExtractImagePatches::get_rates() const {
  auto value_ptr = GetAttr(kRates);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

std::string ExtractImagePatches::get_padding() const {
  auto value_ptr = GetAttr(kPadding);
  return GetValue<std::string>(value_ptr);
}

MIND_API_OPERATOR_IMPL(ExtractImagePatches, BaseOperator);
class ExtractImagePatchesInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1,
                                             primitive->name());
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
    auto x_shape = x_shape_map[kShape];
    // ToSupport Dynamic rank
    if (IsDynamicRank(x_shape)) {
      // The input tensor of Primitive ExtractImagePatches must be a 4-D tensor and the data format is NHWC/NCHW.
      // So DynamicRank can transfer to 4-D dynamic shape
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{-1, -1, -1, -1});
    }

    (void)CheckAndConvertUtils::CheckInteger("input shape", SizeToLong(x_shape.size()), kEqual, extract_image_rank_num,
                                             primitive->name());

    std::vector<int64_t> kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKsizes));
    std::vector<int64_t> strides = GetValue<std::vector<int64_t>>(primitive->GetAttr(kStrides));
    std::vector<int64_t> rates = GetValue<std::vector<int64_t>>(primitive->GetAttr(kRates));
    auto padding = GetValue<std::string>(primitive->GetAttr(kPadding));

    for (auto &item : strides) {
      CheckAndConvertUtils::Check("strides", item, kGreaterThan, 0, primitive->name());
    }
    for (auto &item : kernel_size) {
      CheckAndConvertUtils::Check("kernel_size", item, kGreaterThan, 0, primitive->name());
    }

    for (auto &item : rates) {
      CheckAndConvertUtils::Check("rates", item, kGreaterThan, 0, primitive->name());
    }

    std::vector<int64_t> y_shape(extract_image_rank_num);
    y_shape[0] = x_shape[0];
    y_shape[1] = x_shape[1] == abstract::Shape::kShapeDimAny ? abstract::Shape::kShapeDimAny
                                                             : x_shape[1] * kernel_size[kIdx2] * kernel_size[kIdx3];
    if (padding == "VALID") {
      y_shape[kIdx2] =
        x_shape[kIdx2] == abstract::Shape::kShapeDimAny
          ? abstract::Shape::kShapeDimAny
          : (x_shape[kIdx2] - (kernel_size[kIdx2] + (kernel_size[kIdx2] - 1) * (rates[kIdx2] - 1))) / strides[kIdx2] +
              1;
      y_shape[kIdx3] =
        x_shape[kIdx3] == abstract::Shape::kShapeDimAny
          ? abstract::Shape::kShapeDimAny
          : (x_shape[kIdx3] - (kernel_size[kIdx3] + (kernel_size[kIdx3] - 1) * (rates[kIdx3] - 1))) / strides[kIdx3] +
              1;
    } else {
      y_shape[kIdx2] = x_shape[kIdx2] == abstract::Shape::kShapeDimAny ? abstract::Shape::kShapeDimAny
                                                                       : (x_shape[kIdx2] - 1) / strides[kIdx2] + 1;
      y_shape[kIdx3] = x_shape[kIdx3] == abstract::Shape::kShapeDimAny ? abstract::Shape::kShapeDimAny
                                                                       : (x_shape[kIdx3] - 1) / strides[kIdx3] + 1;
    }

    std::vector<std::string> out_names{"out_batch", "out_depth", "out_row", "out_col"};
    for (size_t idx = 0; idx < y_shape.size(); idx++) {
      if (y_shape[idx] == abstract::Shape::kShapeDimAny) {
        continue;
      }
      CheckAndConvertUtils::Check(out_names[idx], y_shape[idx], kGreaterThan, 0, primitive->name());
    }
    return std::make_shared<abstract::Shape>(y_shape);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, prim->name());
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt8,   kInt16,     kInt32,      kInt64,
                                           kUInt8,   kUInt16,  kUInt32,  kUInt64, kComplex64, kComplex128, kBool};
    return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[0]->BuildType(), valid_types, prim->name());
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ExtractImagePatches, prim::kPrimExtractImagePatches, ExtractImagePatchesInfer, false);
}  // namespace ops
}  // namespace mindspore
