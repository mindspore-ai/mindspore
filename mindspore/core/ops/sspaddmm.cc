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

#include <set>
#include <vector>
#include <memory>
#include <complex>
#include <map>
#include <string>
#include <climits>
#include <algorithm>

#include "ops/sspaddmm.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "base/float16.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/type_id.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t MAX_LEN = 1000000;

template <typename T>
int64_t compute_output_indices_unique_size(const T *indices, size_t size) {
  std::set<T> mat1_indices_set;
  size_t half_size = size / 2;
  for (size_t i = 0; i < half_size; i++) {
    (void)mat1_indices_set.insert(indices[i]);
  }
  return mat1_indices_set.size();
}

enum DimNum : size_t {
  dim0Num = 0,
  dim1Num,
  dim2Num,
};

int64_t GetInt64AlphaDataOther(void *values, TypeId tid, const TypePtr expect_dtype, float real) {
  int64_t compute_val = 0;
  bool flag1 = expect_dtype->type_id() == kNumberTypeUInt8 ? true : false;
  bool flag2 = false;
  switch (tid) {
    case kNumberTypeFloat16:
      flag2 = (flag1 && (reinterpret_cast<float16 *>(values)[0] < static_cast<float16>(0))) ? true : false;
      compute_val = static_cast<int64_t>(reinterpret_cast<float16 *>(values)[0]);
      break;
    case kNumberTypeFloat32:
      flag2 = (flag1 && (reinterpret_cast<float *>(values)[0] < 0)) ? true : false;
      compute_val = static_cast<int64_t>(reinterpret_cast<float *>(values)[0]);
      break;
    case kNumberTypeFloat64:
      flag2 = (flag1 && (reinterpret_cast<double *>(values)[0] < 0)) ? true : false;
      compute_val = static_cast<int64_t>(reinterpret_cast<double *>(values)[0]);
      break;
    case kNumberTypeBool:
      compute_val = static_cast<int64_t>(reinterpret_cast<bool *>(values)[0]);
      break;
    case kNumberTypeComplex64:
    case kNumberTypeComplex128:
      compute_val = static_cast<int64_t>(real);
      break;
    default:
      MS_EXCEPTION(TypeError) << "For Sspaddmm, alpha dtype is not support, only support"
                              << " number type and bool, complex64, complex128. ";
  }
  if (flag2) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, alpha value cannot be converted to type uint8 without overflow. ";
  }
  return compute_val;
}

int64_t GetInt64AlphaData(void *values, TypeId tid, const TypePtr expect_dtype, float real) {
  int64_t compute_val = 0;
  switch (tid) {
    case kNumberTypeUInt8:
      compute_val = static_cast<int64_t>(reinterpret_cast<uint8_t *>(values)[0]);
      break;
    case kNumberTypeUInt16:
      compute_val = static_cast<int64_t>(reinterpret_cast<uint16_t *>(values)[0]);
      break;
    case kNumberTypeUInt32:
      compute_val = static_cast<int64_t>(reinterpret_cast<uint32_t *>(values)[0]);
      break;
    case kNumberTypeUInt64:
      compute_val = static_cast<int64_t>(reinterpret_cast<uint64_t *>(values)[0]);
      break;
    case kNumberTypeInt8:
      compute_val = static_cast<int64_t>(reinterpret_cast<int8_t *>(values)[0]);
      break;
    case kNumberTypeInt16:
      compute_val = static_cast<int64_t>(reinterpret_cast<int16_t *>(values)[0]);
      break;
    case kNumberTypeInt32:
      compute_val = static_cast<int64_t>(reinterpret_cast<int32_t *>(values)[0]);
      break;
    case kNumberTypeInt64:
      compute_val = static_cast<int64_t>(reinterpret_cast<int64_t *>(values)[0]);
      break;
    default:
      compute_val = GetInt64AlphaDataOther(values, tid, expect_dtype, real);
      break;
  }
  return compute_val;
}

void CheckAlphaBeta(const std::vector<AbstractBasePtr> &input_args) {
  auto alpha_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex7]->BuildShape())[kShape];
  auto beta_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex8]->BuildShape())[kShape];
  if (!IsDynamic(alpha_shape) &&
      !((alpha_shape.size() == dim1Num && alpha_shape[0] == SizeToLong(dim1Num)) || (alpha_shape.size() == dim0Num))) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, alpha shape should be (1,) or ()"
                             << ", but get dim num is " << alpha_shape.size() << ", dim0 size is " << alpha_shape[0]
                             << ".";
  }
  if (!IsDynamic(beta_shape) &&
      !((beta_shape.size() == dim1Num && beta_shape[0] == SizeToLong(dim1Num)) || (beta_shape.size() == dim0Num))) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, beta shape should be (1,) or ()"
                             << ", but get dim num is " << beta_shape.size() << ", dim0 size is " << beta_shape[0]
                             << ".";
  }
}

void CheckInputTensorShapeSize(const std::vector<AbstractBasePtr> &input_args, const bool &is_dynamic_rank) {
  if (is_dynamic_rank) {
    return;
  }
  auto x1_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x1_values_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto x1_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto x2_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto x2_values_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto x2_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  auto x3_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];

  if (x1_indices_shape.size() != dim2Num) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, x1_indices should be a 2-D tensor"
                             << ", while x1_indices dim num is " << x1_indices_shape.size() << ".";
  }
  if (x1_values_shape.size() != dim1Num) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, x1_values should be a 1-D tensor"
                             << ",  while x1_values dim num is " << x1_values_shape.size() << ".";
  }
  if (x1_shape_shape.size() != dim1Num) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm"
                             << ", x1_shape should be a 1-D tensor, while x1_shape dim num is " << x1_shape_shape.size()
                             << ".";
  }
  if (x2_indices_shape.size() != dim2Num) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, x2_indices should be a 2-D tensor"
                             << ", while x2_indices dim num is " << x2_indices_shape.size() << ".";
  }
  if (x2_values_shape.size() != dim1Num) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, x2_values should be a 1-D tensor"
                             << ",  while x2_values dim num is " << x2_values_shape.size() << ".";
  }
  if (x2_shape_shape.size() != dim1Num) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm"
                             << ", x2_shape should be a 1-D tensor, while x2_shape dim num is " << x2_shape_shape.size()
                             << ".";
  }
  if (x3_shape.size() != dim2Num) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, x3_dense should be a 2-D tensor"
                             << ", while dim num is " << x3_shape.size() << ".";
  }
}

void CheckInputTensorShapeValue(const std::vector<AbstractBasePtr> &input_args, const bool &is_dynamic) {
  if (is_dynamic) {
    return;
  }
  auto x1_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x1_values_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto x1_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto x2_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto x2_values_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto x2_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];

  if (x1_indices_shape[0] != SizeToLong(dim2Num)) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, x1_indices shape should be (2, n)"
                             << ", while x1_indices shape dim0 is " << x1_indices_shape[0] << ".";
  }
  if (x1_indices_shape[1] != x1_values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm"
                             << ", dim1 size of `x1_indices` and dim0 size of `x1_values` should be the same"
                             << " while x1_indices dim1 size is " << x1_indices_shape[1]
                             << ", x1_values_shape dim0 size is " << x1_values_shape[0] << ".";
  }
  if (x1_shape_shape[0] != SizeToLong(dim2Num)) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm"
                             << ", the shape of x1_shape should be [2] but got shape [" << x1_shape_shape[0] << "].";
  }
  if (x2_indices_shape[0] != SizeToLong(dim2Num)) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, x2_indices shape should be (2, n)"
                             << ", while x2_indices shape dim0 is " << x2_indices_shape[0] << ".";
  }
  if (x2_indices_shape[1] != x2_values_shape[0]) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm"
                             << ", dim1 size of `x2_indices` and dim0 size of `x2_values` should be the same"
                             << " while x2_indices dim1 size is " << x2_indices_shape[1]
                             << ", x2_values_shape dim0 size is " << x2_values_shape[0] << ".";
  }
  if (x2_shape_shape[0] != SizeToLong(dim2Num)) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm"
                             << ", the shape of x2_shape should be [2] but got shape [" << x2_shape_shape[0] << "].";
  }
}

void CheckInputTensor(const std::vector<AbstractBasePtr> &input_args) {
  auto x1_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x1_values_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto x1_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto x2_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  auto x2_values_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->BuildShape())[kShape];
  auto x2_shape_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->BuildShape())[kShape];
  auto x3_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];

  std::vector<ShapeVector> all_shapes = {x1_indices_shape, x1_values_shape, x1_shape_shape, x2_indices_shape,
                                         x2_values_shape,  x2_shape_shape,  x3_shape};
  auto is_dynamic_rank = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamicRank);
  auto is_dynamic = std::any_of(all_shapes.begin(), all_shapes.end(), IsDynamic);

  CheckInputTensorShapeSize(input_args, is_dynamic_rank);
  CheckInputTensorShapeValue(input_args, is_dynamic);
  CheckAlphaBeta(input_args);
}

template <typename T>
void IndicesBoundCheck(const T *indices_val, size_t indices_num, const T *shape_val, const std::string &name) {
  if (shape_val[0] <= 0 || shape_val[1] <= 0) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm, " << name << "_shape should be positive, "
                             << "while got shape [" << shape_val[0] << ", " << shape_val[1] << "].";
  }
  size_t half_num = indices_num / dim2Num;
  for (size_t i = 0; i < half_num; i++) {
    if ((indices_val[i] < 0) || (indices_val[i] >= shape_val[0])) {
      MS_EXCEPTION(ValueError) << "For Sspaddmm, " << name << "_indices row index should between [0, " << shape_val[0]
                               << "], while got row index " << indices_val[i] << ".";
    }
    if ((indices_val[i + half_num] < 0) || (indices_val[i + half_num] >= shape_val[1])) {
      MS_EXCEPTION(ValueError) << "For Sspaddmm, " << name << "_indices col index should between [0, " << shape_val[1]
                               << "], while got col index " << indices_val[i + half_num] << ".";
    }
  }
}

void CheckIndices(const std::vector<AbstractBasePtr> &input_args) {
  if ((input_args[kInputIndex0]->isa<abstract::AbstractTensor>() &&
       input_args[kInputIndex0]->BuildValue()->isa<tensor::Tensor>()) &&
      (input_args[kInputIndex2]->isa<abstract::AbstractTensor>() &&
       input_args[kInputIndex2]->BuildValue()->isa<tensor::Tensor>())) {
    auto x1_indices_abstract = input_args[kInputIndex0]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(x1_indices_abstract);
    auto x1_indices_value_ptr = x1_indices_abstract->BuildValue();
    MS_EXCEPTION_IF_NULL(x1_indices_value_ptr);
    auto x1_indices_tensor = x1_indices_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(x1_indices_tensor);
    auto x1_indices_type = input_args[kInputIndex0]->BuildType();
    MS_EXCEPTION_IF_NULL(x1_indices_type);
    auto x1_indices_type_id = x1_indices_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(x1_indices_type_id);
    auto x1_indices_type_element = x1_indices_type_id->element();
    MS_EXCEPTION_IF_NULL(x1_indices_type_element);
    auto x1_shape_abstract = input_args[kInputIndex2]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(x1_shape_abstract);
    auto x1_shape_value_ptr = x1_shape_abstract->BuildValue();
    MS_EXCEPTION_IF_NULL(x1_shape_value_ptr);
    auto x1_shape_tensor = x1_shape_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(x1_shape_tensor);
    if (x1_indices_type_element->type_id() == kNumberTypeInt32) {
      IndicesBoundCheck<int32_t>(reinterpret_cast<int32_t *>(x1_indices_tensor->data_c()),
                                 x1_indices_tensor->DataSize(), reinterpret_cast<int32_t *>(x1_shape_tensor->data_c()),
                                 "x1");
    } else {
      IndicesBoundCheck<int64_t>(reinterpret_cast<int64_t *>(x1_indices_tensor->data_c()),
                                 x1_indices_tensor->DataSize(), reinterpret_cast<int64_t *>(x1_shape_tensor->data_c()),
                                 "x1");
    }
  }
  if ((input_args[kInputIndex3]->isa<abstract::AbstractTensor>() &&
       input_args[kInputIndex3]->BuildValue()->isa<tensor::Tensor>()) &&
      (input_args[kInputIndex5]->isa<abstract::AbstractTensor>() &&
       input_args[kInputIndex5]->BuildValue()->isa<tensor::Tensor>())) {
    auto x2_indices_abstract = input_args[kInputIndex3]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(x2_indices_abstract);
    auto x2_indices_value_ptr = x2_indices_abstract->BuildValue();
    MS_EXCEPTION_IF_NULL(x2_indices_value_ptr);
    auto x2_indices_tensor = x2_indices_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(x2_indices_tensor);
    auto x2_indices_type = input_args[kInputIndex3]->BuildType();
    MS_EXCEPTION_IF_NULL(x2_indices_type);
    auto x2_indices_type_id = x2_indices_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(x2_indices_type_id);
    auto x2_indices_type_element = x2_indices_type_id->element();
    MS_EXCEPTION_IF_NULL(x2_indices_type_element);
    auto x2_shape_abstract = input_args[kInputIndex5]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(x2_shape_abstract);
    auto x2_shape_value_ptr = x2_shape_abstract->BuildValue();
    MS_EXCEPTION_IF_NULL(x2_shape_value_ptr);
    auto x2_shape_tensor = x2_shape_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(x2_shape_tensor);
    if (x2_indices_type_element->type_id() == kNumberTypeInt32) {
      IndicesBoundCheck<int32_t>(reinterpret_cast<int32_t *>(x2_indices_tensor->data_c()),
                                 x2_indices_tensor->DataSize(), reinterpret_cast<int32_t *>(x2_shape_tensor->data_c()),
                                 "x2");
    } else {
      IndicesBoundCheck<int64_t>(reinterpret_cast<int64_t *>(x2_indices_tensor->data_c()),
                                 x2_indices_tensor->DataSize(), reinterpret_cast<int64_t *>(x2_shape_tensor->data_c()),
                                 "x2");
    }
  }
}

bool GetDtypeMinAndMaxAndCheckOverFlow(const TypePtr tid, int64_t compute_val) {
  int64_t min = 0;
  int64_t max = 0;
  switch (tid->type_id()) {
    case kNumberTypeUInt8:
      max = UCHAR_MAX;
      min = -UCHAR_MAX - 1;
      break;
    case kNumberTypeInt8:
      max = SCHAR_MAX;
      min = SCHAR_MIN;
      break;
    case kNumberTypeInt16:
      max = SHRT_MAX;
      min = SHRT_MIN;
      break;
    case kNumberTypeInt32:
      max = INT_MAX;
      min = INT_MIN;
      break;
    default:
      max = LONG_MAX;
      min = LONG_MIN;
      break;
  }
  if (compute_val <= min || compute_val > max) {
    return true;
  } else {
    return false;
  }
}

void PrintAlphaValueError(TypeId aid, const TypePtr expect_dtype, int64_t compute_val, float real, int64_t imag) {
  if (aid == kNumberTypeComplex64 || aid == kNumberTypeComplex128) {
    MS_EXCEPTION(ValueError) << "For Sspaddmm"
                             << ", alpha cannot be converted to expect dtype " << expect_dtype->ToString()
                             << ", without overflow: (" << real << ", " << imag << ").";
  } else {
    MS_EXCEPTION(ValueError) << "For Sspaddmm"
                             << ", alpha cannot be converted to expect x2_values dtype " << expect_dtype->ToString()
                             << ", without overflow: " << compute_val << ".";
  }
}

abstract::TupleShapePtr SspaddmmInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckInputTensor(input_args);
  CheckIndices(input_args);
  if (input_args[kInputIndex7]->isa<abstract::AbstractTensor>() &&
      input_args[kInputIndex7]->BuildValue()->isa<tensor::Tensor>()) {
    auto alpha_abstract = input_args[kInputIndex7]->cast<abstract::AbstractTensorPtr>();
    auto alpha_value_ptr = alpha_abstract->BuildValue();
    MS_EXCEPTION_IF_NULL(alpha_value_ptr);
    auto alpha_tensor = alpha_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(alpha_tensor);
    auto alpha_dtype = input_args[kInputIndex7]->BuildType();
    MS_EXCEPTION_IF_NULL(alpha_dtype);
    auto alpha_type_id = alpha_dtype->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(alpha_type_id);
    auto expect_dtype = input_args[kInputIndex1]->BuildType()->cast<TensorTypePtr>()->element();
    auto alpha_type_element = alpha_type_id->element();
    float real = 0;
    int32_t imag = 0;
    if (alpha_type_element->type_id() == kNumberTypeComplex64) {
      auto value = reinterpret_cast<std::complex<float> *>(alpha_tensor->data_c());
      real = value[0].real();
      imag = value[0].imag();
    } else if (alpha_type_element->type_id() == kNumberTypeComplex128) {
      auto value = reinterpret_cast<std::complex<double> *>(alpha_tensor->data_c());
      real = value[0].real();
      imag = value[0].imag();
    }
    if (imag != 0 || (expect_dtype->type_id() == kNumberTypeUInt8 && real < 0)) {
      MS_EXCEPTION(ValueError) << "For " << op_name
                               << ", alpha value cannot be converted to type uint8 , without overflow: (" << real
                               << ", " << imag << ").";
    }
    if (!(expect_dtype->type_id() == kNumberTypeFloat32 || expect_dtype->type_id() == kNumberTypeFloat64)) {
      int64_t compute_val =
        GetInt64AlphaData(alpha_tensor->data_c(), alpha_type_element->type_id(), expect_dtype, real);
      if (GetDtypeMinAndMaxAndCheckOverFlow(expect_dtype, compute_val)) {
        PrintAlphaValueError(alpha_type_element->type_id(), expect_dtype, compute_val, real, imag);
      }
    }
  }
  auto x1_indices_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto x3_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->BuildShape())[kShape];
  int64_t x2_indices_shape_right = -1;
  if (input_args[kInputIndex3]->isa<abstract::AbstractTensor>() &&
      input_args[kInputIndex3]->BuildValue()->isa<tensor::Tensor>() && !IsDynamic(x3_shape) &&
      !IsDynamic(x1_indices_shape)) {
    auto x2_indices_abstract = input_args[kInputIndex3]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(x2_indices_abstract);
    auto x2_indices_value_ptr = x2_indices_abstract->BuildValue();
    MS_EXCEPTION_IF_NULL(x2_indices_value_ptr);
    auto x2_indices_tensor = x2_indices_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(x2_indices_tensor);
    auto x2_indices_type = input_args[kInputIndex3]->BuildType();
    MS_EXCEPTION_IF_NULL(x2_indices_type);
    auto x2_indices_type_id = x2_indices_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(x2_indices_type_id);
    auto x2_indices_type_element = x2_indices_type_id->element();
    MS_EXCEPTION_IF_NULL(x2_indices_type_element);
    int64_t x2_indices_unique_size = 0;
    if (x2_indices_type_element->type_id() == kNumberTypeInt32) {
      x2_indices_unique_size = compute_output_indices_unique_size<int32_t>(
        reinterpret_cast<int32_t *>(x2_indices_tensor->data_c()), x2_indices_tensor->DataSize());
    } else {
      x2_indices_unique_size = compute_output_indices_unique_size<int64_t>(
        reinterpret_cast<int64_t *>(x2_indices_tensor->data_c()), x2_indices_tensor->DataSize());
    }
    x2_indices_shape_right = x2_indices_unique_size * x3_shape[1] + x1_indices_shape[1];
  }
  std::vector<int64_t> output_indices_shape = {2, x2_indices_shape_right};
  abstract::ShapePtr output_indices_shape_list = std::make_shared<abstract::Shape>(output_indices_shape);
  std::vector<int64_t> output_values_shape = {x2_indices_shape_right};
  abstract::ShapePtr output_values_shape_list = std::make_shared<abstract::Shape>(output_values_shape);
  auto input_shape = input_args[kInputIndex2]->BuildShape();
  abstract::ShapePtr output_shape_shape_list = input_shape->cast<abstract::ShapePtr>();
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{output_indices_shape_list, output_values_shape_list, output_shape_shape_list});
}

TuplePtr SspaddmmInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  std::map<std::string, TypePtr> x1_args = {{"x1_indices", input_args[kInputIndex0]->BuildType()},
                                            {"x1_shape", input_args[kInputIndex2]->BuildType()}};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(x1_args, {kInt32, kInt64}, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1_values", input_args[kInputIndex1]->BuildType(),
                                                   {kUInt8, kInt8, kInt16, kInt32, kInt64, kFloat32, kFloat64},
                                                   op_name);
  std::map<std::string, TypePtr> x2_args = {{"x2_indices", input_args[kInputIndex3]->BuildType()},
                                            {"x2_shape", input_args[kInputIndex5]->BuildType()}};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(x2_args, {kInt32, kInt64}, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2_values", input_args[kInputIndex4]->BuildType(),
                                                   {kUInt8, kInt8, kInt16, kInt32, kInt64, kFloat32, kFloat64},
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x3_dense", input_args[kInputIndex6]->BuildType(),
                                                   {kUInt8, kInt8, kInt16, kInt32, kInt64, kFloat32, kFloat64},
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid(
    "alpha", input_args[kInputIndex7]->BuildType(),
    {kUInt8, kUInt16, kUInt32, kUInt64, kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64}, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid(
    "beta", input_args[kInputIndex8]->BuildType(),
    {kUInt8, kUInt16, kUInt32, kUInt64, kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64}, op_name);
  auto expect_dtype = input_args[kInputIndex1]->BuildType()->cast<TensorTypePtr>()->element();
  auto beta_dtype = input_args[kInputIndex8]->BuildType()->cast<TensorTypePtr>()->element();
  if (!(expect_dtype->type_id() == kNumberTypeFloat32 || expect_dtype->type_id() == kNumberTypeFloat64)) {
    auto beta_dtype_id = beta_dtype->type_id();
    if (beta_dtype_id == kNumberTypeFloat16 || beta_dtype_id == kNumberTypeFloat32 ||
        beta_dtype_id == kNumberTypeFloat64) {
      MS_EXCEPTION(TypeError) << "For " << op_name << ",beta dtype: " << beta_dtype->ToString()
                              << " can't convert to the desired output type: " << expect_dtype->ToString() << ".";
    }
  }
  std::map<std::string, TypePtr> args = {{"x1_values", input_args[kInputIndex1]->BuildType()},
                                         {"x2_values", input_args[kInputIndex4]->BuildType()},
                                         {"x3_dense", input_args[kInputIndex6]->BuildType()}};
  auto output_values_type = CheckAndConvertUtils::CheckTensorTypeSame(
    args, {kInt8, kInt16, kInt32, kInt64, kUInt8, kFloat32, kFloat64}, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{kInt64, output_values_type, kInt64});
}
}  // namespace

AbstractBasePtr SspaddmmInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 9;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = SspaddmmInferType(primitive, input_args);
  auto infer_shape = SspaddmmInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Sspaddmm, BaseOperator);

// AG means auto generated
class MIND_API AGSspaddmmInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SspaddmmInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SspaddmmInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SspaddmmInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {0, 1, 2, 3, 4, 5, 7, 8}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Sspaddmm, prim::kPrimSspaddmm, AGSspaddmmInfer, false);
}  // namespace ops
}  // namespace mindspore
