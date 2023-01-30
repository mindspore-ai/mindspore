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

#include "ops/fill.h"

#include <memory>
#include <set>
#include <string>
#include <vector>
#include <map>
#include <complex>

#include "abstract/abstract_value.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Fill, BaseOperator);
template <typename T>
static tensor::TensorPtr CreateValuedTensor(const TypePtr &type, const std::vector<int64_t> &shape, T num) {
  MS_EXCEPTION_IF_NULL(type);
  auto type_id = type->type_id();
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, shape);
  const size_t &mem_size = IntToSize(tensor->ElementsNum());
  auto tensor_data = tensor->data_c();
  std::map<TypeId, std::function<void()>> type_dict{
    {kNumberTypeBool,
     [&tensor_data, mem_size, &num]() { SetTensorData<bool>(tensor_data, static_cast<bool>(num), mem_size); }},
    {kNumberTypeInt8,
     [&tensor_data, mem_size, &num]() { SetTensorData<int8_t>(tensor_data, static_cast<int8_t>(num), mem_size); }},
    {kNumberTypeInt16,
     [&tensor_data, mem_size, &num]() { SetTensorData<int16_t>(tensor_data, static_cast<int16_t>(num), mem_size); }},
    {kNumberTypeInt32,
     [&tensor_data, mem_size, &num]() { SetTensorData<int32_t>(tensor_data, static_cast<int32_t>(num), mem_size); }},
    {kNumberTypeInt64,
     [&tensor_data, mem_size, &num]() { SetTensorData<int64_t>(tensor_data, static_cast<int64_t>(num), mem_size); }},
    {kNumberTypeUInt8,
     [&tensor_data, mem_size, &num]() { SetTensorData<uint8_t>(tensor_data, static_cast<uint8_t>(num), mem_size); }},
    {kNumberTypeUInt16,
     [&tensor_data, mem_size, &num]() { SetTensorData<uint16_t>(tensor_data, static_cast<uint16_t>(num), mem_size); }},
    {kNumberTypeUInt32,
     [&tensor_data, mem_size, &num]() { SetTensorData<uint32_t>(tensor_data, static_cast<uint32_t>(num), mem_size); }},
    {kNumberTypeUInt64,
     [&tensor_data, mem_size, &num]() { SetTensorData<uint64_t>(tensor_data, static_cast<uint64_t>(num), mem_size); }},
    {kNumberTypeFloat16,
     [&tensor_data, mem_size, &num]() { SetTensorData<float16>(tensor_data, static_cast<float16>(num), mem_size); }},
    {kNumberTypeFloat32,
     [&tensor_data, mem_size, &num]() { SetTensorData<float>(tensor_data, static_cast<float>(num), mem_size); }},
    {kNumberTypeFloat64,
     [&tensor_data, mem_size, &num]() { SetTensorData<double>(tensor_data, static_cast<double>(num), mem_size); }},
  };
  const auto &tensor_type = tensor->data_type();
  auto iter = type_dict.find(tensor_type);
  if (iter == type_dict.end()) {
    MS_LOG(EXCEPTION) << "unsupported data type: " << tensor_type;
  }
  iter->second();
  return tensor;
}

template <typename T>
static tensor::TensorPtr CreateComplexTensor(const TypePtr &type, const std::vector<int64_t> &shape, T num) {
  MS_EXCEPTION_IF_NULL(type);
  auto type_id = type->type_id();
  tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, shape);
  const size_t &mem_size = IntToSize(tensor->ElementsNum());
  auto tensor_data = tensor->data_c();
  std::map<TypeId, std::function<void()>> type_dict{
    {kNumberTypeComplex64,
     [&tensor_data, mem_size, &num]() {
       SetTensorData<std::complex<float>>(tensor_data, static_cast<std::complex<float>>(num), mem_size);
     }},
    {kNumberTypeComplex128,
     [&tensor_data, mem_size, &num]() {
       SetTensorData<std::complex<double>>(tensor_data, static_cast<std::complex<double>>(num), mem_size);
     }},
  };
  const auto &tensor_type = tensor->data_type();
  auto iter = type_dict.find(tensor_type);
  if (iter == type_dict.end()) {
    MS_LOG(EXCEPTION) << "unsupported data type: " << tensor_type;
  }
  iter->second();
  return tensor;
}

BaseShapePtr FillInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::vector<size_t> inputsIndex{kIndex0, kIndex1, kIndex2};
  if (input_args.size() == kIndex2) {
    inputsIndex[kIndex1] = kIndex0;
    inputsIndex[kIndex2] = kIndex1;
  }
  const auto &prim_name = primitive->name();
  if (input_args[inputsIndex[kIndex1]]->isa<abstract::AbstractTuple>()) {
    auto out_shape = GetShapeValue(primitive, input_args[inputsIndex[kIndex1]]);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  if (!input_args[inputsIndex[kIndex1]]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name() << "', input[1] must be tensor.";
  }
  MS_EXCEPTION_IF_NULL(primitive);
  const uint32_t kInputDims = 1;
  auto input1_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[inputsIndex[kIndex1]]->BuildShape())[kShape];
  if (input1_shape.size() != kInputDims) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "', the shape size of 'input1' must be 1, but got: " << input1_shape.size() << ".";
  }
  auto input2_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[inputsIndex[kIndex2]]->BuildShape())[kShape];
  if (input2_shape.size() > 1 || (input2_shape.size() == 1 && input2_shape[0] > 1)) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "', the shape size of 'input2' must be 0, but got: " << input2_shape.size() << ".";
  }

  auto shape_arg = input_args[inputsIndex[1]];
  MS_EXCEPTION_IF_NULL(shape_arg);
  auto output_shape = GetShapeValue(primitive, shape_arg);
  if (input_args.size() == kIndex2) {
    auto input_value = input_args[0]->BuildValue();
    output_shape = CheckAndConvertUtils::CheckTensorIntValue("axis", input_value, prim_name);
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr FillInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  std::vector<size_t> inputsIndex{0, 1, 2};
  if (input_args.size() == kIndex2) {
    inputsIndex[kIndex1] = 0;
    inputsIndex[kIndex2] = 1;
  }
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();
  // check
  ValuePtr dtype_value;
  TypePtr value_dtype;
  auto input2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  auto input2_dtype = input_args[2]->BuildType();
  TypePtr input2_element_dtype;
  if (input2_dtype->isa<TensorType>()) {
    auto tensor_type = input2_dtype->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_type);
    input2_element_dtype = tensor_type->element();
  } else {
    input2_element_dtype = input2_dtype;
  }
  if (input2_shape.size() > 1 || (input2_shape.size() == 1 && input2_shape[0] > 1)) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the value input only takes scalar or scalar within a tensor!";
  }
  dtype_value = input_args[0]->BuildValue();
  MS_EXCEPTION_IF_NULL(dtype_value);
  if (!dtype_value->isa<Type>()) {
    MS_EXCEPTION(TypeError)
      << "For '" << prim_name
      << "', the supported data type is ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16','uint32', "
         "'uint64','float16', 'float32', 'float64'], but got an invalid dtype!";
  }
  auto output_dtype = dtype_value->cast<TypePtr>();

  const std::set<TypePtr> valid_types = {kBool,   kInt8,   kInt16,   kInt32,   kInt64,   kUInt8,     kUInt16,
                                         kUInt32, kUInt64, kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  CheckAndConvertUtils::CheckSubClass("dtype", input2_element_dtype, valid_types, prim_name);
  return CheckAndConvertUtils::CheckSubClass("dtype", output_dtype, valid_types, prim_name);
}

AbstractBasePtr FillInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  auto type = FillInferType(primitive, input_args);
  auto shape = FillInferShape(primitive, input_args);
  return MakeAbstract(shape, type);
}

ValuePtr FillInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, prim->name());
  auto infered_type = FillInferType(prim, input_args);
  MS_EXCEPTION_IF_NULL(infered_type);
  auto input_value_ptr = input_args[2]->BuildValue();
  auto input_value_type_id = input_args[2]->BuildType()->type_id();
  auto tmp_shape = FillInferShape(prim, input_args);
  if (tmp_shape->IsDynamic()) {
    return kAnyValue;
  }
  MS_EXCEPTION_IF_NULL(tmp_shape);
  auto infered_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(tmp_shape)[kShape];
  auto input_value_tensor = input_value_ptr->cast<tensor::TensorPtr>();
  tensor::TensorPtr infer_result;
  if (input_value_type_id == kNumberTypeBool) {
    infer_result = CreateValuedTensor<bool>(infered_type, infered_shape, GetValue<bool>(input_value_ptr));
  } else if (input_value_type_id == kNumberTypeFloat32) {
    infer_result = CreateValuedTensor<float>(infered_type, infered_shape, GetValue<float>(input_value_ptr));
  } else if (input_value_type_id == kNumberTypeInt32) {
    infer_result = CreateValuedTensor<int32_t>(infered_type, infered_shape, GetValue<int32_t>(input_value_ptr));
  } else if (input_value_type_id == kNumberTypeInt64) {
    infer_result = CreateValuedTensor<int64_t>(infered_type, infered_shape, GetValue<int64_t>(input_value_ptr));
  } else if (input_value_type_id == kNumberTypeComplex64) {
    infer_result = CreateComplexTensor<std::complex<float>>(
      infered_type, infered_shape, static_cast<std::complex<float> *>(input_value_tensor->data_c())[0]);
  } else if (input_value_type_id == kNumberTypeComplex128) {
    infer_result = CreateComplexTensor<std::complex<double>>(
      infered_type, infered_shape, static_cast<std::complex<double> *>(input_value_tensor->data_c())[0]);
  }
  return infer_result;
}
REGISTER_PRIMITIVE_EVAL_IMPL(Fill, prim::kPrimFill, FillInfer, FillInferValue, false);
}  // namespace ops
}  // namespace mindspore
