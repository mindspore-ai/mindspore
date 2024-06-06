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

#include "pipeline/pynative/forward/cast_base.h"
#include <memory>
#include <algorithm>
#include "ops/array_ops.h"
#include "frontend/operator/composite/do_signature.h"

namespace mindspore {
namespace pynative {
namespace {
const char kOpsFunctionModelName[] = "mindspore.ops.functional";

template <typename S>
ValuePtr CastScalarToScalar(S in, const TypeId &type_id) {
  switch (type_id) {
    case kNumberTypeInt32:
      return MakeValue(static_cast<int>(in));
    case kNumberTypeFloat16:
      return MakeValue(static_cast<float16>(in).int_value());
    case kNumberTypeFloat32:
      return MakeValue(static_cast<float>(in));
    case kNumberTypeBool:
      return MakeValue(static_cast<bool>(in));
    case kNumberTypeInt64:
      return MakeValue(static_cast<int64_t>(in));
    case kNumberTypeFloat64:
      return MakeValue(static_cast<double>(in));
    case kNumberTypeInt16:
      return MakeValue(static_cast<int16_t>(in));
    case kNumberTypeInt8:
      return MakeValue(static_cast<int8_t>(in));
    case kNumberTypeUInt64:
      return MakeValue(static_cast<uint64_t>(in));
    case kNumberTypeUInt32:
      return MakeValue(static_cast<uint32_t>(in));
    case kNumberTypeUInt16:
      return MakeValue(static_cast<uint16_t>(in));
    case kNumberTypeUInt8:
      return MakeValue(static_cast<uint8_t>(in));
    case kNumberTypeBFloat16:
      return MakeValue(static_cast<float16>(in).int_value());
    default:
      MS_LOG(DEBUG) << "Not support cast to dst type: " << TypeIdToType(type_id)->ToString();
      return nullptr;
  }
}

template <typename S>
ValuePtr CastScalarToTensor(S in, const TypeId &type_id) {
  switch (type_id) {
    case kNumberTypeInt32:
      return std::make_shared<tensor::Tensor>(static_cast<int>(in), kInt32);
    case kNumberTypeFloat16:
      return std::make_shared<tensor::Tensor>(static_cast<float16>(in), kFloat16);
    case kNumberTypeFloat32:
      return std::make_shared<tensor::Tensor>(static_cast<float>(in), kFloat32);
    case kNumberTypeBool:
      return std::make_shared<tensor::Tensor>(static_cast<bool>(in), kBool);
    case kNumberTypeInt64:
      return std::make_shared<tensor::Tensor>(static_cast<int64_t>(in), kInt64);
    case kNumberTypeFloat64:
      return std::make_shared<tensor::Tensor>(static_cast<double>(in), kFloat64);
    case kNumberTypeInt16:
      return std::make_shared<tensor::Tensor>(static_cast<int16_t>(in), kInt16);
    case kNumberTypeInt8:
      return std::make_shared<tensor::Tensor>(static_cast<int8_t>(in), kInt8);
    case kNumberTypeUInt64:
      return std::make_shared<tensor::Tensor>(static_cast<uint64_t>(in), kUInt64);
    case kNumberTypeUInt32:
      return std::make_shared<tensor::Tensor>(static_cast<uint32_t>(in), kUInt32);
    case kNumberTypeUInt16:
      return std::make_shared<tensor::Tensor>(static_cast<uint16_t>(in), kUInt16);
    case kNumberTypeUInt8:
      return std::make_shared<tensor::Tensor>(static_cast<uint8_t>(in), kUInt8);
    case kNumberTypeBFloat16:
      return std::make_shared<tensor::Tensor>(static_cast<bfloat16>(in), kBFloat16);
    default:
      MS_LOG(DEBUG) << "Not support cast to dst type: " << TypeIdToType(type_id)->ToString();
      return nullptr;
  }
}

template <typename S>
ValuePtr Cast(S in, const std::pair<TypeId, bool> &dst_type) {
  bool has_tensor_input = dst_type.second;
  if (has_tensor_input) {
    return CastScalarToTensor(in, dst_type.first);
  }
  return CastScalarToScalar(in, dst_type.first);
}
}  // namespace

PrimitivePtr CastBaseOperation::GetPrimByTypeId(const TypeId &type_id) const {
  const auto &iter = type_prim_cache_.find(type_id);
  if (iter != type_prim_cache_.end()) {
    return iter->second;
  }

#ifndef ENABLE_TEST
  auto cast_prim = std::make_shared<Primitive>(kCastOpName);
  std::vector<std::string> input_names = {"x", "dst_type"};
  std::vector<std::string> output_names = {"output"};
  (void)cast_prim->AddAttr("input_names", MakeValue(input_names));
  (void)cast_prim->AddAttr("output_names", MakeValue(output_names));
  type_prim_cache_[type_id] = cast_prim;
  cast_prim->EnableSharedMutex();
  return cast_prim;
#else
  py::gil_scoped_acquire gil;
  const auto &cast_prim = python_adapter::GetPyFn(kOpsFunctionModelName, "cast");
  auto prim_adapter = cast_prim.cast<PrimitivePyAdapterPtr>();
  MS_EXCEPTION_IF_NULL(prim_adapter);
  auto primitive = prim_adapter->attached_primitive();
  if (primitive == nullptr) {
    primitive = std::make_shared<PrimitivePy>(cast_prim);
    prim_adapter->set_attached_primitive(primitive);
  }
  if (!primitive->HasPyObj()) {
    MS_LOG(EXCEPTION) << "Pyobj is empty";
  }
  type_prim_cache_[type_id] = primitive;
  primitive->EnableSharedMutex();
  return primitive;
#endif
}

bool CastBaseOperation::GetSignatureType(const std::vector<Signature> &signatures,
                                         std::vector<SignatureEnumDType> *dtypes) const {
  MS_EXCEPTION_IF_NULL(dtypes);
  bool has_sig_dtype = false;
  (void)std::transform(signatures.begin(), signatures.end(), std::back_inserter(*dtypes),
                       [&has_sig_dtype](const Signature &sig) {
                         auto dtype = sig.dtype;
                         if (dtype != SignatureEnumDType::kDTypeEmptyDefaultValue) {
                           has_sig_dtype = true;
                         }
                         return dtype;
                       });
  return has_sig_dtype;
}

tensor::BaseTensorPtr CastBaseOperation::TensorToDstDtypeValue(const ValuePtr &src_value,
                                                               const TypeId &dst_type_id) const {
  MS_EXCEPTION_IF_NULL(src_value);
  auto src_tensor = src_value->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(src_tensor);
  (void)src_tensor->set_data_type(dst_type_id);
  return src_tensor;
}

// This function is used to convert scalar value to another scalar value with destination data type.
// The scope of scalar type includes common data types, such as `FP64`, `FP32`, `FP16, `Int64`, `Int32`, ...
// The following sort is based on the hot spots of the data type.
ValuePtr CastBaseOperation::ScalarToDstDtypeValue(const ValuePtr &src_value,
                                                  const std::pair<TypeId, bool> &dst_type) const {
  MS_EXCEPTION_IF_NULL(src_value);
  // Tensor not do scalar cast
  if (src_value->isa<tensor::BaseTensor>()) {
    return nullptr;
  } else if (src_value->isa<Int64Imm>()) {
    const auto &int64_v = src_value->cast<Int64ImmPtr>();
    return Cast<int64_t>(int64_v->value(), dst_type);
  } else if (src_value->isa<FP32Imm>()) {
    const auto &fp32_v = src_value->cast<FP32ImmPtr>();
    return Cast<float>(fp32_v->value(), dst_type);
  } else if (src_value->isa<Int32Imm>()) {
    const auto &int32_v = src_value->cast<Int32ImmPtr>();
    return Cast<int32_t>(int32_v->value(), dst_type);
  } else if (src_value->isa<FP64Imm>()) {
    const auto &fp64_v = src_value->cast<FP64ImmPtr>();
    return Cast<double>(fp64_v->value(), dst_type);
  } else if (src_value->isa<BoolImm>()) {
    const auto &bool_v = src_value->cast<BoolImmPtr>();
    return Cast<bool>(bool_v->value(), dst_type);
  } else if (src_value->isa<Int16Imm>()) {
    const auto &int16_v = src_value->cast<Int16ImmPtr>();
    return Cast<int16_t>(int16_v->value(), dst_type);
  } else {
    MS_LOG(DEBUG) << "Now, the value [" << src_value->ToString() << "] is not supported to cast directly.";
    return nullptr;
  }
}
}  // namespace pynative
}  // namespace mindspore
