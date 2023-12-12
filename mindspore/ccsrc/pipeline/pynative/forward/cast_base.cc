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
#include <utility>
#include <algorithm>
#include "ops/array_ops.h"
#include "frontend/operator/composite/do_signature.h"

namespace mindspore {
namespace pynative {
namespace {
const char kOpsFunctionModelName[] = "mindspore.ops.functional";
}
PrimitivePtr CastBaseOperation::GetPrimByTypeId(const TypeId &type_id) const {
  const auto &iter = type_prim_cache_.find(type_id);
  if (iter != type_prim_cache_.end()) {
    return iter->second;
  }

#ifndef ENABLE_TEST
  auto cast_prim = std::make_shared<Primitive>(kCastOpName);
  std::vector<std::string> input_names = {"x", "dst_type"};
  std::vector<std::string> output_names = {"output"};
  cast_prim->AddAttr("input_names", MakeValue(input_names));
  cast_prim->AddAttr("output_names", MakeValue(output_names));
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

ValuePtr CastBaseOperation::GetDstTypeValue(const TypeId &type_id) const {
  constexpr int k8Bits = 8;
  constexpr int k16Bits = 16;
  constexpr int k32Bits = 32;
  constexpr int k64Bits = 64;
  ValuePtr value = nullptr;
  if (type_id == kNumberTypeFloat16) {
    value = std::make_shared<Float>(k16Bits);
  } else if (type_id == kNumberTypeFloat32) {
    value = std::make_shared<Float>(k32Bits);
  } else if (type_id == kNumberTypeFloat64) {
    value = std::make_shared<Float>(k64Bits);
  } else if (type_id == kNumberTypeBFloat16) {
    value = std::make_shared<BFloat>(k16Bits);
  } else if (type_id == kNumberTypeBool) {
    value = std::make_shared<Bool>();
  } else if (type_id == kNumberTypeInt8) {
    value = std::make_shared<Int>(k8Bits);
  } else if (type_id == kNumberTypeUInt8) {
    value = std::make_shared<UInt>(k8Bits);
  } else if (type_id == kNumberTypeInt16) {
    value = std::make_shared<Int>(k16Bits);
  } else if (type_id == kNumberTypeInt32) {
    value = std::make_shared<Int>(k32Bits);
  } else if (type_id == kNumberTypeInt64) {
    value = std::make_shared<Int>(k64Bits);
  } else {
    MS_LOG(EXCEPTION) << "Not support dst type " << type_id;
  }
  MS_EXCEPTION_IF_NULL(value);
  return value;
}

void CastBaseOperation::GetTypeIndex(const std::vector<SignatureEnumDType> &dtypes,
                                     mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> *type_indexes) const {
  MS_EXCEPTION_IF_NULL(type_indexes);
  for (size_t i = 0; i < dtypes.size(); ++i) {
    auto it = type_indexes->find(dtypes[i]);
    if (it == type_indexes->end()) {
      (void)type_indexes->emplace(std::make_pair(dtypes[i], std::vector<size_t>{i}));
    } else {
      (void)it->second.emplace_back(i);
    }
  }
}

void CastBaseOperation::GetDstType(const FrontendOpRunInfoPtr &op_run_info,
                                   const mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> &type_indexes,
                                   mindspore::HashMap<SignatureEnumDType, TypeId> *dst_type) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  constexpr size_t index_size = 2;
  for (auto it = type_indexes.begin(); it != type_indexes.end(); (void)++it) {
    const auto &type = it->first;
    const auto &indexes = it->second;
    if (type == SignatureEnumDType::kDTypeEmptyDefaultValue || indexes.size() < index_size) {
      continue;
    }
    int64_t priority = INT_MIN;
    TypeId max_type = TypeId::kTypeUnknown;
    bool has_scalar_float32 = false;
    bool has_scalar_int64 = false;
    bool has_tensor_int8 = false;
    // Find the maximum priority of the same dtype
    for (size_t index : indexes) {
      const auto &v = op_run_info->op_grad_info->input_value[index];
      if (v->isa<FloatImm>()) {
        has_scalar_float32 = true;
      }
      if ((!v->isa<BoolImm>() && v->isa<IntegerImm>())) {
        has_scalar_int64 = true;
      }
      if (v->isa<tensor::Tensor>()) {
        auto arg = v->cast<tensor::TensorPtr>();
        TypeId arg_type_id = arg->data_type();
        auto type_priority = prim::type_map.find(arg_type_id);
        if (type_priority == prim::type_map.end()) {
          continue;
        }
        if (arg_type_id == kNumberTypeInt8) {
          has_tensor_int8 = true;
        }
        int64_t cur_priority = type_priority->second;
        if (op_run_info->source_type[index] != ops::OP_DTYPE::DT_BEGIN) {
          cur_priority = cur_priority - kLowerPriority;
          if (arg_type_id == kNumberTypeFloat32) {
            has_scalar_float32 = true;
          }
          if (arg_type_id == kNumberTypeInt32 || arg_type_id == kNumberTypeInt64) {
            has_scalar_int64 = true;
          }
        }
        if (cur_priority > priority) {
          max_type = type_priority->first;
          priority = cur_priority;
        }
      }
    }
    max_type = JudgeMaxType(max_type, has_scalar_float32, has_scalar_int64, has_tensor_int8);
    MS_EXCEPTION_IF_NULL(dst_type);
    (void)dst_type->emplace(std::make_pair(type, max_type));
  }
}

TypeId CastBaseOperation::JudgeMaxType(TypeId max_type, bool has_scalar_float32, bool has_scalar_int64,
                                       bool has_tensor_int8) const {
  if (max_type == TypeId::kNumberTypeBool) {
    if (has_scalar_int64) {
      max_type = TypeId::kNumberTypeInt64;
    }
    if (has_scalar_float32) {
      max_type = TypeId::kNumberTypeFloat32;
    }
  }
  if (max_type != TypeId::kNumberTypeFloat16 && max_type != TypeId::kNumberTypeFloat32 &&
      max_type != TypeId::kNumberTypeFloat64 && max_type != TypeId::kTypeUnknown && has_scalar_float32) {
    max_type = TypeId::kNumberTypeFloat32;
  }
  if (max_type == TypeId::kNumberTypeUInt8 && has_tensor_int8) {
    max_type = TypeId::kNumberTypeInt16;
  }
  return max_type;
}

const std::string &CastBaseOperation::TypeIdToMsTypeStr(const TypeId &type_id) const {
  const auto &type_name = type_name_map().find(type_id);
  if (type_name == type_name_map().cend()) {
    MS_LOG(EXCEPTION) << "For implicit type conversion, not support convert to the type: " << TypeIdToType(type_id);
  }
  return type_name->second;
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

tensor::TensorPtr CastBaseOperation::TensorToDstDtypeValue(const ValuePtr &src_value, const TypeId &dst_type_id) const {
  MS_EXCEPTION_IF_NULL(src_value);
  auto src_tensor = src_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(src_tensor);
  src_tensor->set_data_type(dst_type_id);
  return src_tensor;
}
}  // namespace pynative
}  // namespace mindspore
