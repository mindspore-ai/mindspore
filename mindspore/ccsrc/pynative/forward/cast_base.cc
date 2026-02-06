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

#include "pynative/forward/cast_base.h"
#include <memory>
#include <algorithm>
#include "primitive/array_ops.h"
#include "include/utils/python_adapter.h"
#include "include/frontend/operator/primitive_py.h"

namespace mindspore {
namespace pynative {
namespace {
const char kOpsFunctionModelName[] = "mindspore.ops.functional";
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
  const auto &cast_prim = python_adapter::GetPyFn(kOpsFunctionModelName, "_cast");
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
}  // namespace pynative
}  // namespace mindspore
