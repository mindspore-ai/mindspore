/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "ir/signature.h"
#include "pybind11/operators.h"
#include "include/common/pybind_api/api_register.h"
#include "pipeline/jit/parse/data_converter.h"

namespace py = pybind11;

namespace mindspore {
static ValuePtr PyArgToValue(const py::object &arg) {
  if (py::isinstance<SignatureEnumKind>(arg) &&
      py::cast<SignatureEnumKind>(arg) == SignatureEnumKind::kKindEmptyDefaultValue) {
    return nullptr;
  }
  return parse::data_converter::PyDataToValue(arg);
}
// Bind SignatureEnumRW as a python class.
void RegSignatureEnumRW(const py::module *m) {
  (void)py::class_<Signature>(*m, "Signature")
    .def(py::init([](const std::string &name, SignatureEnumRW rw, SignatureEnumKind kind, const py::object arg_default,
                     SignatureEnumDType dtype) {
      auto default_value = PyArgToValue(arg_default);
      return Signature(name, rw, kind, default_value, dtype);
    }));
  (void)py::enum_<SignatureEnumRW>(*m, "signature_rw", py::arithmetic())
    .value("RW_READ", SignatureEnumRW::kRWRead)
    .value("RW_WRITE", SignatureEnumRW::kRWWrite)
    .value("RW_REF", SignatureEnumRW::kRWRef)
    .value("RW_EMPTY_DEFAULT_VALUE", SignatureEnumRW::kRWEmptyDefaultValue);
  (void)py::enum_<SignatureEnumKind>(*m, "signature_kind", py::arithmetic())
    .value("KIND_POSITIONAL_KEYWORD", SignatureEnumKind::kKindPositionalKeyword)
    .value("KIND_VAR_POSITIONAL", SignatureEnumKind::kKindVarPositional)
    .value("KIND_KEYWORD_ONLY", SignatureEnumKind::kKindKeywordOnly)
    .value("KIND_VAR_KEYWARD", SignatureEnumKind::kKindVarKeyword)
    .value("KIND_EMPTY_DEFAULT_VALUE", SignatureEnumKind::kKindEmptyDefaultValue);
  (void)py::enum_<SignatureEnumDType>(*m, "signature_dtype", py::arithmetic())
    .value("T", SignatureEnumDType::kDType)
    .value("T1", SignatureEnumDType::kDType1)
    .value("T2", SignatureEnumDType::kDType2)
    .value("T3", SignatureEnumDType::kDType3)
    .value("T4", SignatureEnumDType::kDType4)
    .value("T5", SignatureEnumDType::kDType5)
    .value("T6", SignatureEnumDType::kDType6)
    .value("T7", SignatureEnumDType::kDType7)
    .value("T8", SignatureEnumDType::kDType8)
    .value("T9", SignatureEnumDType::kDType9)
    .value("T_EMPTY_DEFAULT_VALUE", SignatureEnumDType::kDTypeEmptyDefaultValue);
}
}  // namespace mindspore
