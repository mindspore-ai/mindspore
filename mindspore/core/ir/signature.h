/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_SIGNATURE_H_
#define MINDSPORE_CORE_IR_SIGNATURE_H_

#include <string>
#include "ir/value.h"

namespace mindspore {
// Input signature, support type
enum SignatureEnumRW {
  // describe the arguments action on read and write
  kRWRead = 0,  // use the value of the input
  kRWWrite,     // use the key of the input
  kRWRef,       // use the ref of the input
  kRWEmptyDefaultValue,
  kRWDefault = kRWRead
};
enum SignatureEnumKind {
  kKindPositionalKeyword = 0,  // use value of the input start from this arg
  kKindVarPositional,          // use key of the input start from this arg
  kKindKeywordOnly,
  kKindVarKeyword,  // use ref of the input start from this arg
  kKindEmptyDefaultValue,
  kKindDefault = kKindPositionalKeyword
};
enum SignatureEnumDType {
  kDType = 0,
  kDType1,
  kDType2,
  kDType3,
  kDType4,
  kDType5,
  kDType6,
  kDType7,
  kDType8,
  kDType9,
  kDTypeEmptyDefaultValue
};
struct Signature {
  std::string name;
  SignatureEnumRW rw;
  SignatureEnumKind kind;
  ValuePtr default_value;  // nullptr for no default value
  SignatureEnumDType dtype;
  Signature(const std::string &arg_name, const SignatureEnumRW &rw_tag, const SignatureEnumKind &arg_kind,
            const ValuePtr &arg_default, const SignatureEnumDType &arg_dtype)
      : name(arg_name), rw(rw_tag), kind(arg_kind), default_value(arg_default), dtype(arg_dtype) {}
  Signature(const std::string &arg_name, const SignatureEnumRW &rw_tag, const SignatureEnumKind &arg_kind)
      : Signature(arg_name, rw_tag, arg_kind, nullptr, SignatureEnumDType::kDTypeEmptyDefaultValue) {}
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_SIGNATURE_H_
