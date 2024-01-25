/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_PI_JIT_OPCODE_UTIL_H
#define MINDSPORE_PI_JIT_OPCODE_UTIL_H

#include <string>

namespace mindspore {
namespace pijit {
namespace code {
enum OpcodeFlag {
  kNamed = 1 << 0,        // has co_names
  kJRel = 1 << 1,         // is jump relative
  kJAbs = 1 << 2,         // is jump absolute
  kCall = 1 << 3,         // call function opcode
  kCellAccess = 1 << 4,   // cell and free access operations
  kLocalAccess = 1 << 5,  // local variable access operations
  kNoFall = 1 << 6,       // jump or return or raise operation
  /**
   * Generally or literally, no object is modified, only return new object.
   * But it maybe changed because of user-defined function
   * this flag used to optimize local variable
   */
  kGeneralNoSideEffect = 1 << 7,
  // can't modify any object
  kNoSideEffect = 1 << 8,
  kLoad = 1 << 9,
  kMsUnsupported = 1 << 10,  // mindspore support operation
  kBinaryMath = 1 << 11,     //+,-,*,**,/,//,@,&,|,^,<<,>>
};

struct Opcode {
  std::string name_ = "unknown opcode";
  int flag_ = 0;
  int code_ = 0;
  bool operator==(const Opcode &o) { return code_ == o.code_; }
  bool operator!=(const Opcode &o) { return code_ != o.code_; }
  constexpr operator int() const { return code_; }
};

const Opcode &GetOpcodeInfo(int opcode);
}  // namespace code
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_OPCODE_UTIL_H
