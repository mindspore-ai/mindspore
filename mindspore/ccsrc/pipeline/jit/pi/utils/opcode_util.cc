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
#include "pipeline/jit/pi/utils/opcode_util.h"
#include "pipeline/jit/pi/pydef.h"

namespace mindspore {
namespace pijit {

Opcode Opcode::opmap[Opcode::kMaxCode];
const Opcode Opcode::k_ILLEGAL_OPCODE = {"ILLEGAL_OPCODE", ILLEGAL_OPCODE, Opcode::Class::kOther, 0};

enum OpcodeFlag {
  kJRel = 1 << 0,      // is jump relative
  kJAbs = 1 << 1,      // is jump relative
  kNotFall = 1 << 2,   // jump directly, return, raise
  kHasConst = 1 << 3,  // has const in co_consts
  kHasName = 1 << 4,   // has name in co_names
  kHasFree = 1 << 5,   // has free variable operations, not is 'free' of this function
  kCanDel = 1 << 6,    // can be remove if result is unused
  /**
   * Maybe remove if result is unused.
   * Generally or literally, it's no side effect, check it and parse
   * all user-defined operation to call function while graph building
   */
  kMayDel = 1 << 7,
};

Opcode::Opcode() { *this = k_ILLEGAL_OPCODE; }

bool Opcode::IsJRel() const { return flag_ & kJRel; }
bool Opcode::IsJAbs() const { return flag_ & kJAbs; }
bool Opcode::IsNotFall() const { return flag_ & kNotFall; }
bool Opcode::HasName() const { return flag_ & kHasName; }
bool Opcode::HasFree() const { return flag_ & kHasFree; }
bool Opcode::HasConst() const { return flag_ & kHasConst; }
bool Opcode::CanDelete(int oparg) const { return (flag_ & kCanDel) || CheckIsOp(oparg); }
bool Opcode::MayDelete(int oparg) const { return (flag_ & kMayDel) || CanDelete(oparg); }
bool Opcode::CheckIsOp(int oparg, bool *invert) const {
#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9)
  if (invert != nullptr) {
    *invert = oparg == PyCmp_IS_NOT;
  }
  return code_ == COMPARE_OP ? oparg == PyCmp_IS : false;
#else
  if (invert != nullptr) {
    *invert = oparg;
  }
  return code_ == IS_OP;
#endif
}
bool Opcode::CheckContainsOp(int oparg, bool *invert) const {
#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9)
  if (invert != nullptr) {
    *invert = oparg == PyCmp_NOT_IN;
  }
  return code_ == COMPARE_OP ? oparg == PyCmp_IN : false;
#else
  if (invert != nullptr) {
    *invert = oparg;
  }
  return code_ == CONTAINS_OP;
#endif
}

bool Opcode::HasArg() const { return HAS_ARG(code_); }

const Opcode *Opcode::Map() {
  static bool init = false;
  if (init) {
    return opmap;
  }
  init = true;

#define DEF_OPCODE(name, cls, flag) \
  opmap[(name)] = (name) == ILLEGAL_OPCODE ? Opcode::k_ILLEGAL_OPCODE : Opcode(#name, (name), (cls), (flag));

#include "./opcode_attr.def"
#undef DEF_OPCODE

  return opmap;
}

int Opcode::JumpTarget(int pc, int off) const {
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 10)
  constexpr int mul = sizeof(_Py_CODEUNIT);
#else
  constexpr int mul = 1;
#endif
  if (IsJRel()) {
    return pc + 1 + off / mul;
  }
  if (IsJAbs()) {
    return off / mul;
  }
  return -1;
}
int Opcode::JumpOffset(int pc, int tar) const {
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 10)
  constexpr int mul = sizeof(_Py_CODEUNIT);
#else
  constexpr int mul = 1;
#endif
  if (IsJRel()) {
    return (tar - pc - 1) * mul;
  }
  if (IsJAbs()) {
    return tar * mul;
  }
  return -1;
}

}  // namespace pijit
}  // namespace mindspore
