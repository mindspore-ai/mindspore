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

namespace mindspore {
namespace pijit {

class Opcode {
 public:
  static const Opcode k_ILLEGAL_OPCODE;
  static const Opcode &Map(int op) { return Map()[op > 0 && op < kMaxCode ? op : 0]; }

 private:
  static const Opcode *Map();

  static constexpr auto kMaxCode = 256;
  static Opcode opmap[kMaxCode];

 public:
  // opcode class
  enum Class : unsigned char {
    kStack = 0,       // stack operations
    kLocal,           // local access
    kGlobal,          // global access
    kCell,            // cell (types.CellType) access
    kItem,            // __getitem__, __setitem__, __delitem__
    kAttr,            // __getattr__, __setattr__, __delattr__
    kUnaryMath,       // +,-,not,~
    kBinaryMath,      // +,-,*,**,/,//,@,&,|,^,<<,>>,<,<=,>,>=,==,!=
    kContainerBuild,  // (),[],{a:b},{a},[:], build a new object
    kContainerMerge,  // with container modify
    kCall,            // with object call
    kControl,         // has jump
    kUnpack,          // a,b=c; a,b,*=c;
    kNop,             // generally, no operations
    kException,       // exception raise and handler, with syntax, try syntax
    kOther,
    kCount,
  };

  Opcode();
  Opcode(const char *name, int op, Class cls, int flag) : name_(name), code_(op), class_(cls), flag_(flag) {}
  explicit Opcode(int op) { *this = Map(op); }
  Opcode &operator=(int &&op) {
    *this = Map(op);
    return *this;
  }
  Opcode &operator=(const Opcode &) = default;
  bool operator==(int op) const { return code_ == op; }
  bool operator!=(int op) const { return code_ != op; }
  bool operator==(const Opcode &o) const { return code_ == o.code_; }
  bool operator!=(const Opcode &o) const { return code_ != o.code_; }
  const char *name() const { return name_; }
  int flag() const { return flag_; }

  bool IsJRel() const;
  bool IsJAbs() const;
  bool IsNotFall() const;
  bool HasName() const;
  bool HasFree() const;
  bool HasConst() const;
  bool CanDelete(int oparg = 0) const;
  bool MayDelete(int oparg = 0) const;

  bool IsCall() const { return class_ == Class::kCall; }
  bool IsBinaryMath() const { return class_ == Class::kBinaryMath; }
  bool IsUnaryMath() const { return class_ == Class::kUnaryMath; }
  bool IsCellAccess() const { return class_ == Class::kCell; }
  bool IsLocalAccess() const { return class_ == Class::kLocal; }

  // python3.9 explicit IS_OP from COMPARE_OP
  bool CheckIsOp(int oparg, bool *invert = nullptr) const;
  // python3.9 explicit CONTAINS_OP from COMPARE_OP
  bool CheckContainsOp(int oparg, bool *invert = nullptr) const;
  // python3.11 merge binary math opcode to BINARY_OP
  // CheckInplaceBinaryOp...

  Class GetClass() const { return class_; }

  bool HasArg() const;
  int JumpTarget(int pc, int off) const;
  int JumpOffset(int pc, int tar) const;

  constexpr operator int() const { return code_; }

 private:
  const char *name_;
  unsigned char code_;
  Class class_;
  int flag_;
};

}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_OPCODE_UTIL_H
