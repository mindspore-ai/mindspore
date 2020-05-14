/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CCSRC_VM_VM_H_
#define MINDSPORE_CCSRC_VM_VM_H_

#include <map>
#include <memory>
#include <stack>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <deque>
#include <unordered_map>

#include "ir/anf.h"
#include "utils/base_ref.h"

namespace mindspore {
namespace compile {

class Backend;
using BackendPtr = std::shared_ptr<Backend>;

enum Instruction {
  kCall = 0,
  kTailCall,
  kReturn,
  kPartial,
  kSwitch,
  kSwitchReturn,
  kTuple,
  kInput,
  kExternal,
  kPush,
  kPrim,
  kGraph,
  kPadStack
};

using InstType = std::pair<Instruction, VectorRef>;
using InstSet = std::vector<InstType>;
using InstFunctionMap = std::map<Instruction, std::function<void(const VectorRef &)>>;

const std::vector<std::string> inst_str{"call",  "tail_call", "return", "partial",   "switch", "switch_return", "tuple",
                                        "input", "external",  "push",   "primitive", "graph",  "pad_stack"};
class StructPartial : public Base {
 public:
  // Initialize StructPartial.
  StructPartial(int fn, const VectorRef &args, const FuncGraphPtr &fg = nullptr);

  virtual ~StructPartial() = default;
  MS_DECLARE_PARENT(StructPartial, Base)

  int fn_;
  VectorRef args_;
  FuncGraphPtr fg_;
};

std::ostream &operator<<(std::ostream &os, const StructPartial &other);
bool operator==(const StructPartial &lhs, const StructPartial &rhs);

class StructSimuSwitch : public Base {
 public:
  StructSimuSwitch(const BaseRef &fn, const BaseRef &value);

  virtual ~StructSimuSwitch() = default;
  MS_DECLARE_PARENT(StructSimuSwitch, Base)

  BaseRef fn_;
  BaseRef value_;
};

std::ostream &operator<<(std::ostream &os, const StructSimuSwitch &other);
bool operator==(const StructSimuSwitch &lhs, const StructSimuSwitch &rhs);

class FinalVM {
 public:
  // Create a VM with the specified instructions and backend.
  explicit FinalVM(const InstSet &insts, const BackendPtr &backend);

  virtual ~FinalVM() = default;

  BaseRef Eval(const VectorRef &args);
  void InstCall(const VectorRef &args);
  void InstTailCall(const VectorRef &args);
  void InstReturn(const VectorRef &args);
  void InstPartial(const VectorRef &args);
  void InstSimuPartial(const VectorRef &args);
  void InstRealPartial(const VectorRef &args);
  void InstSwitch(const VectorRef &args);
  void InstSimuSwitch(const VectorRef &args);
  void InstRealSwitch(const VectorRef &args);
  void InstTuple(const VectorRef &args);
  void InstPush(const VectorRef &args);
  void InstInput(const VectorRef &args);
  void InstPadStack(const VectorRef &args);
  void InstExternal(const VectorRef &args);
  void InstPushPrim(const VectorRef &args);
  void InstSwitchReturn(const VectorRef &args);
  void set_insts(const InstSet &value) { insts_ = value; }
  BaseRef RunHook(const PrimitivePtr &prim, const VectorRef &args);

 protected:
  BaseRef Ref(int i);
  void Push(const BaseRef &v);
  void Pop(int n = 1);
  void MoveStack(int nitems, int height);
  void Pushp();
  void Popp();
  void Pushsp();
  void Popsp();
  void PushStatus(bool is_switch_call);
  bool PopStatus();
  void DoJmp(const BaseRef &jmp);
  void MergeJmpArgs(const BaseRef &jmp, const BaseRef &c);
  BaseRef MergeArgs(const BaseRef &first, const BaseRef &second);

 private:
  InstSet insts_;
  std::deque<BaseRef> insts_stack_;
  std::stack<int> retp_;
  std::stack<int> retsp_;
  std::stack<bool> ret_status_;
  int pc_;
  int sp_;
  std::unordered_map<BaseRef, BaseRef, BaseRefHash> cond_jmp_;
  std::unordered_map<BaseRef, BaseRef, BaseRefHash> cond_out_;
  BackendPtr backend_;
  const InstFunctionMap inst_function_map = {
    {Instruction::kCall, [this](const VectorRef &args) { InstCall(args); }},
    {Instruction::kTailCall, [this](const VectorRef &args) { InstTailCall(args); }},
    {Instruction::kReturn, [this](const VectorRef &args) { InstReturn(args); }},
    {Instruction::kPartial, [this](const VectorRef &args) { InstPartial(args); }},
    {Instruction::kSwitch, [this](const VectorRef &args) { InstSwitch(args); }},
    {Instruction::kTuple, [this](const VectorRef &args) { InstTuple(args); }},
    {Instruction::kPush, [this](const VectorRef &args) { InstPush(args); }},
    {Instruction::kInput, [this](const VectorRef &args) { InstInput(args); }},
    {Instruction::kPadStack, [this](const VectorRef &args) { InstPadStack(args); }},
    {Instruction::kExternal, [this](const VectorRef &args) { InstExternal(args); }},
    {Instruction::kPrim, [this](const VectorRef &args) { InstPushPrim(args); }},
    {Instruction::kSwitchReturn, [this](const VectorRef &args) { InstSwitchReturn(args); }},
  };
  std::map<std::string, py::object> _hook_grad;
};

using FinalVMPtr = std::shared_ptr<FinalVM>;

}  // namespace compile
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_VM_VM_H_
