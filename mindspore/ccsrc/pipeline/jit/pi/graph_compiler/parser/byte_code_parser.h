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

#ifndef MINDSPORE_PI_JIT_BYTECODE_PARSER_H_
#define MINDSPORE_PI_JIT_BYTECODE_PARSER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "pybind11/pybind11.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/ctrl_flow.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/custom_nodes.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/debug_info.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/value.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace pijit {
namespace py = pybind11;

// ByteCodeParser to parse python byte code
class ByteCodeParser {
 public:
  explicit ByteCodeParser(const py::object &func);
  explicit ByteCodeParser(const PyFunctionObject &func);
  ir::FunctionNodePtr Parse();
  void SetEnableTupleBroaden(bool enable) { func_->SetAttr("enable_tuple_broaden", enable); }

 private:
  class Instr {
   public:
    explicit Instr(const py::object &instr)
        : op_code_(CastToInt(instr.attr("opcode"))),
          op_name_(instr.attr("opname").cast<std::string>()),
          arg_(CastToInt(instr.attr("arg"))),
          arg_repr_(instr.attr("argrepr").cast<std::string>()),
          offset_(CastToInt(instr.attr("offset"))),
          starts_line_(CastToInt(instr.attr("starts_line"))),
          is_jump_target_(instr.attr("is_jump_target").cast<bool>()) {}

    int GetOpCode() const { return op_code_; }
    const std::string &GetOpName() const { return op_name_; }
    int GetArg() const { return arg_; }
    void SetArg(int arg) { arg_ = arg; }
    const std::string &GetArgRepr() const { return arg_repr_; }
    int GetOffset() const { return offset_; }
    int GetStartsLine() const { return starts_line_; }
    bool IsJumpTarget() const { return is_jump_target_; }
    std::string ToString() const {
      return op_name_ + " " + std::to_string(arg_) + (arg_repr_.empty() ? "" : (" (" + arg_repr_) + ")");
    }

   private:
    int CastToInt(const py::object &obj) const { return py::isinstance<py::none>(obj) ? 0 : obj.cast<int>(); }

    int op_code_;
    std::string op_name_;
    int arg_;
    std::string arg_repr_;
    int offset_;
    int starts_line_;
    bool is_jump_target_;
  };
  using InstrPtr = std::unique_ptr<Instr>;

  // Bind op code to it's handle function
  void BuildMethodMap();
  void BuildStackMethodMap();
  void BuildLoadStoreMethodMap();
  void BuildMathMethodMap();
  void BuildBitwiseMethodMap();
  void BuildContainerMethodMap();
  void BuildContrlFlowMethodMap();
  void BuildOtherMethodMap();
  void CallInstrMethod(const InstrPtr &instr);
  void Register(ir::NodePtrList *list) { nodes_.push_back(list); }
  void Restore() { nodes_.pop_back(); }
  void SaveNode(const ir::NodePtr &node);
  bool IsConditionJump(ir::OpCode op);
  void ParseInstructions(const py::list &instrs);
  void ParseIf(const InstrPtr &cond, const py::list &then, const py::list &els);
  void ParseWhile(const InstrPtr &cond, const py::list &body);
  ir::NodePtr PopStack();
  void PushStack(const ir::NodePtr &node);
  // Process parameters, create place holder and set abstract(type info)
  void GeneratePostionalParameters();
  void GenerateVariableParameter();
  void GenerateKeywordOnlyParameters();
  void GenerateKeywordParameter();
  void GenerateFunctionParameters();
  void ParsePopTop(const InstrPtr &instr);
  void DoRot(const int &cnt);
  void ParseRotTwo(const InstrPtr &instr);
  void ParseRotThree(const InstrPtr &instr);
  void ParseRotFour(const InstrPtr &instr);
  void ParseNop(const InstrPtr &instr);
  void ParseDupTop(const InstrPtr &instr);
  void ParseDupTwo(const InstrPtr &instr);
  void ParseUnaryOpertion(const InstrPtr &instr);
  void ParseUnaryNegative(const InstrPtr &instr);
  void ParseUnaryNot(const InstrPtr &instr);
  void ParseUnaryInvert(const InstrPtr &instr);
  void ParseBinaryOpertion(const InstrPtr &instr);
  void ParseBitwise(const InstrPtr &instr);
  void ParseAdd(const InstrPtr &instr);
  void ParseSub(const InstrPtr &instr);
  void ParseMul(const InstrPtr &instr);
  void ParseDiv(const InstrPtr &instr);
  void ParseBinarySubscr(const InstrPtr &instr);
  void ParseCompareOp(const InstrPtr &instr);
  void ParseJump(const InstrPtr &instr);
  void ParseListToTuple(const InstrPtr &instr);
  void ParseReturnValue(const InstrPtr &instr);
  // Process Load Constant as value node
  void ParseLoadConst(const InstrPtr &instr);
  void ParseLoadName(const InstrPtr &instr);
  void ParseMakeFunction(const InstrPtr &instr);
  void ParseBuild(const InstrPtr &instr);
  void ParseLoadAttr(const InstrPtr &instr);
  void ParseImport(const InstrPtr &instr);
  // Process Load a global module/class/function/method/variable etc.
  void ParseLoadGlobal(const InstrPtr &instr);
  // Whether two objects are the same
  // Fold Python objects into constant values, relying on Gurad
  void ParseIsOp(const InstrPtr &instr);
  void ParseContainsOp(const InstrPtr &instr);
  // Process Load a parameter or a local variable
  void ParseLoadFast(const InstrPtr &instr);
  void ParseStoreName(const InstrPtr &instr);
  void ParseStoreSubscr(const InstrPtr &instr);
  void ParseStoreAttr(const InstrPtr &instr);
  void ParseCallFunction(const InstrPtr &instr);
  void ParseLoadClosure(const InstrPtr &instr);
  void ParseSetupWith(const InstrPtr &instr);
  // Fold Python objects into constant values, relying on Gurad
  void ParseFormatValue(const InstrPtr &instr);
  void ParseLoadMethod(const InstrPtr &instr);
  void ParseContainerUpdate(const InstrPtr &instr);
  void ParseNoArgOperation(const InstrPtr &instr);
  void ParseWithExceptStart(const InstrPtr &instr);
  void ParseGet(const InstrPtr &instr);
  void ParseLoadAssertError(const InstrPtr &instr);
  void ParseDeleteSubscr(const InstrPtr &instr);
  void ParseDeleteName(const InstrPtr &instr);
  void ParseDeleteAttr(const InstrPtr &instr);
  void ParseUnpack(const InstrPtr &instr);
  void ParseRaiseVarargs(const InstrPtr &instr);
  ir::DebugInfoPtr GetNodeDebugInfo(const InstrPtr &instr);

  ir::FunctionNodePtr func_;
  const PyCodeObject &code_;
  const py::dict globals_;
  const py::dict builtins_;
  const py::tuple clousre_;
  const py::dict kwdefaults_;
  const py::tuple defaults_;

  using InstrFunc = void (ByteCodeParser::*)(const InstrPtr &instr);
  // Define the function map to parse ast expression
  std::map<int, InstrFunc> instr_method_map_;
  // variable's buffer used to analysis logic and build graph
  ir::NodePtrList stack_;
  std::map<int, std::vector<ir::JumpNodePtr>> jump_nodes_map_;
  std::map<int, ir::NodePtr> targets_map_;
  std::vector<ir::NodePtrList *> nodes_;
  ir::NodePtr latest_gen_node_{nullptr};
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_BYTECODE_PARSER_H_
