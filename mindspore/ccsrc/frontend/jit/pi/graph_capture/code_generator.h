/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_CODE_GENERATOR_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_CODE_GENERATOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <map>
#include <memory>
#include "frontend/jit/pi/graph_capture/graph_analyzer.h"
#include "frontend/jit/pi/graph_capture/graph_build.h"
#include "frontend/jit/pi/graph_capture/side_effect.h"
#include "frontend/jit/pi/graph_build/func_graph_builder.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

class GraphParameterBuilder;

struct NodeSet {
  std::vector<ValueNode *> inputs;  // index is parameters index
  std::vector<ValueNode *> outputs;
  std::vector<ValueNode *> operations;
};

struct GraphInputInfo {
  std::vector<ValueNode *> args;
  std::vector<ValueNode *> globals;
  ValueNode *vargs = nullptr;
  ValueNode *kwargs = nullptr;
};
using InstructionList = std::vector<std::unique_ptr<Instr>>;

class CodeGenerator {
 public:
  struct ExceptionCodeItem {
    // [begin, end]
    Instr *begin_;
    Instr *end_;
    Instr *jump_;
    int stack_;
    bool lasti_;
  };

  struct Code {
    int co_argcount;
    int co_kwonlyargcount;
    int co_nlocals;
    int co_flags;
    int co_firstlineno;
    std::vector<std::unique_ptr<Instr>> co_code;
    std::vector<std::string> co_varnames;
    std::vector<std::string> co_cellvars;
    std::vector<std::string> co_freevars;
    std::string co_name;
    py::object co_filename;
    std::string co_qualname;
    std::vector<ExceptionCodeItem> co_exceptiontable;
  };

  explicit CodeGenerator(const NodeSet *nodes, bool vm_mode = false)
      : nodes_(nodes),
        globals_(),
        code_(),
        nodes_alive_(),
        locals_map_(),
        missing_value_to_undefine_(false),
        vm_mode_(vm_mode) {}
  explicit CodeGenerator(Code &&ccode) : code_(std::move(ccode)) {}

  void set_missing_value_to_undefine(bool v) { missing_value_to_undefine_ = v; }

  void SetGlobals(const py::dict &dict) { globals_ = dict; }
  const py::dict &GetGlobals() const { return globals_; }
  const std::unordered_map<ValueNode *, int> &GetLocalsMap() const { return locals_map_; }
  std::unordered_map<ValueNode *, int> &GetLocalsMap() { return locals_map_; }
  const Code &GetCode() const { return code_; }
  void SetArgsInfo(int argcount, int kwonlyargcount) {
    code_.co_argcount = argcount;
    code_.co_kwonlyargcount = kwonlyargcount;
  }
  void SetCodeFlags(unsigned flags) { code_.co_flags |= flags; }
  void SetLocalsCount(int nlocals) { code_.co_nlocals = std::max(nlocals, code_.co_nlocals); }
  void SetFirstLineNumber(int line) { code_.co_firstlineno = line; }
  void SetVariableNames(const std::vector<std::string> &names) { code_.co_varnames = names; }
  void SetCellVariableNames(const std::vector<std::string> &names) { code_.co_cellvars = names; }
  void SetFreeVariableNames(const std::vector<std::string> &names) { code_.co_freevars = names; }
  void SetCodeName(const std::string &name) { code_.co_name = name; }
  void SetQualName(const std::string &qualname) { code_.co_qualname = qualname; }
  void SetFileName(const py::object &file) { code_.co_filename = file; }

  void ClearAlive(ValueNode *node) { nodes_alive_.erase(node); }
  void ClearAlive() { nodes_alive_.clear(); }
  void MarkAlive(ValueNode *node) { MarkAlive(node, INT_MAX); }
  void MarkAlive();
  // make the node same as other node, use same local index, if the node not in locals, try to load it
  void MakeSameLocal(ValueNode *new_node, ValueNode *old_node, bool clear = false);
  void NewInstr(int op, int arg = 0, int line = -1);
  void AddInstrs(std::vector<std::unique_ptr<Instr>> &&list);
  void AddInstructionWithExceptionTable(const CFG *cfg, int start, int end);
  void CollectExceptionTableItem(const CFG *cfg, int start, int end);
  void AddInstr(std::unique_ptr<Instr> &&instr);
  void AddCallInstr(size_t load_args_offset, int oparg);
  py::object NewCode();

  // initialize local map of parameters
  void Init();

  // build bytecode by nodes
  void Build();

  // generate return operations of outputs
  void GenReturn();

  // build single node
  void BuildOper(ValueNode *node, int index);

  // generator local operations of node
  void LoadValue(ValueNode *node);

  void LoadConst(const py::object &);

  // add node to locals map
  int AllocLocal(ValueNode *node, int index = INT_MAX);

  /**
   * Copy instruction list at range [start, end).
   * NOTE: reset opcode:
   *       LOAD_METHOD -> LOAD_ATTR,
   *       CALL_METHOD -> CALL_FUNCTION
   *
   * \param list instruct nodes list
   * \param start
   * \param end
   * \return instruction list
   */
  static std::vector<std::unique_ptr<Instr>> CopyInstr(const std::vector<std::unique_ptr<Instr>> &list, size_t start,
                                                       size_t end = -1, bool erase_invalid_jump = false,
                                                       bool is_loop_body = false);

  /**
   * Function to copy and replace instructions in a specified bytecode range.
   * This function copies the instructions from the original list and replaces the instructions
   * in the range between start_bci and end_bci with the provided replacement instructions.
   * (Only for Loop Encapsulation)
   * @param list The original list of instructions.
   * @param start_bci The starting bytecode index where replacement begins.
   * @param end_bci The ending bytecode index where replacement ends.
   * @param replacement The list of instructions that will replace the original instructions in the specified range.
   * @return A new vector containing the modified instructions with the specified replacements applied.
   */
  static std::vector<std::unique_ptr<Instr>> CopyAndReplaceInstr(
    const std::vector<std::unique_ptr<Instr>> &list, size_t start_bci, size_t end_bci,
    const std::vector<std::unique_ptr<Instr>> &replacement);

  /**
   * generate rot instructions
   */
  static std::vector<std::unique_ptr<Instr>> RotStack(int stack);

 private:
  void MarkAlive(ValueNode *node, int order);
  void EraseUnusedInstr();
  void ResetExceptionCodeItem(InstructionList::const_iterator erased);

  static py::object Transform(const Code &ccode);
  static std::pair<py::bytes, py::bytes> ConvertToCodeBytes(const Code &ccode);
  static int CalculateStackSize(const Code &ccode, int sp = 0);

  // python3.11+ only
  std::vector<std::unique_ptr<Instr>> ByteCodePrefix() const;
  void FixOffset();
  void FixLocalOffset(int invalid_index);
  void FixFreeOffset(const std::vector<std::string> &other_closure_names);
  static int ExceptionStackRequired(const Code &ccode);

  const NodeSet *nodes_{nullptr};
  py::dict globals_;
  Code code_;
  std::unordered_map<ValueNode *, int> nodes_alive_;
  std::unordered_map<ValueNode *, int> locals_map_;
  bool missing_value_to_undefine_{false};
  bool vm_mode_{false};
};

class LoopBodyReCaptureCodeGenerator {
 public:
  explicit LoopBodyReCaptureCodeGenerator(Graph *graph) : graph_(graph), co_(graph->GetCodeObj()) {}
  bool Prepare();
  py::object Build();

 protected:
  std::vector<std::string> GetClosureNames() const;

  std::string makeLoopBodyFuncName(int loopBodyStartBci, int loopBodyEndBci) const;

  std::string makeFuncName(int loopBodyStartBci, int loopBodyEndBci) const;

  py::object MakeLoopBodyCode(int loopBodyStartBci, int loopBodyEndBci, const std::vector<int> &inputLocals,
                              const std::vector<int> &outputLocals, bool ifForLoop) const;
  Graph *graph_;
  PyCodeObject *co_;
  bool is_for_loop_ = false;
  int loopBodyStartBci_{0};
  int loopBodyEndBci_{0};
};

class CodeBreakGenerator;
using CodeBreakGeneratorPtr = std::shared_ptr<CodeBreakGenerator>;

class CodeBreakGenerator {
 public:
  CodeBreakGenerator(const GraphBuilderPtr &graph_builder, const py::dict &globals, PyCodeObject *co)
      : fg_builder_(graph_builder->GetGraph()->func_graph_builder()),
        co_(co),
        cfg_(nullptr),
        globals_(globals),
        break_bci_(-1),
        extra_local_(-1),
        no_graph_(false) {}

  // collect nodes inputs and outputs at graph analyze
  void Init(const GraphAnalyzer &, Graph *);

  // generate a code to call graph, unsupported operations, and untracked operations that will be compiled
  py::object MakeDispatchCode();

  // used to replace origin code, extend attribute from origin code.
  py::object MakeCapturedCode() const;

 private:
  const CFG *GetCFG() const { return cfg_; }

  // for python3.11+ copy origin instruction need pop the null pointer (which is only consume by call instruction)
  // from stack. here generate code from node avoid pop null
  bool IsCopyCapturedInstructions() const;

  void ExtendCodeInfo(CodeGenerator *cg, bool merge_kw_only) const;

  // rebuild parameters of graph, identify parameters that graph only support as constant
  void BuildGraphParameters(const std::unordered_map<ValueNode *, int> &locals, GraphParameterBuilder *);

  py::object MakeInterpretCapturedCode() const;

  // rebuild captured nodes to bytecode, build parameters load operations
  py::object MakeCapturedCode(std::vector<std::unique_ptr<Instr>> &&sort, int argc, unsigned flag) const;

  // make call operations of graph, build parameters load operations
  void CallCapturedCode(CodeGenerator *code_gen);

  void FixInterpretOuput(CodeGenerator *code_gen);

  void HandleOutputOpt(CodeGenerator *code_gen);

  // make function of untracked bytecode, build restore frame operations of untracked bytecode
  py::object MakeUntrackedCode(int untracked_bci, int untracked_stack_effect, int *argc) const;

  void ReconstructStack(CodeGenerator *code_gen, int untracked_bci, int untracked_stack_effect) const;

  // make call operations of untracked bytecode
  void CallUntrackedCode(CodeGenerator *code_gen);

  void MakeReturn(CodeGenerator *code_gen) const;

  // build operations of block, build restore frame operations of block
  void BreakAtBlock(CodeGenerator *code_gen, int untracked_bci, int untracked_stack_effect);

  // make call operations of untracked bytecode for each branch
  void BreakAtIf(CodeGenerator *code_gen) const;

  // generate specialize code if break point is call
  void BreakAtCall(CodeGenerator *code_gen) const;
  bool NeedHandleBreakAtCall() const;
  // Similar to BreakAtIf(), but it breaks at conditional-statement of a callee function (or we say a subgraph).
  void BreakAtCalleeIfCondition(CodeGenerator *code_gen) const;
  py::object MakeUntrackedCodeForNestedCalls(const std::vector<Graph *> &call_stack, int top_argc, int untracked_bci,
                                             int stack_effect) const;

  void RestoreStack(CodeGenerator *code_gen) const;

  void RestoreLocals(CodeGenerator *code_gen, bool load) const;

  FuncGraphBuilderPtr FGBuilder() const { return fg_builder_; }

  void Compile(const std::string &name, int argc, int kw_only, int flags, const py::object &stub) const;

 private:
  // The FuncGraphBuilder of top-graph
  FuncGraphBuilderPtr fg_builder_;

  // root function
  PyCodeObject *const co_;

  // instructions for break graph
  const CFG *cfg_;

  // function globals
  py::dict globals_;

  /**
   * first execute node,
   * inputs must be same as the start of function locals(include unbound local)
   * outputs is alive values
   **/
  NodeSet interpret_;

  // followed interpret execute node
  NodeSet captured_;

  // interpret execute node after graph
  NodeSet outputs_optimize_;

  // used to record the value nodes and the nodes that replaced them
  std::map<ValueNode *, ValueNode *> replaced_nodes_;

  // break bci alive locals
  std::vector<int> alive_locals_;

  // break bci
  int break_bci_;

  // used to store graph outputs
  int extra_local_;

  bool no_graph_;

  bool is_break_at_call_ = false;
  // The top-graph is at the beginning of the vector and the bottommost subgraph at the end.
  std::vector<Graph *> call_stack_{};
};

// add a key and value to py::dict, check key conflict or rename the key
void MapAdd(const py::dict &dict, const std::string &key, const py::object &value, std::string *rename = nullptr);

// make new code by graph and captured information
py::object MakeCodeFromCodeGen(const GraphBuilderPtr &builder, const GraphAnalyzerPtr &analyzer, PyObject *globals);
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_CODE_GEN_H
