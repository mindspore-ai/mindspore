/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_PI_JIT_GRAPH_CAPTURE_CODE_GENERATOR_H
#define MINDSPORE_PI_JIT_GRAPH_CAPTURE_CODE_GENERATOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <memory>
#include "pipeline/jit/pi/graph_capture/graph_analyzer.h"
#include "pipeline/jit/pi/graph_capture/graph_build.h"
#include "pipeline/jit/pi/graph_build/func_graph_builder.h"

namespace mindspore {
namespace pijit {

namespace py = pybind11;

class ValueNode;
class FrameStates;
class CodeExtra;
class GraphParameterBuilder;

struct NodeSet {
  std::vector<ValueNode *> inputs;  // index is parameters index
  std::vector<ValueNode *> outputs;
  std::vector<ValueNode *> operations;
};

class CodeGenerator {
 public:
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
  };

  explicit CodeGenerator(const NodeSet *nodes) : nodes_(nodes), globals_(), code_(), nodes_alive_(), locals_map_() {}

  void SetGlobals(const py::dict &dict) { globals_ = dict; }
  std::vector<std::unique_ptr<Instr>> MoveCode() { return std::move(code_.co_code); }
  const py::dict &GetGlobals() const { return globals_; }
  const std::unordered_map<ValueNode *, int> &GetLocalsMap() const { return locals_map_; }
  bool EarseLocal(ValueNode *item) {
    auto it = GetLocalsMap().find(item);
    if (it != GetLocalsMap().end()) {
      locals_map_.erase(it);
    } else {
      return false;
    }
    return true;
  }
  const Code &GetCode() const { return code_; }
  void SetArgsInfo(int argcount, int kwonlyargcount) {
    code_.co_argcount = argcount;
    code_.co_kwonlyargcount = kwonlyargcount;
  }
  void SetCodeFlags(int flags) { code_.co_flags |= flags; }
  void SetLocalsCount(int nlocals) { code_.co_nlocals = std::max(nlocals, code_.co_nlocals); }
  void SetFirstLineNumber(int line) { code_.co_firstlineno = line; }
  void SetVariableNames(const std::vector<std::string> &names) { code_.co_varnames = names; }
  void SetCellVariableNames(const std::vector<std::string> &names) { code_.co_cellvars = names; }
  void SetFreeVariableNames(const std::vector<std::string> &names) { code_.co_freevars = names; }
  void SetCodeName(const std::string &name) { code_.co_name = name; }
  void SetFileName(const py::object &file) { code_.co_filename = file; }

  void MarkAlive(ValueNode *node) { nodes_alive_[node] = INT_MAX; }
  void MarkAlive();
  void NewInstr(int op, int arg = 0, int line = -1);
  void AddInstrs(std::vector<std::unique_ptr<Instr>> &&list);
  void EraseUnusedInstr();

  // initialize local map of parameters
  void Init();

  // build bytecode by nodes
  void Build();

  // generate return operations of outputs
  void GenReturn();

  // build single node
  void BuildOper(ValueNode *node, int index);

  // generator local operations of node
  void LoadValue(ValueNode *node, bool is_side_effect = false);

  // add node to locals map
  int AllocLocal(ValueNode *node, int index = INT_MAX);

  std::string PrintAlive() const;

  /**
   * Transform code info to PyCodeObject
   *
   * \param ccode code info
   * \return PyCodeObject
   */
  static py::object Transform(const Code &ccode);

  /**
   * Calculate max stack size
   *
   * \param list instruct nodes list
   * \param sp start of stack depth
   * \return max depth of stack, or -1 if stack out of bound
   */
  static int CalculateStackSize(const std::vector<std::unique_ptr<Instr>> &list, int sp = 0);

  /**
   * Convert instruction list to bytes object. generate line table.
   *
   * \param list instruct nodes list
   * \param first_line first line
   * \return first is co_code, second is co_lnotab
   */
  static std::pair<py::bytes, py::bytes> ConvertToCodeBytes(const std::vector<std::unique_ptr<Instr>> &list,
                                                            int first_line);

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
                                                       size_t end = -1);

  /**
   * Erase unused instr
   *
   * \param list instruction list
   */
  static void EraseUnusedInstr(std::vector<std::unique_ptr<Instr>> *list);

  /**
   * generate rot instructions
   */
  static std::vector<std::unique_ptr<Instr>> RotStack(int stack);

 private:
  const NodeSet *nodes_;
  py::dict globals_;
  Code code_;
  std::unordered_map<ValueNode *, int> nodes_alive_;
  std::unordered_map<ValueNode *, int> locals_map_;
};

class CodeBreakGenerator;
class MindCodeBreakGenerator;
using CodeBreakGeneratorPtr = std::shared_ptr<CodeBreakGenerator>;
using MindCodeBreakGeneratorPtr = std::shared_ptr<MindCodeBreakGenerator>;
class CodeBreakGenerator {
 public:
  explicit CodeBreakGenerator(PyCodeObject *co) : co_(co), cfg_(nullptr), break_bci_(-1), extra_local_(-1) {}
  static CodeBreakGeneratorPtr Creator(const GraphBuilderPtr &builder, PyCodeObject *co) {
    return builder->trace_flag()
             ? std::static_pointer_cast<CodeBreakGenerator>(std::make_shared<MindCodeBreakGenerator>(builder, co))
             : std::make_shared<CodeBreakGenerator>(co);
  }

  void SetGlobals(const py::dict &dict) { globals_ = dict; }
  const py::dict &GetGlobals() const { return globals_; }

  // TODO(chaiyouheng): collect nodes inputs and outputs at graph analyze
  void Init(const Graph *, const GraphAnalyzer::CapturedInfo *);

  virtual py::object MakeCode(bool make_graph, Graph *graph);
  const CFG *GetCFG() const;

 protected:
  // rebuild parameters of graph, identify parameters that graph only support as constant
  void BuildGraphParameters(const std::unordered_map<ValueNode *, int> &locals, GraphParameterBuilder *);

  // rebuild captured nodes to bytecode, build parameters load operations
  virtual py::object MakeCapturedCode(std::vector<std::unique_ptr<Instr>> &&sort, int argc, int flag) const;

  // make call operations of graph, build parameters load operations
  void CallCapturedCode(CodeGenerator *code_gen);

  void FixInterpretOuput(CodeGenerator *code_gen);

  // make call operations of side effect bytecode
  void CallSideEffectCode(CodeGenerator *code_gen, Graph *graph);

  // make function of untracked bytecode, build restore frame operations of untracked bytecode
  py::object MakeUntrackedCode(int untracked_bci, int untracked_stack_effect) const;

  void ReconstructStack(CodeGenerator *code_gen, int untracked_bci, int untracked_stack_effect) const;

  // make call operations of untracked bytecode
  void CallUntrackedCode(CodeGenerator *code_gen);

  void MakeReturn(CodeGenerator *code_gen) const;

  // build operations of block, build restore frame operations of block
  void BreakAtBlock(CodeGenerator *code_gen, int untracked_bci, int untracked_stack_effect);

  // make call operations of untracked bytecode for each branch
  void BreakAtIf(CodeGenerator *code_gen) const;

  void RestoreStack(CodeGenerator *code_gen) const;

  void RestoreLocals(CodeGenerator *code_gen, bool load) const;

  // return co_cellvars and co_freevars
  std::vector<std::string> GetClosureNames() const;

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

  // break bci alive locals
  std::vector<int> alive_locals_;

  // break bci
  int break_bci_;

  // used to store graph outputs
  int extra_local_;
};

class MindCodeBreakGenerator : public CodeBreakGenerator {
 public:
  MindCodeBreakGenerator(const GraphBuilderPtr &builder, PyCodeObject *co)
      : CodeBreakGenerator(co), builder_(builder) {}
  py::object MakeCode(bool make_graph, Graph *graph) override;
  mindspore::FuncGraphBuilderPtr FGBuilder() const {
    return std::dynamic_pointer_cast<MindGraphBuilder>(builder_)->FGBuilder();
  }


  py::object MakeCapturedCode(std::vector<std::unique_ptr<Instr>> &&, int argc, int code_flag) const override;

 private:
  py::object MakeCopyCode(const std::string &co_name, int co_argcount, int co_kwonlyargcount, int co_flags,
                          bool make_graph = false) const;

  GraphBuilderPtr builder_;
};
// add a key and value to py::dict, check key conflict or rename the key
void MapAdd(const py::dict &dict, const std::string &key, const py::object &value, std::string *rename = nullptr);

// make new code by graph and captured information
py::object MakeCodeFromCodeGen(const GraphBuilderPtr &builder, const GraphAnalyzerPtr &analyzer, PyObject *globals);
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_CODE_GEN_H
