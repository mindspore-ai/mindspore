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
#ifndef MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_CODE_GEN_H
#define MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_CODE_GEN_H

#include <string>
#include <set>
#include <unordered_map>
#include <vector>
#include "pipeline/jit/pi_jit/graph_capture/graph.h"
#include "pipeline/jit/pi_jit/graph_capture/graph_analyzer.h"
#include "pipeline/jit/pi_jit/common.h"
#include "pipeline/jit/pi_jit/utils/allocator.h"

#ifndef _Py_MAKECODEUNIT
#ifdef WORDS_BIGENDIAN
#define _Py_MAKECODEUNIT(opcode, oparg) (MS_ASSERT((opcode) < NO_IMPL_OPCODE), ((opcode) << 8) | (oparg))
#else
#define _Py_MAKECODEUNIT(opcode, oparg) (MS_ASSERT((opcode) < NO_IMPL_OPCODE), (opcode) | ((oparg) << 8))
#endif
#endif

namespace mindspore {
namespace jit {
namespace graph {
class CodeGenerator {
 public:
  struct Code {
    int co_argcount;
    int co_kwonlyargcount;
    int co_nlocals;
    int co_stacksize;
    int co_flags;
    int co_firstlineno;
    std::vector<std::string> co_varnames;
    std::vector<std::string> co_freevars;
    std::vector<std::string> co_cellvars;
    std::vector<_Py_CODEUNIT> co_code;
    std::unordered_map<std::string, int> co_names;
    std::unordered_map<PyObject *, int> co_consts;
    py::object co_filename;
    std::string co_name;
    std::vector<char> co_lnotab;
    JitCompileResults::State stat;
  };

  // will generate instrs list
  CodeGenerator(Graph *g, GraphAnalyzer::CapturedInfo &);

  /**
   * if graph break, will generate code to call compiled block and untracked block. rewrite instrs list.
   * return tracked instrs list
   **/

  void CutoffBytecodesIfGraphBreak();

  // use instrs list to generate PyCodeObject
  PyCodeObject *MakeCodeObj();

  void UpdateGlobals(const py::object &g) {
    if (used_globals_.ptr()) {
      PyDict_Update(used_globals_.ptr(), g.ptr());
    } else {
      used_globals_ = g;
    }
  }

  bool IsCodeChanged() const { return code_changed_; }
  auto &GetGlobals() { return used_globals_; }

  // NOTE: if break graph, do this in CutoffBytecodesIfGraphBreak
  // return true if must be interpret these code
  bool TryToBreakGraphIfParameterUnsupported();

  // NOTE: graph inputs will passed by globals for mindspore
  // should produce guard for mindspore
  static std::string GetGraphInputsKey(const ValueNode *v);

  // debug function
  static void Print(const AbstractNode *list_beg, const char *marker_info);

  // alloc id to code object
  static void SetId(Code *c) {
    static int id = 0;
    char buf[16];
    snprintf(buf, sizeof(buf), "%d_", id++);
    std::string s(buf);
    s.append(c->co_name);
    c->co_name = s;
  }

 private:
  /**
   * fix local index after inlined function
   * return true if should insert this InstrNode
   **/
  bool FixInstr(Graph *, InstrNode *, int local_off);

  void InlineCall(CallNode *, int local_off);

  // generate graph to instrs list
  void ProcessGraph(Graph *, int local_off = 0);

  void PushInstr(InstrNode *n) {
    n->GetPres().clear();
    n->SetNext(nullptr);
    instrs_.push_back(n);
  }

  InstrNode *NewInstr(int op, int arg = 0) { return alloc_.NewInstrNode(op, arg); }

  // NOTE: will modify node->marker_ for link jump node
  AbstractNodeList CopyInstrList(const AbstractNodeList &instrs);

  // break graph actions

  void BreakAtUnsupportedOperation(InstrNode *, const FrameStates &f);
  void BreakAtIf(InstrNode *, const FrameStates &);
  bool BreakAtInterpretBlocks(InstrNode *, const FrameStates &f, bool is_loop);

  // return make function bytecodes, generate code object for LOAD_CONST instruction node
  AbstractNodeList GenerateMakeFunc(Code *ccode, const std::set<int> &freevars, JitCompileResults::State);

  // copy instruction node and rewrite local index, cell index, add load actions for inputs and outputs
  AbstractNodeList GenerateNewInstrs(const AbstractNodeList &list, int in_stack, int out_stack,
                                     const std::set<int> &inputs, const std::set<int> &output,
                                     const std::set<int> &visit_cells, Code *ccode);

  // return unpack operations, container = [stack1, stakc2,...,local1, local3,...]
  // assume container is top of stack
  AbstractNodeList UnpackToStackAndLocal(int tmp_local, int stack, const std::set<int> &locals);

  // return call operations bytecodes
  AbstractNodeList CallSubList(const AbstractNodeList &list, int in_stack, int out_stack, const std::set<int> &inputs,
                               const std::set<int> &output, const std::set<int> &deletes,
                               const std::set<int> &visit_cells, JitCompileResults::State t);

  // return call operations bytecodes, not outputs and end with return
  AbstractNodeList CallSubListWithReturn(const AbstractNodeList &list, int stack_off, const FrameStates &f);

  // special version of CallSubList, will handle mindspore unsupported operation
  // return call action bytecodes
  AbstractNodeList ReshapeCapturedBytecodes(const FrameStates &last_frame, const std::set<int> &outputs,
                                            ValueNode *ret = nullptr);

  // NOTE: use ValueNode::marker_ as usage count
  bool LoadValue(std::unordered_map<ValueNode *, int> *local_map, ValueNode *n, AbstractNodeList *res, bool = true);
  void BuildValue(std::unordered_map<ValueNode *, int> *local_map, ValueNode *n, int order, AbstractNodeList *res);

  AbstractNodeList BuildValues(std::unordered_map<ValueNode *, int> *local_map, const std::vector<ValueNode *> &);

  void BuildGraphParameters(std::unordered_map<ValueNode *, int> *graph_local_map,
                            std::unordered_map<ValueNode *, int> *interpret_local_map,
                            AbstractNodeList *graph_load_inputs, AbstractNodeList *interpret_load_inputs, int *argc,
                            int *code_flag);

  // build mindspore operation
  AbstractNodeList ReshapeGraphBytecodes(std::unordered_map<ValueNode *, int> *graph_local_map,
                                         std::unordered_map<ValueNode *, int> *interpret_local_map,
                                         std::unordered_map<ValueNode *, int> *graph_outputs);

  AbstractNodeList instrs_;
  Graph *graph_;
  GraphAnalyzer::CapturedInfo &captured_info_;

  // new locals
  int nlocals_;

  // code status
  bool code_changed_;

  // used globals
  py::object used_globals_;

  // node allocator
  Allocator alloc_;
};
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_CODE_GEN_H
