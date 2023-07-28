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
#include "pipeline/jit/graph_jit/graph_capture/graph.h"
#include "pipeline/jit/graph_jit/graph_capture/graph_analyzer.h"
#include "pipeline/jit/graph_jit/common.h"
#include "pipeline/jit/graph_jit/utils/allocator.h"

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
    int nfrees;
    std::vector<int> co_cell2arg;
    std::vector<_Py_CODEUNIT> co_code;
    std::unordered_map<std::string, int> co_names;
    std::unordered_map<PyObject *, int> co_consts;
    py::object co_filename;
    std::string co_name;
    std::vector<char> co_lnotab;
    bool ms_mode_;
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

  py::object GetClosure() {
    PyObject *c = PyTuple_New(closure_.size());
    for (auto i : closure_) {
      Py_INCREF(i.first);
      PyTuple_SET_ITEM(c, i.second, i.first);
    }
    return py::reinterpret_steal<py::object>(c);
  }

  void UpdateGlobals(const py::object &g) {
    if (used_globals_.ptr()) {
      PyDict_Update(used_globals_.ptr(), g.ptr());
    } else {
      used_globals_ = g;
    }
  }

  bool IsCodeChanged() const { return code_changed_; }
  auto &GetGlobals() { return used_globals_; }
  int getNcells() const { return cell2arg_.size(); }
  int getNfrees() const { return closure_.size(); }

  // NOTE: if break graph, do this in CutoffBytecodesIfGraphBreak
  // return true if must be interpret these code
  bool TryToBreakGraphIfUnsupported();

  // NOTE: graph inputs will passed by globals for mindspore
  // should produce guard for mindspore
  static std::string getGraphInputsKey(const ValueNode *v);

  /**
   * use list to generate code info, update co_co_names, co_consts, so LOAD_CONST node and LOAD_GLOBAL node
   * must has value. will optimize list, e.g delete NOP, delete jump to next instruction
   * return used globals
   **/
  static py::object generateCode(const AbstractNodeList &list, Code *);

  // debug function
  static void Print(const AbstractNode *list_beg, const char *marker_info);

  // alloc id to code object
  static void setId(Code *c) {
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
  bool fixInstr(Graph *, InstrNode *, int local_off);

  void inlineCall(CallNode *, int local_off);

  // generate graph to instrs list
  void processGraph(Graph *, int local_off = 0);

  void pushInstr(InstrNode *n) {
    n->getPres().clear();
    n->setNext(nullptr);
    instrs_.push_back(n);
  }

  InstrNode *NewInstr(int op, int arg = 0) { return alloc_.NewInstrNode(op, arg); }

  // NOTE: will modify node->marker_ for link jump node
  AbstractNodeList copyInstrList(const AbstractNodeList &instrs);

  int addClosure(const py::object &o) {
    auto pos = closure_.find(o.ptr());
    if (pos != closure_.end()) {
      return pos->second;
    }
    int i = closure_.size();
    closure_.insert({o.ptr(), i});
    return i;
  }

  // break graph actions

  void breakAtUnsupportedOperation(InstrNode *, const FrameStates &f);
  void breakAtIf(InstrNode *, const FrameStates &);
  bool breakAtInterpretBlocks(InstrNode *, const FrameStates &f, bool is_loop);

  // return make function bytecodes, generate code object for LOAD_CONST instruction node
  AbstractNodeList GenerateMakeFunc(Code *ccode, const std::set<int> &freevars, JitCompileResults::State);

  // copy instruction node and rewrite local index, cell index, add load actions for inputs and outputs
  AbstractNodeList generateNewInstrs(const AbstractNodeList &list, int in_stack, int out_stack,
                                     const std::set<int> &inputs, const std::set<int> &output,
                                     const std::set<int> &visit_cells, Code *ccode);

  // return unpack operations, container = [stack1, stakc2,...,local1, local3,...]
  // assume container is top of stack
  AbstractNodeList unpackToStackAndLocal(int tmp_local, int stack, const std::set<int> &locals);

  // return call operations bytecodes
  AbstractNodeList callSubList(const AbstractNodeList &list, int in_stack, int out_stack, const std::set<int> &inputs,
                               const std::set<int> &output, const std::set<int> &deletes,
                               const std::set<int> &visit_cells, JitCompileResults::State t);

  // return call operations bytecodes, not outputs and end with return
  AbstractNodeList callSubListWithReturn(const AbstractNodeList &list, int stack_off, const FrameStates &f);

  // special version of callSubList, will handle mindspore unsupported operation
  // return call action bytecodes
  AbstractNodeList reshapeCapturedBytecodes(const FrameStates &last_frame, const std::set<int> &outputs,
                                            ValueNode *ret = nullptr);

  // NOTE: use ValueNode::marker_ as usage count
  bool loadValue(std::unordered_map<ValueNode *, int> *local_map, ValueNode *n, AbstractNodeList *res, bool = true);
  void buildValue(std::unordered_map<ValueNode *, int> *local_map, ValueNode *n, int order, AbstractNodeList *res);

  AbstractNodeList buildValues(std::unordered_map<ValueNode *, int> *local_map, const std::vector<ValueNode *> &);

  // build mindspore operation
  AbstractNodeList reshapeGraphBytecodes(std::unordered_map<ValueNode *, int> *graph_local_map,
                                         std::unordered_map<ValueNode *, int> *interpret_local_map,
                                         std::unordered_map<ValueNode *, int> *graph_outputs);

  AbstractNodeList instrs_;
  Graph *graph_;
  GraphAnalyzer::CapturedInfo &captured_info_;

  // new locals
  int nlocals_;

  // code status
  bool code_changed_;

  std::vector<Graph *> inlined_call_;

  // information for rewrite cell index

  std::set<CellVarNode *> cells_nodes_;
  std::set<CellVarNode *> frees_nodes_;
  std::unordered_map<PyObject *, int> closure_;
  std::vector<int> cell2arg_;

  // used globals
  py::object used_globals_;

  // node allocator
  Allocator alloc_;

  // handle make function gloabls
  py::object make_func_handler_;
};
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_GRAPH_JIT_GRAPH_CAPTURE_CODE_GEN_H
