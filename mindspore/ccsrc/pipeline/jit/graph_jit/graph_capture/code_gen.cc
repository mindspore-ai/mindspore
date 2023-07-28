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

#include "pipeline/jit/graph_jit/graph_capture/code_gen.h"
#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "pipeline/jit/graph_jit/graph_capture/graph_build.h"

namespace mindspore {
namespace jit {
namespace graph {
static bool IsOperationWithoutOperatorStackEffects(int op) {
  return op == STORE_DEREF || op == DELETE_DEREF || op == STORE_GLOBAL || op == DELETE_GLOBAL || op == STORE_ATTR ||
         op == DELETE_ATTR || op == STORE_SUBSCR || op == DELETE_SUBSCR || op == IMPORT_STAR || op == RAISE_VARARGS ||
         op == RERAISE;
}

// restore MAKE_FUNCTION globals after inlined function
static PyObject *MakeFuncHandler(PyObject *, PyObject *args) {
  MS_ASSERT(PyTuple_Check(args) && PyTuple_GET_SIZE(args) == 2);
  PyObject *inner_func = PyTuple_GET_ITEM(args, 0);
  PyObject *outer_func = PyTuple_GET_ITEM(args, 1);
  PyObject *globals = nullptr;
  PyObject *object_call = nullptr;
  if (PyMethod_Check(outer_func)) {
    outer_func = PyInstanceMethod_GET_FUNCTION(outer_func);
  }
  if (PyFunction_Check(outer_func)) {
    globals = PyFunction_GET_GLOBALS(outer_func);
  } else {
    object_call = PyObject_GetAttrString(reinterpret_cast<PyObject *>(Py_TYPE(outer_func)), "__call__");
    MS_EXCEPTION_IF_CHECK_FAIL(object_call != nullptr, "unknown callable object " + std::string(py::str(outer_func)));
    globals = PyFunction_GET_GLOBALS(object_call);
  }
  PyObject *module_name = PyDict_GetItemString(globals, "__name__");
  REPLACE_PY_MEMBER(PyFunction_GET_GLOBALS(inner_func), globals);
  REPLACE_PY_MEMBER(PyFunction_GET_MODULE(inner_func), module_name);
  Py_XDECREF(object_call);
  Py_RETURN_NONE;
}

static PyMethodDef makeFuncHandlerDef = {"MAKE_FUNCTION", MakeFuncHandler, METH_VARARGS,
                                         "restore MAKE_FUNCTION globals"};

static void resetJumpToUnusedInstr(InstrNode *n) {
  if (!n->getJump()) {
    return;
  }
  auto tar = reinterpret_cast<InstrNode *>(n->getJump());
  do {
    int op = tar->getOpcode();
    bool jump_next = (op == JUMP_ABSOLUTE || op == JUMP_FORWARD) && tar->getJump() == tar->getNext();
    if (op != NOP && op != EXTENDED_ARG && !jump_next) {
      break;
    }
    n->setJump(tar->getNext());
    tar = reinterpret_cast<InstrNode *>(tar->getNext());
  } while (true);
}

static void eraseUnusedInstr(AbstractNodeList *list) {
  std::set<InstrNode *> removed;
  for (auto n = reinterpret_cast<InstrNode *>(list->head()); n; n = reinterpret_cast<InstrNode *>(n->getNext())) {
    // jump target shouldn't be a removed node
    MS_ASSERT(!n->getJump() || removed.find(reinterpret_cast<InstrNode *>(n->getJump())) == removed.end());
    resetJumpToUnusedInstr(n);
    int op = n->getOpcode();
    while (((op == JUMP_ABSOLUTE || op == JUMP_FORWARD) && n->getJump() == n->getNext()) || n->getOpcode() == NOP ||
           n->getOpcode() == EXTENDED_ARG) {
      auto tmp = n;
      n = reinterpret_cast<InstrNode *>(n->getNext());
      list->erase(tmp);
      removed.insert(tmp);
      op = n->getOpcode();
    }
  }
}

static PyCodeObject *TransformCode(const CodeGenerator::Code &ccode) {
  PyObject *t = PyBytes_FromStringAndSize(reinterpret_cast<const char *>(ccode.co_code.data()),
                                          ccode.co_code.size() * sizeof(ccode.co_code[0]));
  auto code = py::reinterpret_steal<py::object>(t);
  auto pyconsts = py::reinterpret_steal<py::object>(PyTuple_New(ccode.co_consts.size()));
  auto pynames = py::reinterpret_steal<py::object>(PyTuple_New(ccode.co_names.size()));
  for (auto i : ccode.co_consts) {
    PyTuple_SET_ITEM(pyconsts.ptr(), i.second, i.first);
    Py_XINCREF(i.first);
  }
  for (const auto &i : ccode.co_names) {
    PyTuple_SET_ITEM(pynames.ptr(), i.second, PyUnicode_FromString(i.first.c_str()));
  }

  auto pyvarnames = py::reinterpret_steal<py::object>(PyTuple_New(ccode.co_nlocals));
  auto pycellvars = py::reinterpret_steal<py::object>(PyTuple_New(ccode.co_cell2arg.size()));
  auto pyfreevars = py::reinterpret_steal<py::object>(PyTuple_New(ccode.nfrees));
  char buf[32];
  for (int i = 0; i < ccode.nfrees; ++i) {
    snprintf(buf, sizeof(buf), "%d_free", i);
    PyTuple_SET_ITEM(pyfreevars.ptr(), i, PyUnicode_FromString(buf));
  }
  for (int i = ccode.co_cell2arg.size() - 1; i >= 0; --i) {
    if (ccode.co_cell2arg[i] == CO_CELL_NOT_AN_ARG) {
      snprintf(buf, sizeof(buf), "%d_cell", i);
    } else {
      snprintf(buf, sizeof(buf), "%d_cell_to_%d_arg", i, ccode.co_cell2arg[i]);
    }
    PyObject *str = PyUnicode_FromString(buf);
    PyTuple_SET_ITEM(pycellvars.ptr(), i, str);
    if (ccode.co_cell2arg[i] == CO_CELL_NOT_AN_ARG) {
      continue;
    }
    Py_INCREF(str);
    PyTuple_SET_ITEM(pyvarnames.ptr(), ccode.co_cell2arg[i], str);
  }
  for (int i = 0; i < ccode.co_nlocals; ++i) {
    if (!PyTuple_GET_ITEM(pyvarnames.ptr(), i)) {
      snprintf(buf, sizeof(buf), "%d_local", i);
      PyTuple_SET_ITEM(pyvarnames.ptr(), i, PyUnicode_FromString(buf));
    }
  }

  t = PyBytes_FromStringAndSize(ccode.co_lnotab.data(), ccode.co_lnotab.size() * sizeof(ccode.co_lnotab[0]));
  auto lnotab = py::reinterpret_steal<py::object>(t);
  PyCodeObject *newCodeObj =
    PyCode_New(ccode.co_argcount, ccode.co_kwonlyargcount, ccode.co_nlocals, ccode.co_stacksize, ccode.co_flags,
               code.ptr(), pyconsts.ptr(), pynames.ptr(), pyvarnames.ptr(), pyfreevars.ptr(), pycellvars.ptr(),
               ccode.co_filename.ptr(), py::str(ccode.co_name).ptr(), ccode.co_firstlineno, lnotab.ptr());
  MS_EXCEPTION_IF_CHECK_FAIL(newCodeObj, "check code");
  return newCodeObj;
}

CodeGenerator::CodeGenerator(Graph *g, GraphAnalyzer::CapturedInfo &info)
    : graph_(g), captured_info_(info), nlocals_(g->getNlocals()), code_changed_(false) {
  make_func_handler_ = py::reinterpret_steal<py::object>(PyCFunction_New(&makeFuncHandlerDef, nullptr));

  processGraph(this->graph_);

  eraseUnusedInstr(&instrs_);

  // rewrite cell index
  int ncells = 0;
  for (auto i : this->cells_nodes_) {
    for (auto j : i->getCellOper()) {
      j->setOparg(ncells);
    }
    cell2arg_.push_back(i->getFromParam());
    i->setIndex(ncells);
    ++ncells;
  }

  for (auto i : this->frees_nodes_) {
    MS_EXCEPTION_IF_CHECK_FAIL(i->getVobj()->GetPyObject().ptr(), "check InstrNode");
    int pos = addClosure(i->getVobj()->GetPyObject());
    for (auto j : i->getCellOper()) {
      j->setOparg(ncells + pos);
    }
    i->setIndex(ncells + pos);
  }
}

void CodeGenerator::processGraph(Graph *graph, int local_off) {
  // collect inlined graph cell_free;
  for (auto i : graph->GetFrame(0).GetClosures()) {
    if (i->getType() == ValueNode::CellVar) {
      cells_nodes_.insert(i);
    } else {
      frees_nodes_.insert(i);
    }
  }

  const std::vector<InstrNode *> &instrs = graph->getInstrs();
  for (size_t bci = 0; bci < instrs.size(); ++bci) {
    if (instrs[bci] == nullptr) {
      continue;
    }
    InstrNode *i = instrs[bci];
    if (i->getType() == ValueNode::Call && reinterpret_cast<CallNode *>(i)->getSubGraph()) {
      inlineCall(reinterpret_cast<CallNode *>(i), local_off + graph->getNlocals());
      continue;
    }
    if (!fixInstr(graph, i, local_off)) {
      continue;
    }
    pushInstr(i);
  }
}

void CodeGenerator::inlineCall(CallNode *call_node, int local_off) {
  for (InstrNode *beg = call_node->getExtraOper(); beg;) {
    InstrNode *cur = beg;
    beg = reinterpret_cast<InstrNode *>(beg->getNext());
    // locals of callee
    fixInstr(call_node->getSubGraph(), cur, local_off);
    pushInstr(cur);
  }

  InstrNode *tmp = reinterpret_cast<InstrNode *>(instrs_.back());
  processGraph(call_node->getSubGraph(), local_off);
  for (; tmp != instrs_.back(); tmp = reinterpret_cast<ValueNode *>(tmp->getNext())) {
    if (tmp->getOpcode() == RETURN_VALUE) {
      tmp->setOpcode(JUMP_ABSOLUTE);
      tmp->setJump(instrs_.back());
    }
  }
  if (tmp->getOpcode() == RETURN_VALUE) {
    tmp->setOpcode(NOP);
  }
  // maybe builtin func
  int t = local_off + call_node->getSubGraph()->getNlocals();
  nlocals_ = t > nlocals_ ? t : nlocals_;
  if (call_node->GetGraph()->getInstrs().size()) {
    inlined_call_.push_back(call_node->getSubGraph());
    code_changed_ = true;
  }
}

bool CodeGenerator::fixInstr(Graph *graph, InstrNode *i, int local_off) {
  if (Utils::IsLocalAccessOp(i->getOpcode())) {
    i->setOparg(i->getOparg() + local_off);
    return true;
  }
  if (graph->GetGlobals().ptr() == this->graph_->GetGlobals().ptr()) {
    return true;
  }
  if (i->getOpcode() == MAKE_FUNCTION) {
    pushInstr(i);
    pushInstr(NewInstr(DUP_TOP));
    pushInstr(alloc_.NewValueNode(AObject::Convert(make_func_handler_), LOAD_CONST, -1, {}));
    pushInstr(NewInstr(ROT_TWO));
    pushInstr(NewInstr(LOAD_FAST, local_off + graph->getExtraLocalIndex()));
    pushInstr(NewInstr(CALL_FUNCTION, 2));
    pushInstr(NewInstr(POP_TOP));
    return false;
  }
  return true;
}

AbstractNodeList CodeGenerator::copyInstrList(const AbstractNodeList &instrs) {
  std::vector<AbstractNode *> copy_nodes;
  AbstractNodeList new_list;
  int bci = 0;
  for (auto n = reinterpret_cast<InstrNode *>(instrs.head()); n;
       n = reinterpret_cast<InstrNode *>(n->getNext()), ++bci) {
    int op = n->getOpcode();
    int arg = n->getOparg();
    n->marker_ = bci;
    auto new_node = n->getType() == ValueNode::Value
                      ? alloc_.NewValueNode(reinterpret_cast<ValueNode *>(n)->getVobj(), op, arg, {})
                      : NewInstr(op, arg);
    new_node->setName(n->getName());
    new_node->setLineNo(n->getLineNo());
    new_node->setGraph(n->GetGraph());
    copy_nodes.push_back(new_node);
    new_list.push_back(new_node);
  }
  for (auto n = reinterpret_cast<InstrNode *>(instrs.head()); n; n = reinterpret_cast<InstrNode *>(n->getNext())) {
    if (n->getJump()) {
      copy_nodes[n->marker_]->setJump(copy_nodes[n->getJump()->marker_]);
    }
  }
  return new_list;
}

static int GetOpcodeMaxStackEffect(int op, int arg, bool jump) {
  int off;
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 8
  off = PyCompile_OpcodeStackEffect(op, arg);
  if (op == NOP || op == EXTENDED_ARG) {
    return 0;
  }
#else
  off = PyCompile_OpcodeStackEffectWithJump(op, arg, jump ? 1 : -1);
#endif
  return off;
}

/**
 * Calculate max stack size, and set stack position for each instruction node to node->marker_
 *
 * \param sp start of stack depth
 * \param list instruct nodes list
 * \param clear_marker=true clear nodes marker
 * \return max depth of stack, or -1 if stack out of bound
 */
static int CalculateStackSize(const AbstractNodeList &list, int sp = 0) {
  std::unordered_map<AbstractNode *, int> blocks;
  int max_depth = 0;
  AbstractNode *head = list.head();
  for (auto i = head; sp >= 0 && i; i = i->getNext()) {
    InstrNode *instr = reinterpret_cast<InstrNode *>(i);
    int op = instr->getOpcode();
    int arg = instr->getOparg();
    AbstractNode *jump = i->getJump();
    auto iter = blocks.find(i);
    if (iter != blocks.end()) {
      sp = iter->second;
    }
    i->marker_ = sp;
    if (jump != nullptr) {
      iter = blocks.find(jump);
      int jump_sp = sp + GetOpcodeMaxStackEffect(op, arg, true);
      blocks[jump] = (iter == blocks.end()) ? jump_sp : std::max(iter->second, jump_sp);
    }
    sp += GetOpcodeMaxStackEffect(op, arg, false);
    max_depth = std::max(sp, max_depth);
  }
  return sp < 0 ? -1 : max_depth;
}

static int InstrSize(unsigned int oparg) { return oparg <= 0xff ? 1 : oparg <= 0xffff ? 2 : oparg <= 0xffffff ? 3 : 4; }

static void insert_EXTENDED_ARG(InstrNode *list_beg) {
  // insert EXTENDED_ARG
  bool re_calc;
  do {
    re_calc = false;
    InstrNode *n = list_beg;
    for (int bci = -1; n; n = reinterpret_cast<InstrNode *>(n->getNext())) {
      bci += InstrSize(n->getOparg());
      n->marker_ = bci;
    }
    n = list_beg;
    for (; n; n = reinterpret_cast<InstrNode *>(n->getNext())) {
      int isize = InstrSize(n->getOparg());
      if (n->getJump()) {
        InstrNode *tar = reinterpret_cast<InstrNode *>(n->getJump());
        // fix jump oparg
        int oparg = Utils::IsRelativeJump(n->getOpcode()) ? tar->marker_ - n->marker_ - 1 : tar->marker_;
        oparg -= (InstrSize(tar->getOparg()) - 1);
        n->setOparg(oparg * sizeof(_Py_CODEUNIT));
      }
      re_calc |= isize != InstrSize(n->getOparg());
    }
  } while (re_calc);
}

static void SetGlobal(InstrNode *n, const py::dict &used_globals) {
  MS_EXCEPTION_IF_CHECK_FAIL(strlen(n->getName()) && n->GetGraph(), "check LOAD_GLOBAL node" + n->to_str());
  PyObject *dict = n->GetGraph()->GetGlobals().ptr();
  PyObject *val = PyDict_GetItemString(dict, n->getName());
  if (val == nullptr) {
    return;
  }
  char buf[20];
  snprintf(buf, sizeof(buf), "%p", val);  // hex string
  n->setName(buf);
  py::str key(buf);
  if (!used_globals.contains(key)) {
    used_globals[key] = val;
  }
}

py::object CodeGenerator::generateCode(const AbstractNodeList &list, Code *ccode) {
  ccode->co_names.clear();
  ccode->co_consts.clear();
  ccode->co_code.clear();
  py::dict used_globals;

  for (auto i = list.head(); i; i = i->getNext()) {
    InstrNode *n = reinterpret_cast<InstrNode *>(i);
    // collect globals, names, consts, and rewrite oparg
    int op = n->getOpcode();
    if (op == LOAD_GLOBAL && n->GetGraph()) {
      SetGlobal(n, used_globals);
    }
    if (Utils::IsNameRelated(op)) {
      MS_EXCEPTION_IF_CHECK_FAIL(strlen(n->getName()), "check");
      ccode->co_names.insert({n->getName(), ccode->co_names.size()});
      n->setOparg(ccode->co_names[n->getName()]);
    }
    if (op == LOAD_CONST) {
      PyObject *o = reinterpret_cast<ValueNode *>(n)->getVobj()->GetPyObject().ptr();
      MS_EXCEPTION_IF_CHECK_FAIL(o, "check LOAD_CONST instruction node");
      ccode->co_consts.insert({o, ccode->co_consts.size()});
      n->setOparg(ccode->co_consts[o]);
    }
    MS_EXCEPTION_IF_CHECK_FAIL(op != LOAD_METHOD && op != CALL_METHOD, "must be rewrite to LOAD_ATTR, CALL_FUNCTION");
  }
  ccode->co_stacksize = CalculateStackSize(list);
  if (ccode->co_stacksize < 0) {
    CodeGenerator::Print(reinterpret_cast<InstrNode *>(list.head()), "sp");
    MS_LOG(EXCEPTION) << "check instruction list, computer stack size failed";
  }

  insert_EXTENDED_ARG(reinterpret_cast<InstrNode *>(list.head()));

  // NOTE: insert_EXTENDED_ARG use the ValueNode::marker_ as bci
  int line = ccode->co_firstlineno;
  int bci = 0;
  Graph *first_line_func = nullptr;
  ccode->co_lnotab.clear();
  for (InstrNode *n = reinterpret_cast<InstrNode *>(list.head()); n; n = reinterpret_cast<InstrNode *>(n->getNext())) {
    first_line_func = first_line_func ? first_line_func : n->GetGraph();
    if (n->GetGraph() == first_line_func && n->getLineNo() != -1 && (n->getLineNo() - line) != 0) {
      ccode->co_lnotab.push_back(sizeof(_Py_CODEUNIT) * (n->marker_ - bci));
      ccode->co_lnotab.push_back(n->getLineNo() - line);
      bci = n->marker_;
      line = n->getLineNo();
    }
    int oparg = n->getOparg();
    for (unsigned c = 0, exa = (unsigned)oparg >> 8; exa > 0; exa >>= 8, ++c) {
      ccode->co_code.insert(ccode->co_code.end() - c, _Py_MAKECODEUNIT(EXTENDED_ARG, exa & 0xff));
    }
    ccode->co_code.push_back(_Py_MAKECODEUNIT(n->getOpcode(), oparg & 0xff));
  }
  return used_globals;
}

PyCodeObject *CodeGenerator::MakeCodeObj() {
  PyCodeObject *origin_co = this->graph_->getCodeObj();
  std::string co_name = PyUnicode_AsUTF8(origin_co->co_name);
  Code ccode = {origin_co->co_argcount,
                origin_co->co_kwonlyargcount,
                nlocals_,
                0,
                origin_co->co_flags,
                origin_co->co_firstlineno,
                getNfrees(),
                cell2arg_,
                std::vector<_Py_CODEUNIT>(),
                std::unordered_map<std::string, int>(),
                std::unordered_map<PyObject *, int>(),
                py::reinterpret_borrow<py::object>(origin_co->co_filename),
                co_name};
  py::object g = generateCode(instrs_, &ccode);
  UpdateGlobals(g);
  setId(&ccode);

  return TransformCode(ccode);
}

// it is right only no branch if you remove the values of maybe_writes from maybe_reads
static void build_rws(InstrNode *beg, std::set<int> *maybe_reads, std::set<int> *maybe_writes,
                      std::set<int> *maybe_deletes, std::set<int> *maybe_visit_cells, const FrameStates &start_f) {
  for (InstrNode *i = beg; i; i = reinterpret_cast<InstrNode *>(i->getNext())) {
    int oparg = i->getOparg();
    switch (i->getOpcode()) {
      case LOAD_FAST:
        if (oparg < static_cast<int>(start_f.GetLocals().size()) &&
            start_f.Local(oparg)->getType() != ValueNode::Unbound) {
          maybe_reads->insert(oparg);
        }
        break;
      case STORE_FAST:
        maybe_writes->insert(oparg);
        break;
      case DELETE_FAST:
        if (oparg < static_cast<int>(start_f.GetLocals().size()) &&
            start_f.Local(oparg)->getType() != ValueNode::Unbound) {
          maybe_deletes->insert(oparg);
        }
        break;
      case LOAD_CLOSURE:
      case STORE_DEREF:
      case LOAD_DEREF:
      case DELETE_DEREF:
        maybe_visit_cells->insert(oparg);
        break;
      default:
        break;
    }
  }
}

AbstractNodeList CodeGenerator::GenerateMakeFunc(Code *ccode, const std::set<int> &freevars,
                                                 JitCompileResults::State compile_state) {
  AbstractNodeList res = {nullptr, nullptr};
  JitCompileResults *r = getJitCompileResults(reinterpret_cast<PyObject *>(graph_->getCodeObj()), false);
  MS_EXCEPTION_IF_NULL(r);
  ccode->co_name.append("_" + r->tbs->raw_func_name());
  setId(ccode);

  std::for_each(freevars.begin(), freevars.end(), [&res, this](int i) { res.push_back(NewInstr(LOAD_CLOSURE, i)); });
  int make_oparg = 0;
  if (freevars.size()) {
    make_oparg |= 0x08;
    res.push_back(NewInstr(BUILD_TUPLE, freevars.size()));
  }

  PyCodeObject *co = TransformCode(*ccode);
  auto code = py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(co));
  if (compile_state != JitCompileResults::NEVER_COMPILE) {
    JitCompileResults *c = getJitCompileResults(code.ptr());
    MS_EXCEPTION_IF_NULL(c);
    c->stat = compile_state;
    c->sub_routine = true;
    c->conf = r->conf;
    c->tbs = r->tbs;
    c->ms_mode_ = ccode->ms_mode_ && compile_state == JitCompileResults::GRAPH_CAPTURED;
  }

  ValueNode *load_compiled = alloc_.NewValueNode(AObject::Convert(code), LOAD_CONST, -1, {});
  res.push_back(load_compiled);
  ValueNode *load_qualname = alloc_.NewValueNode(AObject::Convert(co->co_name), LOAD_CONST, -1, {});
  res.push_back(load_qualname);
  res.push_back(NewInstr(MAKE_FUNCTION, make_oparg));
  return res;
}

AbstractNodeList CodeGenerator::generateNewInstrs(const AbstractNodeList &list, int in_stack, int out_stack,
                                                  const std::set<int> &inputs, const std::set<int> &outputs,
                                                  const std::set<int> &visit_cells, Code *ccode) {
  // local = [ stack_val1, stack_v2...local1, local3... ]
  AbstractNodeList restore_frame, graph_out, new_list;
  new_list = copyInstrList(list);
  // local stack_val
  int c = 0;
  for (; c < in_stack; ++c) {
    restore_frame.push_back(NewInstr(LOAD_FAST, c));
  }
  ccode->co_nlocals = in_stack + inputs.size();
  // key is original local index, value is new local index
  std::unordered_map<int, int> local_map;
  std::unordered_map<int, int> cells_map;
  for (auto i : inputs) {
    local_map[i] = c++;
  }
  c = 0;
  for (auto i : visit_cells) {
    cells_map[i] = c++;
  }
  c = 0;
  // rewrite index
  for (auto n = reinterpret_cast<InstrNode *>(new_list.head()); n;
       n = reinterpret_cast<InstrNode *>(n->getNext()), ++c) {
    int arg = n->getOparg();
    if (Utils::IsLocalAccessOp(n->getOpcode())) {
      if (local_map.find(arg) != local_map.end()) {
        n->setOparg(local_map[arg]);
      } else if (in_stack > 0) {
        --in_stack;
        local_map[arg] = in_stack;
        n->setOparg(in_stack);
      } else {
        local_map[arg] = ccode->co_nlocals;
        n->setOparg(ccode->co_nlocals);
        ccode->co_nlocals++;
      }
    }
    if (Utils::IsCellAccessOp(n->getOpcode())) {
      MS_EXCEPTION_IF_CHECK_FAIL(cells_map.find(arg) != cells_map.end(), "check build_rws");
      n->setOparg(cells_map[arg]);
    }
  }

  // graph out
  for (auto i : outputs) {
    MS_ASSERT(local_map.find(i) != local_map.end());
    graph_out.push_back(NewInstr(LOAD_FAST, local_map[i]));
  }
  if (out_stack + outputs.size() != 1) {
    graph_out.push_back(NewInstr(BUILD_TUPLE, out_stack + outputs.size()));
  }

  if ((reinterpret_cast<InstrNode *>(new_list.back()))->getOpcode() != RETURN_VALUE) {
    new_list.push_back(NewInstr(RETURN_VALUE, 0));
  }
  new_list.insert(new_list.head(), &restore_frame);
  new_list.insert(new_list.back(), &graph_out);

  return new_list;
}

AbstractNodeList CodeGenerator::unpackToStackAndLocal(int tmp_local, int stacks, const std::set<int> &locals) {
  AbstractNodeList res;
  if (stacks == 0) {  // optimize
    if (locals.size() == 0) {
      res.push_back(NewInstr(POP_TOP, 0));
    } else {
      res.push_back(NewInstr(UNPACK_SEQUENCE, 0 + locals.size()));
      std::for_each(locals.begin(), locals.end(), [&res, this](int i) { res.push_back(NewInstr(STORE_FAST, i)); });
    }
    return res;
  } else if (stacks == 1) {
    res.push_back(NewInstr(UNPACK_SEQUENCE, 1 + static_cast<int>(locals.size())));
    if (locals.size() != 0) {
      res.push_back(NewInstr(STORE_FAST, tmp_local));
      std::for_each(locals.begin(), locals.end(), [&res, this](int i) { res.push_back(NewInstr(STORE_FAST, i)); });
      res.push_back(NewInstr(LOAD_FAST, tmp_local));
    }
    return res;
  }
  // reverse tuple
  res.push_back(NewInstr(UNPACK_SEQUENCE, stacks + locals.size()));
  res.push_back(NewInstr(BUILD_TUPLE, stacks + locals.size()));
  res.push_back(NewInstr(UNPACK_SEQUENCE, stacks + locals.size()));
  for (auto i = locals.rbegin(); i != locals.rend(); ++i) {
    res.push_back(NewInstr(STORE_FAST, *i));
  }
  return res;
}

AbstractNodeList CodeGenerator::callSubList(const AbstractNodeList &list, int in_stack, int out_stack,
                                            const std::set<int> &inputs, const std::set<int> &outputs,
                                            const std::set<int> &deletes, const std::set<int> &visit_cells,
                                            JitCompileResults::State stat) {
  MS_EXCEPTION_IF_CHECK_FAIL(in_stack >= 0 && out_stack >= 0, "check");

  // code info
  PyCodeObject *origin_co = this->graph_->getCodeObj();
  PyObject *file_name = origin_co->co_filename;
  Code ccode = {
    in_stack + static_cast<int>(inputs.size()),                 // co_argcount
    0,                                                          // co_kwonlyargcount
    0,                                                          // co_nlocals, will set by generateNewInstrs
    0,                                                          // co_stacksize
    (origin_co->co_flags & ~(CO_VARARGS | CO_VARKEYWORDS)),     // co_flags
    (reinterpret_cast<InstrNode *>(list.head()))->getLineNo(),  // co_firstlineno
    static_cast<int>(visit_cells.size()),                       // nfrees
    std::vector<int>(),                                         // co_cell2arg
    std::vector<_Py_CODEUNIT>(),                                // co_code
    std::unordered_map<std::string, int>(),                     // co_names
    std::unordered_map<PyObject *, int>(),                      // co_consts
    py::reinterpret_borrow<py::object>(file_name),              // co_filename
    "",                                                         // co_name
    std::vector<char>(),                                        // co_lnotab
    0,                                                          // ms_mode_
    stat,                                                       // compile stat
  };

  AbstractNodeList instrs = generateNewInstrs(list, in_stack, out_stack, inputs, outputs, visit_cells, &ccode);

  // make const code
  py::object globals = generateCode(instrs, &ccode);
  UpdateGlobals(globals);

  // make func and call
  AbstractNodeList res = GenerateMakeFunc(&ccode, visit_cells, stat);
  // produce sub-func inputs
  switch (in_stack) {
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 7)
    case 0:  // optimize
      break;
    case 1:
      res.push_back(NewInstr(ROT_TWO, 0));
      break;
    case 2:
      res.push_back(NewInstr(ROT_THREE, 0));
      break;
#if (PY_MINOR_VERSION > 7)
    case 3:
      res.push_back(NewInstr(ROT_FOUR, 0));
      break;
#endif
#endif
    default:
      MS_LOG(DEBUG) << ("too many stack value, will build tuple to process\n");
      res.push_front(NewInstr(BUILD_TUPLE, in_stack));
      res.push_front(NewInstr(UNPACK_SEQUENCE, in_stack));
      res.push_front(NewInstr(BUILD_TUPLE, in_stack));  // reverse tuple
      res.push_back(NewInstr(ROT_TWO, 0));
      res.push_back(NewInstr(UNPACK_SEQUENCE, in_stack));
      break;
  }
  std::for_each(inputs.begin(), inputs.end(), [&res, this](int i) { res.push_back(NewInstr(LOAD_FAST, i)); });
  res.push_back(NewInstr(CALL_FUNCTION, ccode.co_argcount));
  // restore frame
  std::for_each(deletes.begin(), deletes.end(), [&res, this](int i) { res.push_back(NewInstr(DELETE_FAST, i)); });
  // handle sub-func outputs
  if (out_stack == 0 && outputs.size() == 1) {
    res.push_back(NewInstr(STORE_FAST, *outputs.begin()));
  }
  if (out_stack + outputs.size() != 1) {
    auto tmp = unpackToStackAndLocal(this->graph_->getExtraLocalIndex(), out_stack, outputs);
    res.insert(nullptr, &tmp);
  }
  return res;
}

AbstractNodeList CodeGenerator::callSubListWithReturn(const AbstractNodeList &list, int stack_off,
                                                      const FrameStates &f) {
  AbstractNode *node = list.head();
  for (int nodes_count = 4; node && nodes_count > 0; node = node->getNext(), --nodes_count) {
    InstrNode *instr = reinterpret_cast<InstrNode *>(node);
    if (Utils::IsCallOp(instr->getOpcode())) {
      break;
    }
  }
  if (node == nullptr) {
    // too few nodes, maybe only return
    return copyInstrList(list);
  }

  std::set<int> reads, unused, cell_visit;
  build_rws(reinterpret_cast<InstrNode *>(list.head()), &reads, &unused, &unused, &cell_visit, f);
  unused.clear();
  AbstractNodeList call_sub = callSubList(list, f.GetStacks().size() + stack_off, 1, reads, unused, unused, cell_visit,
                                          JitCompileResults::GRAPH_CANDIDATE);

  int opcode = reinterpret_cast<InstrNode *>(call_sub.back())->getOpcode();
  if (opcode != RETURN_VALUE && opcode != RAISE_VARARGS && opcode != RERAISE) {
    call_sub.push_back(NewInstr(RETURN_VALUE, 0));
  }
  return call_sub;
}

static FrameStates buildLastFrame(Graph *g, InstrNode **stop_trace_at, bool *is_loop) {
  FrameStates f = g->getLastFrame();
  f.ResizeLocal(g->getNlocals());
  InstrNode *v = g->GetStopTraceAt();
  *is_loop = g->getLoopInfo();
  while (v && v->getType() == InstrNode::Call) {
    g = reinterpret_cast<CallNode *>(v)->getSubGraph();
    if (!g) {
      break;
    }
    f.Popn(reinterpret_cast<CallNode *>(v)->getInputs().size());
    int count = f.GetLocals().size();
    f.ResizeLocal(count + g->getNlocals());
    for (auto i : g->getLastFrame().GetLocals()) {
      f.SetLocal(count++, i);
    }
    for (auto i : g->getLastFrame().GetStacks()) {
      f.Push(i);
    }
    v = g->GetStopTraceAt();
    *is_loop = g->getLoopInfo();
  }
  *stop_trace_at = v;
  return f;
}

void CodeGenerator::CutoffBytecodesIfGraphBreak() {
  InstrNode *stop_trace_at = this->graph_->GetStopTraceAt();
  if (!stop_trace_at) {
    return;
  }
  code_changed_ = true;
  bool is_loop = false;
  auto f = buildLastFrame(this->graph_, &stop_trace_at, &is_loop);

  if (this->graph_->Config().GetBoolConfig(GraphJitConfig::kPrintLastFrameIfBreakGraph)) {
    f.print();
  }
  f.ResizeLocal(nlocals_);

  // no operations captured, break at code start
  if (stop_trace_at == instrs_.head()) {
    bool succ = breakAtInterpretBlocks(stop_trace_at, f, is_loop);
    MS_EXCEPTION_IF_CHECK_FAIL(succ, "break graph failed");
    return;
  }

  std::set<int> inputs, outputs, deletes, reads, writes, cell_visit1, cell_visit2, unused1, unused2;
  AbstractNodeList tracked = instrs_;
  tracked.cutList(stop_trace_at, &instrs_);
  build_rws(reinterpret_cast<InstrNode *>(tracked.head()), &inputs, &writes, &deletes, &cell_visit1,
            this->graph_->GetFrame(0));
  build_rws(reinterpret_cast<InstrNode *>(instrs_.head()), &reads, &unused1, &unused2, &cell_visit2, f);
  for (auto i : writes) {
    if (f.Local(i)->getType() == ValueNode::Unbound) {
      continue;
    }
    if (reads.find(i) != reads.end()) {
      outputs.insert(i);
    }
    if (deletes.find(i) != deletes.end()) {
      deletes.erase(deletes.find(i));
    }
  }

  AbstractNodeList call_compiled;
  if (!this->graph_->Config().GetBoolConfig(GraphJitConfig::kDebugGraphBreakAtUnsupportedOperations)) {
    call_compiled = callSubList(tracked, 0, f.GetStacks().size(), inputs, outputs, deletes, cell_visit1,
                                JitCompileResults::GRAPH_CAPTURED);
  } else {
    // due to reorder locals, the outputs is all reads after
    call_compiled = reshapeCapturedBytecodes(f, reads);
  }
  instrs_.insert(instrs_.head(), &call_compiled);

  if (!GraphBuilder::IsByteCodeImplemented(stop_trace_at->getOpcode())) {
    // break graph directly when bytecode is not implemented
    return;
  }

  switch (stop_trace_at->getOpcode()) {
    case JUMP_IF_FALSE_OR_POP: /* fall-through */
    case JUMP_IF_TRUE_OR_POP:  /* fall-through */
    case POP_JUMP_IF_FALSE:    /* fall-through */
    case POP_JUMP_IF_TRUE:
      breakAtIf(stop_trace_at, f);
      return;
    case JUMP_IF_NOT_EXC_MATCH: /* fall-through */
    case JUMP_ABSOLUTE:         /* fall-through */
    case JUMP_FORWARD:
      MS_LOG(EXCEPTION) << "not implement break at " << Utils::GetOpName(stop_trace_at->getOpcode());
      break;
    default: {
      bool succ = breakAtInterpretBlocks(stop_trace_at, f, is_loop);
      if (succ) {
        return;
      }
      breakAtUnsupportedOperation(stop_trace_at, f);
    }
  }
}

// set break reason while build graph, avoid recursion trace fail
void CodeGenerator::breakAtUnsupportedOperation(InstrNode *stop_trace_at, const FrameStates &f) {
  int opcode = stop_trace_at->getOpcode();
  int oparg = stop_trace_at->getOparg();
#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION == 9)
  if (stop_trace_at->getType() != ValueNode::Value && stop_trace_at->getType() != ValueNode::Call &&
      opcode != UNPACK_SEQUENCE && opcode != UNPACK_EX) {
    MS_LOG(EXCEPTION) << "unimplemented break graph at " << Utils::GetOpName(stop_trace_at->getOpcode());
    return;
  }
#endif
  AbstractNodeList untracked;
  instrs_.cutList(stop_trace_at->getNext(), &untracked);

  int stack_off = PyCompile_OpcodeStackEffect(opcode, oparg);

  auto tmp = callSubListWithReturn(untracked, stack_off, f);
  instrs_.insert(nullptr, &tmp);
}

void CodeGenerator::breakAtIf(InstrNode *stop_trace_at, const FrameStates &last_f) {
  AbstractNodeList fall_block, jump_block, tmp;

  instrs_.cutList(stop_trace_at->getNext(), &fall_block);
  jump_block = {stop_trace_at->getJump(), fall_block.back()};

  tmp = callSubListWithReturn(fall_block, -1, last_f);
  instrs_.insert(nullptr, &tmp);

  int off = stop_trace_at->getOpcode() != JUMP_IF_FALSE_OR_POP && stop_trace_at->getOpcode() != JUMP_IF_TRUE_OR_POP;
  tmp = callSubListWithReturn(jump_block, -off, last_f);
  instrs_.insert(nullptr, &tmp);

  stop_trace_at->setJump(tmp.head());
}

// e.g. while..., for..., while...else..., for...else...,
static AbstractNode *FindLoopEnd(AbstractNode *loop_begin) {
  AbstractNode *loop_exit = loop_begin;
  AbstractNode *target = nullptr;
  // find loop last exit
  for (; loop_exit != target; loop_exit = loop_exit->getNext()) {
    if (loop_exit->getJump() == nullptr) {
      continue;
    }
    // if jump forward, reset target
    if (target == nullptr || target->marker_ < loop_exit->getJump()->marker_) {
      target = loop_exit->getJump();
    }
  }
  // get last exit target node, it is loop blocks end
  AbstractNode *result = target->getPres().size() ? target->getPres()[0] : nullptr;
  return result;
}

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 9)
static InstrNode *FindBlock(InstrNode *stop_trace_at, int *stack_off, bool is_loop) {
  InstrNode *block_end = nullptr;
  switch (stop_trace_at->getOpcode()) {
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 7)
    case SETUP_EXCEPT:
      block_end = reinterpret_cast<InstrNode *>(stop_trace_at->getJump());
      if (!block_end || block_end->getOpcode() != DUP_TOP) {
        break;
      }
      while (block_end && block_end->getOpcode() != END_FINALLY) {
        block_end = reinterpret_cast<InstrNode *>(block_end->getNext());
      }
      break;
    case SETUP_LOOP:
      block_end = reinterpret_cast<InstrNode *>(stop_trace_at->getJump()->getPres()[0]);
      break;
    case FOR_ITER:
      MS_EXCEPTION_IF_CHECK_FAIL(false, "shouldn't reach here, must be break at SETUP_LOOP");
      break;
#endif
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 8)
    case BEGIN_FINALLY:
    case CALL_FINALLY:
      MS_EXCEPTION_IF_CHECK_FAIL(false, "shouldn't reach here, must be break at SETUP_FINALLY");
      break;
    case FOR_ITER:
      block_end = reinterpret_cast<InstrNode *>(FindLoopEnd(stop_trace_at));
      *stack_off = -1;
      break;
#endif
    case SETUP_FINALLY: /* fall-through */
    case SETUP_WITH:
      block_end = reinterpret_cast<InstrNode *>(stop_trace_at->getJump());
      while (block_end && block_end->getOpcode() != END_FINALLY) {
        block_end = reinterpret_cast<InstrNode *>(block_end->getNext());
      }
      *stack_off = -1;
      break;
    default:
      block_end = is_loop ? reinterpret_cast<InstrNode *>(FindLoopEnd(stop_trace_at)) : nullptr;
      break;
  }
  return block_end;
}
#else
static InstrNode *findWithBlockEnd(InstrNode *n) {
  auto tar = reinterpret_cast<InstrNode *>(n->getJump());
  if (!tar || tar->getOpcode() != WITH_EXCEPT_START || !tar->getNext() || !tar->getNext()->getJump() ||
      !tar->getPres().size()) {
    return nullptr;
  }
  return reinterpret_cast<InstrNode *>(tar->getPres()[0]->getJump()->getPres()[0]);
}

// finally block has two copies in bytecodes
// only test for Python3.9
static InstrNode *findFinallyBlockEnd(InstrNode *raise_block, InstrNode *normal_block) {
  if (normal_block->getOpcode() != POP_BLOCK) {
    return nullptr;
  }
  auto i = reinterpret_cast<InstrNode *>(normal_block->getNext());
  auto j = raise_block;
  while (i->getOpcode() == j->getOpcode()) {
    i = reinterpret_cast<InstrNode *>(i->getNext());
    j = reinterpret_cast<InstrNode *>(j->getNext());
  }
  MS_ASSERT(i->getOpcode() == JUMP_FORWARD && j->getOpcode() == RERAISE && i->getJump() == j->getNext());
  return j;
}

static InstrNode *findTryBlockEnd(InstrNode *n) {
  auto tar = reinterpret_cast<InstrNode *>(n->getJump());
  if (!tar) {
    return nullptr;
  }
  int opcode = (reinterpret_cast<InstrNode *>(tar)->getOpcode());
  InstrNode *res = tar;
  if (opcode == DUP_TOP) {
    // try block without finally
    while (res && res->getOpcode() != RERAISE) {
      MS_ASSERT(res->getNext() && res->getNext()->getNext());
      res = reinterpret_cast<InstrNode *>(res->getNext()->getNext()->getJump());
    }
    return res;
  }
  // finally block has two copies in bytecodes, first is normally and end with JUMP_FORWARD, second is end with RERAISE
  InstrNode *reraise_finally_block_start = tar;
  InstrNode *normally_finally_block_start = nullptr;
  MS_ASSERT(n && n->getNext() && (reinterpret_cast<InstrNode *>(n->getNext()))->getOpcode() == SETUP_FINALLY);
  res = reinterpret_cast<InstrNode *>(n->getNext()->getJump());
  MS_ASSERT(res->getOpcode() == DUP_TOP);
  while (res && res->getOpcode() != RERAISE) {
    MS_ASSERT(res->getNext() && res->getNext()->getNext());
    res = reinterpret_cast<InstrNode *>(res->getNext()->getNext()->getJump());
  }
  MS_ASSERT(res && res->getNext() && (reinterpret_cast<InstrNode *>(res->getNext()))->getOpcode() == POP_BLOCK);
  normally_finally_block_start = reinterpret_cast<InstrNode *>(res->getNext());
  return findFinallyBlockEnd(reraise_finally_block_start, normally_finally_block_start);
}

static InstrNode *FindBlock(InstrNode *stop_trace_at, int *stack_off, bool is_loop) {
  InstrNode *block_end = nullptr;
  switch (stop_trace_at->getOpcode()) {
    case SETUP_FINALLY:
      block_end = findTryBlockEnd(stop_trace_at);
      break;
    case SETUP_WITH:
      block_end = findWithBlockEnd(stop_trace_at);
      *stack_off = -1;
      break;
    case FOR_ITER:
      *stack_off = -1;
    /* fall-through */
    default:
      block_end = is_loop ? reinterpret_cast<InstrNode *>(FindLoopEnd(stop_trace_at)) : nullptr;
      break;
  }
  return block_end;
}
#endif

// BUG: loop variable analyze
bool CodeGenerator::breakAtInterpretBlocks(InstrNode *stop_trace_at, const FrameStates &f, bool is_loop) {
  InstrNode *block_end;
  int stack_off = 0;

  // reset bci
  int bci = 0;
  for (auto i = instrs_.head(); i; i = i->getNext()) {
    i->marker_ = bci++;
  }
  block_end = FindBlock(stop_trace_at, &stack_off, is_loop);

  if (block_end == nullptr) {
    return false;
  }
  if (block_end == instrs_.back()) {
    return true;
  }
  AbstractNodeList fall_block;
  instrs_.cutList(block_end->getNext(), &fall_block);
  // reset jump
  instrs_.push_back(NewInstr(NOP, 0));
  for (auto i = instrs_.head(); i; i = i->getNext()) {
    if (i->getJump() && i->getJump() == fall_block.head()) {
      i->setJump(instrs_.back());
    }
  }

  AbstractNodeList call_sub = callSubListWithReturn(fall_block, stack_off, f);
  instrs_.insert(nullptr, &call_sub);
  return true;
}

// NOTE: use node->marker_ as use count
static int allocLocal(std::unordered_map<ValueNode *, int> *local_map, ValueNode *n, int alive_time) {
  auto pos = local_map->find(n);
  if (pos != local_map->end()) {
    return pos->second;
  }
  int res;
  for (pos = local_map->begin(); pos != local_map->end(); ++pos) {
    if (pos->first->marker_ < alive_time) {
      res = pos->second;
      local_map->erase(pos);
      local_map->insert({n, res});
      return res;
    }
  }
  res = local_map->size();
  local_map->insert({n, res});
  return res;
}

bool CodeGenerator::loadValue(std::unordered_map<ValueNode *, int> *local_map, ValueNode *i, AbstractNodeList *res,
                              bool build_value) {
  if (local_map->find(i) != local_map->end()) {
    res->push_back(NewInstr(LOAD_FAST, (*local_map)[i]));
    return true;
  }
  if (i->getType() == ValueNode::CellVar || i->getType() == ValueNode::FreeVar) {
    res->push_back(NewInstr(LOAD_CLOSURE, reinterpret_cast<CellVarNode *>(i)->getIndex()));
  } else if (i->getOpcode() == LOAD_DEREF || i->getOpcode() == LOAD_GLOBAL) {
    auto tmp = NewInstr(i->getOpcode(), i->getOparg());
    res->push_back(tmp);
    tmp->setName(i->getName());
    tmp->setGraph(i->GetGraph());
  } else if (i->getOpcode() == LOAD_CONST) {
    res->push_back(alloc_.NewValueNode(i->getVobj(), LOAD_CONST, -1, {}));
  } else if (build_value) {
    buildValue(local_map, i, 0, res);
    MS_EXCEPTION_IF_CHECK_FAIL(local_map->find(i) != local_map->end(), "check build values");
    res->push_back(NewInstr(LOAD_FAST, (*local_map)[i]));
  } else {
    return false;
  }
  return true;
}

static ValueNode *TraceMakeFuncGlobals(ValueNode *func) {
  while (func->getOpcode() == MAKE_FUNCTION) {
    int outer_func = func->GetGraph()->getExtraLocalIndex();
    const auto &frame = func->GetGraph()->GetFrame(0);
    if (static_cast<int>(frame.GetLocals().size()) <= outer_func) {
      func = nullptr;
      break;
    }
    func = frame.Local(outer_func);
  }
  return func;
}

void CodeGenerator::buildValue(std::unordered_map<ValueNode *, int> *local_map, ValueNode *n, int order,
                               AbstractNodeList *res) {
  int op = n->getOpcode();
  if (IsNonLocalValue(n) || local_map->find(n) != local_map->end()) {
    return;
  }
  int arg = n->getOparg();
  for (auto i : n->getInputs()) {
    MS_EXCEPTION_IF_CHECK_FAIL(i != &ValueNode::UnboundLocal,
                               "used before define, here " +
                                 std::string(py::str(n->GetGraph()->getCodeObj()->co_filename)) + ":" +
                                 std::to_string(n->getLineNo()));
    loadValue(local_map, i, res);
  }
  // NOTE: if support UNPACK_SEQUENCE, should check here
  auto v = NewInstr(op, arg);
  res->push_back(v);
  v->setName(n->getName());
  v->setLineNo(n->getLineNo());
  v->setGraph(n->GetGraph());
  if (IsOperationWithoutOperatorStackEffects(op)) {
    return;
  }
  if (n->marker_ == 0) {
    res->push_back(NewInstr(POP_TOP, 0));
    return;
  }
  res->push_back(NewInstr(STORE_FAST, allocLocal(local_map, n, order)));
  if (op == MAKE_FUNCTION && n->GetGraph()->GetGlobals().ptr() != this->graph_->GetGlobals().ptr()) {
    res->push_back(alloc_.NewValueNode(AObject::Convert(make_func_handler_), LOAD_CONST, -1, {}));
    res->push_back(NewInstr(LOAD_FAST, (*local_map)[n]));
    bool succ = loadValue(local_map, TraceMakeFuncGlobals(n), res, false);
    MS_EXCEPTION_IF_CHECK_FAIL(succ, "check MarkLocalAliveTime");
    res->push_back(NewInstr(CALL_FUNCTION, 2));
    res->push_back(NewInstr(POP_TOP));
  }
}

// if these values not used, allowed delete these operations
// NOTE: use marker_ as usage count
static bool no_side_effect_op(ValueNode *v) {
  if (v->marker_ != 0) {
    return false;
  }
  int op = v->getOpcode();
  if (Utils::IsNoSideEffectOp(op)) {
    return true;
  }
  if ((op == BUILD_MAP || op == BUILD_CONST_KEY_MAP || op == BUILD_SET) && v->getOparg() == 0) {
    // BUILD_MAP will call __hash__
    return true;
  }
  return false;
}

/**
 * Mark values live time
 * traverse all values in reverse order, set alive time for each input value
 * the inputs of all values is before these values
 */
static void MarkLocalAliveTime(const std::vector<ValueNode *> &order_values) {
  for (int i = order_values.size() - 1; i >= 0; --i) {
    for (auto node : order_values[i]->getInputs()) {
      node->marker_ = node->marker_ < i ? i : node->marker_;
    }
    // extra used for MAKE_FUNCTION, use it to restore globals dictionary
    if (order_values[i]->getOpcode() != MAKE_FUNCTION) {
      continue;
    }
    ValueNode *root = TraceMakeFuncGlobals(order_values[i]);
    if (root) {
      root->marker_ = root->marker_ < i ? i : root->marker_;
    }
  }
}

// call this must be after call MarkLocalAliveTime
static void EraseDeadLocal(std::vector<ValueNode *> *values) {
  size_t len;
  do {
    len = values->size();
    auto find_func = [](ValueNode *i) { return i->marker_ == 0 && Utils::IsGeneralNoSideEffectOp(i->getOpcode()); };
    values->erase(std::remove_if(values->begin(), values->end(), find_func), values->end());
    MarkLocalAliveTime(*values);
  } while (len > values->size());
}

AbstractNodeList CodeGenerator::buildValues(std::unordered_map<ValueNode *, int> *local_map,
                                            const std::vector<ValueNode *> &order_values) {
  for (auto node : order_values) {
    node->marker_ = (node->marker_ == INT_MAX) ? INT_MAX : 0;  // clear marker
  }
  // mark alive times
  MarkLocalAliveTime(order_values);
  // dead local erase
  std::vector<ValueNode *> values = order_values;
  if (graph_->Config().GetBoolConfig(GraphJitConfig::kEnableEliminateUnusedOperation)) {
    EraseDeadLocal(&values);
  }
  AbstractNodeList res;
  for (size_t i = 0; i < values.size(); ++i) {
    bool is_unused = values[i]->marker_ == 0;
    if (is_unused && no_side_effect_op(values[i])) {
      continue;
    }
    AbstractNodeList op;
    buildValue(local_map, values[i], i, &op);
    res.insert(nullptr, &op);
  }
  return res;
}

/**
 * mindspore func_graph assume all scale and tuple, list, dict parameter is const,
 * is same as globals value. so if top func not guard these value, it should passed
 * to graph by global value and guard all inputs that passed by global.
 */
static bool IsPassedByParameter(ValueNode *param) {
  AObject *value_info = param->getVobj();
  return value_info->IsMindSporeSupportedType();
}

AbstractNodeList CodeGenerator::reshapeGraphBytecodes(std::unordered_map<ValueNode *, int> *graph_local_map,
                                                      std::unordered_map<ValueNode *, int> *interpret_local_map,
                                                      std::unordered_map<ValueNode *, int> *graph_outputs) {
  auto &inputs = this->captured_info_.captured_locals.inputs;
  int argc = 0;
  int global_input_pos = inputs.size();
  AbstractNodeList graph_load_inputs;
  AbstractNodeList interpret_load_inputs;
  // prepare inputs local map
  for (auto i : inputs) {
    if (IsNonLocalValue(i)) {
      continue;
    }
    {  // interpret load arguments
      AbstractNodeList ld;
      bool has_input = loadValue(interpret_local_map, i, &ld, false);
      MS_EXCEPTION_IF_CHECK_FAIL(has_input, "must be has input");
      interpret_load_inputs.insert(nullptr, &ld);
    }
    // Identify parameters and global variables
    bool pass_parameter = IsPassedByParameter(i);
    if (pass_parameter) {
      graph_local_map->insert({i, argc++});
      continue;
    }
    // guard globals input for func graph when the func_graph called
    --global_input_pos;
    graph_local_map->insert({i, global_input_pos});
    // interpret load arguments
    InstrNode *n = NewInstr(STORE_GLOBAL);
    auto nam = getGraphInputsKey(i);
    n->setName(nam.c_str());
    interpret_load_inputs.push_back(n);
    // graph load input
    n = NewInstr(LOAD_GLOBAL);
    n->setName(nam.c_str());
    graph_load_inputs.push_back(n);
    graph_load_inputs.push_back(NewInstr(STORE_FAST, global_input_pos));
  }

  AbstractNodeList compile = buildValues(graph_local_map, this->captured_info_.captured_locals.order);
  if (compile.head() == nullptr) {
    // no value produce
    MS_EXCEPTION_IF_CHECK_FAIL(graph_outputs->empty(), "check record");
    return {nullptr, nullptr};
  }
  compile.insert(compile.head(), &graph_load_inputs);

  std::vector<InstrNode *> order_out(graph_outputs->size());
  for (auto i : *graph_outputs) {
    MS_EXCEPTION_IF_CHECK_FAIL(graph_local_map->find(i.first) != graph_local_map->end(), "check local map build");
    order_out[i.second] = NewInstr(LOAD_FAST, (*graph_local_map)[i.first]);
  }
  std::for_each(order_out.begin(), order_out.end(), [&compile](InstrNode *node) { compile.push_back(node); });
  if (graph_outputs->size() != 1) {
    compile.push_back(NewInstr(BUILD_TUPLE, graph_outputs->size()));
  }
  compile.push_back(NewInstr(RETURN_VALUE));

  // code info
  PyCodeObject *origin_co = this->graph_->getCodeObj();
  Code ccode = {
    argc,                                                        // co_argcount
    0,                                                           // co_kwonlyargcount
    static_cast<int>(graph_local_map->size()),                   // co_nlocals
    0,                                                           // co_stacksize
    (origin_co->co_flags & ~(CO_VARARGS | CO_VARKEYWORDS)),      // co_flags
    origin_co->co_firstlineno,                                   // co_firstlineno
    0,                                                           // nfrees
    std::vector<int>(),                                          // co_cell2arg, closure is mindspore unsupported
    std::vector<_Py_CODEUNIT>(),                                 // co_code
    std::unordered_map<std::string, int>(),                      // co_names
    std::unordered_map<PyObject *, int>(),                       // co_consts
    py::reinterpret_borrow<py::object>(origin_co->co_filename),  // co_filename
    "_compile",                                                  // co_name
    std::vector<char>(),                                         // co_lnotab
    this->captured_info_.must_be_graph_mode_,                    // ms_mode_
    JitCompileResults::GRAPH_CAPTURED,                           // compile stat
  };
  py::object globals = generateCode(compile, &ccode);
  UpdateGlobals(globals);
  std::set<int> unused;
  AbstractNodeList make_func = GenerateMakeFunc(&ccode, unused, JitCompileResults::GRAPH_CAPTURED);

  make_func.insert(nullptr, &interpret_load_inputs);
  make_func.push_back(NewInstr(CALL_FUNCTION, argc));
  return make_func;
}

// NOTE: use ValueNode::marker_ as usage count
static void MarkAliveValue(const std::vector<std::set<ValueNode *> *> &values) {
  for (auto &i : values) {
    for (auto &j : *i) {
      j->marker_ = INT_MAX;
    }
  }
}

static void collectAliveValue(const FrameStates &last_frame, const std::set<int> &outputs, ValueNode *ret_v,
                              std::set<ValueNode *> *alive_v) {
  if (ret_v) {
    alive_v->insert(ret_v);
  }
  for (auto i : outputs) {
    alive_v->insert(last_frame.Local(i));
  }
  for (auto j : last_frame.GetStacks()) {
    alive_v->insert(j);
  }
  alive_v->erase(&ValueNode::UnboundLocal);
}

AbstractNodeList CodeGenerator::reshapeCapturedBytecodes(const FrameStates &last_frame, const std::set<int> &outputs,
                                                         ValueNode *ret_v) {
  const auto &nodes = this->captured_info_;

  std::set<ValueNode *> alive_v;
  collectAliveValue(last_frame, outputs, ret_v, &alive_v);
  MarkAliveValue({&alive_v, &this->captured_info_.captured_locals.inputs});

  std::unordered_map<ValueNode *, int> graph_outputs;
  // collect graph outputs, ensure all alive value could be found in escaped_locals or captured.values
  for (auto i : alive_v) {
    if (nodes.escaped_locals.find(i) == nodes.escaped_locals.end() && !IsNonLocalValue(i)) {
      graph_outputs.insert({i, graph_outputs.size()});
    }
  }

  std::unordered_map<ValueNode *, int> local_map;
  // prepare local_map
  for (auto i : this->graph_->GetFrame(0).GetLocals()) {
    if (i->getType() != AbstractNode::Param) {
      continue;
    }
    local_map.insert({i, i->getOparg()});
  }
  // produce interpret values
  AbstractNodeList res = buildValues(&local_map, nodes.ordered_escaped_locals);
  // placeholder for graph_output
  int graph_out_index = local_map.size();
  local_map.insert({nullptr, graph_out_index});
  local_map.insert({&ValueNode::UnboundLocal, local_map.size()});

  std::unordered_map<ValueNode *, int> graph_local_map;
  AbstractNodeList call_func = reshapeGraphBytecodes(&graph_local_map, &local_map, &graph_outputs);
  // has operations compiled
  if (call_func.head()) {
    res.insert(nullptr, &call_func);
    res.push_back(NewInstr(STORE_FAST, graph_out_index));
  }

  {  // restore frame
    auto getValueWithAssert = [this, &graph_outputs, &local_map, graph_out_index](ValueNode *i) -> AbstractNodeList {
      AbstractNodeList instrs;
      bool has_input = this->loadValue(&local_map, i, &instrs, false);
      if (has_input) {
        return instrs;
      }
      MS_EXCEPTION_IF_CHECK_FAIL(graph_outputs.find(i) != graph_outputs.end(), "check record");
      instrs.push_back(this->NewInstr(LOAD_FAST, graph_out_index));
      if (graph_outputs.size() == 1) {
        return instrs;
      }
      instrs.push_back(this->alloc_.NewValueNode(AObject::Convert(py::int_(graph_outputs[i])), LOAD_CONST, -1, {}));
      instrs.push_back(this->NewInstr(BINARY_SUBSCR));
      return instrs;
    };
    // first, restore stack
    for (auto i : last_frame.GetStacks()) {
      AbstractNodeList ld = getValueWithAssert(i);
      res.insert(nullptr, &ld);
    }
    // restore reordered local
    std::vector<int> tmp;
    for (auto i : outputs) {
      auto v = last_frame.Local(i);
      if (local_map.find(v) != local_map.end() && i == local_map[v]) {
        continue;
      }
      AbstractNodeList ld = getValueWithAssert(v);
      res.insert(nullptr, &ld);
      tmp.push_back(i);
    }
    for (auto i = tmp.rbegin(); i != tmp.rend(); ++i) {
      res.push_back(NewInstr(STORE_FAST, *i));
    }
    if (ret_v) {
      AbstractNodeList ld = getValueWithAssert(ret_v);
      res.insert(nullptr, &ld);
      res.push_back(NewInstr(RETURN_VALUE));
    }
  }

  nlocals_ = std::max((size_t)nlocals_, local_map.size());
  return res;
}

bool CodeGenerator::TryToBreakGraphIfUnsupported() {
  if (!this->graph_->Config().GetBoolConfig(GraphJitConfig::kDebugGraphBreakAtUnsupportedOperations)) {
    return false;
  }
  if (this->graph_->GetStopTraceAt()) {
    // do this in fix break graph
    return false;
  }
  bool support_param = true;
  for (auto i : this->graph_->GetFrame(0).GetLocals()) {
    if (!i || i->getType() != AbstractNode::Param) {
      continue;
    }
    auto t = i->getVobj()->GetType();
    if (t == AObject::kTypeCell || t == AObject::kTypePrimitive || !i->IsMindsporeSupportedOperation()) {
      support_param = false;
      break;
    }
  }
  if (this->captured_info_.ordered_escaped_locals.empty() && support_param) {
    // no unsupported operations
    return false;
  }
  FrameStates empty;
  std::set<int> out;
  instrs_ = reshapeCapturedBytecodes(empty, out, this->graph_->getRetVal());
  code_changed_ = true;
  return true;
}

void CodeGenerator::Print(const AbstractNode *list_beg, const char *marker_info) {
  for (auto i = list_beg; i; i = i->getNext()) {
    GRAPH_JIT_LOG_F("%s %s:%d\n", i->to_str().c_str(), marker_info, i->marker_);
  }
  GRAPH_JIT_LOG_F("\n");
}

std::string CodeGenerator::getGraphInputsKey(const ValueNode *v) {
  std::stringstream s;
  s << ".graph.inputs." << v << '.' << v->getName();
  return s.str();
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
