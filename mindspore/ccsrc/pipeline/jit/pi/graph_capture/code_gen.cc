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

#include "pipeline/jit/pi/graph_capture/code_gen.h"
#include <algorithm>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "pipeline/jit/pi/graph_capture/graph_build.h"

namespace mindspore {
namespace jit {
namespace graph {
static bool IsOperationWithoutOperatorStackEffects(int op) {
  return op == STORE_DEREF || op == DELETE_DEREF || op == STORE_GLOBAL || op == DELETE_GLOBAL || op == STORE_ATTR ||
         op == DELETE_ATTR || op == STORE_SUBSCR || op == DELETE_SUBSCR || op == IMPORT_STAR || op == RAISE_VARARGS ||
         op == RERAISE;
}

static void reSetJumpToUnusedInstr(InstrNode *n) {
  if (!n->GetJump()) {
    return;
  }
  auto tar = reinterpret_cast<InstrNode *>(n->GetJump());
  do {
    int op = tar->GetOpcode();
    bool jump_next = (op == JUMP_ABSOLUTE || op == JUMP_FORWARD) && tar->GetJump() == tar->GetNext();
    if (op != NOP && op != EXTENDED_ARG && !jump_next) {
      break;
    }
    n->SetJump(tar->GetNext());
    tar = reinterpret_cast<InstrNode *>(tar->GetNext());
  } while (true);
}

static void eraseUnusedInstr(AbstractNodeList *list) {
  std::set<InstrNode *> removed;
  for (auto n = reinterpret_cast<InstrNode *>(list->head()); n; n = reinterpret_cast<InstrNode *>(n->GetNext())) {
    // jump target shouldn't be a removed node
    MS_ASSERT(!n->GetJump() || removed.find(reinterpret_cast<InstrNode *>(n->GetJump())) == removed.end());
    reSetJumpToUnusedInstr(n);
    int op = n->GetOpcode();
    while (((op == JUMP_ABSOLUTE || op == JUMP_FORWARD) && n->GetJump() == n->GetNext()) || n->GetOpcode() == NOP ||
           n->GetOpcode() == EXTENDED_ARG) {
      auto tmp = n;
      n = reinterpret_cast<InstrNode *>(n->GetNext());
      list->erase(tmp);
      removed.insert(tmp);
      op = n->GetOpcode();
    }
  }
}

static PyCodeObject *TransformCode(const CodeGenerator::Code &ccode) {
  const char *bytecode = reinterpret_cast<const char *>(ccode.co_code.data());
  auto code = py::bytes(bytecode, ccode.co_code.size() * sizeof(ccode.co_code[0]));
  auto pyconsts = py::tuple(ccode.co_consts.size());
  auto pynames = py::tuple(ccode.co_names.size());
  for (auto i : ccode.co_consts) {
    Py_XINCREF(i.first);
    PyTuple_SET_ITEM(pyconsts.ptr(), i.second, i.first);
  }
  for (const auto &i : ccode.co_names) {
    PyTuple_SET_ITEM(pynames.ptr(), i.second, PyUnicode_FromString(i.first.c_str()));
  }

  std::set<std::string> vars;
  auto pyvarnames = py::tuple(ccode.co_nlocals);
  for (int i = 0; i < ccode.co_nlocals; ++i) {
    std::string n;
    if (i < static_cast<int>(ccode.co_varnames.size())) {
      n = ccode.co_varnames[i];
    } else {
      n = std::to_string(i) + "_local";
    }
    while (vars.find(n) != vars.end()) {
      n = n + "_" + std::to_string(i);
    }
    vars.insert(n);
    pyvarnames[i] = py::str(n);
  }
  auto pyfreevars = py::tuple(ccode.co_freevars.size());
  for (int i = ccode.co_freevars.size() - 1; i >= 0; --i) {
    PyTuple_SET_ITEM(pyfreevars.ptr(), i, PyUnicode_FromString(ccode.co_freevars[i].c_str()));
  }
  auto pycellvars = py::tuple(ccode.co_cellvars.size());
  for (int i = ccode.co_cellvars.size() - 1; i >= 0; --i) {
    PyTuple_SET_ITEM(pycellvars.ptr(), i, PyUnicode_FromString(ccode.co_cellvars[i].c_str()));
  }

  auto lnotab = py::bytes(ccode.co_lnotab.data(), ccode.co_lnotab.size() * sizeof(ccode.co_lnotab[0]));
  PyCodeObject *newCodeObj =
    PyCode_New(ccode.co_argcount, ccode.co_kwonlyargcount, ccode.co_nlocals, ccode.co_stacksize, ccode.co_flags,
               code.ptr(), pyconsts.ptr(), pynames.ptr(), pyvarnames.ptr(), pyfreevars.ptr(), pycellvars.ptr(),
               ccode.co_filename.ptr(), py::str(ccode.co_name).ptr(), ccode.co_firstlineno, lnotab.ptr());
  MS_EXCEPTION_IF_CHECK_FAIL(newCodeObj, std::string() + "check code because of " + py::error_already_set().what());
  return newCodeObj;
}

CodeGenerator::CodeGenerator(Graph *g, GraphAnalyzer::CapturedInfo &info)
    : graph_(g), captured_info_(info), nlocals_(g->GetNlocals()), code_changed_(false) {
  ProcessGraph(this->graph_);

  eraseUnusedInstr(&instrs_);

  int cell_index = 0;
  for (auto i : this->graph_->GetFrame(0).GetClosures()) {
    MS_EXCEPTION_IF_CHECK_FAIL(i->GetIndex() == cell_index++,
                               "not implement inline function with free variable merge, check cell index");
  }
}

void CodeGenerator::ProcessGraph(Graph *graph, int local_off) {
  if (graph != this->graph_) {
    MS_EXCEPTION_IF_CHECK_FAIL(!graph->GetFrame(0).GetClosures().size(),
                               "not implement inline function with free variable merge, check inline policy");
  }

  const std::vector<InstrNode *> &instrs = graph->GetInstrs();
  for (size_t bci = 0; bci < instrs.size(); ++bci) {
    if (instrs[bci] == nullptr) {
      continue;
    }
    InstrNode *i = instrs[bci];
    if (i->GetType() == ValueNode::Call && reinterpret_cast<CallNode *>(i)->GetSubGraph()) {
      InlineCall(reinterpret_cast<CallNode *>(i), local_off + graph->GetNlocals());
      continue;
    }
    if (!FixInstr(graph, i, local_off)) {
      continue;
    }
    PushInstr(i);
  }
  if ((reinterpret_cast<InstrNode *>(instrs_.back()))->GetOpcode() != RETURN_VALUE) {
    instrs_.push_back(NewInstr(RETURN_VALUE, 0));
  }
}

void CodeGenerator::InlineCall(CallNode *call_node, int local_off) {
  for (InstrNode *beg = call_node->GetExtraOper(); beg;) {
    InstrNode *cur = beg;
    beg = reinterpret_cast<InstrNode *>(beg->GetNext());
    // locals of callee
    FixInstr(call_node->GetSubGraph(), cur, local_off);
    PushInstr(cur);
  }

  InstrNode *tmp = reinterpret_cast<InstrNode *>(instrs_.back());
  ProcessGraph(call_node->GetSubGraph(), local_off);
  for (; tmp != instrs_.back(); tmp = reinterpret_cast<ValueNode *>(tmp->GetNext())) {
    if (tmp->GetOpcode() == RETURN_VALUE) {
      tmp->SetOpcode(JUMP_ABSOLUTE);
      tmp->SetJump(instrs_.back());
    }
  }
  if (tmp->GetOpcode() == RETURN_VALUE) {
    tmp->SetOpcode(NOP);
  }
  // maybe builtin func
  int t = local_off + call_node->GetSubGraph()->GetNlocals();
  nlocals_ = t > nlocals_ ? t : nlocals_;
  if (call_node->GetGraph()->GetInstrs().size()) {
    code_changed_ = true;
  }
}

bool CodeGenerator::FixInstr(Graph *graph, InstrNode *i, int local_off) {
  if (Utils::IsLocalAccessOp(i->GetOpcode())) {
    i->SetOparg(i->GetOparg() + local_off);
    return true;
  }
  if (graph->GetGlobals().ptr() == this->graph_->GetGlobals().ptr()) {
    return true;
  }
  return true;
}

AbstractNodeList CodeGenerator::CopyInstrList(const AbstractNodeList &instrs) {
  std::vector<AbstractNode *> copy_nodes;
  AbstractNodeList new_list;
  int bci = 0;
  for (auto n = reinterpret_cast<InstrNode *>(instrs.head()); n;
       n = reinterpret_cast<InstrNode *>(n->GetNext()), ++bci) {
    int op = n->GetOpcode();
    int arg = n->GetOparg();
    n->marker_ = bci;
    auto new_node = n->GetType() == ValueNode::Value
                      ? alloc_.NewValueNode(reinterpret_cast<ValueNode *>(n)->GetVobj(), op, arg, {})
                      : NewInstr(op, arg);
    new_node->SetName(n->GetName());
    new_node->SetLineNo(n->GetLineNo());
    new_node->SetGraph(n->GetGraph());
    copy_nodes.push_back(new_node);
    new_list.push_back(new_node);
  }
  for (auto n = reinterpret_cast<InstrNode *>(instrs.head()); n; n = reinterpret_cast<InstrNode *>(n->GetNext())) {
    if (n->GetJump()) {
      copy_nodes[n->marker_]->SetJump(copy_nodes[n->GetJump()->marker_]);
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
  int flag = 0;
  AbstractNode *head = list.head();
  for (auto i = head; sp >= 0 && i; i = i->GetNext()) {
    InstrNode *instr = reinterpret_cast<InstrNode *>(i);
    int op = instr->GetOpcode();
    int arg = instr->GetOparg();
    AbstractNode *jump = i->GetJump();
    auto iter = blocks.find(i);
    if (iter != blocks.end()) {
      flag = 0;
      sp = iter->second;
    } else if (flag == 1) {
      continue;
    }
    i->marker_ = sp;
    if (op == RAISE_VARARGS || op == RETURN_VALUE || op == RERAISE) {
      flag = 1;
    }
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
    for (int bci = -1; n; n = reinterpret_cast<InstrNode *>(n->GetNext())) {
      bci += InstrSize(n->GetOparg());
      n->marker_ = bci;
    }
    n = list_beg;
    for (; n; n = reinterpret_cast<InstrNode *>(n->GetNext())) {
      int isize = InstrSize(n->GetOparg());
      if (n->GetJump()) {
        InstrNode *tar = reinterpret_cast<InstrNode *>(n->GetJump());
        // fix jump oparg
        int oparg = Utils::IsRelativeJump(n->GetOpcode()) ? tar->marker_ - n->marker_ - 1 : tar->marker_;
        oparg -= (InstrSize(tar->GetOparg()) - 1);
        n->SetOparg(oparg * sizeof(_Py_CODEUNIT));
      }
      re_calc |= isize != InstrSize(n->GetOparg());
    }
  } while (re_calc);
}

static void SetGlobal(InstrNode *n, const py::dict &used_globals) {
  MS_EXCEPTION_IF_CHECK_FAIL(!n->GetName().empty() && n->GetGraph(), "check LOAD_GLOBAL node" + n->to_str());
  PyObject *dict = n->GetGraph()->GetGlobals().ptr();
  PyObject *val = PyDict_GetItemString(dict, n->GetName().c_str());
  if (val == nullptr) {
    return;
  }
  std::stringstream name;
  int len = n->GetName().size();
  constexpr const int max_len = 40;
  name << n->GetName().substr(0, std::min(len, max_len)) << "<" << val << ">";

  n->SetName(name.str().c_str());
  py::str key(n->GetName());

  PyObject *old = PyDict_GetItem(used_globals.ptr(), key.ptr());
  MS_EXCEPTION_IF_CHECK_FAIL(old == nullptr || old == val, "duplicate global value " + std::string(py::str(key)) + ":" +
                                                             std::string(py::str(old)) + "->" +
                                                             std::string(py::str(val)));
  PyDict_SetItem(used_globals.ptr(), key.ptr(), val);
}

py::object GenerateCode(const AbstractNodeList &list, CodeGenerator::Code *ccode) {
  ccode->co_names.clear();
  ccode->co_consts.clear();
  ccode->co_code.clear();
  py::dict used_globals;

  for (auto i = list.head(); i; i = i->GetNext()) {
    InstrNode *n = reinterpret_cast<InstrNode *>(i);
    // collect globals, names, consts, and rewrite oparg
    int op = n->GetOpcode();
    if (op == LOAD_GLOBAL && n->GetGraph()) {
      SetGlobal(n, used_globals);
    }
    if (Utils::IsNameRelated(op)) {
      MS_EXCEPTION_IF_CHECK_FAIL(!n->GetName().empty(), "check");
      ccode->co_names.insert({n->GetName(), ccode->co_names.size()});
      n->SetOparg(ccode->co_names[n->GetName()]);
    }
    if (op == LOAD_CONST) {
      PyObject *o = reinterpret_cast<ValueNode *>(n)->GetVobj()->GetPyObject().ptr();
      MS_EXCEPTION_IF_CHECK_FAIL(o, "check LOAD_CONST instruction node");
      ccode->co_consts.insert({o, ccode->co_consts.size()});
      n->SetOparg(ccode->co_consts[o]);
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
  for (InstrNode *n = reinterpret_cast<InstrNode *>(list.head()); n; n = reinterpret_cast<InstrNode *>(n->GetNext())) {
    first_line_func = first_line_func ? first_line_func : n->GetGraph();
    if (n->GetGraph() == first_line_func && n->GetLineNo() != -1 && (n->GetLineNo() - line) != 0) {
      ccode->co_lnotab.push_back(sizeof(_Py_CODEUNIT) * (n->marker_ - bci));
      ccode->co_lnotab.push_back(n->GetLineNo() - line);
      bci = n->marker_;
      line = n->GetLineNo();
    }
    int oparg = n->GetOparg();
    for (unsigned c = 0, exa = (unsigned)oparg >> 8; exa > 0; exa >>= 8, ++c) {
      ccode->co_code.insert(ccode->co_code.end() - c, _Py_MAKECODEUNIT(EXTENDED_ARG, exa & 0xff));
    }
    ccode->co_code.push_back(_Py_MAKECODEUNIT(n->GetOpcode(), oparg & 0xff));
  }
  return used_globals;
}

PyCodeObject *CodeGenerator::MakeCodeObj() {
  PyCodeObject *origin_co = this->graph_->GetCodeObj();
  std::string co_name = PyUnicode_AsUTF8(origin_co->co_name);
  Code ccode = {
    origin_co->co_argcount,
    origin_co->co_kwonlyargcount,
    nlocals_,
    0,
    origin_co->co_flags,
    origin_co->co_firstlineno,
    py::cast<std::vector<std::string>>(origin_co->co_varnames),
    py::cast<std::vector<std::string>>(origin_co->co_freevars),
    py::cast<std::vector<std::string>>(origin_co->co_cellvars),
    std::vector<_Py_CODEUNIT>(),
    std::unordered_map<std::string, int>(),
    std::unordered_map<PyObject *, int>(),
    py::reinterpret_borrow<py::object>(origin_co->co_filename),
    co_name,
  };
  py::object g = GenerateCode(instrs_, &ccode);
  UpdateGlobals(g);
  SetId(&ccode);

  return TransformCode(ccode);
}

// it is right only no branch if you remove the values of maybe_writes from maybe_reads
static void build_rws(InstrNode *beg, std::set<int> *maybe_reads, std::set<int> *maybe_writes,
                      std::set<int> *maybe_deletes, std::set<int> *maybe_visit_cells, const FrameStates &start_f) {
  for (InstrNode *i = beg; i; i = reinterpret_cast<InstrNode *>(i->GetNext())) {
    int oparg = i->GetOparg();
    switch (i->GetOpcode()) {
      case LOAD_FAST:
        if (oparg < static_cast<int>(start_f.GetLocals().size()) &&
            start_f.Local(oparg)->GetType() != ValueNode::Unbound) {
          maybe_reads->insert(oparg);
        }
        break;
      case STORE_FAST:
        maybe_writes->insert(oparg);
        break;
      case DELETE_FAST:
        if (oparg < static_cast<int>(start_f.GetLocals().size()) &&
            start_f.Local(oparg)->GetType() != ValueNode::Unbound) {
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
  JitCompileResults *r = getJitCompileResults(reinterpret_cast<PyObject *>(graph_->GetCodeObj()), false);
  MS_EXCEPTION_IF_NULL(r);
  ccode->co_name.append("_" + r->tbs->raw_func_name());
  SetId(ccode);

  MS_EXCEPTION_IF_CHECK_FAIL(freevars.size() == ccode->co_freevars.size(), "code must be generate free variable names");
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
    JitCompileResults *parent = getJitCompileResults(reinterpret_cast<PyObject *>(this->graph_->GetCodeObj()), false);
    MS_EXCEPTION_IF_NULL(c);
    c->stat = compile_state;
    c->conf = r->conf;
    c->tbs = r->tbs;
    c->parent_ = parent;
    parent->children_.push_back(c);
  }

  ValueNode *load_compiled = alloc_.NewValueNode(AObject::Convert(code), LOAD_CONST, -1, {});
  res.push_back(load_compiled);
  ValueNode *load_qualname = alloc_.NewValueNode(AObject::Convert(co->co_name), LOAD_CONST, -1, {});
  res.push_back(load_qualname);
  res.push_back(NewInstr(MAKE_FUNCTION, make_oparg));
  return res;
}

AbstractNodeList CodeGenerator::GenerateNewInstrs(const AbstractNodeList &list, int in_stack, int out_stack,
                                                  const std::set<int> &inputs, const std::set<int> &outputs,
                                                  const std::set<int> &visit_cells, Code *ccode) {
  PyCodeObject *co = this->graph_->GetCodeObj();
  int ncells = PyTuple_GET_SIZE(co->co_cellvars);

  // local = [ stack_val1, stack_v2...local1, local3... ]
  AbstractNodeList restore_frame, graph_out, new_list;
  new_list = CopyInstrList(list);
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
    if (i < ncells) {
      ccode->co_freevars.push_back(PyUnicode_AsUTF8(PyTuple_GET_ITEM(co->co_cellvars, i)));
    } else {
      ccode->co_freevars.push_back(PyUnicode_AsUTF8(PyTuple_GET_ITEM(co->co_freevars, i - ncells)));
    }
  }
  c = 0;
  // rewrite index
  for (auto n = reinterpret_cast<InstrNode *>(new_list.head()); n;
       n = reinterpret_cast<InstrNode *>(n->GetNext()), ++c) {
    int arg = n->GetOparg();
    if (Utils::IsLocalAccessOp(n->GetOpcode())) {
      if (local_map.find(arg) != local_map.end()) {
        n->SetOparg(local_map[arg]);
      } else if (in_stack > 0) {
        --in_stack;
        local_map[arg] = in_stack;
        n->SetOparg(in_stack);
      } else {
        local_map[arg] = ccode->co_nlocals;
        n->SetOparg(ccode->co_nlocals);
        ccode->co_nlocals++;
      }
    }
    if (Utils::IsCellAccessOp(n->GetOpcode())) {
      MS_EXCEPTION_IF_CHECK_FAIL(cells_map.find(arg) != cells_map.end(), "check build_rws");
      n->SetOparg(cells_map[arg]);
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

  if ((reinterpret_cast<InstrNode *>(new_list.back()))->GetOpcode() != RETURN_VALUE) {
    new_list.push_back(NewInstr(RETURN_VALUE, 0));
  }
  new_list.insert(new_list.head(), &restore_frame);
  new_list.insert(new_list.back(), &graph_out);

  return new_list;
}

AbstractNodeList CodeGenerator::UnpackToStackAndLocal(int tmp_local, int stacks, const std::set<int> &locals) {
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

AbstractNodeList CodeGenerator::CallSubList(const AbstractNodeList &list, int in_stack, int out_stack,
                                            const std::set<int> &inputs, const std::set<int> &outputs,
                                            const std::set<int> &deletes, const std::set<int> &visit_cells,
                                            JitCompileResults::State stat) {
  MS_EXCEPTION_IF_CHECK_FAIL(in_stack >= 0 && out_stack >= 0, "check");

  // code info
  PyCodeObject *origin_co = this->graph_->GetCodeObj();
  PyObject *file_name = origin_co->co_filename;
  Code ccode = {
    in_stack + static_cast<int>(inputs.size()),                 // co_argcount
    0,                                                          // co_kwonlyargcount
    0,                                                          // co_nlocals, will set by GenerateNewInstrs
    0,                                                          // co_stacksize
    (origin_co->co_flags & ~(CO_VARARGS | CO_VARKEYWORDS)),     // co_flags
    (reinterpret_cast<InstrNode *>(list.head()))->GetLineNo(),  // co_firstlineno
    std::vector<std::string>(),                                 // co_varnames
    std::vector<std::string>(),                                 // co_freevars
    std::vector<std::string>(),                                 // co_cellvars
    std::vector<_Py_CODEUNIT>(),                                // co_code
    std::unordered_map<std::string, int>(),                     // co_names
    std::unordered_map<PyObject *, int>(),                      // co_consts
    py::reinterpret_borrow<py::object>(file_name),              // co_filename
    "",                                                         // co_name
    std::vector<char>(),                                        // co_lnotab
    stat,                                                       // compile stat
  };
  for (auto i = 0; i < ccode.co_argcount; ++i) {
    ccode.co_varnames.push_back(std::to_string(i) + "_local");
  }

  // rewrite cell index
  AbstractNodeList instrs = GenerateNewInstrs(list, in_stack, out_stack, inputs, outputs, visit_cells, &ccode);

  // make const code
  py::object globals = GenerateCode(instrs, &ccode);
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
    auto tmp = UnpackToStackAndLocal(this->graph_->GetExtraLocalIndex(), out_stack, outputs);
    res.insert(nullptr, &tmp);
  }
  return res;
}

AbstractNodeList CodeGenerator::CallSubListWithReturn(const AbstractNodeList &list, int stack_off,
                                                      const FrameStates &f) {
  AbstractNode *node = list.head();
  for (int nodes_count = 4; node && nodes_count > 0; node = node->GetNext(), --nodes_count) {
    InstrNode *instr = reinterpret_cast<InstrNode *>(node);
    if (Utils::IsCallOp(instr->GetOpcode())) {
      break;
    }
  }
  if (node == nullptr) {
    // too few nodes, maybe only return
    return CopyInstrList(list);
  }

  std::set<int> reads, unused, cell_visit;
  build_rws(reinterpret_cast<InstrNode *>(list.head()), &reads, &unused, &unused, &cell_visit, f);
  unused.clear();
  AbstractNodeList call_sub = CallSubList(list, f.GetStacks().size() + stack_off, 1, reads, unused, unused, cell_visit,
                                          JitCompileResults::GRAPH_CANDIDATE);

  int opcode = reinterpret_cast<InstrNode *>(call_sub.back())->GetOpcode();
  if (opcode != RETURN_VALUE && opcode != RAISE_VARARGS && opcode != RERAISE) {
    call_sub.push_back(NewInstr(RETURN_VALUE, 0));
  }
  return call_sub;
}

static FrameStates buildLastFrame(Graph *g, InstrNode **stop_trace_at, bool *is_loop) {
  FrameStates f = g->GetLastFrame();
  f.ResizeLocal(g->GetNlocals());
  InstrNode *v = g->GetStopTraceAt();
  *is_loop = g->GetLoopInfo();
  while (v && v->GetType() == InstrNode::Call) {
    g = reinterpret_cast<CallNode *>(v)->GetSubGraph();
    if (!g) {
      break;
    }
    f.Popn(reinterpret_cast<CallNode *>(v)->getInputs().size());
    int count = f.GetLocals().size();
    f.ResizeLocal(count + g->GetNlocals());
    for (auto i : g->GetLastFrame().GetLocals()) {
      f.SetLocal(count++, i);
    }
    for (auto i : g->GetLastFrame().GetStacks()) {
      f.Push(i);
    }
    v = g->GetStopTraceAt();
    *is_loop = g->GetLoopInfo();
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

  if (stop_trace_at->GetGraph() && this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    auto sub = stop_trace_at->GetGraph();
    PyObject *file = sub->GetCodeObj()->co_filename;
    GRAPH_JIT_LOG_F("break graph at [ %U : %d ] operation %s", file, stop_trace_at->GetLineNo(),
                    GetStopTraceReasonDesc(sub->GetStopTraceReason()).c_str());
  }

  f.ResizeLocal(nlocals_);

  // no operations captured, break at code start
  if (stop_trace_at == instrs_.head()) {
    bool succ = BreakAtInterpretBlocks(stop_trace_at, f, is_loop);
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
    if (f.Local(i)->GetType() == ValueNode::Unbound) {
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
  // due to reorder locals, the outputs is all reads after
  call_compiled = ReshapeCapturedBytecodes(f, reads);
  instrs_.insert(instrs_.head(), &call_compiled);

  if (!GraphBuilder::IsByteCodeImplemented(stop_trace_at->GetOpcode())) {
    // break graph directly when bytecode is not implemented
    return;
  }

  switch (stop_trace_at->GetOpcode()) {
    case JUMP_IF_FALSE_OR_POP: /* fall-through */
    case JUMP_IF_TRUE_OR_POP:  /* fall-through */
    case POP_JUMP_IF_FALSE:    /* fall-through */
    case POP_JUMP_IF_TRUE:
      BreakAtIf(stop_trace_at, f);
      return;
    case JUMP_IF_NOT_EXC_MATCH: /* fall-through */
    case JUMP_ABSOLUTE:         /* fall-through */
    case JUMP_FORWARD:
      MS_LOG(EXCEPTION) << "not implement break at " << Utils::GetOpName(stop_trace_at->GetOpcode());
      break;
    default: {
      bool succ = BreakAtInterpretBlocks(stop_trace_at, f, is_loop);
      if (succ) {
        return;
      }
      BreakAtUnsupportedOperation(stop_trace_at, f);
    }
  }
}

// set break reason while build graph, avoid recursion trace fail
void CodeGenerator::BreakAtUnsupportedOperation(InstrNode *stop_trace_at, const FrameStates &f) {
  int opcode = stop_trace_at->GetOpcode();
  int oparg = stop_trace_at->GetOparg();
  AbstractNodeList untracked;
  instrs_.cutList(stop_trace_at->GetNext(), &untracked);

  int stack_off = PyCompile_OpcodeStackEffect(opcode, oparg);

  auto tmp = CallSubListWithReturn(untracked, stack_off, f);
  instrs_.insert(nullptr, &tmp);
}

void CodeGenerator::BreakAtIf(InstrNode *stop_trace_at, const FrameStates &last_f) {
  AbstractNodeList fall_block, jump_block, tmp;

  instrs_.cutList(stop_trace_at->GetNext(), &fall_block);
  jump_block = {stop_trace_at->GetJump(), fall_block.back()};

  tmp = CallSubListWithReturn(fall_block, -1, last_f);
  instrs_.insert(nullptr, &tmp);

  int off = stop_trace_at->GetOpcode() != JUMP_IF_FALSE_OR_POP && stop_trace_at->GetOpcode() != JUMP_IF_TRUE_OR_POP;
  tmp = CallSubListWithReturn(jump_block, -off, last_f);
  instrs_.insert(nullptr, &tmp);

  stop_trace_at->SetJump(tmp.head());
}

// e.g. while..., for..., while...else..., for...else...,
static AbstractNode *FindLoopEnd(AbstractNode *loop_begin) {
  AbstractNode *loop_exit = loop_begin;
  AbstractNode *target = nullptr;
  AbstractNode *result = nullptr;
  // find loop last exit
  for (; loop_exit != target; loop_exit = loop_exit->GetNext()) {
    if (loop_exit->GetJump() == nullptr) {
      continue;
    }
    // if jump forward, reset target
    if (target == nullptr || target->marker_ < loop_exit->GetJump()->marker_) {
      target = loop_exit->GetJump();
    }
  }
  loop_exit = loop_begin;
  for (; loop_exit != nullptr; loop_exit = loop_exit->GetNext()) {
    if (loop_exit->GetJump() == loop_begin) {
      result = loop_exit;
    }
  }
  // get last exit target node, it is loop blocks end
  if (target->GetPres().size()) {
    if (!result || result->marker_ < target->GetPres()[0]->marker_) {
      return target->GetPres()[0];
    }
  }
  return result;
}

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 9)
static InstrNode *FindBlock(InstrNode *stop_trace_at, int *stack_off, bool is_loop) {
  InstrNode *block_end = nullptr;
  switch (stop_trace_at->GetOpcode()) {
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 7)
    case SETUP_EXCEPT:
      block_end = reinterpret_cast<InstrNode *>(stop_trace_at->GetJump());
      if (!block_end || block_end->GetOpcode() != DUP_TOP) {
        break;
      }
      while (block_end && block_end->GetOpcode() != END_FINALLY) {
        block_end = reinterpret_cast<InstrNode *>(block_end->GetNext());
      }
      break;
    case SETUP_LOOP:
      block_end = reinterpret_cast<InstrNode *>(stop_trace_at->GetJump()->GetPres()[0]);
      break;
    case FOR_ITER:
      block_end = reinterpret_cast<InstrNode *>(FindLoopEnd(stop_trace_at));
      *stack_off = -1;
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
      block_end = reinterpret_cast<InstrNode *>(stop_trace_at->GetJump());
      while (block_end && block_end->GetOpcode() != END_FINALLY) {
        block_end = reinterpret_cast<InstrNode *>(block_end->GetNext());
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
  auto tar = reinterpret_cast<InstrNode *>(n->GetJump());
  if (!tar || tar->GetOpcode() != WITH_EXCEPT_START || !tar->GetNext() || !tar->GetNext()->GetJump() ||
      !tar->GetPres().size()) {
    return nullptr;
  }
  return reinterpret_cast<InstrNode *>(tar->GetPres()[0]->GetJump()->GetPres()[0]);
}

// finally block has two copies in bytecodes
// only test for Python3.9
static InstrNode *findFinallyBlockEnd(InstrNode *raise_block, InstrNode *normal_block) {
  if (normal_block->GetOpcode() != POP_BLOCK) {
    return nullptr;
  }
  auto i = reinterpret_cast<InstrNode *>(normal_block->GetNext());
  auto j = raise_block;
  while (i->GetOpcode() == j->GetOpcode()) {
    i = reinterpret_cast<InstrNode *>(i->GetNext());
    j = reinterpret_cast<InstrNode *>(j->GetNext());
  }
  MS_ASSERT(i->GetOpcode() == JUMP_FORWARD && j->GetOpcode() == RERAISE && i->GetJump() == j->GetNext());
  return j;
}

static InstrNode *findTryBlockEnd(InstrNode *n) {
  auto tar = reinterpret_cast<InstrNode *>(n->GetJump());
  if (!tar) {
    return nullptr;
  }
  int opcode = (reinterpret_cast<InstrNode *>(tar)->GetOpcode());
  InstrNode *res = tar;
  if (opcode == DUP_TOP) {
    // try block without finally
    while (res && res->GetOpcode() != RERAISE) {
      MS_ASSERT(res->GetNext() && res->GetNext()->GetNext());
      res = reinterpret_cast<InstrNode *>(res->GetNext()->GetNext()->GetJump());
    }
    return res;
  }
  // finally block has two copies in bytecodes, first is normally and end with JUMP_FORWARD, second is end with RERAISE
  InstrNode *reraise_finally_block_start = tar;
  InstrNode *normally_finally_block_start = nullptr;
  MS_ASSERT(n && n->GetNext() && (reinterpret_cast<InstrNode *>(n->GetNext()))->GetOpcode() == SETUP_FINALLY);
  res = reinterpret_cast<InstrNode *>(n->GetNext()->GetJump());
  MS_ASSERT(res->GetOpcode() == DUP_TOP);
  while (res && res->GetOpcode() != RERAISE) {
    MS_ASSERT(res->GetNext() && res->GetNext()->GetNext());
    res = reinterpret_cast<InstrNode *>(res->GetNext()->GetNext()->GetJump());
  }
  MS_ASSERT(res && res->GetNext() && (reinterpret_cast<InstrNode *>(res->GetNext()))->GetOpcode() == POP_BLOCK);
  normally_finally_block_start = reinterpret_cast<InstrNode *>(res->GetNext());
  return findFinallyBlockEnd(reraise_finally_block_start, normally_finally_block_start);
}

static InstrNode *FindBlock(InstrNode *stop_trace_at, int *stack_off, bool is_loop) {
  InstrNode *block_end = nullptr;
  switch (stop_trace_at->GetOpcode()) {
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
bool CodeGenerator::BreakAtInterpretBlocks(InstrNode *stop_trace_at, const FrameStates &f, bool is_loop) {
  InstrNode *block_end;
  int stack_off = 0;

  // reset bci
  int bci = 0;
  for (auto i = instrs_.head(); i; i = i->GetNext()) {
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
  instrs_.cutList(block_end->GetNext(), &fall_block);
  // reset jump
  instrs_.push_back(NewInstr(NOP, 0));
  for (auto i = instrs_.head(); i; i = i->GetNext()) {
    if (i->GetJump() && i->GetJump() == fall_block.head()) {
      i->SetJump(instrs_.back());
    }
  }

  AbstractNodeList call_sub = CallSubListWithReturn(fall_block, stack_off, f);
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
  std::set<int> used_slots;  // order set
  for (pos = local_map->begin(); pos != local_map->end(); ++pos) {
    if (pos->first->marker_ < alive_time) {
      res = pos->second;
      local_map->erase(pos);
      local_map->insert({n, res});
      return res;
    }
    used_slots.insert(pos->second);
  }
  res = 0;
  for (auto i : used_slots) {
    if (res != i) {
      break;
    }
    res++;
  }
  local_map->insert({n, res});
  return res;
}

bool CodeGenerator::LoadValue(std::unordered_map<ValueNode *, int> *local_map, ValueNode *i, AbstractNodeList *res,
                              bool build_value) {
  if (local_map->find(i) != local_map->end()) {
    res->push_back(NewInstr(LOAD_FAST, (*local_map)[i]));
    return true;
  }
  if (i->GetType() == ValueNode::CellVar || i->GetType() == ValueNode::FreeVar) {
    res->push_back(NewInstr(LOAD_CLOSURE, reinterpret_cast<CellVarNode *>(i)->GetIndex()));
  } else if (i->GetOpcode() == LOAD_DEREF || i->GetOpcode() == LOAD_GLOBAL) {
    auto tmp = NewInstr(i->GetOpcode(), i->GetOparg());
    res->push_back(tmp);
    tmp->SetName(i->GetName());
    tmp->SetGraph(i->GetGraph());
  } else if (i->GetOpcode() == LOAD_CONST) {
    res->push_back(alloc_.NewValueNode(i->GetVobj(), LOAD_CONST, -1, {}));
  } else if (build_value) {
    BuildValue(local_map, i, 0, res);
    MS_EXCEPTION_IF_CHECK_FAIL(local_map->find(i) != local_map->end(), "check build values");
    res->push_back(NewInstr(LOAD_FAST, (*local_map)[i]));
  } else {
    return false;
  }
  return true;
}

void CodeGenerator::BuildValue(std::unordered_map<ValueNode *, int> *local_map, ValueNode *n, int order,
                               AbstractNodeList *res) {
  int op = n->GetOpcode();
  if (IsNonLocalValue(n) || local_map->find(n) != local_map->end()) {
    return;
  }
  int arg = n->GetOparg();
  for (auto i : n->getInputs()) {
    MS_EXCEPTION_IF_CHECK_FAIL(i != &ValueNode::UnboundLocal,
                               "used before define, here " +
                                 std::string(py::str(n->GetGraph()->GetCodeObj()->co_filename)) + ":" +
                                 std::to_string(n->GetLineNo()));
    LoadValue(local_map, i, res);
  }
  // NOTE: if support UNPACK_SEQUENCE, should check here
  auto v = NewInstr(op, arg);
  res->push_back(v);
  v->SetName(n->GetName());
  v->SetLineNo(n->GetLineNo());
  v->SetGraph(n->GetGraph());
  if (IsOperationWithoutOperatorStackEffects(op)) {
    return;
  }
  if (n->marker_ == 0) {
    res->push_back(NewInstr(POP_TOP, 0));
    return;
  }
  res->push_back(NewInstr(STORE_FAST, allocLocal(local_map, n, order)));
}

// if these values not used, allowed delete these operations
// NOTE: use marker_ as usage count
static bool no_side_effect_op(ValueNode *v) {
  if (v->marker_ != 0) {
    return false;
  }
  int op = v->GetOpcode();
  if (Utils::IsNoSideEffectOp(op)) {
    return true;
  }
  if (op == BUILD_MAP || op == BUILD_CONST_KEY_MAP) {
    // BUILD_MAP will call __hash__
    if (v->GetOparg() == 0) {
      return true;
    }
    if (v->GetVobj()) {
      return static_cast<AbstractDict *>(v->GetVobj())->KeyType() != AObject::kTypeAnyValue;
    }
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
  }
}

// call this must be after call MarkLocalAliveTime
static void EraseDeadLocal(std::vector<ValueNode *> *values) {
  size_t len;
  do {
    len = values->size();
    auto find_func = [](ValueNode *i) { return i->marker_ == 0 && Utils::IsGeneralNoSideEffectOp(i->GetOpcode()); };
    values->erase(std::remove_if(values->begin(), values->end(), find_func), values->end());
    MarkLocalAliveTime(*values);
  } while (len > values->size());
}

AbstractNodeList CodeGenerator::BuildValues(std::unordered_map<ValueNode *, int> *local_map,
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
    if (no_side_effect_op(values[i])) {
      continue;
    }
    AbstractNodeList op;
    BuildValue(local_map, values[i], i, &op);
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
  AObject *value_info = param->GetVobj();
  return value_info ? value_info->IsMindSporeSupportedType() : false;
}

void CodeGenerator::BuildGraphParameters(std::unordered_map<ValueNode *, int> *graph_local_map,
                                         std::unordered_map<ValueNode *, int> *interpret_local_map,
                                         AbstractNodeList *graph_load_inputs, AbstractNodeList *interpret_load_inputs,
                                         int *graph_argc, int *graph_code_flag) {
  PyCodeObject *co = this->graph_->GetCodeObj();
  auto &inputs = this->captured_info_.captured_locals.inputs;
  int argc = 0;
  int global_input_pos = inputs.size();

  // NOTE: if *vargs is cell variable, it is not parameter node
  ValueNode *vargs = nullptr;
  ValueNode *kwargs = nullptr;
  int arg_index = co->co_argcount + co->co_kwonlyargcount;
  if (co->co_flags & CO_VARARGS) {
    vargs = graph_->GetFrame(0).Local(arg_index);
    if (vargs == &ValueNode::UnboundLocal || !IsPassedByParameter(vargs)) {
      vargs = nullptr;
    }
  }
  if (co->co_flags & CO_VARKEYWORDS) {
    kwargs = graph_->GetFrame(0).Local(arg_index + (static_cast<bool>(co->co_flags & CO_VARARGS)));
    if (kwargs == &ValueNode::UnboundLocal || !IsPassedByParameter(kwargs)) {
      kwargs = nullptr;
    }
  }

  // prepare inputs local map
  for (auto i : inputs) {
    if (IsNonLocalValue(i) || i == vargs || i == kwargs) {
      continue;
    }
    {  // interpret load arguments
      AbstractNodeList ld;
      bool has_input = LoadValue(interpret_local_map, i, &ld, false);
      MS_EXCEPTION_IF_CHECK_FAIL(has_input, "must be has input");
      interpret_load_inputs->insert(nullptr, &ld);
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
    auto nam = GetGraphInputsKey(i);
    n->SetName(nam.c_str());
    interpret_load_inputs->push_back(n);
    // graph load input
    n = NewInstr(LOAD_GLOBAL);
    n->SetName(nam.c_str());
    graph_load_inputs->push_back(n);
    graph_load_inputs->push_back(NewInstr(STORE_FAST, global_input_pos));
  }

  *graph_argc = argc;
  if (inputs.find(vargs) != inputs.end()) {
    *graph_code_flag |= CO_VARARGS;
    interpret_load_inputs->push_back(NewInstr(BUILD_LIST, argc));
    AbstractNodeList ld;
    bool has_input = LoadValue(interpret_local_map, vargs, &ld, false);
    MS_EXCEPTION_IF_CHECK_FAIL(has_input, "must be has input");
    interpret_load_inputs->insert(nullptr, &ld);
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9
    interpret_load_inputs->push_back(NewInstr(BUILD_TUPLE_UNPACK, 2));
#else
    interpret_load_inputs->push_back(NewInstr(LIST_EXTEND, 1));
    interpret_load_inputs->push_back(NewInstr(LIST_TO_TUPLE, 0));
#endif
    graph_local_map->insert({vargs, argc++});
  }
  if (inputs.find(kwargs) != inputs.end()) {
    if (!(*graph_code_flag & CO_VARARGS)) {
      // only kwargs
      interpret_load_inputs->push_back(NewInstr(BUILD_TUPLE, argc));
    }
    *graph_code_flag |= CO_VARKEYWORDS;
    AbstractNodeList ld;
    bool has_input = LoadValue(interpret_local_map, kwargs, &ld, false);
    MS_EXCEPTION_IF_CHECK_FAIL(has_input, "must be has input");
    interpret_load_inputs->insert(nullptr, &ld);
    graph_local_map->insert({kwargs, argc++});
  }
}

AbstractNodeList CodeGenerator::ReshapeGraphBytecodes(std::unordered_map<ValueNode *, int> *graph_local_map,
                                                      std::unordered_map<ValueNode *, int> *interpret_local_map,
                                                      std::unordered_map<ValueNode *, int> *graph_outputs) {
  PyCodeObject *origin_co = this->graph_->GetCodeObj();
  AbstractNodeList graph_load_inputs;
  AbstractNodeList interpret_load_inputs;
  int argc;
  int code_flag = origin_co->co_flags & ~(CO_VARARGS | CO_VARKEYWORDS);
  // prepare inputs local map
  BuildGraphParameters(graph_local_map, interpret_local_map, &graph_load_inputs, &interpret_load_inputs, &argc,
                       &code_flag);

  AbstractNodeList compile = BuildValues(graph_local_map, this->captured_info_.captured_locals.order);
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
  Code ccode = {
    argc,                                                        // co_argcount
    0,                                                           // co_kwonlyargcount
    static_cast<int>(graph_local_map->size()),                   // co_nlocals
    0,                                                           // co_stacksize
    code_flag,                                                   // co_flags
    origin_co->co_firstlineno,                                   // co_firstlineno
    std::vector<std::string>(),                                  // co_varnames
    std::vector<std::string>(),                                  // co_freevars
    std::vector<std::string>(),                                  // co_cellvars
    std::vector<_Py_CODEUNIT>(),                                 // co_code
    std::unordered_map<std::string, int>(),                      // co_names
    std::unordered_map<PyObject *, int>(),                       // co_consts
    py::reinterpret_borrow<py::object>(origin_co->co_filename),  // co_filename
    "_compile",                                                  // co_name
    std::vector<char>(),                                         // co_lnotab
    JitCompileResults::GRAPH_CAPTURED,                           // compile stat
  };
  for (auto i = 0; i < ccode.co_argcount; ++i) {
    ccode.co_varnames.push_back(std::to_string(i) + "_local");
  }

  py::object globals = GenerateCode(compile, &ccode);
  UpdateGlobals(globals);
  std::set<int> unused;
  AbstractNodeList make_func = GenerateMakeFunc(&ccode, unused, JitCompileResults::GRAPH_CAPTURED);

  make_func.insert(nullptr, &interpret_load_inputs);
  int opcode = CALL_FUNCTION;
  int oparg = argc;
  if (code_flag & (CO_VARKEYWORDS | CO_VARARGS)) {
    opcode = CALL_FUNCTION_EX;
    oparg = static_cast<bool>(code_flag & CO_VARKEYWORDS);
  }
  make_func.push_back(NewInstr(opcode, oparg));
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

AbstractNodeList CodeGenerator::ReshapeCapturedBytecodes(const FrameStates &last_frame, const std::set<int> &outputs,
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
  const auto &params = this->graph_->GetFrame(0).GetLocals();
  for (int i = 0; i < static_cast<int>(params.size()); ++i) {
    if (params[i] != &ValueNode::UnboundLocal) {
      MS_EXCEPTION_IF_CHECK_FAIL(params[i]->GetType() == ValueNode::Param && params[i]->GetOparg() == i,
                                 "must be parameter node");

      local_map.insert({params[i], i});
    }
  }
  // produce interpret values
  AbstractNodeList res = BuildValues(&local_map, nodes.ordered_escaped_locals);
  // placeholder for graph_output
  int graph_out_index = local_map.size();
  local_map.insert({nullptr, graph_out_index});
  local_map.insert({&ValueNode::UnboundLocal, local_map.size()});

  std::unordered_map<ValueNode *, int> graph_local_map;
  AbstractNodeList call_func = ReshapeGraphBytecodes(&graph_local_map, &local_map, &graph_outputs);
  // has operations compiled
  if (call_func.head()) {
    res.insert(nullptr, &call_func);
    res.push_back(NewInstr(STORE_FAST, graph_out_index));
  }

  {  // restore frame
    auto GetValueWithAssert = [this, &graph_outputs, &local_map, graph_out_index](ValueNode *i) -> AbstractNodeList {
      AbstractNodeList instrs;
      bool has_input = this->LoadValue(&local_map, i, &instrs, false);
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
      AbstractNodeList ld = GetValueWithAssert(i);
      res.insert(nullptr, &ld);
    }
    // restore reordered local
    std::vector<int> tmp;
    for (auto i : outputs) {
      auto v = last_frame.Local(i);
      if (local_map.find(v) != local_map.end() && i == local_map[v]) {
        continue;
      }
      AbstractNodeList ld = GetValueWithAssert(v);
      res.insert(nullptr, &ld);
      tmp.push_back(i);
    }
    for (auto i = tmp.rbegin(); i != tmp.rend(); ++i) {
      res.push_back(NewInstr(STORE_FAST, *i));
    }
    if (ret_v) {
      AbstractNodeList ld = GetValueWithAssert(ret_v);
      res.insert(nullptr, &ld);
      res.push_back(NewInstr(RETURN_VALUE));
    }
  }

  nlocals_ = std::max((size_t)nlocals_, local_map.size());
  return res;
}

bool CodeGenerator::TryToBreakGraphIfParameterUnsupported() {
  if (this->graph_->GetStopTraceAt()) {
    // do this in fix break graph
    return false;
  }
  bool support_param = true;
  for (auto i : this->graph_->GetFrame(0).GetLocals()) {
    if (i->GetType() == AbstractNode::Param && !i->GetVobj()->IsMindSporeSupportedType()) {
      if (this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
        GRAPH_JIT_LOG_F("rewrite bytecode because of parameter is unsupported [%s]", i->GetVobj()->ToString().c_str());
      }
      support_param = false;
      break;
    }
    // cell2arg parameter maybe unsupported, check it
  }
  if (this->captured_info_.ordered_escaped_locals.empty() && support_param) {
    // no unsupported operations
    return false;
  }
  FrameStates empty;
  std::set<int> out;
  instrs_ = ReshapeCapturedBytecodes(empty, out, this->graph_->GetRetVal());
  code_changed_ = true;
  return true;
}

void CodeGenerator::Print(const AbstractNode *list_beg, const char *marker_info) {
  for (auto i = list_beg; i; i = i->GetNext()) {
    GRAPH_JIT_LOG_F("%s %s:%d\n", i->to_str().c_str(), marker_info, i->marker_);
  }
  GRAPH_JIT_LOG_F("\n");
}

std::string CodeGenerator::GetGraphInputsKey(const ValueNode *v) {
  std::stringstream s;
  s << ".graph.inputs." << v << '.' << v->GetName();
  return s.str();
}
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
