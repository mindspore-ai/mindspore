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
#include "pipeline/jit/pi/graph_capture/graph.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include "pipeline/jit/pi/common.h"

namespace mindspore {
namespace jit {
namespace graph {
Graph::Graph(PyCodeObject *co, PyObject *globals, const GraphJitConfig &conf)
    : ret_val_(nullptr),
      co_(py::cast<py::object>(reinterpret_cast<PyObject *>(co))),
      f_globals_(py::cast<py::object>(globals)),
      conf_(conf),
      guard_(nullptr),
      prune_branch_count_(0) {
  stop_trace_info_ = {-1, StopTraceReason::kNonStopTrace};
  if (!co) {
    frame_states_.emplace_back(std::make_unique<FrameStates>());  // emplace empty frame
    module_name_ = "";
    return;
  }
  cfg_ = std::make_unique<CFG>(co, &alloc_, conf);
  cfg_->GenerateCFG();
  for (auto &i : cfg_->bb_pool()) {
    i->SetGraph(this);
  }
  Init();
  auto pyname = PyDict_GetItemString(globals, "__name__");
  if (pyname) {
    module_name_ = PyUnicode_AsUTF8(pyname);
  } else {
    module_name_ = "";
    PyErr_Clear();
  }

  if (conf_.GetBoolConfig(GraphJitConfig::kLoopUnrolling)) {
    LoopFinder loop_finder(this);
    loop_finder.FormSimpleLoopInfo();
    if (conf_.GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
      GRAPH_JIT_LOG_F("%s", DumpLoops().c_str());
    }
  }
}

Block *Graph::GetBlockByBci(int bci) const {
  if (bci < 0 || bci >= static_cast<int>(bb_cache_.size())) {
    return nullptr;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(bb_cache_.find(bci) != bb_cache_.end(), "not found)");
  Block *r = bb_cache_.find(bci)->second;
  MS_EXCEPTION_IF_CHECK_FAIL(bci >= r->begin_ci() && bci < r->end_ci(),
                             "check block id: " + std::to_string(r->id()) + " curr_bci: " + std::to_string(bci));
  return r;
}

void Graph::Init() {
  // all bb
  for (const auto &bb : cfg_->bb_pool()) {
    for (auto &instr : bb->instrs()) {
      bb->GetNodes().clear();
      instrs_.insert(std::make_pair(instr.bci(), &instr));
      bb_cache_.insert(std::make_pair(instr.bci(), bb.get()));
    }
  }
  frame_states_.clear();
  frame_states_.resize(instrs_.size());
  instr_nodes_.clear();
  instr_nodes_.resize(instrs_.size());
  for (size_t i = 0; i < instr_nodes_.size(); ++i) {
    instr_nodes_[i] = nullptr;
  }
}

// for loop_unrolling
void Graph::Reset() {
  loops_.clear();
  stop_trace_info_ = {-1, StopTraceReason::kNonStopTrace};
  instrs_.clear();
  bb_cache_.clear();
  Init();
  for (auto &i : cfg_->bb_pool()) {
    i->SetGraph(this);
  }
}

void Graph::SetFrame(int bci, const FrameStates &f) {
  MS_ASSERT(bci >= 0 && bci < (int)frame_states_.size());
  if (!frame_states_[bci].get()) {
    auto i = std::make_unique<FrameStates>();
    frame_states_[bci].swap(i);
  }
  *frame_states_[bci] = f;
}

const FrameStates &Graph::GetFrame(int bci) const {
  MS_ASSERT(bci >= 0 && bci < (int)frame_states_.size() && frame_states_[bci].get());
  return *frame_states_[bci];
}

// if sub graph, extra stack size for handle parameters operation
int Graph::GetStackSize() const {
  PyCodeObject *c = reinterpret_cast<PyCodeObject *>(co_.ptr());
  return c ? (c->co_stacksize + 2 + ((c->co_flags & CO_VARKEYWORDS) ? 1 : 0)) : 0;
}

TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);
static bool PrepareTraceParam(std::vector<ValueNode *> *inputs, TraceVector *tv, int depth, int max_depth,
                              bool *has_unsupported, bool strict, bool print) {
  for (auto it : *inputs) {
    auto t = GetTrace(it, strict, print, depth + 1, max_depth);
    if (t == nullptr) {
      return false;
    } else if (t->GetTraceType() == TraceType::Unsupported) {
      *has_unsupported = true;
    }
    tv->push_back(t);
  }
  return true;
}

static bool CheckDepth(int depth, int max_depth) { return depth < max_depth || max_depth == -1; }

static bool CheckObjPtr(ValueNode *node) {
  return node->GetVobj() == nullptr || node->GetVobj()->GetPyObject().ptr() == nullptr;
}

TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth) {
  if (!CheckDepth(depth, max_depth)) {
    MS_LOG(DEBUG) << "too deep trace for guard";
    return nullptr;
  }
  TraceVector tv;
  ValueNode *p = node;
  bool has_unsupported = false;
  if (!PrepareTraceParam(&(node->getInputs()), &tv, depth, max_depth, &has_unsupported, strict, print) ||
      CheckObjPtr(node)) {
    return strict ? nullptr : std::make_shared<UnsupportedTrace>(nullptr, tv, p->GetOpcode(), p->GetOparg());
  }
  TracePtr ret = nullptr;
  switch (node->GetType()) {
    case AbstractNode::Type::Value: {
      if (!has_unsupported) {
        ret = CreateOpTrace(p->GetVobj()->GetPyObject().ptr(), p->GetOpcode(), p->GetOparg(), tv,
                            node->GetGraph()->GetModuleName(), p->GetName(), strict, print);
      }
      if (ret == nullptr && !strict) {
        ret = std::make_shared<UnsupportedTrace>(p->GetVobj()->GetPyObject().ptr(), tv, p->GetOpcode(), p->GetOparg());
      }
      return ret;
    } break;
    case AbstractNode::Type::Call: {
      if (!has_unsupported) {
        if (p->GetOpcode() == CALL_FUNCTION) {
          ret = CreateOpTrace(p->GetVobj()->GetPyObject().ptr(), p->GetOpcode(), p->GetOparg(), tv,
                              node->GetGraph()->GetModuleName(), p->getInputs()[0]->GetName(), strict, print);
        } else {
          ret = CreateOpTrace(p->GetVobj()->GetPyObject().ptr(), p->GetOpcode(), p->GetOparg(), tv,
                              node->GetGraph()->GetModuleName(), p->GetName(), strict, print);
        }
      }
      if (ret == nullptr && !strict) {
        ret = std::make_shared<UnsupportedTrace>(p->GetVobj()->GetPyObject().ptr(), tv, p->GetOpcode(), p->GetOparg());
      }
      return ret;
    } break;
    case AbstractNode::Type::Param: {
      return std::make_shared<RootTrace>(p->GetVobj()->GetPyObject().ptr(), mindspore::jit::graph::TraceType::Param,
                                         p->GetOparg(), p->GetName());
    }
    case AbstractNode::Type::CellVar:
    case AbstractNode::Type::FreeVar: {
      return std::make_shared<RootTrace>(p->GetVobj()->GetPyObject().ptr(), mindspore::jit::graph::TraceType::Param,
                                         p->GetOparg(), p->GetName());
    }
    case AbstractNode::Type::Merged:
    case AbstractNode::Type::Unbound:
      break;
    default:
      break;
  }
  return nullptr;
}

bool Graph::GuardValueNode(ValueNode *node) {
  AObject *vo = node->GetVobj();
  if (guard_ == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    return false;
  }
  if (node->GetOpcode() == LOAD_CONST) {
    return true;
  }
  TracePtr t = GetTrace(node, Config().GetBoolConfig(GraphJitConfig::kStrictTrace),
                        Config().GetBoolConfig(GraphJitConfig::kPrintGuard), 0,
                        Config().getIntConfig(GraphJitConfig::GraphJitConfig::kMaxTraceDepth));
  if (t != nullptr) {
    bool ret = guard_->GetGuard()->GuardOn(t, mindspore::jit::graph::GuardLevel::GEqual);
    if (Config().GetBoolConfig(GraphJitConfig::kGuardDetachObject)) {
      t->Detach();
    }
    return ret;
  }
  return false;
}

TracePtr Graph::TraceValueNode(ValueNode *node, int max_trace_depth) {
  AObject *vo = node->GetVobj();
  if (guard_ == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    return nullptr;
  }
  if (max_trace_depth < 0) {
    max_trace_depth = Config().getIntConfig(GraphJitConfig::GraphJitConfig::kMaxTraceDepth);
  }
  return GetTrace(node, Config().GetBoolConfig(GraphJitConfig::kStrictTrace),
                  Config().GetBoolConfig(GraphJitConfig::kPrintGuard), 0, max_trace_depth);
}

void Graph::print(int depth) const {
  std::string prefix(depth << 1, ' ');
  if (!cfg_.get()) {
    GRAPH_JIT_LOG_F("%s empty graph\n", prefix.c_str());
    return;
  }
  if (depth == 0) {
    GRAPH_JIT_LOG_F("%s params:\n", prefix.c_str());
    const FrameStates &f = GetFrame(0);
    int param_cnt = 0;
    for (auto i : f.GetLocals()) {
      if (i != &ValueNode::UnboundLocal) {
        GRAPH_JIT_LOG_F("%s%d:%s\n", prefix.c_str(), param_cnt++, i->to_str().c_str());
      }
    }
  }
  for (Block *b : *cfg_) {
    if (b->GetNodes().empty()) {
      continue;
    }
    GRAPH_JIT_LOG_F("\n%s--- %s ---\n", prefix.c_str(), b->Dump(false).c_str());
    for (auto i : b->GetNodes()) {
      GRAPH_JIT_LOG_F("%s%s\n", prefix.c_str(), i->to_str().c_str());
      if (i->GetType() != AbstractNode::Call) {
        continue;
      }
      CallNode *node = reinterpret_cast<CallNode *>(i);
      bool has_sub_graph = node->GetSubGraph();
      PyCodeObject *co = has_sub_graph ? node->GetSubGraph()->GetCodeObj() : nullptr;
      std::string code_name = co ? py::str(reinterpret_cast<PyObject *>(co)) : "";
      GRAPH_JIT_LOG_F("%s{--inline %s stat %s --\n", prefix.c_str(), code_name.c_str(),
                      GetInlineReasonDesc(node->GetInlineReason()).c_str());
      if (!has_sub_graph) {
        GRAPH_JIT_LOG_F("%s}\n", prefix.c_str());
        continue;
      }
      node->GetSubGraph()->print(depth + 1);
      GRAPH_JIT_LOG_F("%s}--inlined--\n", prefix.c_str());
    }
  }
  GRAPH_JIT_LOG_F("\n%s return: %s\n", prefix.c_str(),
                  GetRetVal() ? GetRetVal()->to_str().c_str() : "track break or no return");
  GRAPH_JIT_LOG_F("%s%s\n", prefix.c_str(), GetStopTraceReasonDesc(stop_trace_info_.reason).c_str());
  GRAPH_JIT_LOG_F("%sbreak bci: %d\n", prefix.c_str(), stop_trace_info_.bci);
  GRAPH_JIT_LOG_F("\n\n");
}

void FrameStates::print() {
  GRAPH_JIT_LOG_F("locals:\n");
  int c = 0;
  for (auto i : locals) {
    GRAPH_JIT_LOG_F("%d:%s\n", c++, i == &ValueNode::UnboundLocal ? "(UnboundLocal)" : i->to_str().c_str());
  }
  GRAPH_JIT_LOG_F("\nstacks:\n");
  for (auto i : stack) {
    GRAPH_JIT_LOG_F("%d:%s\n", c++, i == &ValueNode::UnboundLocal ? "(UnboundLocal)" : i->to_str().c_str());
  }
  GRAPH_JIT_LOG_F("\ncell_free:\n");
  for (auto i : cell_free) {
    GRAPH_JIT_LOG_F("%s\n", i == &ValueNode::UnboundLocal ? "(UnboundLocal)" : i->to_str().c_str());
  }
  GRAPH_JIT_LOG_F("\n");
}

std::string Graph::DumpLoops() const {
  std::ostringstream os;
  if (loops_.empty()) {
    return os.str();
  }
  os << "*** Dump Loops on [" << py::str(co_.ptr()).cast<std::string>() << "] ***\n";
  for (const auto *lp : loops_) {
    os << lp->Dump();
  }
  os << '\n';
  return os.str();
}

}  // namespace graph
}  // namespace jit
}  // namespace mindspore
