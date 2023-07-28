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
#include "pipeline/jit/graph_jit/graph_capture/graph.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include "pipeline/jit/graph_jit/common.h"

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

static TracePtr getTrace(ValueNode *node, bool print, int depth = 0) {
  if (depth > 8) {
    MS_LOG(DEBUG) << "too deep trace for guard";
    return nullptr;
  }
  switch (node->getType()) {
    case AbstractNode::Type::Value: {
      ValueNode *p = reinterpret_cast<ValueNode *>(node);
      TraceVector tv;
      for (auto it : p->getInputs()) {
        auto t = getTrace(it, print, depth + 1);
        if (t == nullptr) {
          return nullptr;
        }
        tv.push_back(t);
      }
      return CreateOpTrace(p->getVobj()->GetPyObject().ptr(), p->getOpcode(), p->getOparg(), tv,
                           node->GetGraph()->getModuleName(), p->getName(), print);
    } break;
    case AbstractNode::Type::Call: {
      CallNode *p = reinterpret_cast<CallNode *>(node);
      TraceVector tv;
      for (auto it : p->getInputs()) {
        auto t = getTrace(it, print, depth + 1);
        if (t == nullptr) {
          return nullptr;
        }
        tv.push_back(t);
      }
      if (p->getOpcode() == CALL_FUNCTION) {
        return CreateOpTrace(p->getVobj()->GetPyObject().ptr(), p->getOpcode(), p->getOparg(), tv,
                             node->GetGraph()->getModuleName(), p->getInputs()[0]->getName(), print);
      } else {
        return CreateOpTrace(p->getVobj()->GetPyObject().ptr(), p->getOpcode(), p->getOparg(), tv,
                             node->GetGraph()->getModuleName(), p->getName(), print);
      }
    } break;
    case AbstractNode::Type::Merged:
      break;
    case AbstractNode::Type::Param: {
      ParamNode *p = reinterpret_cast<ParamNode *>(node);
      return std::make_shared<RootTrace>(p->getVobj()->GetPyObject().ptr(), mindspore::jit::graph::TraceType::Param,
                                         p->getOparg(), p->getName());
    }
    case AbstractNode::Type::CellVar:
    case AbstractNode::Type::FreeVar: {
      CellVarNode *p = reinterpret_cast<CellVarNode *>(node);
      return std::make_shared<RootTrace>(p->getVobj()->GetPyObject().ptr(), mindspore::jit::graph::TraceType::Param,
                                         p->getOparg(), p->getName());
    }
    case AbstractNode::Type::Unbound:
      break;
    default:
      break;
  }
  return nullptr;
}

bool Graph::GuardValueNode(ValueNode *node) {
  AObject *vo = node->getVobj();
  if (guard_ == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    return false;
  }
  TracePtr t = getTrace(node, Config().GetBoolConfig(GraphJitConfig::kPrintGuard));
  if (t != nullptr) {
    return guard_->GetGuard()->GuardOn(t, mindspore::jit::graph::GuardLevel::GEqual);
  }
  return false;
}

TracePtr Graph::TraceValueNode(ValueNode *node) {
  AObject *vo = node->getVobj();
  if (guard_ == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    return nullptr;
  }
  return getTrace(node, Config().GetBoolConfig(GraphJitConfig::kPrintGuard));
}

void Graph::print(int depth) const {
  std::string prefix(depth, ' ');
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
      if (i->getType() != AbstractNode::Call) {
        continue;
      }
      CallNode *node = reinterpret_cast<CallNode *>(i);
      bool has_sub_graph = node->getSubGraph();
      PyCodeObject *co = has_sub_graph ? node->getSubGraph()->getCodeObj() : nullptr;
      std::string code_name = co ? py::str(reinterpret_cast<PyObject *>(co)) : "";
      GRAPH_JIT_LOG_F("%s{--inline %s stat %s --\n", prefix.c_str(), code_name.c_str(),
                      GetInlineReasonDesc(node->getInlineReason()).c_str());
      if (!has_sub_graph) {
        GRAPH_JIT_LOG_F("%s}\n", prefix.c_str());
        continue;
      }
      node->getSubGraph()->print(depth + 1);
      GRAPH_JIT_LOG_F("%s}--inlined--\n", prefix.c_str());
    }
  }
  GRAPH_JIT_LOG_F("\n%s return: %s\n", prefix.c_str(),
                  getRetVal() ? getRetVal()->to_str().c_str() : "track break or no return");
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
