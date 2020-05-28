/**
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

#ifndef MINDSPORE_CCSRC_DEBUG_TRACE_INFO_H_
#define MINDSPORE_CCSRC_DEBUG_TRACE_INFO_H_

#include <iostream>
#include <string>
#include <memory>
#include <stack>
#include <utility>
#include <vector>

#include "ir/base.h"

namespace mindspore {
class TraceInfo;
using TraceInfoPtr = std::shared_ptr<TraceInfo>;
class Location;
using LocationPtr = std::shared_ptr<Location>;
class DebugInfo;
using DebugInfoPtr = std::shared_ptr<DebugInfo>;

// namespace to support intermediate representation definition
class TraceInfo : public Base {
 public:
  TraceInfo(const DebugInfoPtr &info, const std::string &full_name, const std::string &symbol) {
    symbol_ = symbol;
    full_name_ = full_name;
    name_ = full_name_;
    debug_info_ = info;
  }
  TraceInfo(const TraceInfo &info)
      : Base(), debug_info_(info.debug_info_), symbol_(info.symbol_), full_name_(info.full_name_), name_(info.name_) {}
  virtual ~TraceInfo() = default;
  MS_DECLARE_PARENT(TraceInfo, Base);
  virtual std::string name() { return name_; }
  virtual std::string symbol() { return symbol_; }
  virtual std::string full_name() { return full_name_; }
  virtual TraceInfoPtr clone() { return shared_from_base<TraceInfo>(); }
  virtual std::string action_name() { return ""; }
  virtual std::string GetActionBetweenNode(const DebugInfoPtr &info);
  void set_debug_info(const DebugInfoPtr &info) { debug_info_ = info; }
  DebugInfoPtr debug_info() { return debug_info_; }
  DebugInfoPtr DebugInfoHasLoc();
  std::vector<std::pair<DebugInfoPtr, TraceInfoPtr>> GetSourceCodeDebugInfo();

 protected:
  DebugInfoPtr debug_info_;
  std::string symbol_;
  std::string full_name_;
  std::string name_;
};

class TracePhi : public TraceInfo {
 public:
  explicit TracePhi(const DebugInfoPtr &info) : TraceInfo(info, "phi", "Φ") {}
  MS_DECLARE_PARENT(TracePhi, TraceInfo);
  ~TracePhi() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TracePhi>(*shared_from_base<TracePhi>()); }
};

class TraceIfStmtTrueBranch : public TraceInfo {
 public:
  TraceIfStmtTrueBranch(const TraceIfStmtTrueBranch &) = default;
  explicit TraceIfStmtTrueBranch(const DebugInfoPtr &info) : TraceInfo(info, "if_true", "✓") {}
  MS_DECLARE_PARENT(TraceIfStmtTrueBranch, TraceInfo);
  ~TraceIfStmtTrueBranch() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TraceIfStmtTrueBranch>(*shared_from_base<TraceIfStmtTrueBranch>());
  }
};

class TraceIfStmtFalseBranch : public TraceInfo {
 public:
  TraceIfStmtFalseBranch(const TraceIfStmtFalseBranch &) = default;
  explicit TraceIfStmtFalseBranch(const DebugInfoPtr &info) : TraceInfo(info, "if_false", "✗") {}
  MS_DECLARE_PARENT(TraceIfStmtFalseBranch, TraceInfo);
  ~TraceIfStmtFalseBranch() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TraceIfStmtFalseBranch>(*shared_from_base<TraceIfStmtFalseBranch>());
  }
};

class TraceIfStmtAfterBranch : public TraceInfo {
 public:
  explicit TraceIfStmtAfterBranch(const DebugInfoPtr &info) : TraceInfo(info, "if_after", "↓") {}
  MS_DECLARE_PARENT(TraceIfStmtAfterBranch, TraceInfo);
  ~TraceIfStmtAfterBranch() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TraceIfStmtAfterBranch>(*shared_from_base<TraceIfStmtAfterBranch>());
  }
};

class TraceIfExpTrueBranch : public TraceInfo {
 public:
  explicit TraceIfExpTrueBranch(const DebugInfoPtr &info) : TraceInfo(info, "ifexp_true", "↰") {}
  MS_DECLARE_PARENT(TraceIfExpTrueBranch, TraceInfo);
  ~TraceIfExpTrueBranch() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TraceIfExpTrueBranch>(*shared_from_base<TraceIfExpTrueBranch>());
  }
};

class TraceIfExpFalseBranch : public TraceInfo {
 public:
  explicit TraceIfExpFalseBranch(const DebugInfoPtr &info) : TraceInfo(info, "ifexp_false", "↱") {}
  MS_DECLARE_PARENT(TraceIfExpFalseBranch, TraceInfo);
  ~TraceIfExpFalseBranch() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TraceIfExpFalseBranch>(*shared_from_base<TraceIfExpFalseBranch>());
  }
};

class TraceCopy : public TraceInfo {
 public:
  TraceCopy() : TraceInfo(nullptr, "copy", "") {}
  explicit TraceCopy(const DebugInfoPtr &info) : TraceInfo(info, "copy", "") {}
  MS_DECLARE_PARENT(TraceCopy, TraceInfo);
  ~TraceCopy() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceCopy>(*shared_from_base<TraceCopy>()); }
};

class TraceIterator : public TraceInfo {
 public:
  explicit TraceIterator(const DebugInfoPtr &info) : TraceInfo(info, "iterator", "@") {}
  MS_DECLARE_PARENT(TraceIterator, TraceInfo);
  ~TraceIterator() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceIterator>(*shared_from_base<TraceIterator>()); }
};

class TraceWhileHeader : public TraceInfo {
 public:
  explicit TraceWhileHeader(const DebugInfoPtr &info) : TraceInfo(info, "while_header", "⤾") {}
  MS_DECLARE_PARENT(TraceWhileHeader, TraceInfo);
  ~TraceWhileHeader() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceWhileHeader>(*shared_from_base<TraceWhileHeader>()); }
};

class TraceWhileBody : public TraceInfo {
 public:
  explicit TraceWhileBody(const DebugInfoPtr &info) : TraceInfo(info, "while_body", "⥁") {}
  MS_DECLARE_PARENT(TraceWhileBody, TraceInfo);
  ~TraceWhileBody() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceWhileBody>(*shared_from_base<TraceWhileBody>()); }
};

class TraceWhileAfter : public TraceInfo {
 public:
  explicit TraceWhileAfter(const DebugInfoPtr &info) : TraceInfo(info, "while_after", "↓") {}
  MS_DECLARE_PARENT(TraceWhileAfter, TraceInfo);
  ~TraceWhileAfter() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceWhileAfter>(*shared_from_base<TraceWhileAfter>()); }
};

class TraceForHeader : public TraceInfo {
 public:
  explicit TraceForHeader(const DebugInfoPtr &info) : TraceInfo(info, "for_header", "⤾") {}
  MS_DECLARE_PARENT(TraceForHeader, TraceInfo);
  ~TraceForHeader() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceForHeader>(*shared_from_base<TraceForHeader>()); }
};

class TraceForBody : public TraceInfo {
 public:
  explicit TraceForBody(const DebugInfoPtr &info) : TraceInfo(info, "for_body", "⥁") {}
  MS_DECLARE_PARENT(TraceForBody, TraceInfo);
  ~TraceForBody() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceForBody>(*shared_from_base<TraceForBody>()); }
};

class TraceForAfter : public TraceInfo {
 public:
  explicit TraceForAfter(const DebugInfoPtr &info) : TraceInfo(info, "for_after", "↓") {}
  MS_DECLARE_PARENT(TraceForAfter, TraceInfo);
  ~TraceForAfter() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceForAfter>(*shared_from_base<TraceForAfter>()); }
};

class TraceLoopEnd : public TraceInfo {
 public:
  explicit TraceLoopEnd(const DebugInfoPtr &info) : TraceInfo(info, "loop_end", "↓↓") {}
  MS_DECLARE_PARENT(TraceLoopEnd, TraceInfo);
  ~TraceLoopEnd() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceLoopEnd>(*shared_from_base<TraceLoopEnd>()); }
};

class TraceEquiv : public TraceInfo {
 public:
  explicit TraceEquiv(const DebugInfoPtr &info) : TraceInfo(info, "equiv", "equiv") {}
  MS_DECLARE_PARENT(TraceEquiv, TraceInfo);
  ~TraceEquiv() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceEquiv>(*shared_from_base<TraceEquiv>()); }
};

class TraceGradFpropApp : public TraceInfo {
 public:
  TraceGradFpropApp() : TraceInfo(nullptr, "grad_fprop_app", "▲") {}
  explicit TraceGradFpropApp(const DebugInfoPtr &info) : TraceInfo(info, "grad_fprop_app", "▲") {}
  MS_DECLARE_PARENT(TraceGradFpropApp, TraceInfo);
  ~TraceGradFpropApp() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceGradFpropApp>(*shared_from_base<TraceGradFpropApp>()); }
};

class TraceGradBpropApp : public TraceInfo {
 public:
  TraceGradBpropApp() : TraceInfo(nullptr, "grad_bprop_app", "▼") {}
  explicit TraceGradBpropApp(const DebugInfoPtr &info) : TraceInfo(info, "grad_bprop_app", "▼") {}
  MS_DECLARE_PARENT(TraceGradBpropApp, TraceInfo);
  ~TraceGradBpropApp() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceGradBpropApp>(*shared_from_base<TraceGradBpropApp>()); }
};

class TraceGradFprop : public TraceInfo {
 public:
  TraceGradFprop() : TraceInfo(nullptr, "grad_fprop", "▶") {}
  explicit TraceGradFprop(const DebugInfoPtr &info) : TraceInfo(info, "grad_fprop", "▶") {}
  MS_DECLARE_PARENT(TraceGradFprop, TraceInfo);
  ~TraceGradFprop() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceGradFprop>(*shared_from_base<TraceGradFprop>()); }
};

class TraceGradBprop : public TraceInfo {
 public:
  TraceGradBprop() : TraceInfo(nullptr, "grad_bprop", "◀") {}
  explicit TraceGradBprop(const DebugInfoPtr &info) : TraceInfo(info, "grad_bprop", "◀") {}
  MS_DECLARE_PARENT(TraceGradBprop, TraceInfo);
  ~TraceGradBprop() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceGradBprop>(*shared_from_base<TraceGradBprop>()); }
};

class TraceGradSens : public TraceInfo {
 public:
  TraceGradSens() : TraceInfo(nullptr, "grad_sens", "∇") {}
  explicit TraceGradSens(const DebugInfoPtr &info) : TraceInfo(info, "grad_sens", "∇") {}
  MS_DECLARE_PARENT(TraceGradSens, TraceInfo);
  ~TraceGradSens() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceGradSens>(*shared_from_base<TraceGradSens>()); }
};

class TraceSpecialize : public TraceInfo {
 public:
  explicit TraceSpecialize(const std::string &counter) : TraceInfo(nullptr, "specialize", "") { counter_ = counter; }
  MS_DECLARE_PARENT(TraceSpecialize, TraceInfo);
  std::string name() override { return full_name_ + counter_; }
  std::string symbol() override { return counter_ + "_"; }
  std::string full_name() override { return full_name_ + counter_ + "_"; }
  ~TraceSpecialize() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceSpecialize>(*shared_from_base<TraceSpecialize>()); }
  std::string counter_;
};

class TraceGradOperation : public TraceInfo {
 public:
  explicit TraceGradOperation(const DebugInfoPtr &info) : TraceInfo(info, "grad_ops", "") {}
  MS_DECLARE_PARENT(TraceGradOperation, TraceInfo);
  ~TraceGradOperation() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TraceGradOperation>(*shared_from_base<TraceGradOperation>());
  }
};

class TraceForceBool : public TraceInfo {
 public:
  explicit TraceForceBool(const DebugInfoPtr &info) : TraceInfo(info, "force_bool", "") {}
  MS_DECLARE_PARENT(TraceForceBool, TraceInfo);
  ~TraceForceBool() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceForceBool>(*shared_from_base<TraceForceBool>()); }
};

class TraceExpandJ : public TraceInfo {
 public:
  explicit TraceExpandJ(const DebugInfoPtr &info) : TraceInfo(info, "expand_j", "") {}
  MS_DECLARE_PARENT(TraceExpandJ, TraceInfo);
  ~TraceExpandJ() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceExpandJ>(*shared_from_base<TraceExpandJ>()); }
};

class TraceGenMetaFuncGraph : public TraceInfo {
 public:
  explicit TraceGenMetaFuncGraph(const DebugInfoPtr &info) : TraceInfo(info, "GenMetaFuncGraph", "") {}
  MS_DECLARE_PARENT(TraceGenMetaFuncGraph, TraceInfo);
  ~TraceGenMetaFuncGraph() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TraceGenMetaFuncGraph>(*shared_from_base<TraceGenMetaFuncGraph>());
  }
};

class TraceEvaluatorGenGraph : public TraceInfo {
 public:
  explicit TraceEvaluatorGenGraph(const DebugInfoPtr &info) : TraceInfo(info, "GenEvaluatorGraph", "") {}
  MS_DECLARE_PARENT(TraceEvaluatorGenGraph, TraceInfo);
  ~TraceEvaluatorGenGraph() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TraceEvaluatorGenGraph>(*shared_from_base<TraceEvaluatorGenGraph>());
  }
};

class TraceResolve : public TraceInfo {
 public:
  explicit TraceResolve(const DebugInfoPtr &info) : TraceInfo(info, "resolve", "") {}
  MS_DECLARE_PARENT(TraceResolve, TraceInfo);
  ~TraceResolve() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceResolve>(*shared_from_base<TraceResolve>()); }
};

class TraceTransform : public TraceInfo {
 public:
  TraceTransform() : TraceInfo(nullptr, "transform", "") { transform_name_ = ""; }
  explicit TraceTransform(const std::string &transform_name) : TraceInfo(nullptr, "transform", "") {
    transform_name_ = transform_name;
  }

  std::string full_name() override { return full_name_ + transform_name_; }
  MS_DECLARE_PARENT(TraceTransform, TraceInfo);
  std::string symbol() override {
    if (transform_name_.empty()) {
      return "";
    }
    return transform_name_ + "_";
  }

  ~TraceTransform() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceTransform>(*shared_from_base<TraceTransform>()); }
  std::string transform_name_;
};

class TraceGenerateVarArg : public TraceInfo {
 public:
  explicit TraceGenerateVarArg(const DebugInfoPtr &info) : TraceInfo(info, "GenerateVarArg", "") {}
  MS_DECLARE_PARENT(TraceGenerateVarArg, TraceInfo);
  ~TraceGenerateVarArg() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TraceGenerateVarArg>(*shared_from_base<TraceGenerateVarArg>());
  }
};

class TraceGenerateKwArg : public TraceInfo {
 public:
  explicit TraceGenerateKwArg(const DebugInfoPtr &info) : TraceInfo(info, "GenerateKwArg", "") {}
  MS_DECLARE_PARENT(TraceGenerateKwArg, TraceInfo);
  ~TraceGenerateKwArg() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TraceGenerateKwArg>(*shared_from_base<TraceGenerateKwArg>());
  }
};

class TraceTrasformK : public TraceInfo {
 public:
  explicit TraceTrasformK(const DebugInfoPtr &info) : TraceInfo(info, "TraceTrasformK", "") {}
  MS_DECLARE_PARENT(TraceTrasformK, TraceInfo);
  ~TraceTrasformK() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceTrasformK>(*shared_from_base<TraceTrasformK>()); }
};

class TracePartialTransform : public TraceInfo {
 public:
  explicit TracePartialTransform(const DebugInfoPtr &info) : TraceInfo(info, "PartialTransform", "") {}
  MS_DECLARE_PARENT(TracePartialTransform, TraceInfo);
  ~TracePartialTransform() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TracePartialTransform>(*shared_from_base<TracePartialTransform>());
  }
};

class TraceGetEnv : public TraceInfo {
 public:
  explicit TraceGetEnv(const DebugInfoPtr &info) : TraceInfo(info, "get_env", "") {}
  MS_DECLARE_PARENT(TraceGetEnv, TraceInfo);
  ~TraceGetEnv() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceGetEnv>(*shared_from_base<TraceGetEnv>()); }
};

class TraceDoSignature : public TraceInfo {
 public:
  explicit TraceDoSignature(const DebugInfoPtr &info) : TraceInfo(info, "DoSignature", "") {}
  MS_DECLARE_PARENT(TraceDoSignature, TraceInfo);
  ~TraceDoSignature() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceDoSignature>(*shared_from_base<TraceDoSignature>()); }
};

class TraceCombileLikeGraphs : public TraceInfo {
 public:
  TraceCombileLikeGraphs() : TraceInfo(nullptr, "CombileLike", "L-") {}
  explicit TraceCombileLikeGraphs(const DebugInfoPtr &info) : TraceInfo(info, "CombileLike", "L-") {}
  MS_DECLARE_PARENT(TraceCombileLikeGraphs, TraceInfo);
  ~TraceCombileLikeGraphs() override = default;
  TraceInfoPtr clone() override {
    return std::make_shared<TraceCombileLikeGraphs>(*shared_from_base<TraceCombileLikeGraphs>());
  }
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_TRACE_INFO_H_
