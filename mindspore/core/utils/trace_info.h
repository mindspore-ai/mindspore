/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_TRACE_INFO_H_
#define MINDSPORE_CORE_UTILS_TRACE_INFO_H_

#include <string>
#include <memory>

namespace mindspore {
class TraceInfo;
using TraceInfoPtr = std::shared_ptr<TraceInfo>;
class Location;
using LocationPtr = std::shared_ptr<Location>;
class DebugInfo;
using DebugInfoPtr = std::shared_ptr<DebugInfo>;

class TraceInfo {
 public:
  explicit TraceInfo(const DebugInfoPtr &info) : debug_info_(info) {}
  TraceInfo(const TraceInfo &other) = default;
  TraceInfo &operator=(const TraceInfo &) = default;
  virtual ~TraceInfo() = default;
  virtual std::string name() const { return ""; }
  virtual std::string symbol() const { return ""; }
  virtual std::string full_name() const { return name(); }
  virtual TraceInfoPtr clone() { return std::make_shared<TraceInfo>(*this); }
  virtual std::string action_name() const { return ""; }
  void set_debug_info(const DebugInfoPtr &info) { debug_info_ = info; }
  const DebugInfoPtr &debug_info() const { return debug_info_; }
  template <typename T>
  bool isa() const {
    return dynamic_cast<const T *>(this) != nullptr;
  }

 protected:
  DebugInfoPtr debug_info_;
};

#define MS_DECLARE_TRACE_NAME_SYMBOL(trace_name, trace_symbol) \
  std::string name() const override { return trace_name; }     \
  std::string symbol() const override { return trace_symbol; }

class TracePhi : public TraceInfo {
 public:
  explicit TracePhi(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TracePhi() override = default;
  // phi: Φ
#ifdef _WIN32
  MS_DECLARE_TRACE_NAME_SYMBOL("phi", "phi_");
#else
  MS_DECLARE_TRACE_NAME_SYMBOL("phi", "\u0444");
#endif
  TraceInfoPtr clone() override { return std::make_shared<TracePhi>(*this); }
};

class TraceIfStmtTrueBranch : public TraceInfo {
 public:
  TraceIfStmtTrueBranch(const TraceIfStmtTrueBranch &) = default;
  TraceIfStmtTrueBranch &operator=(const TraceIfStmtTrueBranch &) = default;
  explicit TraceIfStmtTrueBranch(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceIfStmtTrueBranch() override = default;
  // if_true: ✓
  MS_DECLARE_TRACE_NAME_SYMBOL("if_true", "\u2713");
  TraceInfoPtr clone() override { return std::make_shared<TraceIfStmtTrueBranch>(*this); }
};

class TraceIfStmtFalseBranch : public TraceInfo {
 public:
  TraceIfStmtFalseBranch(const TraceIfStmtFalseBranch &) = default;
  TraceIfStmtFalseBranch &operator=(const TraceIfStmtFalseBranch &) = default;
  explicit TraceIfStmtFalseBranch(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceIfStmtFalseBranch() override = default;
  // if_false: ✗
  MS_DECLARE_TRACE_NAME_SYMBOL("if_false", "\u2717");
  TraceInfoPtr clone() override { return std::make_shared<TraceIfStmtFalseBranch>(*this); }
};

class TraceIfStmtAfterBranch : public TraceInfo {
 public:
  explicit TraceIfStmtAfterBranch(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceIfStmtAfterBranch() override = default;
  // if_after: ↓
  MS_DECLARE_TRACE_NAME_SYMBOL("if_after", "\u2193");
  TraceInfoPtr clone() override { return std::make_shared<TraceIfStmtAfterBranch>(*this); }
};

class TraceIfExpTrueBranch : public TraceInfo {
 public:
  explicit TraceIfExpTrueBranch(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceIfExpTrueBranch() override = default;
  // ifexp_true: ↰
  MS_DECLARE_TRACE_NAME_SYMBOL("ifexp_true", "\u21B0");
  TraceInfoPtr clone() override { return std::make_shared<TraceIfExpTrueBranch>(*this); }
};

class TraceIfExpFalseBranch : public TraceInfo {
 public:
  explicit TraceIfExpFalseBranch(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceIfExpFalseBranch() override = default;
  // ifexp_false: ↱
  MS_DECLARE_TRACE_NAME_SYMBOL("ifexp_false", "\u21B1");
  TraceInfoPtr clone() override { return std::make_shared<TraceIfExpFalseBranch>(*this); }
};

class TraceCopy : public TraceInfo {
 public:
  TraceCopy() : TraceInfo(nullptr) {}
  explicit TraceCopy(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceCopy() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("copy", "");
  TraceInfoPtr clone() override { return std::make_shared<TraceCopy>(*this); }
};

class TraceIterator : public TraceInfo {
 public:
  explicit TraceIterator(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceIterator() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("iterator", "@");
  TraceInfoPtr clone() override { return std::make_shared<TraceIterator>(*this); }
};

class TraceWhileHeader : public TraceInfo {
 public:
  explicit TraceWhileHeader(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceWhileHeader() override = default;
  // while_header: ↵
  MS_DECLARE_TRACE_NAME_SYMBOL("while_header", "\u21B5");
  TraceInfoPtr clone() override { return std::make_shared<TraceWhileHeader>(*this); }
};

class TraceWhileBody : public TraceInfo {
 public:
  explicit TraceWhileBody(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceWhileBody() override = default;
  // while_body: ↻
  MS_DECLARE_TRACE_NAME_SYMBOL("while_body", "\u21BB");
  TraceInfoPtr clone() override { return std::make_shared<TraceWhileBody>(*this); }
};

class TraceWhileAfter : public TraceInfo {
 public:
  explicit TraceWhileAfter(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceWhileAfter() override = default;
  // while_after: ↓
  MS_DECLARE_TRACE_NAME_SYMBOL("while_after", "\u2193");
  TraceInfoPtr clone() override { return std::make_shared<TraceWhileAfter>(*this); }
};

class TraceForHeader : public TraceInfo {
 public:
  explicit TraceForHeader(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceForHeader() override = default;
  // for_header: ↵
  MS_DECLARE_TRACE_NAME_SYMBOL("for_header", "\u21B5");
  TraceInfoPtr clone() override { return std::make_shared<TraceForHeader>(*this); }
};

class TraceForBody : public TraceInfo {
 public:
  explicit TraceForBody(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceForBody() override = default;
  // for_body: ↻
  MS_DECLARE_TRACE_NAME_SYMBOL("for_body", "\u21BB");
  TraceInfoPtr clone() override { return std::make_shared<TraceForBody>(*this); }
};

class TraceForRolledBody : public TraceInfo {
 public:
  explicit TraceForRolledBody(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceForRolledBody() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("for_rolled_body", "R_");
  TraceInfoPtr clone() override { return std::make_shared<TraceForRolledBody>(*this); }
};

class TraceForAfter : public TraceInfo {
 public:
  explicit TraceForAfter(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceForAfter() override = default;
  // for_after: ↓
  MS_DECLARE_TRACE_NAME_SYMBOL("for_after", "\u2193");
  TraceInfoPtr clone() override { return std::make_shared<TraceForAfter>(*this); }
};

class TraceLoopEnd : public TraceInfo {
 public:
  explicit TraceLoopEnd(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceLoopEnd() override = default;
  // loop_end: ⇊
  MS_DECLARE_TRACE_NAME_SYMBOL("loop_end", "\u21CA");
  TraceInfoPtr clone() override { return std::make_shared<TraceLoopEnd>(*this); }
};

class TraceEquiv : public TraceInfo {
 public:
  explicit TraceEquiv(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceEquiv() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("equiv", "equiv_");
  TraceInfoPtr clone() override { return std::make_shared<TraceEquiv>(*this); }
};

class TraceGradFpropApp : public TraceInfo {
 public:
  TraceGradFpropApp() : TraceInfo(nullptr) {}
  explicit TraceGradFpropApp(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceGradFpropApp() override = default;
  // grad_fprop_app: ▲
  MS_DECLARE_TRACE_NAME_SYMBOL("grad_fprop_app", "\u25B2");
  TraceInfoPtr clone() override { return std::make_shared<TraceGradFpropApp>(*this); }
};

class TraceGradBpropApp : public TraceInfo {
 public:
  TraceGradBpropApp() : TraceInfo(nullptr) {}
  explicit TraceGradBpropApp(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceGradBpropApp() override = default;
  // grad_bprop_app: ▼
  MS_DECLARE_TRACE_NAME_SYMBOL("grad_bprop_app", "\u25BC");
  TraceInfoPtr clone() override { return std::make_shared<TraceGradBpropApp>(*this); }
};

class TraceGradFprop : public TraceInfo {
 public:
  TraceGradFprop() : TraceInfo(nullptr) {}
  explicit TraceGradFprop(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceGradFprop() override = default;
  // grad_fprop: ▶
  MS_DECLARE_TRACE_NAME_SYMBOL("grad_fprop", "\u25B8");
  TraceInfoPtr clone() override { return std::make_shared<TraceGradFprop>(*this); }
};

class TraceGradBprop : public TraceInfo {
 public:
  TraceGradBprop() : TraceInfo(nullptr) {}
  explicit TraceGradBprop(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceGradBprop() override = default;
  // grad_bprop: ◀
  MS_DECLARE_TRACE_NAME_SYMBOL("grad_bprop", "\u25C2");
  TraceInfoPtr clone() override { return std::make_shared<TraceGradBprop>(*this); }
};

class TraceGradSens : public TraceInfo {
 public:
  TraceGradSens() : TraceInfo(nullptr) {}
  explicit TraceGradSens(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceGradSens() override = default;
  // grad_sens: ∇
  MS_DECLARE_TRACE_NAME_SYMBOL("grad_sens", "\u25BD");
  TraceInfoPtr clone() override { return std::make_shared<TraceGradSens>(*this); }
};

class TraceSpecialize : public TraceInfo {
 public:
  explicit TraceSpecialize(int64_t counter) : TraceInfo(nullptr), counter_(counter) {}
  MS_DECLARE_TRACE_NAME_SYMBOL("specialize" + std::to_string(counter_), std::to_string(counter_) + "_");
  std::string full_name() const override { return "specialize" + std::to_string(counter_) + "_"; }
  ~TraceSpecialize() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceSpecialize>(*this); }

 private:
  int64_t counter_;
};

class TraceGradOperation : public TraceInfo {
 public:
  explicit TraceGradOperation(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceGradOperation() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("grad_ops", "grad_");
  TraceInfoPtr clone() override { return std::make_shared<TraceGradOperation>(*this); }
};

class TraceVmapOperation : public TraceInfo {
 public:
  explicit TraceVmapOperation(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceVmapOperation() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("vmap_ops", "vmap_");
  TraceInfoPtr clone() override { return std::make_shared<TraceVmapOperation>(*this); }
};

class TraceForceBool : public TraceInfo {
 public:
  explicit TraceForceBool(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceForceBool() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("force_bool", "force_bool_");
  TraceInfoPtr clone() override { return std::make_shared<TraceForceBool>(*this); }
};

class TraceForceWhileCond : public TraceInfo {
 public:
  explicit TraceForceWhileCond(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceForceWhileCond() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("force_while_cond", "force_while_cond_");
  TraceInfoPtr clone() override { return std::make_shared<TraceForceWhileCond>(*this); }
};

class TraceExpandJ : public TraceInfo {
 public:
  explicit TraceExpandJ(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceExpandJ() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("expand_j", "j_");
  TraceInfoPtr clone() override { return std::make_shared<TraceExpandJ>(*this); }
};

class TraceGenMetaFuncGraph : public TraceInfo {
 public:
  explicit TraceGenMetaFuncGraph(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceGenMetaFuncGraph() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("GenMetaFuncGraph", "meta_");
  TraceInfoPtr clone() override { return std::make_shared<TraceGenMetaFuncGraph>(*this); }
};

class TraceEvaluatorGenGraph : public TraceInfo {
 public:
  explicit TraceEvaluatorGenGraph(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceEvaluatorGenGraph() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("GenEvaluatorGraph", "gen_evaluator_graph_");
  TraceInfoPtr clone() override { return std::make_shared<TraceEvaluatorGenGraph>(*this); }
};

class TraceParse : public TraceInfo {
 public:
  explicit TraceParse(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceParse() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("parse", "");
  TraceInfoPtr clone() override { return std::make_shared<TraceParse>(*this); }
};

class TraceResolve : public TraceInfo {
 public:
  explicit TraceResolve(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceResolve() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("resolve", "resolve_");
  TraceInfoPtr clone() override { return std::make_shared<TraceResolve>(*this); }
};

class TraceTransform : public TraceInfo {
 public:
  TraceTransform() : TraceTransform("") {}
  explicit TraceTransform(const std::string &transform_name) : TraceInfo(nullptr), transform_name_(transform_name) {}
  ~TraceTransform() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("transform", transform_name_.empty() ? "" : (transform_name_ + "_"));
  std::string full_name() const override { return "transform" + transform_name_; }
  TraceInfoPtr clone() override { return std::make_shared<TraceTransform>(*this); }

 private:
  std::string transform_name_;
};

class TraceGenerateVarArg : public TraceInfo {
 public:
  explicit TraceGenerateVarArg(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceGenerateVarArg() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("GenerateVarArg", "gen_var_arg_");
  TraceInfoPtr clone() override { return std::make_shared<TraceGenerateVarArg>(*this); }
};

class TraceGenerateKwArg : public TraceInfo {
 public:
  explicit TraceGenerateKwArg(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceGenerateKwArg() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("GenerateKwArg", "gen_kw_arg_");
  TraceInfoPtr clone() override { return std::make_shared<TraceGenerateKwArg>(*this); }
};

class TraceTransformK : public TraceInfo {
 public:
  explicit TraceTransformK(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceTransformK() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("TraceTransformK", "k_");
  TraceInfoPtr clone() override { return std::make_shared<TraceTransformK>(*this); }
};

class TracePartialTransform : public TraceInfo {
 public:
  explicit TracePartialTransform(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TracePartialTransform() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("PartialTransform", "partial_trans_");
  TraceInfoPtr clone() override { return std::make_shared<TracePartialTransform>(*this); }
};

class TraceGetEnv : public TraceInfo {
 public:
  explicit TraceGetEnv(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceGetEnv() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("get_env", "get_env_");
  TraceInfoPtr clone() override { return std::make_shared<TraceGetEnv>(*this); }
};

class TraceDoSignature : public TraceInfo {
 public:
  explicit TraceDoSignature(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceDoSignature() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("DoSignature", "");
  TraceInfoPtr clone() override { return std::make_shared<TraceDoSignature>(*this); }
};

class TraceCombileLikeGraphs : public TraceInfo {
 public:
  TraceCombileLikeGraphs() : TraceInfo(nullptr) {}
  explicit TraceCombileLikeGraphs(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceCombileLikeGraphs() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("CombileLike", "L_");
  TraceInfoPtr clone() override { return std::make_shared<TraceCombileLikeGraphs>(*this); }
};

class TraceGraphReusing : public TraceInfo {
 public:
  TraceGraphReusing() : TraceInfo(nullptr) {}
  explicit TraceGraphReusing(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceGraphReusing() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("CellReusing", "CR_");
  TraceInfoPtr clone() override { return std::make_shared<TraceGraphReusing>(*this); }
};

class TraceSegmentTransform : public TraceInfo {
 public:
  explicit TraceSegmentTransform(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceSegmentTransform() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("segment_transform", "seg_trans_");
  TraceInfoPtr clone() override { return std::make_shared<TraceSegmentTransform>(*this); }
};

class TraceOpt : public TraceInfo {
 public:
  explicit TraceOpt(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceOpt() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("opt", "");
  TraceInfoPtr clone() override { return std::make_shared<TraceOpt>(*this); }
};

class TraceListComp : public TraceInfo {
 public:
  explicit TraceListComp(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceListComp() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("ListComp", "G_");
  TraceInfoPtr clone() override { return std::make_shared<TraceListComp>(*this); }
};

class TraceDictComp : public TraceInfo {
 public:
  explicit TraceDictComp(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceDictComp() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("DictComp", "GD_");
  TraceInfoPtr clone() override { return std::make_shared<TraceDictComp>(*this); }
};

class TraceMixedPrecision : public TraceInfo {
 public:
  explicit TraceMixedPrecision(const DebugInfoPtr &info) : TraceInfo(info) {}
  MS_DECLARE_TRACE_NAME_SYMBOL("MixedPrecision", "C_");
  ~TraceMixedPrecision() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceMixedPrecision>(*this); }
};

class TraceShard : public TraceInfo {
 public:
  explicit TraceShard(const DebugInfoPtr &info) : TraceInfo(info) {}
  ~TraceShard() override = default;
  MS_DECLARE_TRACE_NAME_SYMBOL("shard_ops", "shard_");
  TraceInfoPtr clone() override { return std::make_shared<TraceShard>(*this); }
};

class TraceAssert : public TraceInfo {
 public:
  explicit TraceAssert(const DebugInfoPtr &info) : TraceInfo(info) {}
  MS_DECLARE_TRACE_NAME_SYMBOL("Assert", "assert_");
  ~TraceAssert() override = default;
  TraceInfoPtr clone() override { return std::make_shared<TraceAssert>(*this); }
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_TRACE_INFO_H_
