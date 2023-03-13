/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_EVALUATOR_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_EVALUATOR_H_

#include <memory>
#include <string>
#include <vector>
#include <stack>
#include <unordered_map>

#include "utils/ms_context.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "pipeline/jit/static_analysis/async_eval_result.h"

namespace mindspore {
namespace abstract {
using EvaluatorCacheMgrPtr = std::shared_ptr<EvaluatorCacheMgr>;
using EvaluatorAttrMap =
  std::unordered_map<AbstractBasePtrList, AttrValueMapPtr, AbstractBasePtrListHasher, AbstractBasePtrListEqual>;
using EvaluatorAttrCache = MultiThreadCache<AbstractBasePtrList, AttrValueMapPtr, EvaluatorAttrMap>;
using EvaluatorAttrCachePtr = std::shared_ptr<EvaluatorAttrCache>;

class Evaluator : public Base {
 public:
  explicit Evaluator(const std::string &id)
      : identifier_(id),
        evaluator_cache_mgr_(std::make_shared<EvaluatorCacheMgr>()),
        attr_cache_(std::make_shared<EvaluatorAttrCache>()) {}
  ~Evaluator() override = default;
  MS_DECLARE_PARENT(Evaluator, Base);

  // Difference between Run() and Eval():
  // Run() will be called with ConfigPtrList, but Eval() will be called with AbstractBasePtr.
  // Run() will modify cache_ member, so it cannot marked as const;
  virtual EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                            const AnfNodeConfigPtr &out_conf);

  virtual EvalResultPtr Eval(AnalysisEnginePtr engine, const AbstractBasePtrList &args_abs_list,
                             const AnfNodeConfigPtr &out_conf) = 0;

  virtual EvalResultPtr SingleRun(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                  const AnfNodeConfigPtr &out_conf);

  virtual AbstractBasePtrList NormalizeArgs(const AbstractBasePtrList &args_abs_list) const { return args_abs_list; }

  virtual AbstractBasePtrList BroadenUndeterminedArgs(const AbstractBasePtrList &args_abs_list,
                                                      const AnalysisEnginePtr &) {
    return args_abs_list;
  }

  virtual EvalResultPtr EvalUndeterminedArgs(const AbstractBasePtrList &args_abs_list) {
    auto is_undetermined = std::any_of(args_abs_list.begin(), args_abs_list.end(), [](auto &arg) -> bool {
      if (arg->BuildType()->type_id() == kObjectTypeUndeterminedType) {
        return true;
      }
      return false;
    });
    if (is_undetermined) {
      MS_LOG(DEBUG) << "Eval " << identifier_ << " return undetermined abstract result";
      return std::make_shared<EvalResult>(std::make_shared<AbstractUndetermined>(), std::make_shared<AttrValueMap>());
    }
    return nullptr;
  }

  std::string ToString() const override { return identifier_; }

  virtual AnfNodePtr bound_node() const { return bound_node_.lock(); }

  virtual void set_bound_node(const AnfNodePtr &node) { bound_node_ = AnfNodeWeakPtr(node); }

  EvaluatorCacheMgrPtr evaluator_cache_mgr() const { return evaluator_cache_mgr_; }
  EvaluatorAttrCachePtr attr_cache() const { return attr_cache_; }

  const std::recursive_timed_mutex &eval_lock() const { return eval_lock_; }

 protected:
  std::string identifier_;
  AnfNodeWeakPtr bound_node_;
  EvaluatorCacheMgrPtr evaluator_cache_mgr_;
  std::recursive_timed_mutex eval_lock_;

 private:
  EvaluatorAttrCachePtr attr_cache_;
};

class PrimEvaluator : public Evaluator {
 public:
  explicit PrimEvaluator(const std::string &id) : Evaluator(id) {}
  ~PrimEvaluator() override = default;
  MS_DECLARE_PARENT(PrimEvaluator, Evaluator);
  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) final {
    MS_LOG(EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }
};

class TrivialPrimEvaluator : public PrimEvaluator {
 public:
  explicit TrivialPrimEvaluator(const std::string &id)
      : PrimEvaluator(id), eval_cache_(AnalysisResultCacheMgr::GetInstance().prim_eval_cache()) {}
  ~TrivialPrimEvaluator() override = default;
  MS_DECLARE_PARENT(TrivialPrimEvaluator, PrimEvaluator);
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, const AnfNodeConfigPtr &) final;
  virtual EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list) = 0;

 protected:
  PrimitiveEvalCachePtr eval_cache_;
};

class TransitionPrimEvaluator : public PrimEvaluator {
 public:
  explicit TransitionPrimEvaluator(const std::string &id) : PrimEvaluator(id) {}
  ~TransitionPrimEvaluator() override = default;
  MS_DECLARE_PARENT(TransitionPrimEvaluator, PrimEvaluator);
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) final;
  // Parameter in_conf0 : the first element in args_conf_list;
  virtual EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_abs_list,
                                 const ConfigPtr &in_conf, const AnfNodeConfigPtr &out_conf) = 0;
};

class SymbolicPrimEvaluator : public PrimEvaluator {
 public:
  explicit SymbolicPrimEvaluator(const std::string &id) : PrimEvaluator(id) {}
  ~SymbolicPrimEvaluator() override = default;
  MS_DECLARE_PARENT(SymbolicPrimEvaluator, PrimEvaluator);
  EvalResultPtr Run(AnalysisEnginePtr, const ConfigPtrList &args_conf_list, const AnfNodeConfigPtr &) final;
  virtual EvalResultPtr EvalPrim(const ConfigPtrList &args_conf_list) = 0;
};

// Evaluator will be stored in AnalysisEngine.evaluators_
using EvaluatorPtrList = std::vector<EvaluatorPtr>;

class DummyEvaluator : public Evaluator {
 public:
  DummyEvaluator() : Evaluator("dummy") {}
  ~DummyEvaluator() override = default;
  MS_DECLARE_PARENT(DummyEvaluator, Evaluator);
  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    return nullptr;
  }
};

// Wrap another evaluator to track a subset of uses.
// A TrackedEvaluator has its own cache that maps possible calls to
// their results, but is ultimately backed by a different evaluator.
// Multiple TrackedEvaluators can be backed by the same Evaluator.
class TrackedEvaluator : public Evaluator {
 public:
  explicit TrackedEvaluator(const EvaluatorPtr &subinf) : Evaluator("TrackedEvaluator"), sub_evaluator_(subinf) {}
  ~TrackedEvaluator() override = default;
  MS_DECLARE_PARENT(TrackedEvaluator, Evaluator);
  AnfNodePtr bound_node() const override {
    if (sub_evaluator_ != nullptr) {
      return sub_evaluator_->bound_node();
    }
    return bound_node_.lock();
  }

  void set_bound_node(const AnfNodePtr &node) override {
    if (sub_evaluator_ != nullptr) {
      sub_evaluator_->set_bound_node(node);
    }
    bound_node_ = AnfNodeWeakPtr(node);
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override;
  std::string ToString() const override { return identifier_ + "_" + sub_evaluator_->ToString(); }

 private:
  EvaluatorPtr sub_evaluator_;
};

using FuncGraphCacheMap =
  std::unordered_map<AbstractBasePtrList, FuncGraphPtr, AbstractBasePtrListHasher, AbstractBasePtrListEqual>;
class StackFrame;
using StackFramePtr = std::shared_ptr<StackFrame>;

class BaseFuncGraphEvaluator : public Evaluator {
 public:
  explicit BaseFuncGraphEvaluator(const AnalysisContextPtr &context)
      : Evaluator("basegraph"), parent_context_(context) {}

  ~BaseFuncGraphEvaluator() override = default;
  MS_DECLARE_PARENT(BaseFuncGraphEvaluator, Evaluator);

  EvalResultPtr Eval(AnalysisEnginePtr engine, const AbstractBasePtrList &args_abs_list,
                     const AnfNodeConfigPtr &out_conf) override;

  virtual FuncGraphPtr GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_abs_list) = 0;

  AnalysisContextPtr parent_context() const { return parent_context_; }
  void set_parent_context(const AnalysisContextPtr &parent_context) { parent_context_ = parent_context; }

  void PushAlwaysEvalFlag(bool flag) { always_eval_flags_.push_back(flag); }
  void PopAlwaysEvalFlag() { always_eval_flags_.pop_back(); }
  bool always_eval_flag() const {
    if (always_eval_flags_.empty()) {
      MS_LOG(EXCEPTION) << "Always_eval_flag should not be empty";
    }
    return always_eval_flags_.back();
  }

  virtual void SyncFuncGraphSideEffectFlag(const FuncGraphPtr &func_graph) = 0;

 protected:
  AnalysisContextPtr parent_context_;

 private:
  // As evaluator can be recursively called, so use a vector to simulate a stack of flags.
  std::vector<bool> always_eval_flags_;
  AbstractBasePtr LaunchRecursiveEval(const AnalysisEnginePtr &engine, const FuncGraphPtr &fg,
                                      const AnalysisContextPtr &context) const;
  // Add functions for stack frame routine.
  AbstractBasePtr LaunchStackFrame(const AnalysisEnginePtr &engine, const FuncGraphPtr &fg,
                                   const AnalysisContextPtr &context);
  static void EnterStackFrame(const AnalysisEnginePtr &engine, const StackFramePtr &current_stack_frame,
                              const StackFramePtr &new_stack_frame);
  static void LeaveStackFrame(const AnalysisEnginePtr &, const StackFramePtr &current_stack_frame);
};

class FuncGraphEvaluator : public BaseFuncGraphEvaluator {
 public:
  FuncGraphEvaluator(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context)
      : BaseFuncGraphEvaluator(context), func_graph_(func_graph) {}

  ~FuncGraphEvaluator() override = default;
  MS_DECLARE_PARENT(FuncGraphEvaluator, BaseFuncGraphEvaluator);

  FuncGraphPtr GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_abs_list) override;

  FuncGraphPtr func_graph() { return func_graph_; }

  AbstractBasePtrList NormalizeArgs(const AbstractBasePtrList &args_abs_list) const override;
  AbstractBasePtrList BroadenUndeterminedArgs(const AbstractBasePtrList &args_abs_list,
                                              const AnalysisEnginePtr &engine) override;
  std::string ToString() const override { return identifier_ + "_" + func_graph_->ToString(); }

  void SyncFuncGraphSideEffectFlag(const FuncGraphPtr &func_graph) override {
    if (func_graph->has_side_effect_node()) {
      func_graph_->set_has_side_effect_node(true);
    }
  }

 private:
  FuncGraphPtr func_graph_;
  FuncGraphCacheMap func_graph_cache_;
  std::vector<AbstractBasePtrList> trace_;
};
using FuncGraphEvaluatorPtr = std::shared_ptr<FuncGraphEvaluator>;

class MetaFuncGraphEvaluator : public BaseFuncGraphEvaluator {
 public:
  // Note: context parameter is not used;
  MetaFuncGraphEvaluator(const MetaFuncGraphPtr &meta_func_graph, const ScopePtr &scope)
      : BaseFuncGraphEvaluator(AnalysisContext::DummyContext()), meta_func_graph_(meta_func_graph), scope_(scope) {}
  ~MetaFuncGraphEvaluator() override = default;
  MS_DECLARE_PARENT(MetaFuncGraphEvaluator, BaseFuncGraphEvaluator);

  FuncGraphPtr GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_abs_list) override;

  // Return normalized versions of the arguments.
  AbstractBasePtrList NormalizeArgs(const AbstractBasePtrList &args_abs_list) const override {
    return meta_func_graph_->NormalizeArgs(args_abs_list);
  }
  std::string ToString() const override { return identifier_ + "_" + meta_func_graph_->ToString(); }

  void SyncFuncGraphSideEffectFlag(const FuncGraphPtr &func_graph) override {
    if (func_graph->has_side_effect_node()) {
      meta_func_graph_->set_has_side_effect_node(true);
    }
  }

 private:
  MetaFuncGraphPtr meta_func_graph_;
  FuncGraphCacheMap func_graph_cache_;
  ScopePtr scope_;
};

class PartialAppEvaluator : public Evaluator {
 public:
  PartialAppEvaluator(const EvaluatorPtr &evaluator, const AbstractBasePtrList &args)
      : Evaluator("PartialAppEvaluator"), evaluator_(evaluator), args_spec_list_(args) {}
  ~PartialAppEvaluator() override = default;
  MS_DECLARE_PARENT(PartialAppEvaluator, Evaluator);
  AnfNodePtr bound_node() const override {
    if (evaluator_ != nullptr) {
      return evaluator_->bound_node();
    }
    return bound_node_.lock();
  }

  void set_bound_node(const AnfNodePtr &node) override {
    if (evaluator_ != nullptr) {
      evaluator_->set_bound_node(node);
    }
    bound_node_ = AnfNodeWeakPtr(node);
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(EXCEPTION) << "Should not be called, Run() method should be called";
  }

  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                    const AnfNodeConfigPtr &out_conf) override;
  std::string ToString() const override { return identifier_ + "_" + evaluator_->ToString(); }

 private:
  EvaluatorPtr evaluator_;
  AbstractBasePtrList args_spec_list_;
};

class VirtualEvaluator : public Evaluator {
 public:
  VirtualEvaluator(const AbstractBasePtrList &args_abs_list, const AbstractBasePtr &output)
      : Evaluator("virtual"), args_spec_list_(args_abs_list), output_(output) {}
  ~VirtualEvaluator() override = default;
  MS_DECLARE_PARENT(VirtualEvaluator, Evaluator);

  EvalResultPtr Eval(AnalysisEnginePtr engine, const AbstractBasePtrList &args_abs_list,
                     const AnfNodeConfigPtr &out_conf) override;
  std::string ToString() const override { return identifier_; }

 private:
  AbstractBasePtrList args_spec_list_;
  AbstractBasePtr output_;
};

class JEvaluator : public Evaluator {
 public:
  JEvaluator(const EvaluatorPtr &evaluator, const AbstractFunctionPtr &orig_func)
      : Evaluator("JEvaluator"), evaluator_(evaluator), primal_func_(orig_func) {}
  ~JEvaluator() override = default;
  MS_DECLARE_PARENT(JEvaluator, Evaluator);
  AnfNodePtr bound_node() const override {
    if (evaluator_ != nullptr) {
      return evaluator_->bound_node();
    }
    return bound_node_.lock();
  }

  void set_bound_node(const AnfNodePtr &node) override {
    if (evaluator_ != nullptr) {
      evaluator_->set_bound_node(node);
    }
    bound_node_ = AnfNodeWeakPtr(node);
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(EXCEPTION) << "Should not be called, Run() method should be called";
  }
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, const AnfNodeConfigPtr &) override;
  std::string ToString() const override { return identifier_ + "_" + evaluator_->ToString(); }

 private:
  EvaluatorPtr evaluator_;
  AbstractFunctionPtr primal_func_;
};

class TaylorEvaluator : public Evaluator {
 public:
  TaylorEvaluator(const EvaluatorPtr &evaluator, const AbstractFunctionPtr &orig_func)
      : Evaluator("TaylorEvaluator"), evaluator_(evaluator), primal_func_(orig_func) {}
  ~TaylorEvaluator() override = default;
  MS_DECLARE_PARENT(TaylorEvaluator, Evaluator);
  AnfNodePtr bound_node() const override {
    if (evaluator_ != nullptr) {
      return evaluator_->bound_node();
    }
    return bound_node_.lock();
  }

  void set_bound_node(const AnfNodePtr &node) override {
    if (evaluator_ != nullptr) {
      evaluator_->set_bound_node(node);
    }
    bound_node_ = AnfNodeWeakPtr(node);
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(EXCEPTION) << "Should not be called, Run() method should be called";
  }
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, const AnfNodeConfigPtr &) override;
  std::string ToString() const override { return identifier_ + "_" + evaluator_->ToString(); }

 private:
  EvaluatorPtr evaluator_;
  AbstractFunctionPtr primal_func_;
};

class ShardEvaluator : public Evaluator {
 public:
  ShardEvaluator(const EvaluatorPtr &evaluator, const AbstractFunctionPtr &orig_func)
      : Evaluator("ShardEvaluator"), evaluator_(evaluator), primal_func_(orig_func) {}
  ~ShardEvaluator() override = default;
  MS_DECLARE_PARENT(ShardEvaluator, Evaluator);

  AnfNodePtr bound_node() const override {
    if (evaluator_ != nullptr) {
      return evaluator_->bound_node();
    }
    return bound_node_.lock();
  }

  void set_bound_node(const AnfNodePtr &node) override {
    if (evaluator_ != nullptr) {
      evaluator_->set_bound_node(node);
    }
    bound_node_ = AnfNodeWeakPtr(node);
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(EXCEPTION) << "Should not be called, Run() method should be called";
  }

  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, const AnfNodeConfigPtr &) override;

  std::string ToString() const override { return identifier_ + "_" + evaluator_->ToString(); }

 private:
  EvaluatorPtr evaluator_;
  AbstractFunctionPtr primal_func_;
};

class VmapEvaluator : public Evaluator {
 public:
  VmapEvaluator(const EvaluatorPtr &evaluator, const AbstractFunctionPtr &orig_func, const ValuePtr &in_axes,
                const ValuePtr &out_axes, size_t cell_size)
      : Evaluator("VmapEvaluator"),
        evaluator_(evaluator),
        primal_func_(orig_func),
        in_axes_(in_axes),
        out_axes_(out_axes),
        cell_size_(cell_size) {}
  ~VmapEvaluator() override = default;
  MS_DECLARE_PARENT(VmapEvaluator, Evaluator);
  AnfNodePtr bound_node() const override {
    if (evaluator_ != nullptr) {
      return evaluator_->bound_node();
    }
    return bound_node_.lock();
  }

  void set_bound_node(const AnfNodePtr &node) override {
    if (evaluator_ != nullptr) {
      evaluator_->set_bound_node(node);
    }
    bound_node_ = AnfNodeWeakPtr(node);
  }

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &, const AnfNodeConfigPtr &) override {
    MS_LOG(EXCEPTION) << "Should not be called, Run() method should be called";
  }
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, const AnfNodeConfigPtr &) override;
  std::string ToString() const override { return identifier_ + "_" + evaluator_->ToString(); }

 private:
  EvaluatorPtr evaluator_;
  AbstractFunctionPtr primal_func_;
  ValuePtr in_axes_;
  ValuePtr out_axes_;
  size_t cell_size_;
};

AbstractBasePtrList EvaluateArguments(const ConfigPtrList &args_conf_list);

bool CheckIfAlwaysEval(const AnfNodeConfigPtr &conf, const AbstractBasePtr &arg);
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_EVALUATOR_H_
