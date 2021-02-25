/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include <vector>

#include "pipeline/jit/static_analysis/static_analysis.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace abstract {
using EvaluatorCacheMap =
  std::unordered_map<AbstractBasePtrList, EvalResultPtr, AbstractBasePtrListHasher, AbstractBasePtrListEqual>;
using EvaluatorCacheMapPtr = std::shared_ptr<EvaluatorCacheMap>;

using EvaluatorAttrMap =
  std::unordered_map<AbstractBasePtrList, AttrValueMapPtr, AbstractBasePtrListHasher, AbstractBasePtrListEqual>;
using EvaluatorAttrMapPtr = std::shared_ptr<EvaluatorAttrMap>;

class Evaluator : public Base {
 public:
  explicit Evaluator(const std::string &id)
      : evaluator_cache_map_(std::make_shared<EvaluatorCacheMap>()),
        attr_cache_(std::make_shared<EvaluatorAttrMap>()),
        identifier_(id) {}
  ~Evaluator() override = default;
  MS_DECLARE_PARENT(Evaluator, Base);

  // difference between Run() and Eval():
  // Run() will be called with ConfigPtrList, but Eval() will be called with AbstractBasePtr.
  // Run() will modify cache_ member, so it cannot marked as const;
  virtual EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, AnfNodeConfigPtr out_conf);

  virtual EvalResultPtr Eval(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) = 0;

  virtual AbstractBasePtrList NormalizeArgs(const AbstractBasePtrList &args_spec_list) const { return args_spec_list; }

  virtual AbstractBasePtrList BroadenUndeterminedArgs(const AbstractBasePtrList &args_spec_list) {
    return args_spec_list;
  }

  virtual EvalResultPtr AbstractEval(const AbstractBasePtrList &args_spec_list) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    bool enable_sparse = context->get_param<bool>(MS_CTX_ENABLE_SPARSE);
    if (!enable_sparse) {
      return nullptr;
    }

    auto is_abstract = std::any_of(args_spec_list.begin(), args_spec_list.end(), [](auto &arg) {
      if (arg->BuildType()->type_id() == kObjectTypeUndeterminedType) {
        return true;
      }
      return false;
    });
    if (is_abstract) {
      MS_LOG(DEBUG) << "Eval " << identifier_ << " return abstract result";
      return std::make_shared<EvalResult>(std::make_shared<AbstractUndetermined>(), std::make_shared<AttrValueMap>());
    }
    return nullptr;
  }

  std::string ToString() const override { return identifier_; }

  virtual AnfNodePtr bound_node() const { return bound_node_.lock(); }

  virtual void set_bound_node(const AnfNodePtr &node) { bound_node_ = AnfNodeWeakPtr(node); }

  EvaluatorCacheMapPtr &evaluator_cache_map() { return evaluator_cache_map_; }
  EvaluatorAttrMapPtr &attr_cache() { return attr_cache_; }

  EvaluatorCacheMapPtr evaluator_cache_map_;
  EvaluatorAttrMapPtr attr_cache_;
  std::string identifier_;

  AnfNodeWeakPtr bound_node_;
};

class PrimEvaluator : public Evaluator {
 public:
  explicit PrimEvaluator(const std::string &id) : Evaluator(id) {}
  ~PrimEvaluator() override = default;
  MS_DECLARE_PARENT(PrimEvaluator, Evaluator);
  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &) final {
    MS_LOG(EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }
};

class TrivialPrimEvaluator : public PrimEvaluator {
 public:
  explicit TrivialPrimEvaluator(const std::string &id) : PrimEvaluator(id) {}
  ~TrivialPrimEvaluator() override = default;
  MS_DECLARE_PARENT(TrivialPrimEvaluator, PrimEvaluator);
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, AnfNodeConfigPtr out_conf) final;
  virtual EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list) = 0;
};

class TransitionPrimEvaluator : public PrimEvaluator {
 public:
  explicit TransitionPrimEvaluator(const std::string &id) : PrimEvaluator(id) {}
  ~TransitionPrimEvaluator() override = default;
  MS_DECLARE_PARENT(TransitionPrimEvaluator, PrimEvaluator);
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, AnfNodeConfigPtr out_conf) final;
  // Parameter in_conf0 : the first element in args_conf_list;
  virtual EvalResultPtr EvalPrim(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list,
                                 const ConfigPtr &in_conf0, const AnfNodeConfigPtr &out_conf) = 0;
};

class SymbolicPrimEvaluator : public PrimEvaluator {
 public:
  explicit SymbolicPrimEvaluator(const std::string &id) : PrimEvaluator(id) {}
  ~SymbolicPrimEvaluator() override = default;
  MS_DECLARE_PARENT(SymbolicPrimEvaluator, PrimEvaluator);
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, AnfNodeConfigPtr out_conf) final;
  virtual EvalResultPtr EvalPrim(const ConfigPtrList &args_conf_list) = 0;
};

// Evaluator will be stored in AnalysisEngine.constructors_
using EvaluatorPtrList = std::vector<EvaluatorPtr>;

class DummyEvaluator : public Evaluator {
 public:
  DummyEvaluator() : Evaluator("dummy") {}
  ~DummyEvaluator() override = default;
  MS_DECLARE_PARENT(DummyEvaluator, Evaluator);
  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &) override { return nullptr; }
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

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &) override {
    MS_LOG(EXCEPTION) << "Eval() should not be called, Run() method should be called";
  }
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, AnfNodeConfigPtr out_conf) override;
  std::string ToString() const override { return identifier_ + "_" + sub_evaluator_->ToString(); }

 private:
  EvaluatorPtr sub_evaluator_;
};

class BaseFuncGraphEvaluator : public Evaluator {
 public:
  explicit BaseFuncGraphEvaluator(const AnalysisContextPtr &context)
      : Evaluator("basegraph"), parent_context_(context) {}

  ~BaseFuncGraphEvaluator() override = default;
  MS_DECLARE_PARENT(BaseFuncGraphEvaluator, Evaluator);

  EvalResultPtr Eval(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) override;

  virtual FuncGraphPtr GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) = 0;

  AnalysisContextPtr MakeContext(const AnalysisEnginePtr &engine, const AbstractBasePtrList &args_spec_list);
  AnalysisContextPtr graph_context() const { return graph_context_; }

 protected:
  AnalysisContextPtr parent_context_;

 private:
  AnalysisContextPtr graph_context_;
};

class FuncGraphEvaluator : public BaseFuncGraphEvaluator {
 public:
  FuncGraphEvaluator(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context)
      : BaseFuncGraphEvaluator(context->Filter(func_graph)), func_graph_(func_graph) {}

  ~FuncGraphEvaluator() override = default;
  MS_DECLARE_PARENT(FuncGraphEvaluator, BaseFuncGraphEvaluator);

  FuncGraphPtr GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) override;

  FuncGraphPtr func_graph() { return func_graph_; }

  AbstractBasePtrList NormalizeArgs(const AbstractBasePtrList &args_spec_list) const override;
  AbstractBasePtrList BroadenUndeterminedArgs(const AbstractBasePtrList &args_spec_list) override;
  std::string ToString() const override { return identifier_ + "_" + func_graph_->ToString(); }

 private:
  FuncGraphPtr func_graph_;
  std::unordered_map<AbstractBasePtrList, FuncGraphPtr, AbstractBasePtrListHasher, AbstractBasePtrListEqual>
    func_graph_cache_;
  std::vector<AbstractBasePtrList> trace_;
};
using FuncGraphEvaluatorPtr = std::shared_ptr<FuncGraphEvaluator>;

class MetaFuncGraphEvaluator : public BaseFuncGraphEvaluator {
 public:
  // Note: context parameter is not used;
  MetaFuncGraphEvaluator(const MetaFuncGraphPtr &meta_func_graph, AnalysisContextPtr, const ScopePtr &scope)
      : BaseFuncGraphEvaluator(AnalysisContext::DummyContext()), meta_func_graph_(meta_func_graph), scope_(scope) {}
  ~MetaFuncGraphEvaluator() override = default;
  MS_DECLARE_PARENT(MetaFuncGraphEvaluator, BaseFuncGraphEvaluator);

  FuncGraphPtr GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) override;

  // Return normalized versions of the arguments.
  AbstractBasePtrList NormalizeArgs(const AbstractBasePtrList &args_spec_list) const override {
    return meta_func_graph_->NormalizeArgs(args_spec_list);
  }
  std::string ToString() const override { return identifier_ + "_" + meta_func_graph_->ToString(); }

 private:
  MetaFuncGraphPtr meta_func_graph_;
  std::unordered_map<AbstractBasePtrList, FuncGraphPtr, AbstractBasePtrListHasher, AbstractBasePtrListEqual>
    func_graph_cache_;
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

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &) override {
    MS_LOG(EXCEPTION) << "Should not be called, Run() method should be called";
  }

  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, AnfNodeConfigPtr out_conf) override;
  std::string ToString() const override { return identifier_ + "_" + evaluator_->ToString(); }

 private:
  EvaluatorPtr evaluator_;
  AbstractBasePtrList args_spec_list_;
};

class VirtualEvaluator : public Evaluator {
 public:
  VirtualEvaluator(const AbstractBasePtrList &args_spec_list, const AbstractBasePtr &output)
      : Evaluator("virtual"), args_spec_list_(args_spec_list), output_(output) {}
  ~VirtualEvaluator() override = default;
  MS_DECLARE_PARENT(VirtualEvaluator, Evaluator);

  EvalResultPtr Eval(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) override;
  std::string ToString() const override { return identifier_; }

 private:
  AbstractBasePtrList args_spec_list_;
  AbstractBasePtr output_;
};

class JEvaluator : public Evaluator {
 public:
  JEvaluator(const EvaluatorPtr &evaluator, const AbstractFunctionPtr &orig_func)
      : Evaluator("JEvaluator"), evaluator_(evaluator), orig_func_(orig_func) {}
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

  EvalResultPtr Eval(AnalysisEnginePtr, const AbstractBasePtrList &) override {
    MS_LOG(EXCEPTION) << "Should not be called, Run() method should be called";
  }
  EvalResultPtr Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, AnfNodeConfigPtr out_conf) override;
  std::string ToString() const override { return identifier_ + "_" + evaluator_->ToString(); }

 private:
  EvaluatorPtr evaluator_;
  AbstractFunctionPtr orig_func_;
};
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_EVALUATOR_H_
