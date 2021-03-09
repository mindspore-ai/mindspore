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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_STATIC_ANALYSIS_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_STATIC_ANALYSIS_H_

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>
#include <map>
#include <set>
#include <unordered_set>

#ifdef DEBUG
#include <stack>
#endif

#include "utils/log_adapter.h"
#include "ir/anf.h"
#include "pybind_api/ir/primitive_py.h"
#include "abstract/analysis_context.h"
#include "abstract/abstract_function.h"
#include "pipeline/jit/parse/parse.h"

namespace mindspore {
namespace abstract {
// define attribute value map
using AttrValueMap = std::unordered_map<std::string, ValuePtr>;
using AttrValueMapPtr = std::shared_ptr<AttrValueMap>;

// the class to save evaluated result: abstract value and modified attribute
class EvalResult : public Base {
 public:
  EvalResult(AbstractBasePtr abs, AttrValueMapPtr attr) : abstract_(abs), attribute_(attr) {}
  ~EvalResult() override = default;
  MS_DECLARE_PARENT(EvalResult, Base);
  AbstractBasePtr abstract() { return abstract_; }
  AttrValueMapPtr attribute() { return attribute_; }

 private:
  AbstractBasePtr abstract_;
  // Attribute related to PrimEvaluator;
  AttrValueMapPtr attribute_;
};
using EvalResultPtr = std::shared_ptr<EvalResult>;

// Superclass for AnfNodeConfig and VirtualConfig.
class Config : public Base {
 public:
  Config() = default;
  ~Config() override = default;
  MS_DECLARE_PARENT(Config, Base);
  virtual EvalResultPtr ObtainEvalResult() = 0;
};

// Config will be stored in AnalysisCache
using ConfigPtr = std::shared_ptr<Config>;
using ConfigPtrList = std::vector<ConfigPtr>;

// Config to a certain node in a certain context.
class AnfNodeConfig : public Config {
 public:
  AnfNodeConfig(const AnalysisEnginePtr &engine, const AnfNodePtr &node, const AnalysisContextPtr &context)
      : Config(), engine_(std::weak_ptr<AnalysisEngine>(engine)), node_(node) {
    FuncGraphPtr fg;
    if (IsValueNode<FuncGraph>(node)) {
      auto v = node->cast<ValueNodePtr>();
      fg = v->value()->cast<FuncGraphPtr>();
    } else {
      fg = node->func_graph();
    }
    context_ = nullptr;
    if (context != nullptr) {
      context_ = context->Filter(fg);
    }
  }

  ~AnfNodeConfig() override = default;
  MS_DECLARE_PARENT(AnfNodeConfig, Config);

  EvalResultPtr ObtainEvalResult() override;

  AnalysisContextPtr context() const { return context_; }

  AnfNodePtr node() const { return node_; }

  AnalysisEnginePtr engine() const { return engine_.lock(); }

  // used by unordered_map;
  bool operator==(const AnfNodeConfig &other) const {
    // compare node with pointer, context with pointer except DummyContext as it's created by make_shared;
    // context should not be nullptr;
    if (context_->IsDummyContext() && other.context_->IsDummyContext()) {
      return true;
    }
    return (node_ == other.node_) && (context_ == other.context_);
  }

  std::string ToString() const override {
    std::ostringstream buffer;
    buffer << "Node: " << node_->DebugString() << "-uid(" << node_->UniqueId()
           << "), Context: " << context_->ToString();
    return buffer.str();
  }

 private:
  // AnalysisEngine is global.
  // As AnfNodeConfig is cached in AnalysisEngine.AnalysisCache, use
  // weak_ptr to break Config cycle.
  std::weak_ptr<AnalysisEngine> engine_;
  AnfNodePtr node_;
  AnalysisContextPtr context_;
};

using AnfNodeConfigPtr = std::shared_ptr<AnfNodeConfig>;

struct AnfNodeConfigHasher {
  std::size_t operator()(const AnfNodeConfigPtr conf) const;
};

struct AnfNodeConfigEqual {
  bool operator()(const AnfNodeConfigPtr lhs, const AnfNodeConfigPtr rhs) const;
};

class VirtualConfig : public Config {
 public:
  explicit VirtualConfig(const AbstractBasePtr &abstract) : Config(), abstract_(abstract) {}

  ~VirtualConfig() override = default;
  MS_DECLARE_PARENT(VirtualConfig, Config);
  EvalResultPtr ObtainEvalResult() override {
    return std::make_shared<EvalResult>(abstract_, std::make_shared<AttrValueMap>());
  }

 private:
  AbstractBasePtr abstract_;
};

// AnalysisCache
class AnalysisCache {
 public:
  AnalysisCache() = default;
  ~AnalysisCache() = default;
  void Clear() { analysis_cache_map_.clear(); }
  void set_value(const AnfNodeConfigPtr &conf, const EvalResultPtr &arg);
  EvalResultPtr GetValue(const AnfNodeConfigPtr &conf);

 private:
  std::unordered_map<AnfNodeConfigPtr, EvalResultPtr, AnfNodeConfigHasher, AnfNodeConfigEqual> analysis_cache_map_;
};

using PrimEvaluatorMap = std::unordered_map<PrimitivePtr, EvaluatorPtr, PrimitiveHasher, PrimitiveEqual>;
using AnfNodeConfigMap =
  std::unordered_map<AnfNodeConfigPtr, AnfNodeConfigPtr, AnfNodeConfigHasher, AnfNodeConfigEqual>;

struct AnalysisResult {
  EvalResultPtr inferred;
  AnalysisContextPtr context;
};

struct PartialAppHasher {
  std::size_t operator()(const std::pair<AbstractFunctionPtr, AbstractBasePtrList> &p) const {
    auto h1 = std::hash<AbstractFunctionPtr>{}(p.first);
    auto h2 = AbstractBasePtrListHash(p.second);
    return h1 ^ h2;
  }
};
class AnalysisEngine : public std::enable_shared_from_this<AnalysisEngine> {
 public:
  AnalysisEngine(const PrimEvaluatorMap &prim_evaluator_map, const FuncGraphManagerPtr &func_graph_manager)
      : analysis_cache_(AnalysisCache()),
        prim_constructors_(prim_evaluator_map),
        func_graph_manager_(func_graph_manager) {
    function_call_depth_ = 0;
    forward_count_ = 0;
  }
  ~AnalysisEngine() = default;

  // func_graph: The func_graph to analyze.
  // args_spec_list: The abstracted arguments for the func_graph. Must be a tuple of AbstractBase.
  AnalysisResult Run(const FuncGraphPtr &func_graph, const AbstractBasePtrList &args_spec_list);
  EvalResultPtr ObtainEvalResultWithCache(const AnfNodeConfigPtr &conf);
  // Return the Evaluator for the given function.
  EvaluatorPtr GetEvaluatorFor(const AbstractFunctionPtr &fn);

  AbstractBasePtr EvalValueNode(const ValueNodePtr &value_node, const AnfNodeConfigPtr &conf);
  EvalResultPtr EvalCNode(const CNodePtr &cnode, const AnfNodeConfigPtr &conf);
  // Infer the result of fn(args).
  EvalResultPtr Execute(const AbstractFunctionPtr &fn, const AbstractBasePtrList &args_spec_list);
  void Clear();
  void ClearEvaluatorCache();
  AnalysisCache &analysis_cache() { return analysis_cache_; }
  AnfNodeConfigPtr MakeConfig(const AnfNodePtr &node, const AnalysisContextPtr &context) {
    return std::make_shared<AnfNodeConfig>(shared_from_this(), node, context);
  }
  // Overloaded function.
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<PrimitiveAbstractClosure> &fn);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<PartialAbstractClosure> &fn);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<FuncGraphAbstractClosure> &fn);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<MetaFuncGraphAbstractClosure> &fn);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<VirtualAbstractClosure> &fn);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<TypedPrimitiveAbstractClosure> &);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<JTransformedAbstractClosure> &fn);

  FuncGraphManagerPtr func_graph_manager() { return func_graph_manager_; }
  const AnfNodeConfigMap &anfnode_config_map() const { return anfnode_config_map_; }

  // Set the analysis result for orig to the result for new.
  // This sets an entry in anfnode_config_map from orig to new.
  EvalResultPtr ForwardConfig(const AnfNodeConfigPtr &orig_conf, const AnfNodeConfigPtr new_conf);
  const PrimEvaluatorMap &PrimConstructors() const { return prim_constructors_; }

  AnalysisCache analysis_cache_;
  std::unordered_map<PrimitivePyPtr, EvaluatorPtr> prim_py_evaluators_;

  void ResetFunctionCallDepth() { function_call_depth_ = 0; }

  void IncreaseFunctionCallDepth() { function_call_depth_++; }

  void DecreaseFunctionCallDepth() {
    if (function_call_depth_ == 0) {
      MS_LOG(EXCEPTION) << "Current function call depth is already 0, can not decrease it.";
    }
    function_call_depth_--;
  }

  uint64_t function_call_depth() { return function_call_depth_; }

  void CheckNoStackInSameFuncGraph(const AnfNodeConfigPtr &conf);

 private:
  // Should compare Args based on value other than pointer;
  struct EvaluatorArgs {
    EvaluatorArgs(const EvaluatorPtr &eval, const AbstractBasePtrList &args) : evaluator_(eval), args_(args) {}
    bool operator==(const EvaluatorArgs &other) const {
      if (evaluator_ != other.evaluator_) {
        return false;
      }
      if (AbstractBasePtrListDeepEqual(args_, other.args_)) {
        return true;
      }
      return false;
    }
    bool operator!=(const EvaluatorArgs &other) { return !(*this == other); }

    EvaluatorPtr evaluator_;
    AbstractBasePtrList args_;
  };
  using EvalTraceRevIter = std::list<EvaluatorArgs>::reverse_iterator;
  struct EvaluatorArgsHasher {
    std::size_t operator()(const EvaluatorArgs &eval_args) const {
      return hash_combine(std::hash<EvaluatorPtr>{}(eval_args.evaluator_), AbstractBasePtrListHash(eval_args.args_));
    }
  };
  struct EvaluatorArgsEqual {
    bool operator()(const EvaluatorArgs &lhs, const EvaluatorArgs &rhs) const { return lhs == rhs; }
  };

  void SetUndeterminedFlag(const EvaluatorPtr &evaluator);
  EvaluatorPtr HandleNestedRecursion(const std::vector<EvaluatorPtr> &evaluators, const EvaluatorPtr &eval,
                                     const AbstractBasePtrList &args_spec_list, const EvalTraceRevIter &it,
                                     bool *continue_flag);
  EvalResultPtr ProcessEvalResults(const AbstractBasePtrList &out_specs);

  const PrimEvaluatorMap &prim_constructors_;
  FuncGraphManagerPtr func_graph_manager_;
  std::unordered_map<AbstractFunctionPtr, EvaluatorPtr, AbstractFunctionHasher, AbstractFunctionEqual> constructors_;
  std::unordered_map<std::pair<AbstractFunctionPtr, AbstractBasePtrList>, EvaluatorPtr, PartialAppHasher>
    constructors_app_;
  AnfNodeConfigMap anfnode_config_map_;
  // Use a list to trace multiple evaluators.
  std::list<EvaluatorArgs> eval_trace_;
  std::map<EvaluatorPtr, EvaluatorPtr> multi_poss_;
  std::unordered_set<EvaluatorArgs, EvaluatorArgsHasher, EvaluatorArgsEqual> continued_evals_;

  AnalysisContextPtr Run(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                         const ConfigPtrList &args_conf_list);
  EvalResultPtr Eval(const AnfNodeConfigPtr &conf);
  EvaluatorPtr _GetEvaluatorFor(const AbstractFunctionPtr &fn);
  EvalResultPtr ExecuteEvaluators(const std::vector<EvaluatorPtr> &evaluators, const AnfNodeConfigPtr &out_conf,
                                  const ConfigPtrList &args_conf_list);
  EvalResultPtr ExecuteMultipleEvaluators(const std::vector<EvaluatorPtr> &evaluators, const AnfNodeConfigPtr &out_conf,
                                          const ConfigPtrList &args_conf_list);
  // record current depth of function call statck
  uint64_t function_call_depth_;

  uint64_t forward_count_;

#ifdef DEBUG
  std::vector<AnfNodePtr> compute_conf_stack_;
#endif
};

// Translate the value to an abstract value.
// Arguments:
// value:   The value to convert.
// context: The context in which the value was found, used if the value is a Graph.
// conf:     The Config to the valuenode we are converting, if there is one,
// so that we can generate a tracking_id.
AbstractBasePtr ToAbstract(const ValuePtr &value, const AnalysisContextPtr &context = nullptr,
                           const AnfNodeConfigPtr &conf = nullptr);

// Convert a value to an abstract value.
// Arguments:
// v:       The value to convert.
// broaden: If True, concrete values will be made more abstract, so e.g.
// the value 1234 would become ANYTHING.
AbstractBasePtr FromValueInside(const ValuePtr &value, bool broaden = false);

template <typename T>
AbstractBasePtr FromValue(const T &value, bool broaden = false) {
  return FromValueInside(MakeValue(value), broaden);
}

EvalResultPtr EvalOnePrim(const PrimitivePtr &p, const AbstractBasePtrList &arg_specs);
}  // namespace abstract
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_STATIC_ANALYSIS_H_
