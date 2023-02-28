/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_STATIC_ANALYSIS_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_STATIC_ANALYSIS_H_

#include <list>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include "utils/ms_utils.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "utils/log_adapter.h"
#include "ir/anf.h"
#include "pybind_api/ir/primitive_py.h"
#include "abstract/abstract_value.h"
#include "abstract/analysis_context.h"
#include "abstract/abstract_function.h"
#include "pipeline/jit/parse/parse.h"

namespace mindspore {
namespace abstract {
void ResetFunctionCallDepth();
void IncreaseFunctionCallDepth();
void DecreaseFunctionCallDepth();
size_t FunctionCallDepth();

void ResetStackFrameDepth();
void IncreaseStackFrameDepth();
void DecreaseStackFrameDepth();
size_t StackFrameDepth();

// define attribute value map
using AttrValueMap = mindspore::HashMap<std::string, ValuePtr>;
using AttrValueMapPtr = std::shared_ptr<AttrValueMap>;

// the class to save evaluated result: abstract value and modified attribute
class EvalResult : public Base {
 public:
  EvalResult(const AbstractBasePtr &abs, const AttrValueMapPtr &attr) : abstract_(abs), attribute_(attr) {}
  ~EvalResult() override = default;
  MS_DECLARE_PARENT(EvalResult, Base);
  const AbstractBasePtr &abstract() const { return abstract_; }
  const AttrValueMapPtr &attribute() const { return attribute_; }

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
class AnfNodeConfig final : public Config {
 public:
  AnfNodeConfig(const AnalysisEnginePtr &engine, const AnfNodePtr &node, const AnalysisContextPtr &context,
                const FuncGraphPtr &func_graph)
      : Config(),
        engine_(std::weak_ptr<AnalysisEngine>(engine)),
        node_(node),
        context_(nullptr),
        func_graph_(func_graph) {
    if (context == nullptr) {
      return;
    }
    auto fg = GetValuePtr<FuncGraph>(node);
    if (fg == nullptr && node != nullptr) {
      fg = node->func_graph().get();
    }
    if (context->func_graph().get() == fg) {
      // Usually `node` is CNode and not a FV, or top graph's ValueNodes.
      context_ = context;
    } else {
      // If `node` is FV, FuncGraph, or other graph ValueNodes.
      // Non-FuncGraph ValueNodes will always get a DummyContext since `fg` is null.
      context_ = context->FindOwnOrParentContext(fg);
    }
  }

  ~AnfNodeConfig() override = default;
  MS_DECLARE_PARENT(AnfNodeConfig, Config);

  EvalResultPtr ObtainEvalResult() override;

  const AnalysisContextPtr &context() const { return context_; }

  const AnfNodePtr &node() const { return node_; }

  const FuncGraphPtr &func_graph() const { return func_graph_; }

  AnalysisEnginePtr engine() const { return engine_.lock(); }

  size_t hash() const override {
    std::size_t node_hash = PointerHash<AnfNodePtr>{}(node_);
    return hash_combine(node_hash, PointerHash<AnalysisContextPtr>{}(context_));
  }

  bool operator==(const AnfNodeConfig &other) const {
    if (this == &other) {
      return true;
    }
    // Compare node with pointer.
    if (node_ != other.node_) {
      return false;
    }
    // Compare context with pointer.
    return context_ == other.context_;
  }

  std::string ToString() const override {
    std::ostringstream buffer;
    constexpr int recursive_level = 2;
    buffer << "Node: " << node_ << "/" << node_->DebugString(recursive_level) << "-uid(" << node_->UniqueId()
           << "), Context: " << context_ << "/" << context_->ToString() << ", FuncGraph: " << func_graph_ << "/"
           << func_graph_->ToString();
    return buffer.str();
  }

 private:
  // AnalysisEngine is global.
  // As AnfNodeConfig is cached in AnalysisEngine.AnalysisCache, use
  // weak_ptr to break Config cycle.
  std::weak_ptr<AnalysisEngine> engine_;
  AnfNodePtr node_;
  // Which context the node would be called, usually in owner func graph context.
  AnalysisContextPtr context_;
  // Where to call the node.
  FuncGraphPtr func_graph_;
};

using AnfNodeConfigPtr = std::shared_ptr<AnfNodeConfig>;

struct AnfNodeConfigHasher {
  std::size_t operator()(const AnfNodeConfigPtr &conf) const {
    MS_EXCEPTION_IF_NULL(conf);
    return conf->hash();
  }
};

struct AnfNodeConfigEqual {
  bool operator()(const AnfNodeConfigPtr &lhs, const AnfNodeConfigPtr &rhs) const {
    if (lhs == nullptr || rhs == nullptr) {
      return false;
    }
    if (lhs == rhs) {
      return true;
    }
    return (*lhs == *rhs);
  }
};

class VirtualConfig final : public Config {
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

using PrimEvaluatorMap = mindspore::HashMap<PrimitivePtr, EvaluatorPtr, PrimitiveHasher, PrimitiveEqual>;
using AnfNodeConfigMap =
  mindspore::HashMap<AnfNodeConfigPtr, AnfNodeConfigPtr, AnfNodeConfigHasher, AnfNodeConfigEqual>;

struct AnalysisResult {
  EvalResultPtr eval_result;
  AnalysisContextPtr context;
};

struct PartialAppHasher {
  std::size_t operator()(const std::pair<AbstractFunctionPtr, AbstractBasePtrList> &p) const {
    auto hash_value = PointerHash<AbstractFunctionPtr>{}(p.first);
    for (const auto &abs : p.second) {
      hash_value = hash_combine(hash_value, PointerHash<AbstractBasePtr>{}(abs));
    }
    return hash_value;
  }
};

// Should compare Args based on value other than pointer;
struct EvaluatorArgs {
  EvaluatorArgs(const EvaluatorPtr &eval, const AbstractBasePtrList &args) : evaluator_(eval), args_(args) {}
  bool operator==(const EvaluatorArgs &other) const {
    return (this == &other) || ((evaluator_ == other.evaluator_) && AbstractBasePtrListDeepEqual(args_, other.args_));
  }
  bool operator!=(const EvaluatorArgs &other) const { return !(*this == other); }

  EvaluatorPtr evaluator_;
  AbstractBasePtrList args_;
};
using EvalTraceRevIter = std::list<EvaluatorArgs>::const_reverse_iterator;
struct EvaluatorArgsHasher {
  std::size_t operator()(const EvaluatorArgs &eval_args) const {
    return hash_combine(PointerHash<EvaluatorPtr>{}(eval_args.evaluator_), AbstractBasePtrListHash(eval_args.args_));
  }
};
struct EvaluatorArgsEqual {
  bool operator()(const EvaluatorArgs &lhs, const EvaluatorArgs &rhs) const { return lhs == rhs; }
};

struct PrimitiveEvalCacheKey {
  AttrValueMap attrs;
  AbstractBasePtrList args;
};

struct PrimitiveEvalCacheHash {
  std::size_t operator()(const PrimitiveEvalCacheKey &key) const {
    auto hash_value = key.attrs.size();
    for (const auto &attr : key.attrs) {
      hash_value = hash_combine(hash_value, std::hash<std::string>{}(attr.first));
      if (attr.second != nullptr) {
        hash_value = hash_combine(hash_value, attr.second->hash());
      }
    }
    return hash_combine(hash_value, AbstractBasePtrListHash(key.args));
  }
};

struct PrimitiveEvalCacheEqual {
  bool operator()(const PrimitiveEvalCacheKey &a, const PrimitiveEvalCacheKey &b) const {
    if (!common::IsAttrsEqual(a.attrs, b.attrs)) {
      return false;
    }
    return AbstractBasePtrListDeepEqual(a.args, b.args);
  }
};

class PrimitiveEvalCache {
 public:
  using EvalCache =
    std::unordered_map<PrimitiveEvalCacheKey, EvalResultPtr, PrimitiveEvalCacheHash, PrimitiveEvalCacheEqual>;
  using PrimToEvalCache = mindspore::HashMap<std::string, EvalCache>;
  EvalResultPtr Get(const PrimitivePtr &prim, const AbstractBasePtrList &args) const;
  void Put(const PrimitivePtr &prim, AttrValueMap &&attrs, const AbstractBasePtrList &args,
           const EvalResultPtr &result);
  void Clear();

 private:
  mutable std::mutex mutex_;
  PrimToEvalCache prim_cache_;
};

using PrimitiveEvalCachePtr = std::shared_ptr<PrimitiveEvalCache>;

class AnalysisEngine : public std::enable_shared_from_this<AnalysisEngine> {
 public:
  AnalysisEngine(const PrimEvaluatorMap &prim_evaluator_map, const FuncGraphManagerPtr &func_graph_manager)
      : prim_constructors_(prim_evaluator_map),
        func_graph_manager_(func_graph_manager),
        forward_count_(0),
        enable_recursive_eval_(common::GetEnv("MS_DEV_RECURSIVE_EVAL") == "1"),
        check_isolated_side_effect_(false) {}
  virtual ~AnalysisEngine() = default;

  // func_graph: The func_graph to analyze.
  // args_spec_list: The abstracted arguments for the func_graph. Must be a tuple of AbstractBase.
  AnalysisResult Run(const FuncGraphPtr &func_graph, const AbstractBasePtrList &args_spec_list);
  void SaveEvalResultInCache(const AnfNodeConfigPtr &conf, const EvalResultPtr &result) const;
  EvalResultPtr ObtainEvalResultWithCache(const AnfNodeConfigPtr &conf);
  // Evaluate a CNode without look up cache.
  EvalResultPtr ObtainEvalResultWithoutCache(const AnfNodeConfigPtr &conf);
  // Return the Evaluator for the given function.
  EvaluatorPtr GetEvaluatorFor(const AbstractFunctionPtr &func);

  AnfNodeConfigPtr GetForwardConfig(const AnfNodeConfigPtr &conf) const;
  EvalResultPtr InterpretedNodeCall(const CNodePtr &cnode, const AnfNodeConfigPtr &conf);
  AbstractBasePtr GetCNodeOperatorAbstract(const CNodePtr &cnode, const AnalysisContextPtr &context,
                                           const FuncGraphPtr &func_graph);
  AbstractBasePtr EvalValueNode(const ValueNodePtr &value_node, const AnfNodeConfigPtr &conf) const;
  EvalResultPtr EvalCNode(const CNodePtr &cnode, const AnfNodeConfigPtr &conf);
  // Infer the result of fn(args).
  EvalResultPtr Execute(const AbstractFunctionPtr &func, const AbstractBasePtrList &args_spec_list);
  void Clear();
  void ClearEvaluatorCache();
  AnfNodeConfigPtr MakeConfig(const AnfNodePtr &node, const AnalysisContextPtr &context,
                              const FuncGraphPtr &func_graph) {
    return std::make_shared<AnfNodeConfig>(shared_from_this(), node, context, func_graph);
  }

  FuncGraphManagerPtr func_graph_manager() { return func_graph_manager_; }
  const AnfNodeConfigMap &anfnode_config_map() const { return anfnode_config_map_; }

  // Set the analysis result for orig to the result for new.
  // This sets an entry in anfnode_config_map from orig to new.
  EvalResultPtr ForwardConfig(const AnfNodeConfigPtr &orig_conf, const AnfNodeConfigPtr new_conf,
                              bool need_erase = true);
  const PrimEvaluatorMap &PrimConstructors() const { return prim_constructors_; }

  FuncGraphPtr root_func_graph() const { return root_func_graph_; }
  AnalysisContextPtr root_context() const { return root_context_; }
  void set_root_context(const AnalysisContextPtr &context) { root_context_ = context; }

  mindspore::HashMap<PrimitivePyPtr, EvaluatorPtr> prim_py_evaluators_;

  bool enable_recursive_eval() const { return enable_recursive_eval_; }
  static EvalResultPtr ProcessEvalResults(const AbstractBasePtrList &out_specs, const AnfNodePtr &node);

  bool check_isolated_side_effect() const { return check_isolated_side_effect_; }
  void set_check_isolated_side_effect(bool check_isolated_side_effect) {
    check_isolated_side_effect_ = check_isolated_side_effect;
  }
  void SetUndeterminedFlag(const std::string &thread_id, const FuncGraph &fg);
  void SetIgnoreValueFlag(const std::string &thread_id, FuncGraph *fg);

 private:
  // Overloaded function.
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<PrimitiveAbstractClosure> &func);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<PartialAbstractClosure> &func);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<FuncGraphAbstractClosure> &func);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<MetaFuncGraphAbstractClosure> &func);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<VirtualAbstractClosure> &func);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<JTransformedAbstractClosure> &func);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<TaylorTransformedAbstractClosure> &func);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<ShardTransformedAbstractClosure> &func);
  EvaluatorPtr _GetEvaluatorFor(const std::shared_ptr<VmapTransformedAbstractClosure> &func);

  EvaluatorPtr HandleNestedRecursion(const std::vector<EvaluatorPtr> &evaluators, const EvaluatorPtr &eval,
                                     const AbstractBasePtrList &args_spec_list, const EvalTraceRevIter &it,
                                     bool *continue_flag);

  const PrimEvaluatorMap &prim_constructors_;
  FuncGraphManagerPtr func_graph_manager_;
  std::unordered_map<AbstractFunctionPtr, EvaluatorPtr, AbstractFunctionHasher, AbstractFunctionEqual> evaluators_;
  // Record the func_graph which should be set as undetermined and the setting thread id.
  mindspore::HashMap<const FuncGraph *, std::forward_list<std::string>> func_graph_undetermined_flags_;
  std::unordered_map<std::pair<AbstractFunctionPtr, AbstractBasePtrList>, EvaluatorPtr, PartialAppHasher>
    constructors_app_;

  AnfNodeConfigMap anfnode_config_map_;
  // Use a list to trace multiple evaluators.
  std::list<EvaluatorArgs> eval_trace_;
  std::map<EvaluatorPtr, EvaluatorPtr> multi_poss_;
  std::unordered_set<EvaluatorArgs, EvaluatorArgsHasher, EvaluatorArgsEqual> continued_evals_;
  // Root or top func_graph for static analysis;
  FuncGraphPtr root_func_graph_{nullptr};
  AnalysisContextPtr root_context_{nullptr};

  AnalysisContextPtr Run(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                         const ConfigPtrList &args_conf_list);
  EvalResultPtr Eval(const AnfNodeConfigPtr &conf);
  EvalResultPtr ExecuteEvaluators(const std::vector<EvaluatorPtr> &evaluators, const AnfNodeConfigPtr &out_conf,
                                  const ConfigPtrList &args_conf_list);
  EvalResultPtr ExecuteMultipleEvaluators(const std::vector<EvaluatorPtr> &evaluators, const AnfNodeConfigPtr &out_conf,
                                          const ConfigPtrList &args_conf_list);
  EvalResultPtr ExecuteMultipleEvaluatorsMultiThread(const std::vector<EvaluatorPtr> &evaluators,
                                                     const AnfNodeConfigPtr &out_conf,
                                                     const ConfigPtrList &args_conf_list);

  std::atomic_long forward_count_;

  bool enable_recursive_eval_;

  bool check_isolated_side_effect_;

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
EvaluatorPtr GetPrimEvaluator(const PrimitivePtr &prim, const AnalysisEnginePtr &engine);

EvalResultPtr ObtainEvalResultFromCache(const AnfNodeConfigPtr &conf);
// If the config of CNode(funcgraph/metafuncgraph) can be found in the cache, evaluation of the config of that CNode
// is not required, but the use flags of arguments should be synchronized as if the flags will be set when the
// evaluation is executed.
void SynchronizeSequenceElementsUseFlagsForFuncGraphArgs(const AnalysisEnginePtr &engine, const FuncGraphPtr &fg,
                                                         const CNodePtr &cnode,
                                                         const AbstractFunctionPtr &base_func_graph_func,
                                                         const AnalysisContextPtr &fg_context);
}  // namespace abstract
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_STATIC_ANALYSIS_H_
