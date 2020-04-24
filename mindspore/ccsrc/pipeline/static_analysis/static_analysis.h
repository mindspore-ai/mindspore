/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef PIPELINE_STATIC_ANALYSIS_STATIC_ANALYSIS_H_
#define PIPELINE_STATIC_ANALYSIS_STATIC_ANALYSIS_H_

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

#ifdef DEBUG
#include <stack>
#endif

#include "utils/log_adapter.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "pipeline/static_analysis/analysis_context.h"
#include "pipeline/static_analysis/abstract_function.h"
#include "pipeline/parse/parse.h"

namespace mindspore {
namespace abstract {
// Superclass for AnfNodeConfig and VirtualConfig.
class Config : public Base {
 public:
  Config() = default;
  ~Config() override = default;
  MS_DECLARE_PARENT(Config, Base);
  virtual AbstractBasePtr GetEvaluatedValue() = 0;
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

  AbstractBasePtr GetEvaluatedValue() override;

  AnalysisContextPtr context() const { return context_; }

  AnfNodePtr node() const { return node_; }

  AnalysisEnginePtr engine() const { return engine_.lock(); }

  // used by unordered_map;
  bool operator==(const AnfNodeConfig &other) const {
    // compare node with pointer, context with content;
    // context should not be nullptr;
    return (node_ == other.node_) && (*context_ == *other.context_);
  }

  std::string ToString() const override {
    std::ostringstream buffer;
    buffer << "Node: " << node_->DebugString() << ", Context: " << context_->ToString();
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
  AbstractBasePtr GetEvaluatedValue() override { return abstract_; }

 private:
  AbstractBasePtr abstract_;
};

// AnalysisCache
class AnalysisCache {
 public:
  AnalysisCache() = default;
  ~AnalysisCache() = default;
  void Clear() { cache_.clear(); }
  void set_value(const AnfNodeConfigPtr &conf, const AbstractBasePtr &arg);
  AbstractBasePtr GetValue(const AnfNodeConfigPtr &conf);

 private:
  std::unordered_map<AnfNodeConfigPtr, AbstractBasePtr, AnfNodeConfigHasher, AnfNodeConfigEqual> cache_;
};

using PrimEvaluatorMap = std::unordered_map<PrimitivePtr, EvaluatorPtr, PrimitiveHasher, PrimitiveEqual>;
using AnfNodeConfigMap =
  std::unordered_map<AnfNodeConfigPtr, AnfNodeConfigPtr, AnfNodeConfigHasher, AnfNodeConfigEqual>;

struct AnalysisResult {
  AbstractBasePtr inferred;
  AnalysisContextPtr context;
};

class AnalysisEngine : public std::enable_shared_from_this<AnalysisEngine> {
 public:
  AnalysisEngine(const PrimEvaluatorMap &prim_evaluator_map, const FuncGraphManagerPtr &func_graph_manager)
      : cache_(AnalysisCache()), prim_constructors_(prim_evaluator_map), func_graph_manager_(func_graph_manager) {}
  ~AnalysisEngine() = default;

  // func_graph: The func_graph to analyze.
  // args_spec_list: The abstracted arguments for the func_graph. Must be a tuple of AbstractBase.
  AnalysisResult Run(const FuncGraphPtr &func_graph, const AbstractBasePtrList &args_spec_list);
  AbstractBasePtr GetEvaluatedValue(const AnfNodeConfigPtr &conf);
  // Return the Evaluator for the given function.
  EvaluatorPtr GetEvaluatorFor(const AbstractFunctionPtr &fn);

  AbstractBasePtr EvalValueNode(const ValueNodePtr &value_node, const AnfNodeConfigPtr &conf);
  AbstractBasePtr InferCNode(const CNodePtr &cnode, const AnfNodeConfigPtr &conf);
  // Infer the result of fn(args).
  AbstractBasePtr Execute(const AbstractFunctionPtr &fn, const AbstractBasePtrList &args_spec_list);
  void Clear();
  void ClearEvaluatorCache();
  AnalysisCache &cache() { return cache_; }
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
  AbstractBasePtr ForwardConfig(const AnfNodeConfigPtr &orig_conf, const AnfNodeConfigPtr new_conf) {
    // Use anfnode_config_map_[orig_conf] = new_conf will require AnfNodeConfig provide copy constructor.
    (void)anfnode_config_map_.emplace(orig_conf, new_conf);
    MS_LOG(DEBUG) << "Forward orig_conf: " << orig_conf->node()->DebugString()
                  << ", to new_conf: " << new_conf->node()->DebugString();
    return GetEvaluatedValue(new_conf);
  }
  const PrimEvaluatorMap &PrimConstructors() const { return prim_constructors_; }

  AnalysisCache cache_;
  std::unordered_map<PrimitivePyPtr, EvaluatorPtr> prim_py_evaluators_;

 private:
  const PrimEvaluatorMap &prim_constructors_;
  FuncGraphManagerPtr func_graph_manager_;
  std::unordered_map<AbstractFunctionPtr, EvaluatorPtr> constructors_;
  AnfNodeConfigMap anfnode_config_map_;
  // Use a list to trace multiple evaluators.
  std::list<std::pair<EvaluatorPtr, AbstractBasePtrList>> eval_trace_;

  AnalysisContextPtr Run(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                         const ConfigPtrList &args_conf_list);
  AbstractBasePtr Eval(const AnfNodeConfigPtr &conf);
  EvaluatorPtr _GetEvaluatorFor(const AbstractFunctionPtr &fn);
  AbstractBasePtr ExecuteEvaluators(const std::vector<EvaluatorPtr> &evaluators, const AnfNodeConfigPtr &out_conf,
                                    const ConfigPtrList &args_conf_list);
  AbstractBasePtr ExecuteMultipleEvaluators(const std::vector<EvaluatorPtr> &evaluators,
                                            const AnfNodeConfigPtr &out_conf, const ConfigPtrList &args_conf_list);

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

AbstractBasePtr InferOnePrim(const PrimitivePtr &p, const AbstractBasePtrList &arg_specs);
}  // namespace abstract
}  // namespace mindspore

#endif  // PIPELINE_STATIC_ANALYSIS_STATIC_ANALYSIS_H_
