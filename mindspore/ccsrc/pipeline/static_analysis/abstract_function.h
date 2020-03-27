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

#ifndef PIPELINE_STATIC_ANALYSIS_ABSTRACT_FUNCTION_H_
#define PIPELINE_STATIC_ANALYSIS_ABSTRACT_FUNCTION_H_

#include <memory>
#include <string>

#include "pipeline/static_analysis/abstract_value.h"
#include "pipeline/static_analysis/analysis_context.h"
#include "ir/meta_func_graph.h"

namespace mindspore {
namespace abstract {
class AbstractFuncAtom : public AbstractFunction {
 public:
  AbstractFuncAtom() = default;
  ~AbstractFuncAtom() override = default;
  MS_DECLARE_PARENT(AbstractFuncAtom, AbstractFunction)

  AbstractFunctionPtr GetUnique() override { return shared_from_base<AbstractFuncAtom>(); }
  EvaluatorPtr GetEvaluator(AnalysisEnginePtr) override {
    MS_LOG(EXCEPTION) << "Cannot GetEvaluator from AbstractFuncAtom";
  }

  AbstractFunctionPtr Join(const AbstractFunctionPtr &other) final;
  void Visit(std::function<void(const AbstractFuncAtomPtr &)>) const final;
  bool operator==(const AbstractFunction &other) const;

  std::size_t hash() const override { return tid(); }
};

class AbstractFuncUnion : public AbstractFunction {
 public:
  explicit AbstractFuncUnion(const AbstractFuncAtomPtrList &func_list);
  AbstractFuncUnion(const AbstractFunctionPtr &first, const AbstractFunctionPtr &second);
  ~AbstractFuncUnion() override = default;
  MS_DECLARE_PARENT(AbstractFuncUnion, AbstractFunction)

  std::string ToString() const override;

  AbstractFunctionPtr GetUnique() override { MS_LOG(EXCEPTION) << "Cannot get unique from AbstractFuncUnion"; }
  EvaluatorPtr GetEvaluator(AnalysisEnginePtr) override {
    MS_LOG(EXCEPTION) << "Cannot GetEvaluator from AbstractFuncUnion";
  }
  bool IsSuperSet(const AbstractFunctionPtr &other);
  AbstractFunctionPtr Join(const AbstractFunctionPtr &other) final;
  void Visit(std::function<void(const AbstractFuncAtomPtr &)>) const final;
  bool operator==(const AbstractFunction &other) const override;
  std::size_t hash() const override;
  AbstractFunctionPtr Copy() const override { MS_LOG(EXCEPTION) << "Cannot Copy from AbstractFuncUnion"; }

 private:
  AbstractFuncAtomPtrList func_list_;
};

class PrimitiveAbstractClosure : public AbstractFuncAtom {
 public:
  // Represents a Primitive.
  // prim: The primitive
  // tracking_id: Identifies different uses of the same primitive.
  explicit PrimitiveAbstractClosure(const PrimitivePtr &prim, const AnfNodePtr &tracking_id = nullptr)
      : prim_(prim), tracking_id_(AnfNodeWeakPtr(tracking_id)) {}
  ~PrimitiveAbstractClosure() override = default;
  MS_DECLARE_PARENT(PrimitiveAbstractClosure, AbstractFuncAtom)

  EvaluatorPtr GetEvaluator(AnalysisEnginePtr engine) override;

  PrimitivePtr prim() { return prim_; }

  AnfNodePtr tracking_id() const override { return tracking_id_.lock(); }

  void set_tracking_id(AnfNodePtr node) override { tracking_id_ = AnfNodeWeakPtr(node); }

  AbstractFunctionPtr Copy() const override { return std::make_shared<PrimitiveAbstractClosure>(prim_, tracking_id()); }

  bool operator==(const AbstractFunction &other) const override;
  std::size_t hash() const override;

  std::string ToString() const override { return "Prim: " + prim_->name(); }

 private:
  PrimitivePtr prim_;
  // store it as weak_ptr to break reference cycle.
  // one reference cycle example is Graph::set_output() input0 local variable.
  AnfNodeWeakPtr tracking_id_;
};

class FuncGraphAbstractClosure : public AbstractFuncAtom {
 public:
  // Represents a Graph in a certain Context.
  // context: The context, or Context.empty()
  FuncGraphAbstractClosure(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context)
      : func_graph_(func_graph), context_(context) {
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_EXCEPTION_IF_NULL(context);
  }
  ~FuncGraphAbstractClosure() override = default;
  MS_DECLARE_PARENT(FuncGraphAbstractClosure, AbstractFuncAtom)

  EvaluatorPtr GetEvaluator(AnalysisEnginePtr engine) override;

  FuncGraphPtr func_graph() { return func_graph_; }

  AnalysisContextPtr context() const override { return context_; }

  AbstractFunctionPtr Copy() const override {
    return std::make_shared<FuncGraphAbstractClosure>(func_graph_, context_);
  }

  bool operator==(const AbstractFunction &other) const override;
  std::size_t hash() const override;

  std::string ToString() const override;

 private:
  FuncGraphPtr func_graph_;
  AnalysisContextPtr context_;
};

class MetaFuncGraphAbstractClosure : public AbstractFuncAtom {
 public:
  explicit MetaFuncGraphAbstractClosure(const MetaFuncGraphPtr &meta_func_graph, const ScopePtr &scope = kDefaultScope)
      : meta_func_graph_(meta_func_graph), scope_(scope) {}
  ~MetaFuncGraphAbstractClosure() override = default;
  MS_DECLARE_PARENT(MetaFuncGraphAbstractClosure, AbstractFuncAtom)

  MetaFuncGraphPtr meta_func_graph() { return meta_func_graph_; }

  AnalysisContextPtr context() const override { return kDummyAnalysisContext; }

  EvaluatorPtr GetEvaluator(AnalysisEnginePtr engine) override;

  ScopePtr GetScope() { return scope_; }

  AbstractFunctionPtr Copy() const override { return std::make_shared<MetaFuncGraphAbstractClosure>(meta_func_graph_); }
  bool operator==(const AbstractFunction &other) const override;
  std::size_t hash() const override;

  std::string ToString() const override;

 private:
  MetaFuncGraphPtr meta_func_graph_;
  ScopePtr scope_;
};
using MetaFuncGraphAbstractClosurePtr = std::shared_ptr<MetaFuncGraphAbstractClosure>;

class PartialAbstractClosure : public AbstractFuncAtom {
 public:
  // Represents a partial application.
  // args_spec_list: The first few arguments of that function
  PartialAbstractClosure(const AbstractFuncAtomPtr &fn, const AbstractBasePtrList &args_spec_list)
      : fn_(fn), args_spec_list_(args_spec_list) {}
  ~PartialAbstractClosure() override = default;
  MS_DECLARE_PARENT(PartialAbstractClosure, AbstractFuncAtom)

  EvaluatorPtr GetEvaluator(AnalysisEnginePtr engine) override;

  AbstractFunctionPtr fn() { return fn_; }
  AbstractBasePtrList args() { return args_spec_list_; }
  AbstractFunctionPtr Copy() const override { return std::make_shared<PartialAbstractClosure>(fn_, args_spec_list_); }
  bool operator==(const AbstractFunction &other) const override;
  std::size_t hash() const override;

  std::string ToString() const override;

 private:
  AbstractFuncAtomPtr fn_;
  AbstractBasePtrList args_spec_list_;
};

class JTransformedAbstractClosure : public AbstractFuncAtom {
 public:
  // Represents a Function transformed through the application of J.
  explicit JTransformedAbstractClosure(const AbstractFuncAtomPtr &fn) : fn_(fn) {}
  ~JTransformedAbstractClosure() override = default;
  MS_DECLARE_PARENT(JTransformedAbstractClosure, AbstractFuncAtom)
  EvaluatorPtr GetEvaluator(AnalysisEnginePtr engine) override;

  AbstractFuncAtomPtr fn() { return fn_; }
  AbstractFunctionPtr Copy() const override { return std::make_shared<JTransformedAbstractClosure>(fn_); }
  bool operator==(const AbstractFunction &other) const override;
  std::size_t hash() const override;

  std::string ToString() const override { return "J(" + fn_->ToString() + ")"; }

 private:
  AbstractFuncAtomPtr fn_;
};

class VirtualAbstractClosure : public AbstractFuncAtom {
 public:
  // Represents some function with an explicitly fixed type signature.
  // args_spec_list: The arguments as abstract value given to the function
  // output: The output which is abstract value.
  VirtualAbstractClosure(const AbstractBasePtrList &args_spec_list, const AbstractBasePtr &output_spec)
      : args_spec_list_(args_spec_list), output_(output_spec) {}
  VirtualAbstractClosure(const AbstractBasePtr &args_spec, const AbstractBasePtr &output_spec)
      : args_spec_list_({args_spec}), output_(output_spec) {}
  ~VirtualAbstractClosure() override = default;
  MS_DECLARE_PARENT(VirtualAbstractClosure, AbstractFuncAtom)

  EvaluatorPtr GetEvaluator(AnalysisEnginePtr engine) override;

  AbstractBasePtrList args_spec_list() { return args_spec_list_; }

  AbstractBasePtr output() { return output_; }
  AbstractFunctionPtr Copy() const override {
    return std::make_shared<VirtualAbstractClosure>(args_spec_list_, output_);
  }
  bool operator==(const AbstractFunction &other) const override;
  std::size_t hash() const override;

  std::string ToString() const override;

 private:
  AbstractBasePtrList args_spec_list_;
  AbstractBasePtr output_;
};
using VirtualAbstractClosurePtr = std::shared_ptr<VirtualAbstractClosure>;

class TypedPrimitiveAbstractClosure : public AbstractFuncAtom {
 public:
  // Represents a Primitive with an explicitly fixed type signature.
  // args_spec_list: The arguments as abstract value given to the Primitive
  // output: The output which is abstract value.
  TypedPrimitiveAbstractClosure(const PrimitivePtr prim, const AbstractBasePtrList &args_spec_list,
                                const AbstractBasePtr &output_spec)
      : prim_(prim), args_spec_list_(args_spec_list), output_(output_spec) {}
  ~TypedPrimitiveAbstractClosure() override = default;
  MS_DECLARE_PARENT(TypedPrimitiveAbstractClosure, AbstractFuncAtom)

  EvaluatorPtr GetEvaluator(AnalysisEnginePtr engine) override;

  PrimitivePtr prim() { return prim_; }
  AbstractBasePtrList args_spec_list() { return args_spec_list_; }
  AbstractBasePtr output() { return output_; }
  AbstractFunctionPtr Copy() const override {
    return std::make_shared<TypedPrimitiveAbstractClosure>(prim_, args_spec_list_, output_);
  }
  bool operator==(const AbstractFunction &other) const override;
  std::size_t hash() const override;

  std::string ToString() const override;

 private:
  PrimitivePtr prim_;
  AbstractBasePtrList args_spec_list_;
  AbstractBasePtr output_;
};

// Represents a function that can't be called.
class DummyAbstractClosure : public AbstractFuncAtom {
 public:
  DummyAbstractClosure() = default;
  ~DummyAbstractClosure() = default;
  MS_DECLARE_PARENT(DummyAbstractClosure, AbstractFuncAtom)

  EvaluatorPtr GetEvaluator(AnalysisEnginePtr) override { MS_LOG(EXCEPTION) << "A dummy function cannot eval."; }

  AbstractFunctionPtr Copy() const override { return std::make_shared<DummyAbstractClosure>(); }
  bool operator==(const AbstractFunction &other) const override;

  std::string ToString() const override { return "DummyAbstractClosure()"; }
};

struct AbstractFunctionHasher {
  std::size_t operator()(const AbstractFunctionPtr &t) const {
    std::size_t hash = t->hash();
    return hash;
  }
};

struct AbstractFunctionEqual {
  bool operator()(const AbstractFunctionPtr &lhs, const AbstractFunctionPtr &rhs) const { return *lhs == *rhs; }
};
}  // namespace abstract
}  // namespace mindspore
#endif  // PIPELINE_STATIC_ANALYSIS_ABSTRACT_FUNCTION_H_
