/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_ABSTRACT_ABSTRACT_FUNCTION_H_
#define MINDSPORE_CORE_ABSTRACT_ABSTRACT_FUNCTION_H_

#include <cstdint>
#include <memory>
#include <string>

#include "abstract/abstract_value.h"
#include "abstract/analysis_context.h"
#include "ir/meta_func_graph.h"

namespace mindspore {
namespace abstract {
/// \brief AbstractFuncAtom defines interface for abstract of atom function.
class MS_CORE_API AbstractFuncAtom : public AbstractFunction {
 public:
  /// \brief Constructor of AbstractFuncAtom.
  AbstractFuncAtom() = default;

  /// \brief Destructor of AbstractFuncAtom.
  ~AbstractFuncAtom() override = default;
  MS_DECLARE_PARENT(AbstractFuncAtom, AbstractFunction)

  AbstractFunctionPtr GetUnique() override { return shared_from_base<AbstractFuncAtom>(); }

  AbstractFunctionPtr Join(const AbstractFunctionPtr &other) final;

  void Visit(std::function<void(const AbstractFuncAtomPtr &)> visit_func) const final;

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override { return tid(); }
};

/// \brief AbstractFuncUnion defines interface for abstract of union function.
class MS_CORE_API AbstractFuncUnion final : public AbstractFunction {
 public:
  /// \brief Constructor AbstractFuncUnion from AbstractFuncAtom list.
  ///
  /// \param[in] func_list The AbstractFuncAtom list for AbstractFuncUnion.
  explicit AbstractFuncUnion(const AbstractFuncAtomPtrList &func_list);

  /// \brief Constructor AbstractFuncUnion from two AbstractFunction.
  ///
  /// \param[in] first The first AbstractFunction for AbstractFuncUnion.
  /// \param[in] second The second AbstractFunction for AbstractFuncUnion.
  AbstractFuncUnion(const AbstractFunctionPtr &first, const AbstractFunctionPtr &second);

  /// \brief Destructor for AbstractFunction.
  ~AbstractFuncUnion() override = default;
  MS_DECLARE_PARENT(AbstractFuncUnion, AbstractFunction)

  std::string ToString() const override;

  std::string ToString(bool verbose) const override;

  AbstractFunctionPtr GetUnique() override { MS_LOG(INTERNAL_EXCEPTION) << "Cannot get unique from AbstractFuncUnion"; }

  /// \brief Check whether the input AbstractFunction is in AbstractFuncUnion.
  ///
  /// \param[in] other The input AbstractFunction for check.
  ///
  /// \return Return true if other is in AbstractFuncUnion, otherwise return False.
  bool IsSuperSet(const AbstractFunctionPtr &other);

  AbstractFunctionPtr Join(const AbstractFunctionPtr &other) final;

  void Visit(std::function<void(const AbstractFuncAtomPtr &)> visit_func) const final;

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override;

  AbstractFunctionPtr Copy() const override { MS_LOG(INTERNAL_EXCEPTION) << "Cannot Copy from AbstractFuncUnion"; }

  const AbstractFuncAtomPtrList &func_list() const { return func_list_; }

 private:
  AbstractFuncAtomPtrList func_list_;
};

/// \brief PrimitiveAbstractClosure defines interface for abstract of Primitive.
class MS_CORE_API PrimitiveAbstractClosure final : public AbstractFuncAtom {
 public:
  /// \brief Constructor of PrimitiveAbstractClosure
  ///
  /// \param[in] prim The primitive that this PrimitiveAbstractClosure corresponding to.
  /// \param[in] tracking_node A Node identifies different uses of the prim.
  explicit PrimitiveAbstractClosure(const PrimitivePtr &prim, const AnfNodePtr &tracking_node = nullptr)
      : PrimitiveAbstractClosure(prim, ToTrackingId(tracking_node)) {}

  // For internal usage only, make it public so that make_shared can work on it.
  PrimitiveAbstractClosure(const PrimitivePtr &prim, std::uintptr_t tracking_id)
      : prim_(prim), tracking_id_(tracking_id) {}

  /// \brief Destructor of PrimitiveAbstractClosure
  ~PrimitiveAbstractClosure() override = default;
  MS_DECLARE_PARENT(PrimitiveAbstractClosure, AbstractFuncAtom)

  /// \brief Get the Primitive that this PrimitiveAbstractClosure corresponding to.
  ///
  /// \return The Primitive that this PrimitiveAbstractClosure corresponding to.
  const PrimitivePtr &prim() const { return prim_; }

  std::uintptr_t tracking_id() const override { return tracking_id_; }

  AbstractFunctionPtr Copy() const override { return std::make_shared<PrimitiveAbstractClosure>(prim_, tracking_id_); }

  AbstractFunctionPtr CopyWithoutTrackingId() const override {
    return std::make_shared<PrimitiveAbstractClosure>(prim_, 0);
  }

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override;

  std::string ToString() const override { return "PrimitiveAbstractClosure: " + prim_->name(); }

  std::string ToString(bool verbose) const override;

  ValuePtr RealBuildValue() const override { return prim_; }

 private:
  PrimitivePtr prim_;
  // To discriminate different usage of same primitive calls,
  // store it as the memory address of the user node.
  std::uintptr_t tracking_id_;
};
using PrimitiveAbstractClosurePtr = std::shared_ptr<PrimitiveAbstractClosure>;

/// \brief FuncGraphAbstractClosure defines interface for abstract of FuncGraph.
class MS_CORE_API FuncGraphAbstractClosure final : public AbstractFuncAtom {
 public:
  /// \brief Constructor of FuncGraphAbstractClosure.
  ///
  /// \param[in] func_graph The function graph that this PrimitiveAbstractClosure corresponding to.
  /// \param[in] context The context that func_graph corresponding to.
  /// \param[in] tracking_node A Node identifies different uses of the func_graph.
  FuncGraphAbstractClosure(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                           const AnfNodePtr &tracking_node = nullptr, bool specialized = false)
      : FuncGraphAbstractClosure(func_graph, context, ToTrackingId(tracking_node), specialized) {}

  // For internal usage only, make it public so that make_shared can work on it.
  FuncGraphAbstractClosure(const FuncGraphPtr &func_graph, const AnalysisContextPtr &context,
                           std::uintptr_t tracking_id, bool specialized)
      : func_graph_(FuncGraphWeakPtr(func_graph)),
        context_(context),
        tracking_id_(tracking_id),
        specialized_(specialized) {
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_EXCEPTION_IF_NULL(context);
  }

  /// \brief Destructor of FuncGraphAbstractClosure.
  ~FuncGraphAbstractClosure() override = default;
  MS_DECLARE_PARENT(FuncGraphAbstractClosure, AbstractFuncAtom)

  /// \brief Get the FuncGraph that this FuncGraphAbstractClosure corresponding to.
  ///
  /// \return The FuncGraph that this FuncGraphAbstractClosure corresponding to.
  FuncGraphPtr func_graph() const { return func_graph_.lock(); }

  AnalysisContextPtr context() const override { return context_; }

  std::uintptr_t tracking_id() const override { return tracking_id_; }

  bool specialized() const { return specialized_; }

  AbstractFunctionPtr Copy() const override {
    return std::make_shared<FuncGraphAbstractClosure>(func_graph(), context_, tracking_id_, specialized_);
  }

  AbstractFunctionPtr CopyWithoutTrackingId() const override {
    return std::make_shared<FuncGraphAbstractClosure>(func_graph(), context_, 0, specialized_);
  }

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override;

  std::string ToString() const override;

  std::string ToString(bool verbose) const override;

  bool IsEqualExceptTrackingId(const FuncGraphAbstractClosure &other) const;

  std::size_t HashWithoutTrackingId() const;

 private:
  FuncGraphWeakPtr func_graph_;
  AnalysisContextPtr context_;
  // To discriminate different usage of same graph by using this tracking_id,
  // so different tracking_id will produce different FuncGraphAbstractClosure,
  // different FuncGraphEvaluator.
  // Especially useful for recursive func graph call, so it will not mess up
  // the `context_` in FuncGraphEvaluator.
  // store it as the memory address of the user node.
  std::uintptr_t tracking_id_;
  // If the func_graph_ member is the specialized func_graph_ in current IR or
  // it's a old func_graph of IR before renormalized.
  bool specialized_{false};
};
using FuncGraphAbstractClosurePtr = std::shared_ptr<FuncGraphAbstractClosure>;

/// \brief MetaFuncGraphAbstractClosure defines interface for abstract of MetaFuncGraph.
class MS_CORE_API MetaFuncGraphAbstractClosure final : public AbstractFuncAtom {
 public:
  /// \brief Constructor of FuncGraphAbstractClosure.
  ///
  /// \param[in] meta_func_graph The function graph that this MetaFuncGraphAbstractClosure corresponding to.
  /// \param[in] tracking_node A Node identifies different uses of the meta_func_graph.
  /// \param[in] scope The scope to which the tracking_id belong to.
  explicit MetaFuncGraphAbstractClosure(const MetaFuncGraphPtr &meta_func_graph,
                                        const AnfNodePtr &tracking_node = nullptr,
                                        const ScopePtr &scope = kDefaultScope)
      : MetaFuncGraphAbstractClosure(meta_func_graph, ToTrackingId(tracking_node), scope) {}

  // For internal usage only, make it public so that make_shared can work on it.
  MetaFuncGraphAbstractClosure(const MetaFuncGraphPtr &meta_func_graph, std::uintptr_t tracking_id,
                               const ScopePtr &scope)
      : meta_func_graph_(meta_func_graph), tracking_id_(tracking_id), scope_(scope) {}

  /// \brief Destructor of MetaFuncGraphAbstractClosure.
  ~MetaFuncGraphAbstractClosure() override = default;
  MS_DECLARE_PARENT(MetaFuncGraphAbstractClosure, AbstractFuncAtom)

  /// \brief Get the MetaFuncGraph that this MetaFuncGraphAbstractClosure corresponding to.
  ///
  /// \return The MetaFuncGraph that this MetaFuncGraphAbstractClosure corresponding to.
  const MetaFuncGraphPtr &meta_func_graph() const { return meta_func_graph_; }

  AnalysisContextPtr context() const override { return AnalysisContext::DummyContext(); }

  /// \brief Get the Scope that this MetaFuncGraphAbstractClosure corresponding to.
  ///
  /// \return The Scope that this MetaFuncGraphAbstractClosure corresponding to.
  const ScopePtr &GetScope() const { return scope_; }

  std::uintptr_t tracking_id() const override { return tracking_id_; }

  AbstractFunctionPtr Copy() const override {
    return std::make_shared<MetaFuncGraphAbstractClosure>(meta_func_graph_, tracking_id_, kDefaultScope);
  }

  AbstractFunctionPtr CopyWithoutTrackingId() const override {
    return std::make_shared<MetaFuncGraphAbstractClosure>(meta_func_graph_, 0, kDefaultScope);
  }

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override;

  std::string ToString() const override;

 private:
  MetaFuncGraphPtr meta_func_graph_;
  // Refer the comment in FuncGraphAbstractClosure;
  // Store it as memory address of the user node.
  std::uintptr_t tracking_id_;
  ScopePtr scope_;
};
using MetaFuncGraphAbstractClosurePtr = std::shared_ptr<MetaFuncGraphAbstractClosure>;

/// \brief PartialAbstractClosure defines the abstract AbstractFuncAtom interface provided by some args in advance.
class MS_CORE_API PartialAbstractClosure final : public AbstractFuncAtom {
 public:
  /// \brief Constructor of PartialAbstractClosure.
  ///
  /// \param[in] fn The AbstractFuncAtom this PartialAbstractClosure corresponding to.
  /// \param[in] args_abs_list The first few parameters provided for fn in advance.
  /// \param[in] node The CNode which this PartialAbstractClosure evaluated from.
  PartialAbstractClosure(const AbstractFuncAtomPtr &fn, const AbstractBasePtrList &args_abs_list,
                         const AnfNodePtr &node = nullptr)
      : fn_(fn), args_abs_list_(args_abs_list), node_(AnfNodePtr(node)) {}

  /// \brief Destructor of PartialAbstractClosure.
  ~PartialAbstractClosure() override = default;
  MS_DECLARE_PARENT(PartialAbstractClosure, AbstractFuncAtom)

  /// \brief Get the AbstractFuncAtom that this PartialAbstractClosure corresponding to.
  ///
  /// \return The AbstractFuncAtom that this PartialAbstractClosure corresponding to.
  AbstractFuncAtomPtr fn() { return fn_; }

  /// \brief Set the AbstractFuncAtom that this PartialAbstractClosure corresponding to.
  ///
  /// \param[in] fn The AbstractFuncAtom that this PartialAbstractClosure corresponding to.
  void set_fn(const AbstractFuncAtomPtr &fn) { fn_ = fn; }

  /// \brief Get the pre-provided arguments.
  ///
  /// \return The pre-provided arguments.
  const AbstractBasePtrList &args() const { return args_abs_list_; }

  /// \brief Get the CNode this PartialAbstractClosure evaluated from.
  ///
  /// \return The CNode this PartialAbstractClosure evaluated from.
  AnfNodePtr node() const { return node_.lock(); }

  /// \brief Set the CNode this PartialAbstractClosure evaluated from.
  ///
  /// \param[in] node The CNode this PartialAbstractClosure evaluated from.
  void set_node(const AnfNodePtr &node) { node_ = AnfNodeWeakPtr(node); }

  /// \brief Get whether the args need to be appended to the end.
  ///
  /// \return Whether the args need to be appended to the end.
  bool need_append_to_end() const { return need_append_to_end_; }

  /// \brief Set whether the args need to be appended to the end.
  ///
  /// \param[in] flag Whether the args need to be appended to the end.
  void set_need_append_to_end(bool flag) { need_append_to_end_ = flag; }

  AbstractFunctionPtr Copy() const override {
    auto abs = std::make_shared<PartialAbstractClosure>(fn_, args_abs_list_, node_.lock());
    abs->set_need_append_to_end(need_append_to_end_);
    return abs;
  }

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override;

  std::string ToString() const override;

  std::string ToString(bool verbose) const override;

 protected:
  ValuePtr RealBuildValue() const override { return fn_->BuildValue(); }

 private:
  AbstractFuncAtomPtr fn_;
  AbstractBasePtrList args_abs_list_;
  // The ANFNode which this PartialAbstractClosure evaluated from.
  AnfNodeWeakPtr node_;
  bool need_append_to_end_{false};
};
using PartialAbstractClosurePtr = std::shared_ptr<PartialAbstractClosure>;

/// \brief JTransformedAbstractClosure defines interface for abstract of Function
/// transformed through the application of J.
class MS_CORE_API JTransformedAbstractClosure final : public AbstractFuncAtom {
 public:
  /// \brief Constructor of JTransformedAbstractClosure
  ///
  /// \param[in] fn The AbstractFuncAtom transformed through the application of J.
  explicit JTransformedAbstractClosure(const AbstractFuncAtomPtr &fn) : fn_(fn) {}

  /// \brief Destructor of JTransformedAbstractClosure
  ~JTransformedAbstractClosure() override = default;
  MS_DECLARE_PARENT(JTransformedAbstractClosure, AbstractFuncAtom)

  /// \brief Get the AbstractFuncAtom JTransformedAbstractClosure corresponding to.
  ///
  /// \return The AbstractFuncAtom JTransformedAbstractClosure corresponding to.
  const AbstractFuncAtomPtr &fn() const { return fn_; }

  AbstractFunctionPtr Copy() const override { return std::make_shared<JTransformedAbstractClosure>(fn_); }

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override;

  std::string ToString() const override { return "J(" + fn_->ToString() + ")"; }

 private:
  AbstractFuncAtomPtr fn_;
};

/// \brief TaylorTransformedAbstractClosure defines interface for abstract of Function
/// transformed through the application of Taylor.
class MS_CORE_API TaylorTransformedAbstractClosure final : public AbstractFuncAtom {
 public:
  /// \brief Constructor of TaylorTransformedAbstractClosure
  ///
  /// \param[in] fn The AbstractFuncAtom transformed through the application of Taylor.
  explicit TaylorTransformedAbstractClosure(const AbstractFuncAtomPtr &fn) : fn_(fn) {}

  /// \brief Destructor of TaylorTransformedAbstractClosure
  ~TaylorTransformedAbstractClosure() override = default;
  MS_DECLARE_PARENT(TaylorTransformedAbstractClosure, AbstractFuncAtom)

  /// \brief Get the AbstractFuncAtom TaylorTransformedAbstractClosure corresponding to.
  ///
  /// \return The AbstractFuncAtom TaylorTransformedAbstractClosure corresponding to.
  const AbstractFuncAtomPtr &fn() const { return fn_; }

  AbstractFunctionPtr Copy() const override { return std::make_shared<TaylorTransformedAbstractClosure>(fn_); }

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override;

  std::string ToString() const override { return "Taylor(" + fn_->ToString() + ")"; }

 private:
  AbstractFuncAtomPtr fn_;
};

/// \brief ShardTransformedAbstractClosure defines interface for abstract of Function
/// transformed through the application of Shard.
class MS_CORE_API ShardTransformedAbstractClosure final : public AbstractFuncAtom {
 public:
  /// \brief Constructor of ShardTransformedAbstractClosure
  ///
  /// \param[in] fn The AbstractFuncAtom transformed through the application of Shard.
  explicit ShardTransformedAbstractClosure(const AbstractFuncAtomPtr &fn) : fn_(fn) {}

  /// \brief Destructor of ShardTransformedAbstractClosure
  ~ShardTransformedAbstractClosure() override = default;
  MS_DECLARE_PARENT(ShardTransformedAbstractClosure, AbstractFuncAtom)

  /// \brief Get the AbstractFuncAtom ShardTransformedAbstractClosure corresponding to.
  ///
  /// \return The AbstractFuncAtom ShardTransformedAbstractClosure corresponding to.
  const AbstractFuncAtomPtr &fn() const { return fn_; }

  AbstractFunctionPtr Copy() const override { return std::make_shared<ShardTransformedAbstractClosure>(fn_); }

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override;

  std::string ToString() const override { return "Shard(" + fn_->ToString() + ")"; }

 private:
  AbstractFuncAtomPtr fn_;
};

/// \brief VmapTransformedAbstractClosure defines interface for abstract of Function
/// transformed through the application of Vmap.
class MS_CORE_API VmapTransformedAbstractClosure final : public AbstractFuncAtom {
 public:
  /// \brief Constructor of VmapTransformedAbstractClosure
  ///
  /// \param[in] fn The AbstractFuncAtom transformed through the application of Vmap.
  explicit VmapTransformedAbstractClosure(const AbstractFuncAtomPtr &fn, const ValuePtr &in_axes,
                                          const ValuePtr &out_axes, size_t cell_size)
      : fn_(fn), in_axes_(in_axes), out_axes_(out_axes), cell_size_(cell_size) {}

  /// \brief Destructor of VmapTransformedAbstractClosure
  ~VmapTransformedAbstractClosure() override = default;
  MS_DECLARE_PARENT(VmapTransformedAbstractClosure, AbstractFuncAtom)

  /// \brief Get the AbstractFuncAtom VmapTransformedAbstractClosure corresponding to.
  ///
  /// \return The AbstractFuncAtom VmapTransformedAbstractClosure corresponding to.
  const AbstractFuncAtomPtr &fn() const { return fn_; }

  const ValuePtr &in_axes() const { return in_axes_; }

  const ValuePtr &out_axes() const { return out_axes_; }

  size_t cell_size() const { return cell_size_; }

  AbstractFunctionPtr Copy() const override {
    return std::make_shared<VmapTransformedAbstractClosure>(fn_, in_axes_, out_axes_, cell_size_);
  }

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override;

  std::string ToString() const override { return "Vmap(" + fn_->ToString() + ")"; }

 private:
  AbstractFuncAtomPtr fn_;
  ValuePtr in_axes_;
  ValuePtr out_axes_;
  size_t cell_size_;
};

/// \brief VirtualAbstractClosure defines interface for function with an explicitly
/// fixed type signature.
class MS_CORE_API VirtualAbstractClosure final : public AbstractFuncAtom {
 public:
  /// \brief Constructor of VirtualAbstractClosure.
  ///
  /// \param[in] args_abs_list The abstract values of the arguments to the function.
  /// \param[in] output_spec The abstract value of output.
  VirtualAbstractClosure(const AbstractBasePtrList &args_abs_list, const AbstractBasePtr &output_spec)
      : args_abs_list_(args_abs_list), output_(output_spec) {}

  /// \brief Constructor of VirtualAbstractClosure.
  ///
  /// \param[in] args_abs The abstract value of argument to the function.
  /// \param[in] output_spec The abstract value of output.
  VirtualAbstractClosure(const AbstractBasePtr &args_abs, const AbstractBasePtr &output_spec)
      : args_abs_list_({args_abs}), output_(output_spec) {}

  /// \brief Destructor of VirtualAbstractClosure.
  ~VirtualAbstractClosure() override = default;
  MS_DECLARE_PARENT(VirtualAbstractClosure, AbstractFuncAtom)

  /// \brief Get the abstract values of arguments.
  ///
  /// \return The abstract values of arguments.
  const AbstractBasePtrList &args_abs_list() const { return args_abs_list_; }

  /// \brief Get the abstract value of output.
  ///
  /// \return The abstract value of output.
  const AbstractBasePtr &output() const { return output_; }

  AbstractFunctionPtr Copy() const override {
    return std::make_shared<VirtualAbstractClosure>(args_abs_list_, output_);
  }

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override;

  std::string ToString() const override;

 private:
  AbstractBasePtrList args_abs_list_;
  AbstractBasePtr output_;
};
using VirtualAbstractClosurePtr = std::shared_ptr<VirtualAbstractClosure>;

/// \brief TypedPrimitiveAbstractClosure defines interface for Primitive with an explicitly
/// fixed type signature.
class MS_CORE_API TypedPrimitiveAbstractClosure final : public AbstractFuncAtom {
 public:
  /// \brief Constructor of TypedPrimitiveAbstractClosure.
  ///
  /// \param[in] prim The Primitive with an explicitly fixed type signature.
  /// \param[in] args_abs_list The abstract values of arguments to the Primitive.
  /// \param[in] output_spec The abstract value of output.
  TypedPrimitiveAbstractClosure(const PrimitivePtr prim, const AbstractBasePtrList &args_abs_list,
                                const AbstractBasePtr &output_spec)
      : prim_(prim), args_abs_list_(args_abs_list), output_(output_spec) {}

  /// \brief Destructor of TypedPrimitiveAbstractClosure.
  ~TypedPrimitiveAbstractClosure() override = default;
  MS_DECLARE_PARENT(TypedPrimitiveAbstractClosure, AbstractFuncAtom)

  /// \brief Get the Primitive that this TypedPrimitiveAbstractClosure corresponding to.
  ///
  /// \return The Primitive that this TypedPrimitiveAbstractClosure corresponding to.
  const PrimitivePtr &prim() const { return prim_; }

  /// \brief Get the abstract values of arguments this TypedPrimitiveAbstractClosure corresponding to.
  ///
  /// \return The abstract values of arguments this TypedPrimitiveAbstractClosure corresponding to.
  const AbstractBasePtrList &args_abs_list() const { return args_abs_list_; }

  /// \brief Get the abstract value of output this TypedPrimitiveAbstractClosure corresponding to.
  ///
  /// \return The abstract value of output this TypedPrimitiveAbstractClosure corresponding to.
  const AbstractBasePtr &output() const { return output_; }

  AbstractFunctionPtr Copy() const override {
    return std::make_shared<TypedPrimitiveAbstractClosure>(prim_, args_abs_list_, output_);
  }

  bool operator==(const AbstractFunction &other) const override;

  std::size_t hash() const override;

  std::string ToString() const override;

 private:
  PrimitivePtr prim_;
  AbstractBasePtrList args_abs_list_;
  AbstractBasePtr output_;
};

/// \brief Hash operator for AbstractFunction.
struct MS_CORE_API AbstractFunctionHasher {
  /// \brief Implementation of hash operation.
  ///
  /// \param[in] t The AbstractFunction needs to hash.
  ///
  /// \return The hash result.
  std::size_t operator()(const AbstractFunctionPtr &t) const {
    std::size_t hash = t->hash();
    return hash;
  }
};

/// \brief Equal operator for AbstractFunction.
struct MS_CORE_API AbstractFunctionEqual {
  /// \brief Implementation of Equal operation.
  ///
  /// \param[in] lhs The left AbstractFunction for compare.
  /// \param[in] rhs The right AbstractFunction for compare.
  ///
  /// \return Return True if the comparison result is equal, otherwise return False.
  bool operator()(const AbstractFunctionPtr &lhs, const AbstractFunctionPtr &rhs) const { return *lhs == *rhs; }
};
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_ABSTRACT_FUNCTION_H_
