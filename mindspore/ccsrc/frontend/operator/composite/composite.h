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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_H_

#include <vector>
#include <string>
#include <utility>
#include <map>
#include <set>
#include <memory>
#include "utils/hash_map.h"
#include "frontend/operator/composite/zip_operation.h"
#include "frontend/operator/composite/list_operation.h"
#include "frontend/operator/composite/do_signature.h"
#include "frontend/operator/composite/unpack_call.h"
#include "frontend/operator/composite/multitype_funcgraph.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "utils/misc.h"
#include "utils/any.h"
#include "ir/dtype.h"
#include "ir/meta_func_graph.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using AbstractSlicePtr = abstract::AbstractSlicePtr;
using AbstractScalarPtr = abstract::AbstractScalarPtr;
using AbstractTensorPtr = abstract::AbstractTensorPtr;
using ElemwiseMap = mindspore::HashMap<std::string, PrimitivePtr>;
using ArgsPairList = std::vector<std::pair<AnfNodePtr, TypePtr>>;
using AbstractListPtr = abstract::AbstractListPtr;

class HyperMap : public MetaFuncGraph {
 public:
  explicit HyperMap(bool reverse = false, const std::shared_ptr<MultitypeFuncGraph> &fn_leaf = nullptr);
  HyperMap(const HyperMap &h);
  void Init();
  HyperMap &operator=(const HyperMap &h) noexcept {
    if (this != &h) {
      fn_leaf_ = h.fn_leaf_;
      reverse_ = h.reverse_;
      nonleaf_ = h.nonleaf_;
      if (fn_leaf_) {
        name_ = "hyper_map[" + fn_leaf_->name() + "]";
      }
    }
    return *this;
  }
  ~HyperMap() override = default;
  MS_DECLARE_PARENT(HyperMap, MetaFuncGraph)

  abstract::AbstractBasePtrList NormalizeArgs(const abstract::AbstractBasePtrList &args_spec_list) const override;
  FuncGraphPtr GenerateFromTypes(const TypePtrList &args_spec_list) override;
  MetaFuncGraphPtr GetFnLeaf() { return fn_leaf_; }

 private:
  AnfNodePtr FullMake(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_map);
  AnfNodePtr FullMake(const std::shared_ptr<List> &type, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                      const ArgsPairList &arg_map);
  AnfNodePtr FullMake(const std::shared_ptr<Tuple> &type, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                      const ArgsPairList &arg_map);
  AnfNodePtr FullMake(const std::shared_ptr<Dictionary> &type, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                      const ArgsPairList &arg_map);
  AnfNodePtr Make(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_map);
  std::pair<std::string, std::string> GetHyperMapInputIndex(size_t num) const;

  MultitypeFuncGraphPtr fn_leaf_;
  bool reverse_;
  std::set<TypeId> nonleaf_;
};
using HyperMapPtr = std::shared_ptr<HyperMap>;

class HyperMapPy : public HyperMap {
 public:
  explicit HyperMapPy(bool reverse = false, const std::shared_ptr<MultitypeFuncGraph> &fn_leaf = nullptr)
      : HyperMap(reverse, fn_leaf) {}
  ~HyperMapPy() override = default;
  MS_DECLARE_PARENT(HyperMapPy, HyperMap)
};
using HyperMapPyPtr = std::shared_ptr<HyperMapPy>;

extern ValuePtr kCompositeHyperMap;

enum TailType { kGradAll, kGradFirst, kGradByPosition, kNotGrad };

class Tail : public MetaFuncGraph {
 public:
  explicit Tail(const std::string &name, TailType tail_type = kNotGrad, bool return_ids = false)
      : MetaFuncGraph(name), tail_type_(tail_type), enable_tuple_grad_first_(false), return_ids_(return_ids) {}
  ~Tail() override = default;
  MS_DECLARE_PARENT(Tail, MetaFuncGraph)

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;

  friend bool operator==(const Tail &lhs, const Tail &rhs) { return lhs.name_ == rhs.name_; }
  void set_enable_tuple_grad_first(bool enable_tuple_grad_first) { enable_tuple_grad_first_ = enable_tuple_grad_first; }

 private:
  FuncGraphPtr GenerateTailFuncGraph(const abstract::AbstractSequencePtr &sequence_arg) const;
  FuncGraphPtr GenerateGradFuncGraph(const abstract::AbstractTuplePtr &tuple_arg,
                                     const abstract::AbstractTuplePtr &position = nullptr) const;

  TailType tail_type_;
  bool enable_tuple_grad_first_;
  bool return_ids_;
};
using TailPtr = std::shared_ptr<Tail>;

class MakeTupleGradient : public MetaFuncGraph {
 public:
  explicit MakeTupleGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~MakeTupleGradient() override = default;
  MS_DECLARE_PARENT(MakeTupleGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const MakeTupleGradient &lhs, const MakeTupleGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using MakeTupleGradientPtr = std::shared_ptr<MakeTupleGradient>;

class MakeListGradient : public MetaFuncGraph {
 public:
  explicit MakeListGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~MakeListGradient() override = default;
  MS_DECLARE_PARENT(MakeListGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const MakeListGradient &lhs, const MakeListGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using MakeListGradientPtr = std::shared_ptr<MakeListGradient>;

class MakeDictGradient : public MetaFuncGraph {
 public:
  explicit MakeDictGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~MakeDictGradient() override = default;
  MS_DECLARE_PARENT(MakeDictGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const MakeDictGradient &lhs, const MakeDictGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using MakeDictGradientPtr = std::shared_ptr<MakeDictGradient>;

class PyExecuteGradient : public MetaFuncGraph {
 public:
  explicit PyExecuteGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~PyExecuteGradient() override = default;
  MS_DECLARE_PARENT(PyExecuteGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const PyExecuteGradient &lhs, const PyExecuteGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using PyExecuteGradientPtr = std::shared_ptr<PyExecuteGradient>;

class MutableGradient : public MetaFuncGraph {
 public:
  explicit MutableGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~MutableGradient() override = default;
  MS_DECLARE_PARENT(MutableGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const MutableGradient &lhs, const MutableGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using MutableGradientPtr = std::shared_ptr<MutableGradient>;

class GradOperation : public MetaFuncGraph {
 public:
  explicit GradOperation(const std::string &name, bool get_all = false, bool get_by_list = false,
                         bool sens_param = false, bool get_by_position = false, bool has_aux = false,
                         bool get_value = false, bool return_ids = false);
  ~GradOperation() override = default;
  MS_DECLARE_PARENT(GradOperation, MetaFuncGraph)

  FuncGraphPtr GetGrad(const AnfNodePtr &j, const AnfNodePtr &weights, const AnfNodePtr &position,
                       const std::vector<AnfNodePtr> &forward_graph_params, bool enable_tuple_grad,
                       bool is_weights_none) const;

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;

  bool sens_param() const { return sens_param_; }
  bool get_all_;
  bool get_by_list_;
  bool sens_param_;
  bool get_by_position_;
  bool has_aux_;
  bool get_value_;
  bool return_ids_;

 private:
  void GradByParameter(const FuncGraphPtr &k_child, const AnfNodePtr &f_app, const AnfNodePtr &bprop,
                       const AnfNodePtr &weights, const AnfNodePtr &position, bool enable_tuple_grad,
                       bool is_weights_none) const;
  CNodePtr SetNodeByParameter(const CNodePtr &grad, const FuncGraphPtr &fg) const;
  AbstractBasePtr weight_value_;
};
using GradOperationPtr = std::shared_ptr<GradOperation>;

class GradAux : public MetaFuncGraph {
 public:
  explicit GradAux(const std::string &name) : MetaFuncGraph(name) {}
  ~GradAux() override = default;
  MS_DECLARE_PARENT(GradAux, MetaFuncGraph);
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
};
using GradAuxPtr = std::shared_ptr<GradAux>;

class TaylorOperation : public MetaFuncGraph {
 public:
  explicit TaylorOperation(const std::string &name);
  ~TaylorOperation() override = default;
  MS_DECLARE_PARENT(TaylorOperation, MetaFuncGraph);
  FuncGraphPtr GetTaylorGrad(const AnfNodePtr &k, const std::vector<AnfNodePtr> &forward_graph_params) const;

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
};
using TaylorOperationPtr = std::shared_ptr<TaylorOperation>;

class TupleAdd : public MetaFuncGraph {
 public:
  explicit TupleAdd(const std::string &name) : MetaFuncGraph(name) {}
  ~TupleAdd() override = default;
  MS_DECLARE_PARENT(TupleAdd, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const TupleAdd &lhs, const TupleAdd &rhs) { return lhs.name_ == rhs.name_; }
};
using TupleAddPtr = std::shared_ptr<TupleAdd>;

class SequenceSlice : public MetaFuncGraph {
 public:
  explicit SequenceSlice(const std::string &name) : MetaFuncGraph(name) {}
  ~SequenceSlice() override = default;
  MS_DECLARE_PARENT(SequenceSlice, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) final;
  friend bool operator==(const SequenceSlice &lhs, const SequenceSlice &rhs) { return lhs.name_ == rhs.name_; }

 protected:
  virtual void CheckArgs(const AbstractBasePtrList &args_spec_list) = 0;
  virtual FuncGraphPtr BuildFuncGraph(int64_t start_index, int64_t stop_index, int64_t step_value) = 0;
  abstract::AbstractSequencePtr sequence_ = nullptr;
  AbstractSlicePtr slice_ = nullptr;
};

class SequenceSliceGetItem : public SequenceSlice {
 public:
  explicit SequenceSliceGetItem(const std::string &name, const std::string &prim_name, const std::string &get_item_name)
      : SequenceSlice(name),
        prim_(std::make_shared<Primitive>(prim_name)),
        get_item_(std::make_shared<Primitive>(get_item_name)) {}
  ~SequenceSliceGetItem() override = default;
  MS_DECLARE_PARENT(SequenceSliceGetItem, MetaFuncGraph)
  friend bool operator==(const SequenceSliceGetItem &lhs, const SequenceSliceGetItem &rhs) {
    return lhs.name_ == rhs.name_;
  }

 protected:
  void CheckArgs(const AbstractBasePtrList &args_spec_list) override;
  FuncGraphPtr BuildFuncGraph(int64_t start_index, int64_t stop_index, int64_t step_value) override;

 private:
  PrimitivePtr prim_;
  PrimitivePtr get_item_;
};

class ListSliceSetItem : public SequenceSlice {
 public:
  explicit ListSliceSetItem(const std::string &name) : SequenceSlice(name) {}
  ~ListSliceSetItem() override = default;
  MS_DECLARE_PARENT(ListSliceSetItem, MetaFuncGraph)
  friend bool operator==(const ListSliceSetItem &lhs, const ListSliceSetItem &rhs) { return lhs.name_ == rhs.name_; }

 protected:
  void CheckArgs(const AbstractBasePtrList &args_spec_list) override;
  FuncGraphPtr BuildFuncGraph(int64_t start_index, int64_t stop_index, int64_t step_value) override;

 private:
  void CheckAssignRange(int64_t start_index, int64_t stop_index, int64_t step_value);
  AnfNodePtr GetAssignNode(const FuncGraphPtr &func_graph, const AnfNodePtr &assign_node, int64_t step_value);
  AbstractListPtr value_list_ = nullptr;
};

class TupleGetItemTensor : public MetaFuncGraph {
 public:
  explicit TupleGetItemTensor(const std::string &name) : MetaFuncGraph(name) {}
  ~TupleGetItemTensor() override = default;
  MS_DECLARE_PARENT(TupleGetItemTensor, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const TupleGetItemTensor &lhs, const TupleGetItemTensor &rhs) {
    return lhs.name_ == rhs.name_;
  }
};
using TupleGetItemTensorPtr = std::shared_ptr<TupleGetItemTensor>;

class Shard : public MetaFuncGraph {
 public:
  explicit Shard(const string &name) : MetaFuncGraph(name) {
    signatures_ =
      // def shard(func:read, weight_list:read, in_axes:read, out_axes:read, parameter_plan:read, device:read,
      // level:read):
      std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"in_axes", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"out_axes", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"device", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"level", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault}});
    kShardInputSize = signatures_.size();
  }
  ~Shard() override = default;
  MS_DECLARE_PARENT(Shard, MetaFuncGraph)

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;

 private:
  size_t kShardInputSize = 0;
};

class VmapOperation : public MetaFuncGraph {
 public:
  explicit VmapOperation(const std::string &name);
  ~VmapOperation() override = default;
  MS_DECLARE_PARENT(VmapOperation, MetaFuncGraph)

  FuncGraphPtr GetVmap(const AnfNodePtr &vmap, int param_number) const;

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
};
using VmapOperationPtr = std::shared_ptr<VmapOperation>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_H_
