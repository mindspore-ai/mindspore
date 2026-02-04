/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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
#include "frontend/operator/composite/starred_operation.h"
#include "frontend/jit/ps/static_analysis/static_analysis.h"
#include "utils/any.h"
#include "ir/meta_func_graph.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using AbstractSlicePtr = abstract::AbstractSlicePtr;
using AbstractScalarPtr = abstract::AbstractScalarPtr;
using AbstractTensorPtr = abstract::AbstractTensorPtr;
using ElemwiseMap = mindspore::HashMap<std::string, PrimitivePtr>;
using ArgsPairList = std::vector<std::pair<AnfNodePtr, TypePtr>>;
using AbstractListPtr = abstract::AbstractListPtr;

typedef enum OpsType {
  Type_Any = -1,
  Type_Normal = 0,
  Type_View,
  Type_Inplace,
  Type_Variable,
} OpsType;

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

  abstract::AbstractBasePtrList NormalizeArgs(const abstract::AbstractBasePtrList &args_abs_list) const override;
  FuncGraphPtr GenerateFromTypes(const TypePtrList &args_abs_list) override;
  MetaFuncGraphPtr GetFnLeaf() { return fn_leaf_; }
  void SetObjectForFnLeaf(const py::object &leaf_object);

 private:
  AnfNodePtr FullMake(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) const;
  AnfNodePtr FullMake(const std::shared_ptr<List> &type, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                      const ArgsPairList &arg_map) const;
  AnfNodePtr FullMake(const std::shared_ptr<Tuple> &type, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                      const ArgsPairList &arg_map) const;
  AnfNodePtr FullMake(const std::shared_ptr<Dictionary> &type, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                      const ArgsPairList &arg_map) const;
  AnfNodePtr Make(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_map) const;
  std::pair<std::string, std::string> GetHyperMapInputIndex(size_t num) const;
  template <typename T>
  void CheckArgsInSequence(const ArgsPairList &arg_map, TypeId type_id, std::size_t size, bool *contains_dyn) const;
  AnfNodePtr HyperMapConverter(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_map,
                               TypeId type_id, std::size_t size) const;
  template <typename T>
  AnfNodePtr HyperMapDynamicConverter(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                                      const ArgsPairList &arg_map, const TypePtr &element_type) const;

  MultitypeFuncGraphPtr fn_leaf_;
  bool reverse_;
  std::set<TypeId> nonleaf_;
};
using HyperMapPtr = std::shared_ptr<HyperMap>;

class HyperMapPy : public HyperMap {
 public:
  explicit HyperMapPy(bool reverse = false, const py::object &fn_leaf = py::none())
      : HyperMap(reverse, fn_leaf.cast<prim::MultitypeFuncGraphPtr>()) {
    SetObjectForFnLeaf(fn_leaf);
  }
  ~HyperMapPy() override = default;
  MS_DECLARE_PARENT(HyperMapPy, HyperMap)
};
using HyperMapPyPtr = std::shared_ptr<HyperMapPy>;

extern ValuePtr kCompositeHyperMap;

class MakeTupleGradient : public MetaFuncGraph {
 public:
  explicit MakeTupleGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~MakeTupleGradient() override = default;
  MS_DECLARE_PARENT(MakeTupleGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const MakeTupleGradient &lhs, const MakeTupleGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using MakeTupleGradientPtr = std::shared_ptr<MakeTupleGradient>;

class MakeListGradient : public MetaFuncGraph {
 public:
  explicit MakeListGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~MakeListGradient() override = default;
  MS_DECLARE_PARENT(MakeListGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const MakeListGradient &lhs, const MakeListGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using MakeListGradientPtr = std::shared_ptr<MakeListGradient>;

class MakeDictGradient : public MetaFuncGraph {
 public:
  explicit MakeDictGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~MakeDictGradient() override = default;
  MS_DECLARE_PARENT(MakeDictGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const MakeDictGradient &lhs, const MakeDictGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using MakeDictGradientPtr = std::shared_ptr<MakeDictGradient>;

class PyExecuteGradient : public MetaFuncGraph {
 public:
  explicit PyExecuteGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~PyExecuteGradient() override = default;
  MS_DECLARE_PARENT(PyExecuteGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const PyExecuteGradient &lhs, const PyExecuteGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using PyExecuteGradientPtr = std::shared_ptr<PyExecuteGradient>;

class MutableGradient : public MetaFuncGraph {
 public:
  explicit MutableGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~MutableGradient() override = default;
  MS_DECLARE_PARENT(MutableGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const MutableGradient &lhs, const MutableGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using MutableGradientPtr = std::shared_ptr<MutableGradient>;

class GradAux : public MetaFuncGraph {
 public:
  explicit GradAux(const std::string &name) : MetaFuncGraph(name) {}
  ~GradAux() override = default;
  MS_DECLARE_PARENT(GradAux, MetaFuncGraph);
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
};
using GradAuxPtr = std::shared_ptr<GradAux>;

class TaylorOperation : public MetaFuncGraph {
 public:
  explicit TaylorOperation(const std::string &name);
  ~TaylorOperation() override = default;
  MS_DECLARE_PARENT(TaylorOperation, MetaFuncGraph);
  FuncGraphPtr GetTaylorGrad(const AnfNodePtr &k, const std::vector<AnfNodePtr> &forward_graph_params) const;

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
};
using TaylorOperationPtr = std::shared_ptr<TaylorOperation>;

class TupleAdd : public MetaFuncGraph {
 public:
  explicit TupleAdd(const std::string &name) : MetaFuncGraph(name) {}
  ~TupleAdd() override = default;
  MS_DECLARE_PARENT(TupleAdd, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const TupleAdd &lhs, const TupleAdd &rhs) { return lhs.name_ == rhs.name_; }
};
using TupleAddPtr = std::shared_ptr<TupleAdd>;

class ListAdd : public MetaFuncGraph {
 public:
  explicit ListAdd(const std::string &name) : MetaFuncGraph(name) {}
  ~ListAdd() override = default;
  MS_DECLARE_PARENT(ListAdd, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const ListAdd &lhs, const ListAdd &rhs) { return lhs.name_ == rhs.name_; }
};
using ListAddPtr = std::shared_ptr<ListAdd>;

class SequenceSlice : public MetaFuncGraph {
 public:
  explicit SequenceSlice(const std::string &name) : MetaFuncGraph(name) {}
  ~SequenceSlice() override = default;
  MS_DECLARE_PARENT(SequenceSlice, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) final;
  friend bool operator==(const SequenceSlice &lhs, const SequenceSlice &rhs) { return lhs.name_ == rhs.name_; }

 protected:
  virtual void CheckArgs(const AbstractBasePtrList &args_abs_list) = 0;
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
  void CheckArgs(const AbstractBasePtrList &args_abs_list) override;
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
  void CheckArgs(const AbstractBasePtrList &args_abs_list) override;
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
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
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

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;

 private:
  size_t kShardInputSize = 0;
};

class AddAttr : public MetaFuncGraph {
 public:
  explicit AddAttr(const std::string &name) : MetaFuncGraph(name) {
    signatures_ = std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                                          {"attr_dict", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault}});
    kAddAttrInputSize = signatures_.size();
  }
  ~AddAttr() override = default;
  MS_DECLARE_PARENT(AddAttr, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;

 private:
  size_t kAddAttrInputSize = 0;
};

class VmapOperation : public MetaFuncGraph {
 public:
  explicit VmapOperation(const std::string &name);
  ~VmapOperation() override = default;
  MS_DECLARE_PARENT(VmapOperation, MetaFuncGraph)

  FuncGraphPtr GetVmap(const AnfNodePtr &vmap, int param_number) const;

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
};
using VmapOperationPtr = std::shared_ptr<VmapOperation>;

class ZerosLike : public MetaFuncGraph {
 public:
  explicit ZerosLike(const std::string &name, const std::shared_ptr<MultitypeFuncGraph> &fn_leaf = nullptr)
      : MetaFuncGraph(name), fn_leaf_(fn_leaf) {}
  ~ZerosLike() override = default;
  MS_DECLARE_PARENT(ZerosLike, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const ZerosLike &lhs, const ZerosLike &rhs) { return lhs.name_ == rhs.name_; }

 private:
  MultitypeFuncGraphPtr fn_leaf_;
};
using ZerosLikePtr = std::shared_ptr<ZerosLike>;

class IterConverter : public MetaFuncGraph {
 public:
  explicit IterConverter(const std::string &name) : MetaFuncGraph(name) {}
  ~IterConverter() override = default;
  MS_DECLARE_PARENT(IterConverter, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const IterConverter &lhs, const IterConverter &rhs) { return lhs.name_ == rhs.name_; }
};
using IterConverterPtr = std::shared_ptr<IterConverter>;

class HasNext : public MetaFuncGraph {
 public:
  explicit HasNext(const std::string &name) : MetaFuncGraph(name) {}
  ~HasNext() override = default;
  MS_DECLARE_PARENT(HasNext, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const HasNext &lhs, const HasNext &rhs) { return lhs.name_ == rhs.name_; }
};
using HasNextPtr = std::shared_ptr<HasNext>;

class Next : public MetaFuncGraph {
 public:
  explicit Next(const std::string &name) : MetaFuncGraph(name) {}
  ~Next() override = default;
  MS_DECLARE_PARENT(Next, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const Next &lhs, const Next &rhs) { return lhs.name_ == rhs.name_; }
};
using NextPtr = std::shared_ptr<Next>;

class TupleFunc : public MetaFuncGraph {
 public:
  explicit TupleFunc(const std::string &name) : MetaFuncGraph(name) {}
  ~TupleFunc() override = default;
  MS_DECLARE_PARENT(TupleFunc, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const TupleFunc &lhs, const TupleFunc &rhs) { return lhs.name_ == rhs.name_; }
};
using TupleFuncPtr = std::shared_ptr<TupleFunc>;

class ListFunc : public MetaFuncGraph {
 public:
  explicit ListFunc(const std::string &name) : MetaFuncGraph(name) {}
  ~ListFunc() override = default;
  MS_DECLARE_PARENT(ListFunc, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const ListFunc &lhs, const ListFunc &rhs) { return lhs.name_ == rhs.name_; }
};
using ListFuncPtr = std::shared_ptr<ListFunc>;

class DictFunc : public MetaFuncGraph {
 public:
  explicit DictFunc(const std::string &name) : MetaFuncGraph(name) {}
  ~DictFunc() override = default;
  MS_DECLARE_PARENT(DictFunc, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const DictFunc &lhs, const DictFunc &rhs) { return lhs.name_ == rhs.name_; }
};
using DictFuncPtr = std::shared_ptr<DictFunc>;

class ForHalfUnrollLess : public MetaFuncGraph {
 public:
  ForHalfUnrollLess() : MetaFuncGraph("ForHalfUnrollLess") {}
  ~ForHalfUnrollLess() override = default;
  MS_DECLARE_PARENT(ForHalfUnrollLess, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const ForHalfUnrollLess &lhs, const ForHalfUnrollLess &rhs) { return lhs.name_ == rhs.name_; }
};

class RecomputeBlock : public MetaFuncGraph {
 public:
  explicit RecomputeBlock(const std::string &name) : MetaFuncGraph(name) {}
  ~RecomputeBlock() override = default;
  MS_DECLARE_PARENT(RecomputeBlock, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
  friend bool operator==(const RecomputeBlock &lhs, const RecomputeBlock &rhs) { return lhs.name_ == rhs.name_; }
};
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_H_
