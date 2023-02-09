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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_H_

#include <memory>
#include <vector>
#include <string>
#include <map>
#include "utils/hash_map.h"
#include "transform/graph_ir/op_adapter_util.h"
#include "transform/graph_ir/op_adapter_base.h"
#include "include/common/utils/utils.h"
namespace mindspore {
namespace transform {
class OpAdapterImpl {
 public:
  OpAdapterImpl(const mindspore::HashMap<int, InputDesc> &input_map,
                const mindspore::HashMap<int, DynInputDesc> &dyn_input_map, const std::map<int, OutputDesc> &output_map,
                const mindspore::HashMap<int, DynOutputDesc> &dyn_output_map,
                const mindspore::HashMap<int, SubGraphDesc> &subgraph_map,
                const mindspore::HashMap<int, DynSubGraphDesc> &dyn_subgraph_map,
                const mindspore::HashMap<std::string, AttrDesc> &attr_map,
                const mindspore::HashMap<std::string, int> &enum_map,
                const mindspore::HashMap<unsigned int, AttrDesc> &input_attr_map,
                const mindspore::HashMap<std::string, std::string> &attr_input_map,
                mindspore::HashMap<std::string, mindspore::HashMap<int, std::string>> *cus_input_map,
                mindspore::HashMap<std::string, std::map<int, std::string>> *cus_output_map,
                mindspore::HashMap<std::string, ValuePtr> *extra_attr,
                mindspore::HashMap<std::string, int> *name_counts, BaseOpAdapter *adpt)
      : input_map_(input_map),
        dyn_input_map_(dyn_input_map),
        output_map_(output_map),
        dyn_output_map_(dyn_output_map),
        subgraph_map_(subgraph_map),
        dyn_subgraph_map_(dyn_subgraph_map),
        attr_map_(attr_map),
        enum_map_(enum_map),
        input_attr_map_(input_attr_map),
        attr_input_map_(attr_input_map),
        cus_input_map_(cus_input_map),
        cus_output_map_(cus_output_map),
        extra_attr_(extra_attr),
        name_counts_(name_counts),
        adpt_(adpt) {
    MS_EXCEPTION_IF_NULL(cus_input_map_);
    MS_EXCEPTION_IF_NULL(cus_output_map_);
    MS_EXCEPTION_IF_NULL(extra_attr_);
    MS_EXCEPTION_IF_NULL(name_counts_);
    MS_EXCEPTION_IF_NULL(adpt_);
  }
  ~OpAdapterImpl() {}
  bool IsCustomOp(const OperatorPtr &op) const;
  std::string GetCustomOpType(const PrimitivePtr &prim) const;
  Status GenerateCustomOpInputMap(const CusOperatorPtr &op, const PrimitivePtr &prim) const;
  Status GenerateCustomOpOutputMap(const CusOperatorPtr &op, const PrimitivePtr &prim) const;
  OperatorPtr GenerateCustomOp(const AnfNodePtr anf);
  Status SetOpSubgraphFunc(const OperatorPtr &op, const std::shared_ptr<std::vector<DfGraph>> &subgraphs);
  Status SetOpSubgraphFunc(const OperatorPtr &op, int index, const std::shared_ptr<std::vector<DfGraph>> &branches);
  Status SetCustomOpInput(const CusOperatorPtr &op, int index, const OperatorPtr &input) const;
  Status SetNormalOpInput(const OperatorPtr &op, int index, const OperatorPtr &input);
  int setInput(const OperatorPtr &op, int index, const OperatorPtr &input);
  Status SetCustomOpInput(const CusOperatorPtr &op, int index, const OutHandler &handle) const;
  Status SetNormalOpInput(const OperatorPtr &op, int index, const OutHandler &handle);
  int setInput(const OperatorPtr &op, int index, const OutHandler &handle);
  int setInput(const OperatorPtr &op, int index, const std::shared_ptr<std::vector<OutHandler>> &handler_vec);
  OutHandler getOutput(const OperatorPtr &op, int index);
  std::vector<OutHandler> getOutputs(const OperatorPtr &op) const;
  OutHandler getCustomOutput(const OperatorPtr &op, int index) const;
  OutHandler getNormalOutput(const OperatorPtr &op, int index);
  std::vector<OutHandler> getNormalOutputs(const OperatorPtr &op) const;
  std::vector<OutHandler> getCustomOutputs(const OperatorPtr &op) const;
  Status UpdateSingleOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                                const std::string &format);
  size_t GetCustomOpOutputSize(const CusOperatorPtr &cus_op) const;
  std::map<std::string, ValuePtr> GetNormalOpAttrList(const AnfNodePtr &node) const;
  std::shared_ptr<GeTensorDesc> CreateOutputDesc(const abstract::ShapePtr &shape_ptr, const TypePtr &type,
                                                 const std::string &format) const;
  Status UpdateMultiOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                               const std::string &format);
  std::shared_ptr<GeTensorDesc> CreateNodeDesc(const AnfNodePtr &node, const std::string &format) const;
  void UpdateNormalOpInputDesc(const OperatorPtr &op, const AnfNodePtr &node, const std::string format);
  void UpdateCustomOpInputDesc(const CusOperatorPtr &op, const AnfNodePtr &node, const std::string format) const;
  void updateInputDesc(const OperatorPtr &op, const AnfNodePtr &node);
  void updateOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                        const AnfNodePtr &node);
  int setAttr(const OperatorPtr &op, const std::string &attr_key, const ValuePtr &attr_value);
  int SetCustomOpAttr(const CusOperatorPtr &op, const PrimitivePtr &prim) const;
  int SetNormalOpAttr(const OperatorPtr &op, const PrimitivePtr &prim);
  int setAttr(const OperatorPtr &op, const PrimitivePtr &prim);
  int setAttr(const OperatorPtr &op, const AnfNodePtr &node);

 private:
  const mindspore::HashMap<int, InputDesc> &input_map_;
  const mindspore::HashMap<int, DynInputDesc> &dyn_input_map_;
  const std::map<int, OutputDesc> &output_map_;
  const mindspore::HashMap<int, DynOutputDesc> &dyn_output_map_;
  const mindspore::HashMap<int, SubGraphDesc> &subgraph_map_;
  const mindspore::HashMap<int, DynSubGraphDesc> &dyn_subgraph_map_;
  const mindspore::HashMap<std::string, AttrDesc> &attr_map_;
  const mindspore::HashMap<std::string, int> &enum_map_;
  const mindspore::HashMap<unsigned int, AttrDesc> &input_attr_map_;
  const mindspore::HashMap<std::string, std::string> &attr_input_map_;
  mindspore::HashMap<std::string, mindspore::HashMap<int, std::string>> *const cus_input_map_;
  mindspore::HashMap<std::string, std::map<int, std::string>> *const cus_output_map_;
  mindspore::HashMap<std::string, ValuePtr> *const extra_attr_;
  mindspore::HashMap<std::string, int> *const name_counts_;
  BaseOpAdapter *const adpt_;
};

template <typename T>
class OpAdapter : public BaseOpAdapter {
 public:
  using OpType = T;
  OpAdapter()
      : impl_(std::make_shared<OpAdapterImpl>(input_map_, dyn_input_map_, output_map_, dyn_output_map_, subgraph_map_,
                                              dyn_subgraph_map_, attr_map_, enum_map_, input_attr_map_, attr_input_map_,
                                              &cus_input_map_, &cus_output_map_, &extra_attr_, &name_counts_, this)) {
    MS_EXCEPTION_IF_NULL(impl_);
  }
  explicit OpAdapter(const ExtraAttr &extra_attr)
      : extra_attr_(extra_attr),
        impl_(std::make_shared<OpAdapterImpl>(input_map_, dyn_input_map_, output_map_, dyn_output_map_, subgraph_map_,
                                              dyn_subgraph_map_, attr_map_, enum_map_, input_attr_map_, attr_input_map_,
                                              &cus_input_map_, &cus_output_map_, &extra_attr_, &name_counts_, this)) {
    MS_EXCEPTION_IF_NULL(impl_);
  }
  ~OpAdapter() override {}

  bool IsCustomOp(const OperatorPtr &op) { return impl_->IsCustomOp(op); }

  Status GenerateCustomOpInputMap(const CusOperatorPtr &op, const PrimitivePtr &prim) {
    return impl_->GenerateCustomOpInputMap(op, prim);
  }

  Status GenerateCustomOpOutputMap(const CusOperatorPtr &op, const PrimitivePtr &prim) {
    return impl_->GenerateCustomOpOutputMap(op, prim);
  }

  // Convert ME UserCustom AnfNode to GE CustomOp. And set it's attrs.
  OperatorPtr GenerateCustomOp(const AnfNodePtr anf) { return impl_->GenerateCustomOp(anf); }

  OperatorPtr GenerateNormalOp(const AnfNodePtr &anf) const {
    OperatorPtr op = nullptr;
    // There are duplicate names in ANF graph, do not assign ANF node name to GE
    // GE will generate unique name automatically
    if (anf != nullptr && anf->fullname_with_scope() != "") {
      auto name = anf->fullname_with_scope();
      op = std::make_shared<OpType>(name);
    } else {
      MS_LOG(DEBUG) << "no fullname_with_scope";
      op = std::make_shared<OpType>();
    }

    // set dynamic output num if op use DYNAMIC_OUTPUT
    if ((op != nullptr) && (!dyn_output_map_.empty()) && (anf != nullptr)) {
      TypePtr type = anf->Type();
      if (type == nullptr) {
        MS_LOG(EXCEPTION) << "Dynamic output node:" << op->GetName() << "'s Type is a nullptr!";
      }
      auto num = GetOutputSize(type);
      MS_LOG(INFO) << "create_dyn_output for node:" << anf->fullname_with_scope() << ", type:" << type->ToString()
                   << ", num:" << num;
      dyn_output_map_.begin()->second.create_dyn_output(op, static_cast<unsigned int>(num));
    }
    return op;
  }

  OperatorPtr GenerateDynamicOutputOp(const AnfNodePtr &anf) const {
    OperatorPtr op = nullptr;
    // There are duplicate names in ANF graph, do not assign ANF node name to GE
    // GE will generate unique name automatically
    if (anf != nullptr && anf->fullname_with_scope() != "") {
      MS_LOG(DEBUG) << anf->fullname_with_scope();
      op = std::make_shared<OpType>(anf->fullname_with_scope());
    } else {
      MS_LOG(DEBUG) << "no fullname_with_scope";
      op = std::make_shared<OpType>();
    }
    return op;
  }

  void setDynamicOutputNum(const OperatorPtr &op, size_t dyn_output_size) override {
    // set dynamic output num if op use DYNAMIC_OUTPUT
    if ((op != nullptr) && (!dyn_output_map_.empty())) {
      MS_LOG(DEBUG) << "create_dyn_output for node:" << op->GetName() << ", num:" << dyn_output_size;
      dyn_output_map_.begin()->second.create_dyn_output(op, static_cast<unsigned int>(dyn_output_size));
    }
  }

  OperatorPtr generate(const AnfNodePtr &anf) override {
    OperatorPtr op = nullptr;
    if (IsCustomCNode(anf)) {
      op = GenerateCustomOp(anf);
    } else {
      op = GenerateNormalOp(anf);
    }
    if (op == nullptr) {
      MS_LOG(EXCEPTION) << "Can not generate op for " << anf->fullname_with_scope();
    }
    return op;
  }

  OperatorPtr generate(const std::string &op_name) override { return std::make_shared<OpType>(op_name); }

  OperatorPtr generateDynOutputOp(const AnfNodePtr &anf) override {
    OperatorPtr op = nullptr;
    op = GenerateDynamicOutputOp(anf);
    if (op == nullptr) {
      MS_LOG(EXCEPTION) << "Can not generate op for " << anf->fullname_with_scope();
    }
    return op;
  }

  std::string getOpType() override {
    if (op_type_.empty()) {
      auto op = std::make_unique<OpType>();
      op_type_ = op->GetOpType();
    }
    return op_type_;
  }
  const mindspore::HashMap<int, InputDesc> &getInputMap() override { return input_map_; }
  const mindspore::HashMap<unsigned int, AttrDesc> &getInputAttrMap() override { return input_attr_map_; }
  const mindspore::HashMap<std::string, std::string> &getAttrInputMap() override { return attr_input_map_; }
  const mindspore::HashMap<int, DynInputDesc> &getDynInputMap() override { return dyn_input_map_; }
  const mindspore::HashMap<int, SubGraphDesc> &getSubgraphMap() override { return subgraph_map_; }
  const std::map<int, OutputDesc> &getOutputMap() override { return output_map_; }
  const mindspore::HashMap<int, DynOutputDesc> &getDynOutputMap() override { return dyn_output_map_; }
  const mindspore::HashMap<int, DynSubGraphDesc> &getDynSubgraphMap() override { return dyn_subgraph_map_; }
  std::map<std::string, ValuePtr> GetNormalOpAttrList(const AnfNodePtr &node) override {
    return impl_->GetNormalOpAttrList(node);
  }
  bool IsDynInputOp(uint64_t index) override { return dyn_input_map_.find(index) != dyn_input_map_.end(); }
  bool IsDyOutputOp(uint64_t index) override { return dyn_output_map_.find(index) != dyn_output_map_.end(); }
  bool IsMultipleOutputOp() override { return output_map_.size() > 1; }

  Status SetOpSubgraphFunc(const OperatorPtr &op, std::shared_ptr<std::vector<DfGraph>> subgraphs) {
    return impl_->SetOpSubgraphFunc(op, subgraphs);
  }

  void setSubgraph(const OperatorPtr &op, std::shared_ptr<std::vector<DfGraph>> subgraphs) override {
    (void)SetOpSubgraphFunc(op, subgraphs);
  }

  Status SetOpSubgraphFunc(const OperatorPtr &op, int index, const std::shared_ptr<std::vector<DfGraph>> &branches) {
    return impl_->SetOpSubgraphFunc(op, index, branches);
  }

  void setSubgraph(const OperatorPtr &op, int index, const std::shared_ptr<std::vector<DfGraph>> &branches) override {
    (void)SetOpSubgraphFunc(op, index, branches);
  }

  Status SetCustomOpInput(const CusOperatorPtr &op, int index, const OperatorPtr &input) {
    return impl_->SetCustomOpInput(op, index, input);
  }

  Status SetNormalOpInput(const OperatorPtr &op, int index, const OperatorPtr &input) {
    return impl_->SetNormalOpInput(op, index, input);
  }

  int setInput(const OperatorPtr &op, int index, const OperatorPtr &input) override {
    return impl_->setInput(op, index, input);
  }

  Status SetCustomOpInput(const CusOperatorPtr &op, int index, const OutHandler &handle) {
    return impl_->SetCustomOpInput(op, index, handle);
  }

  Status SetNormalOpInput(const OperatorPtr &op, int index, const OutHandler &handle) {
    return impl_->SetNormalOpInput(op, index, handle);
  }

  int setInput(const OperatorPtr &op, int index, const OutHandler &handle) override {
    return impl_->setInput(op, index, handle);
  }

  int setInput(const OperatorPtr &op, int index, const std::shared_ptr<std::vector<OutHandler>> &handler_vec) override {
    return impl_->setInput(op, index, handler_vec);
  }

  OutHandler getOutput(const OperatorPtr &op, int index) override { return impl_->getOutput(op, index); }

  std::vector<OutHandler> getOutputs(const OperatorPtr &op) override { return impl_->getOutputs(op); }

  OutHandler getCustomOutput(const OperatorPtr &op, int index) { return impl_->getCustomOutput(op, index); }

  OutHandler getNormalOutput(const OperatorPtr &op, int index) { return impl_->getNormalOutput(op, index); }

  Status UpdateSingleOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                                const std::string &format) {
    return impl_->UpdateSingleOutputDesc(op, shp, type, format);
  }

  size_t GetCustomOpOutputSize(const CusOperatorPtr &cus_op) { return impl_->GetCustomOpOutputSize(cus_op); }

  std::shared_ptr<GeTensorDesc> CreateOutputDesc(const abstract::ShapePtr &shape_ptr, const TypePtr &type,
                                                 const std::string &format) {
    return impl_->CreateOutputDesc(shape_ptr, type, format);
  }

  Status UpdateMultiOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                               const std::string &format) {
    return impl_->UpdateMultiOutputDesc(op, shp, type, format);
  }

  std::shared_ptr<GeTensorDesc> CreateNodeDesc(const AnfNodePtr &node, const std::string &format) {
    return impl_->CreateNodeDesc(node, format);
  }

  void UpdateNormalOpInputDesc(const OperatorPtr &op, const AnfNodePtr node, const std::string format) {
    return impl_->UpdateNormalOpInputDesc(op, node, format);
  }

  void UpdateCustomOpInputDesc(const CusOperatorPtr &op, const AnfNodePtr &node, const std::string format) {
    return impl_->UpdateCustomOpInputDesc(op, node, format);
  }

  void updateInputDesc(const OperatorPtr &op, const AnfNodePtr &node) { impl_->updateInputDesc(op, node); }

  void updateOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                        const AnfNodePtr &node) override {
    impl_->updateOutputDesc(op, shp, type, node);
  }

  int setAttr(const OperatorPtr &op, const std::string &attrKey, const ValuePtr &attrValue) override {
    return impl_->setAttr(op, attrKey, attrValue);
  }

  int SetCustomOpAttr(const CusOperatorPtr &op, const PrimitivePtr &prim) { return impl_->SetCustomOpAttr(op, prim); }

  int SetNormalOpAttr(const OperatorPtr &op, const PrimitivePtr &prim) { return impl_->SetNormalOpAttr(op, prim); }

  int setAttr(const OperatorPtr &op, const PrimitivePtr &prim) override { return impl_->setAttr(op, prim); }

  int setAttr(const OperatorPtr &op, const AnfNodePtr &node) override { return impl_->setAttr(op, node); }

  mindspore::HashMap<std::string, ValuePtr> GetExtraAttr() override { return extra_attr_; }

 private:
  template <typename S>
  static S ConvertAny(const ValuePtr &value, const AnyTraits<S> &) {
    return GetValue<S>(value);
  }

  // specialization for reverse bool
  static bool ConvertAny(const ValuePtr &value, const AnyTraits<bool> &, bool reverse) {
    return reverse != GetValue<bool>(value);
  }

  template <typename P, typename Q>
  static Q ConvertAny(const ValuePtr &value, const AnyTraits<P> &traits_from, const AnyTraits<Q> &traits_to) {
    return ConvertAnyUtil(value, traits_from, traits_to);
  }

  // specialization for tensor
  static GeTensor ConvertAny(const ValuePtr &value, const AnyTraits<mindspore::tensor::Tensor> &traits) {
    // To-DO the format may read from ME tensor
    return ConvertAnyUtil(value, traits);
  }

  // specialization for int
  static int64_t ConvertAny(const ValuePtr &value, const AnyTraits<int64_t>) {
    return static_cast<int64_t>(GetValue<int64_t>(value));
  }

  // specialization for int or tuple broadcast to Vector
  static std::vector<int64_t> ConvertAny(const ValuePtr &value, const std::string &name,
                                         const AnyTraits<std::vector<int64_t>> anyTraitsInt) {
    return ConvertAnyUtil(value, name, anyTraitsInt);
  }

  static std::vector<std::vector<int64_t>> ConvertAny(const ValuePtr &value,
                                                      const AnyTraits<std::vector<std::vector<int64_t>>>) {
    MS_EXCEPTION_IF_NULL(value);
    MS_LOG(INFO) << "Value: " << value->type_name();
    std::vector<std::vector<int64_t>> list;
    if (!value->isa<ValueTuple>()) {
      MS_LOG(EXCEPTION) << "Value should be ValueTuple, but got " << value->type_name();
    }
    auto vec = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(vec);
    for (auto &it : vec->value()) {
      MS_EXCEPTION_IF_NULL(it);
      std::vector<int64_t> sublist;
      if (!it->isa<ValueTuple>()) {
        if (it->type_name() != "ValueList") {
          MS_LOG(EXCEPTION) << "It should be ValueTuple or ValueList, but got " << it->type_name();
        }
        auto sub_vector = it->cast<ValueListPtr>();
        for (auto &item : sub_vector->value()) {
          sublist.push_back(static_cast<int64_t>(GetValue<int64_t>(item)));
        }
      } else {
        auto sub_vector = it->cast<ValueTuplePtr>();
        for (auto &item : sub_vector->value()) {
          sublist.push_back(static_cast<int64_t>(GetValue<int64_t>(item)));
        }
      }
      list.push_back(sublist);
    }
    return list;
  }

  static std::vector<int64_t> ConvertAny(const ValuePtr &value, const AnyTraits<std::vector<std::vector<int64_t>>>,
                                         const AnyTraits<std::vector<int64_t>>) {
    MS_EXCEPTION_IF_NULL(value);
    MS_LOG(DEBUG) << "Value: " << value->type_name();
    if (!value->isa<ValueSequence>()) {
      MS_LOG(EXCEPTION) << "Value should be ValueSequence, but got " << value->type_name();
    }
    auto vec = value->cast<ValueSequencePtr>();
    std::vector<int64_t> list;
    for (auto &it : vec->value()) {
      MS_EXCEPTION_IF_NULL(it);
      if (!it->isa<ValueSequence>()) {
        MS_LOG(EXCEPTION) << "It should be ValueSequence, but got " << it->type_name();
      }
      auto sub_vector = it->cast<ValueSequencePtr>();
      for (auto &item : sub_vector->value()) {
        list.push_back(static_cast<int64_t>(GetValue<int64_t>(item)));
      }
    }
    return list;
  }

  static std::vector<int64_t> ConvertAny(const ValuePtr &value, const AnyTraits<std::vector<int64_t>>,
                                         const AnyTraits<std::vector<int64_t>>) {
    MS_EXCEPTION_IF_NULL(value);
    MS_LOG(INFO) << "Value: " << value->type_name();
    std::vector<int64_t> list;
    if (value->isa<ValueSequence>()) {
      auto vec = value->cast<ValueSequencePtr>();
      MS_EXCEPTION_IF_NULL(vec);
      for (auto &it : vec->value()) {
        list.push_back(static_cast<int64_t>(GetValue<int64_t>(it)));
      }
      return list;
    }
    if (value->isa<Scalar>()) {
      list.push_back(static_cast<int64_t>(GetValue<int64_t>(value)));
      return list;
    }
    MS_LOG(EXCEPTION) << "Value should be ValueTuple or Scalar, but got " << value->type_name();
  }

  static std::string ConvertAny(const ValuePtr &value, const AnyTraits<std::vector<int64_t>> anyTraitsVec,
                                const AnyTraits<std::string> anyTraitsStr) {
    return ConvertAnyUtil(value, anyTraitsVec, anyTraitsStr);
  }

  static std::vector<float> ConvertAny(const ValuePtr &value, const AnyTraits<std::vector<float>> anyTraitsVec,
                                       const AnyTraits<float> anyTraitsFlo) {
    return ConvertAnyUtil(value, anyTraitsVec, anyTraitsFlo);
  }

  static std::vector<int64_t> ConvertAny(const ValuePtr &value, const std::string &format,
                                         const AnyTraits<std::vector<int64_t>> anyTraitsVec,
                                         const AnyTraits<int64_t> anyTraitsInt) {
    return ConvertAnyUtil(value, format, anyTraitsVec, anyTraitsInt);
  }

  // convert value list for value tuple to vector
  template <typename P, typename Q>
  static std::vector<Q> ConvertAny(const ValuePtr &value, const AnyTraits<P> &anyTraitsP,
                                   const AnyTraits<std::vector<Q>> anyTraitsQ) {
    return ConvertAnyUtil(value, anyTraitsP, anyTraitsQ);
  }

  static int64_t ConvertAny(const ValuePtr &value, const AnyTraits<GeEnum>) {
    auto name = GetValue<std::string>(value);
    auto it = enum_map_.find(name);
    int v = 0;
    if (it != enum_map_.end()) {
      v = it->second;
    }
    return v;
  }

  static GeDataType ConvertAny(const ValuePtr &value, const AnyTraits<GEType> anyTraitsGE) {
    return ConvertAnyUtil(value, anyTraitsGE);
  }

  // convert any value to tensor
  static GeTensor ConvertAny(const ValuePtr &value, const AnyTraits<AnyValue> anyTraitsValue) {
    return ConvertAnyUtil(value, anyTraitsValue);
  }

  size_t GetOutputSize(const TypePtr &type) const {
    if (!type->isa<Tuple>()) {
      return (type->isa<MonadType>() || type->isa<TypeNone>() || type->isa<TypeNull>()) ? 0 : 1;
    }
    size_t output_size = 0;
    auto tuple_type = type->cast<std::shared_ptr<Tuple>>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    auto elements = tuple_type->elements();
    for (const auto &element : elements) {
      if (element->isa<MonadType>() || element->isa<TypeNone>() || element->isa<TypeNull>()) {
        continue;
      }
      output_size = output_size + GetOutputSize(element);
    }
    return output_size;
  }

  static const mindspore::HashMap<int, InputDesc> input_map_;
  static const mindspore::HashMap<int, DynInputDesc> dyn_input_map_;
  // note: To keep the outputs in order, the 'output_map_' and 'cus_output_map_' must be std::map instead of Hashmap.
  static const std::map<int, OutputDesc> output_map_;
  static const mindspore::HashMap<int, DynOutputDesc> dyn_output_map_;
  static const mindspore::HashMap<int, SubGraphDesc> subgraph_map_;
  static const mindspore::HashMap<int, DynSubGraphDesc> dyn_subgraph_map_;
  static const mindspore::HashMap<std::string, AttrDesc> attr_map_;
  static const mindspore::HashMap<std::string, int> enum_map_;
  // convert input from anf graph to Attr in Operators
  static const mindspore::HashMap<unsigned int, AttrDesc> input_attr_map_;
  static const mindspore::HashMap<std::string, std::string> attr_input_map_;
  static mindspore::HashMap<std::string, mindspore::HashMap<int, std::string>> cus_input_map_;
  static mindspore::HashMap<std::string, std::map<int, std::string>> cus_output_map_;
  mindspore::HashMap<std::string, ValuePtr> extra_attr_;
  mindspore::HashMap<std::string, int> name_counts_;
  const std::shared_ptr<OpAdapterImpl> impl_;
  std::string op_type_;
};

template <typename T>
const mindspore::HashMap<int, InputDesc> OpAdapter<T>::input_map_;
template <typename T>
const mindspore::HashMap<int, DynInputDesc> OpAdapter<T>::dyn_input_map_;
template <typename T>
const std::map<int, OutputDesc> OpAdapter<T>::output_map_;
template <typename T>
const mindspore::HashMap<int, DynOutputDesc> OpAdapter<T>::dyn_output_map_;
template <typename T>
const mindspore::HashMap<int, SubGraphDesc> OpAdapter<T>::subgraph_map_;
template <typename T>
const mindspore::HashMap<int, DynSubGraphDesc> OpAdapter<T>::dyn_subgraph_map_;
template <typename T>
const mindspore::HashMap<std::string, AttrDesc> OpAdapter<T>::attr_map_;
template <typename T>
const mindspore::HashMap<std::string, int> OpAdapter<T>::enum_map_;
template <typename T>
const mindspore::HashMap<unsigned int, AttrDesc> OpAdapter<T>::input_attr_map_;
template <typename T>
const mindspore::HashMap<std::string, std::string> OpAdapter<T>::attr_input_map_;
template <typename T>
mindspore::HashMap<std::string, mindspore::HashMap<int, std::string>> OpAdapter<T>::cus_input_map_;
template <typename T>
mindspore::HashMap<std::string, std::map<int, std::string>> OpAdapter<T>::cus_output_map_;

// specialization for method
}  // namespace transform
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_H_
