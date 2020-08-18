/**
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include "transform/graph_ir/op_adapter_util.h"
#include "utils/utils.h"
namespace mindspore {
namespace transform {
class OpAdapterImpl {
 public:
  OpAdapterImpl(const std::unordered_map<int, InputDesc> &input_map,
                const std::unordered_map<int, DynInputDesc> &dyn_input_map,
                const std::unordered_map<int, OutputDesc> &output_map,
                const std::unordered_map<int, DynOutputDesc> &dyn_output_map,
                const std::unordered_map<int, DynSubGraphDesc> &dyn_subgraph_map,
                const std::unordered_map<std::string, AttrDesc> &attr_map,
                const std::unordered_map<std::string, int> &enum_map,
                const std::unordered_map<unsigned int, AttrDesc> &input_attr_map,
                std::unordered_map<std::string, std::unordered_map<int, std::string>> *cus_input_map,
                std::unordered_map<std::string, std::unordered_map<int, std::string>> *cus_output_map,
                std::unordered_map<std::string, ValuePtr> *extra_attr,
                std::unordered_map<std::string, int> *name_counts, BaseOpAdapter *adpt)
      : input_map_(input_map),
        dyn_input_map_(dyn_input_map),
        output_map_(output_map),
        dyn_output_map_(dyn_output_map),
        dyn_subgraph_map_(dyn_subgraph_map),
        attr_map_(attr_map),
        enum_map_(enum_map),
        input_attr_map_(input_attr_map),
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
  bool IsCustomOp(const OperatorPtr &op);
  Status GenerateCustomOpInputMap(const CusOperatorPtr &op, const PrimitivePtr &prim);
  Status GenerateCustomOpOutputMap(const CusOperatorPtr &op, const PrimitivePtr &prim);
  OperatorPtr GenerateCustomOp(const AnfNodePtr anf);
  Status SetOpSubgraphFunc(const OperatorPtr &op, int index, const std::shared_ptr<std::vector<DfGraph>> &branches);
  Status SetCustomOpInput(const CusOperatorPtr &op, int index, const OperatorPtr &input);
  Status SetNormalOpInput(const OperatorPtr &op, int index, const OperatorPtr &input);
  int setInput(const OperatorPtr &op, int index, const OperatorPtr &input);
  Status SetCustomOpInput(const CusOperatorPtr &op, int index, const OutHandler &handle);
  Status SetNormalOpInput(const OperatorPtr &op, int index, const OutHandler &handle);
  int setInput(const OperatorPtr &op, int index, const OutHandler &handle);
  int setInput(const OperatorPtr &op, int index, const std::shared_ptr<std::vector<OutHandler>> &handler_vec);
  OutHandler getOutput(const OperatorPtr &op, int index);
  OutHandler getCustomOutput(const OperatorPtr &op, int index);
  OutHandler getNormalOutput(const OperatorPtr &op, int index);
  Status UpdateSingleOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type);
  size_t GetCustomOpOutputSize(const CusOperatorPtr &cus_op);
  std::shared_ptr<GeTensorDesc> CreateOutputDesc(const abstract::ShapePtr &shape_ptr, const TypePtr &type,
                                                 const std::string &format);
  Status UpdateMultiOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type);
  std::shared_ptr<GeTensorDesc> CreateNodeDesc(const AnfNodePtr &node);
  void UpdateNormalOpInputDesc(const OperatorPtr &op, const AnfNodePtr &node);
  void UpdateCustomOpInputDesc(const CusOperatorPtr &op, const AnfNodePtr &node);
  void updateInputDesc(const OperatorPtr &op, const AnfNodePtr &node);
  void updateOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                        const AnfNodePtr &node);
  int setAttr(const OperatorPtr &op, const std::string &attr_key, const ValuePtr &attr_value);
  int SetCustomOpAttr(const CusOperatorPtr &op, const PrimitivePtr &prim);
  int SetNormalOpAttr(const OperatorPtr &op, const PrimitivePtr &prim);
  int setAttr(const OperatorPtr &op, const PrimitivePtr &prim);
  int setAttr(const OperatorPtr &op, const AnfNodePtr &node);

 private:
  const std::unordered_map<int, InputDesc> &input_map_;
  const std::unordered_map<int, DynInputDesc> &dyn_input_map_;
  const std::unordered_map<int, OutputDesc> &output_map_;
  const std::unordered_map<int, DynOutputDesc> &dyn_output_map_;
  const std::unordered_map<int, DynSubGraphDesc> &dyn_subgraph_map_;
  const std::unordered_map<std::string, AttrDesc> &attr_map_;
  const std::unordered_map<std::string, int> &enum_map_;
  const std::unordered_map<unsigned int, AttrDesc> &input_attr_map_;
  std::unordered_map<std::string, std::unordered_map<int, std::string>> *const cus_input_map_;
  std::unordered_map<std::string, std::unordered_map<int, std::string>> *const cus_output_map_;
  std::unordered_map<std::string, ValuePtr> *const extra_attr_;
  std::unordered_map<std::string, int> *const name_counts_;
  BaseOpAdapter *const adpt_;
};

template <typename T>
class OpAdapter : public BaseOpAdapter {
 public:
  using OpType = T;
  OpAdapter()
      : impl_(std::make_shared<OpAdapterImpl>(input_map_, dyn_input_map_, output_map_, dyn_output_map_,
                                              dyn_subgraph_map_, attr_map_, enum_map_, input_attr_map_, &cus_input_map_,
                                              &cus_output_map_, &extra_attr_, &name_counts_, this)) {
    MS_EXCEPTION_IF_NULL(impl_);
  }
  explicit OpAdapter(const ExtraAttr &extra_attr)
      : extra_attr_(extra_attr),
        impl_(std::make_shared<OpAdapterImpl>(input_map_, dyn_input_map_, output_map_, dyn_output_map_,
                                              dyn_subgraph_map_, attr_map_, enum_map_, input_attr_map_, &cus_input_map_,
                                              &cus_output_map_, &extra_attr_, &name_counts_, this)) {
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

  OperatorPtr GenerateNormalOp(const AnfNodePtr &anf) {
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

    // set dynamic output num if op use DYNAMIC_OUTPUT
    if ((op != nullptr) && (!dyn_output_map_.empty()) && (anf != nullptr)) {
      TypePtr type = anf->Type();
      if (type == nullptr) {
        MS_LOG(EXCEPTION) << "Dynamic output node:" << op->GetName() << "'s Type is a nullptr!";
      }
      size_t num = type->isa<Tuple>() ? (type->cast<std::shared_ptr<Tuple>>()->size()) : 1;
      MS_LOG(INFO) << "create_dyn_output for node:" << anf->ToString() << ", type:" << type->ToString()
                   << ", num:" << num;
      dyn_output_map_.begin()->second.create_dyn_output(op, static_cast<unsigned int>(num));
    }
    return op;
  }

  OperatorPtr generate(const AnfNodePtr &anf) override {
    OperatorPtr op = nullptr;
    if (IsCustomCNode(anf)) {
      op = GenerateCustomOp(anf);
    } else {
      op = GenerateNormalOp(anf);
    }
    return op;
  }

  OperatorPtr generate(const std::string &op_name) override { return std::make_shared<OpType>(op_name); }

  const std::unordered_map<int, InputDesc> &getInputMap() override { return input_map_; }
  const std::unordered_map<unsigned int, AttrDesc> &getInputAttrMap() override { return input_attr_map_; }
  const std::unordered_map<int, DynInputDesc> &getDynInputMap() override { return dyn_input_map_; }
  const std::unordered_map<int, OutputDesc> &getOutputMap() override { return output_map_; }
  const std::unordered_map<int, DynSubGraphDesc> &getDynSubgraphMap() override { return dyn_subgraph_map_; }

  Status SetOpSubgraphFunc(const OperatorPtr &op, int index, std::shared_ptr<std::vector<DfGraph>> branches) {
    return impl_->SetOpSubgraphFunc(op, index, branches);
  }

  int setSubgraph(const OperatorPtr &op, int index, std::shared_ptr<std::vector<DfGraph>> branches) override {
    return static_cast<int>(SetOpSubgraphFunc(op, index, branches));
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

  OutHandler getCustomOutput(const OperatorPtr &op, int index) { return impl_->getCustomOutput(op, index); }

  OutHandler getNormalOutput(const OperatorPtr &op, int index) { return impl_->getNormalOutput(op, index); }

  Status UpdateSingleOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type) {
    return impl_->UpdateSingleOutputDesc(op, shp, type);
  }

  size_t GetCustomOpOutputSize(const CusOperatorPtr &cus_op) { return impl_->GetCustomOpOutputSize(cus_op); }

  std::shared_ptr<GeTensorDesc> CreateOutputDesc(const abstract::ShapePtr &shape_ptr, const TypePtr &type,
                                                 const std::string &format) {
    return impl_->CreateOutputDesc(shape_ptr, type, format);
  }

  Status UpdateMultiOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type) {
    return impl_->UpdateMultiOutputDesc(op, shp, type);
  }

  std::shared_ptr<GeTensorDesc> CreateNodeDesc(const AnfNodePtr &node) { return impl_->CreateNodeDesc(node); }

  void UpdateNormalOpInputDesc(const OperatorPtr &op, const AnfNodePtr node) {
    return impl_->UpdateNormalOpInputDesc(op, node);
  }

  void UpdateCustomOpInputDesc(const CusOperatorPtr &op, const AnfNodePtr &node) {
    return impl_->UpdateCustomOpInputDesc(op, node);
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

  std::unordered_map<std::string, ValuePtr> GetExtraAttr() override { return extra_attr_; }

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
    return static_cast<int64_t>(GetValue<int>(value));
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
      if (!it->isa<ValueTuple>()) {
        MS_LOG(EXCEPTION) << "It should be ValueTuple, but got " << it->type_name();
      }
      auto sub_vector = it->cast<ValueTuplePtr>();
      std::vector<int64_t> sublist;
      for (auto &item : sub_vector->value()) {
        sublist.push_back(static_cast<int64_t>(GetValue<int>(item)));
      }
      list.push_back(sublist);
    }
    return list;
  }

  static std::vector<int64_t> ConvertAny(const ValuePtr &value, const AnyTraits<std::vector<std::vector<int64_t>>>,
                                         const AnyTraits<std::vector<int64_t>>) {
    MS_EXCEPTION_IF_NULL(value);
    MS_LOG(DEBUG) << "Value: " << value->type_name();
    if (!value->isa<ValueList>()) {
      MS_LOG(EXCEPTION) << "Value should be ValueList, but got " << value->type_name();
    }
    auto vec = value->cast<ValueListPtr>();
    std::vector<int64_t> list;
    for (auto &it : vec->value()) {
      MS_EXCEPTION_IF_NULL(it);
      if (!it->isa<ValueList>()) {
        MS_LOG(EXCEPTION) << "It should be ValueList, but got " << it->type_name();
      }
      auto sub_vector = it->cast<ValueListPtr>();
      for (auto &item : sub_vector->value()) {
        list.push_back(static_cast<int64_t>(GetValue<int>(item)));
      }
    }
    return list;
  }

  static std::vector<int64_t> ConvertAny(const ValuePtr &value, const AnyTraits<std::vector<int64_t>>,
                                         const AnyTraits<std::vector<int64_t>>) {
    MS_EXCEPTION_IF_NULL(value);
    MS_LOG(INFO) << "Value: " << value->type_name();
    std::vector<int64_t> list;
    if (value->isa<ValueSequeue>()) {
      auto vec = value->cast<ValueSequeuePtr>();
      MS_EXCEPTION_IF_NULL(vec);
      for (auto &it : vec->value()) {
        list.push_back(static_cast<int64_t>(GetValue<int>(it)));
      }
      return list;
    }
    if (value->isa<Scalar>()) {
      list.push_back(static_cast<int64_t>(GetValue<int>(value)));
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

  static const std::unordered_map<int, InputDesc> input_map_;
  static const std::unordered_map<int, DynInputDesc> dyn_input_map_;
  static const std::unordered_map<int, OutputDesc> output_map_;
  static const std::unordered_map<int, DynOutputDesc> dyn_output_map_;
  static const std::unordered_map<int, DynSubGraphDesc> dyn_subgraph_map_;
  static const std::unordered_map<std::string, AttrDesc> attr_map_;
  static const std::unordered_map<std::string, int> enum_map_;
  // convert input from anf graph to Attr in Operators
  static const std::unordered_map<unsigned int, AttrDesc> input_attr_map_;
  static std::unordered_map<std::string, std::unordered_map<int, std::string>> cus_input_map_;
  static std::unordered_map<std::string, std::unordered_map<int, std::string>> cus_output_map_;
  std::unordered_map<std::string, ValuePtr> extra_attr_;
  std::unordered_map<std::string, int> name_counts_;
  const std::shared_ptr<OpAdapterImpl> impl_;
};

template <typename T>
const std::unordered_map<int, InputDesc> OpAdapter<T>::input_map_;
template <typename T>
const std::unordered_map<int, DynInputDesc> OpAdapter<T>::dyn_input_map_;
template <typename T>
const std::unordered_map<int, OutputDesc> OpAdapter<T>::output_map_;
template <typename T>
const std::unordered_map<int, DynOutputDesc> OpAdapter<T>::dyn_output_map_;
template <typename T>
const std::unordered_map<int, DynSubGraphDesc> OpAdapter<T>::dyn_subgraph_map_;
template <typename T>
const std::unordered_map<std::string, AttrDesc> OpAdapter<T>::attr_map_;
template <typename T>
const std::unordered_map<std::string, int> OpAdapter<T>::enum_map_;
template <typename T>
const std::unordered_map<unsigned int, AttrDesc> OpAdapter<T>::input_attr_map_;
template <typename T>
std::unordered_map<std::string, std::unordered_map<int, std::string>> OpAdapter<T>::cus_input_map_;
template <typename T>
std::unordered_map<std::string, std::unordered_map<int, std::string>> OpAdapter<T>::cus_output_map_;

// specialization for method
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_ADAPTER_H_
