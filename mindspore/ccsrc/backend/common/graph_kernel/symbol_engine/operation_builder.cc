/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/symbol_engine/operation_builder.h"
#include <algorithm>
#include <utility>
#include "ir/primitive.h"
#include "ops/shape_calc.h"
#include "ops/math_ops.h"
#include "ops/sequence_ops.h"
#include "backend/common/graph_kernel/symbol_engine/operations/common_op.h"
#include "backend/common/graph_kernel/symbol_engine/operations/infershape_op.h"
#include "backend/common/graph_kernel/symbol_engine/operations/infervalue_op.h"

namespace mindspore::graphkernel::symbol {
void SymbolCache::InitInputs(const AnfNodePtrList &parameters) {
  inputs_.reserve(parameters.size());
  for (size_t i = 0; i < parameters.size(); i++) {
    auto &inp = inputs_.emplace_back(InputSymbol::Make(parameters[i]->abstract()));
    input_index_[parameters[i]] = i;
    MS_LOG(DEBUG) << "Init input [" << i << "]: " << parameters[i]->DebugString() << ". symbol is " << inp->ToString();
  }
}

bool SymbolCache::UpdateInputs(const AbstractBasePtrList &inputs) {
  if (inputs.size() != inputs_.size()) {
    MS_LOG(WARNING) << "The size of inputs should be equal to input symbol size, but got " << inputs.size() << " vs "
                    << inputs_.size();
    return false;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    inputs_[i]->SetAbstract(inputs[i]);
    MS_LOG(DEBUG) << "Update input [" << i << "] to " << inputs_[i]->ToString();
  }
  return true;
}

namespace ops::builders {
SymbolPtr OperationBuilder::RealShape(const AnfNodePtr &node) const {
  auto smbl = cache_->GetShape(node);
  if (smbl != nullptr) {
    return smbl;
  }
  auto real_node = cache_->RealNode(node);
  InputSymbolPtr inp_symbol = nullptr;
  if (real_node->isa<Parameter>()) {
    inp_symbol = cache_->GetInput(node);
  } else {
    if (real_node->isa<CNode>()) {
      MS_LOG(DEBUG) << "The shape of " << real_node->DebugString() << " does not exist in symbol cache.";
    }
    inp_symbol = InputSymbol::Make(real_node->abstract());
  }
  smbl = Emit(std::make_shared<infershape::RealShape>(inp_symbol));
  cache_->SetShape(node, smbl);
  return smbl;
}

SymbolPtr OperationBuilder::RealValue(const AnfNodePtr &node) const {
  auto smbl = cache_->GetValue(node);
  if (smbl != nullptr) {
    return smbl;
  }
  auto real_node = cache_->RealNode(node);
  InputSymbolPtr inp_symbol = nullptr;
  if (real_node->isa<Parameter>()) {
    inp_symbol = cache_->GetInput(node);
  } else {
    if (real_node->isa<CNode>()) {
      MS_LOG(DEBUG) << "The value of " << real_node->DebugString() << " does not exist in symbol cache.";
    }
    inp_symbol = InputSymbol::Make(real_node->abstract());
  }
  smbl = Emit(std::make_shared<infervalue::RealValue>(inp_symbol));
  cache_->SetValue(node, smbl);
  return smbl;
}

SymbolPtr OperationBuilder::GetAttr(const std::string &attr_name) const {
  auto attr = GetCNodePrimitive(cnode_)->GetAttr(attr_name);
  if (attr == nullptr) {
    return nullptr;
  }
  if (attr->isa<BoolImm>()) {
    return BoolSymbol::Make(GetValue<bool>(attr));
  }
  if (attr->isa<IntegerImm>()) {
    return IntSymbol::Make(GetValue<int64_t>(attr));
  }
  if (attr->isa<StringImm>()) {
    return StrSymbol::Make(GetValue<std::string>(attr));
  }
  if (attr->isa<FloatImm>()) {
    return FloatSymbol::Make(GetValue<double>(attr));
  }
  if (attr->isa<ValueSequence>()) {
    return IListSymbol::FromShape(GetValue<std::vector<int64_t>>(attr), true);
  }
  MS_LOG(EXCEPTION) << "Only support {bool, int, float, string, int-list} attr, but got " << attr->ToString();
}
namespace {
template <int IDX>
SymbolPtr TransparentShape(OperationBuilder *b) {
  return b->GetInputShape(IDX);
}
template <int IDX>
SymbolPtr TransparentValue(OperationBuilder *b) {
  return b->GetInputValue(IDX);
}
SymbolPtr BinElemwiseShape(OperationBuilder *b) {
  auto lhs = b->GetInputShape(1);
  auto rhs = b->GetInputShape(2);
  return b->Emit(std::make_shared<infershape::BinElemwise>(lhs, rhs));
}
SymbolPtr AddnShape(OperationBuilder *b) {
  auto process_func = [b](const CNodePtr &cnode) {
    MS_EXCEPTION_IF_NULL(cnode);
    auto result = b->RealShape(cnode->input(1));
    for (size_t i = 2; i < cnode->size(); i++) {
      auto next = b->RealShape(cnode->input(i));
      result = b->Emit(std::make_shared<infershape::BinElemwise>(result, next));
    }
    return result;
  };
  if (IsPrimitiveCNode(b->cnode()->input(1), prim::kPrimMakeTuple)) {
    return process_func(b->cnode()->input(1)->cast<CNodePtr>());
  }
  return process_func(b->cnode());
}

SymbolPtr ReduceShape(OperationBuilder *b) {
  auto input = b->GetInputShape(1);
  auto axis = b->GetInputValue(2);
  auto keep_dims = b->GetAttr(kAttrKeepDims);
  MS_EXCEPTION_IF_NULL(keep_dims);
  // the skip_mode only exists in ReduceSum
  auto skip_mode = b->GetAttr(kAttrSkipMode);
  if (skip_mode == nullptr) {
    skip_mode = BoolSymbol::Make(false);
  }
  return b->Emit(std::make_shared<infershape::Reduce>(input, axis, keep_dims, skip_mode));
}

template <bool HAS_BATCH>
SymbolPtr MatMulShape(OperationBuilder *b) {
  auto x = b->GetInputShape(1);
  auto y = b->GetInputShape(2);
  auto trans_a = b->GetAttr("transpose_a");
  if (trans_a == nullptr) {
    trans_a = BoolSymbol::Make(false);
  }
  auto trans_b = b->GetAttr("transpose_b");
  if (trans_b == nullptr) {
    trans_b = BoolSymbol::Make(false);
  }
  return b->Emit(std::make_shared<infershape::MatMul>(x, y, trans_a, trans_b, HAS_BATCH));
}

template <typename T>
SymbolPtr BinShapeValue(OperationBuilder *b) {
  auto s = b->GetInputShape(1);
  auto v = b->GetInputValue(2);
  return b->Emit(std::make_shared<T>(s, v));
}
inline const std::initializer_list<DependOn> kDependShapeValue = {DependOn::kShape, DependOn::kValue};

SymbolPtr BiasAddGradShape(OperationBuilder *b) {
  auto x = b->GetInputShape(1);
  auto fmt = b->GetAttr("format");
  if (fmt != nullptr) {
    fmt = StrSymbol::Make("NCHW");
  }
  return b->Emit(std::make_shared<infershape::BiasAddGrad>(x, fmt));
}

SymbolPtr TupleGetItem(OperationBuilder *b) {
  std::shared_ptr<ListSymbol> input = nullptr;
  if (b->is_building_shape()) {
    input = b->GetInputShape(1)->as_sptr<ListSymbol>();
  } else {
    input = b->GetInputValue(1)->as_sptr<ListSymbol>();
  }
  MS_EXCEPTION_IF_NULL(input);
  int64_t index = GetValue<int64_t>(b->GetInput(2)->cast<ValueNodePtr>()->value());
  if (LongToSize(index) >= input->size()) {
    MS_LOG(EXCEPTION) << "The index " << index << " is out of range of input symbol " << input->ToString()
                      << ". node: " << b->cnode()->DebugString();
  }
  return input->symbols()[index];
}
SymbolPtr MakeTuple(OperationBuilder *b) {
  SymbolPtrList result(b->cnode()->size() - 1);
  (void)std::transform(
    b->cnode()->inputs().cbegin() + 1, b->cnode()->inputs().cend(), result.begin(),
    [b](const AnfNodePtr &node) { return b->is_building_shape() ? b->RealShape(node) : b->RealValue(node); });
  if (!b->is_building_shape() &&
      std::all_of(result.cbegin(), result.cend(), [](const SymbolPtr &s) { return s->is<IntSymbol>(); })) {
    return IListSymbol::Make(std::move(result));
  }
  return ListSymbol::Make(std::move(result));
}
SymbolPtr TupleToTensor(OperationBuilder *b) {
  auto list_symbol = b->GetInputValue(kIndex1)->as_sptr<ListSymbol>();
  if (list_symbol->is_dyn_len()) {
    MS_LOG(WARNING) << "The input of TupleToTensor should not be dynamic rank, but got " << list_symbol->ToString();
    return nullptr;
  }
  return IListSymbol::Make(list_symbol->symbols());
}

REG_OP_SYMBOL_BUILDER("Abs").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Cast").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Exp").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Log").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Neg").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Sqrt").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("ReLU").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("ReluGrad").SetShapeDepend({DependOn::kShape, DependOn::kNone}).SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Reciprocal").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Softmax").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("SoftmaxGrad").SetShapeFunc(TransparentShape<2>);
REG_OP_SYMBOL_BUILDER("LogSoftmax").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("LogSoftmaxGrad").SetShapeFunc(TransparentShape<2>);
REG_OP_SYMBOL_BUILDER("BiasAdd").SetShapeDepend({DependOn::kShape, DependOn::kNone}).SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("BiasAddGrad").SetShapeFunc(BiasAddGradShape);
REG_OP_SYMBOL_BUILDER("AddN").SetShapeFunc(AddnShape);
REG_OP_SYMBOL_BUILDER("Pow").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Add").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Sub").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Mul").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Maximum").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Minimum").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Pow").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("RealDiv").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Div").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Less").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("LessEqual").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Greater").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("GreaterEqual").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Equal").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("NotEqual").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("LogicalAnd").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("LogicalOr").SetShapeFunc(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("LogicalNot").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Select").SetShapeFunc(TransparentShape<1>);  // 3 inputs' shape should be same for Select
REG_OP_SYMBOL_BUILDER("ReduceSum").SetShapeDepend(kDependShapeValue).SetShapeFunc(ReduceShape);
REG_OP_SYMBOL_BUILDER("ReduceMax").SetShapeDepend(kDependShapeValue).SetShapeFunc(ReduceShape);
REG_OP_SYMBOL_BUILDER("ReduceMin").SetShapeDepend(kDependShapeValue).SetShapeFunc(ReduceShape);
REG_OP_SYMBOL_BUILDER("Reshape").SetShapeDepend(kDependShapeValue).SetShapeFunc(BinShapeValue<infershape::Reshape>);
REG_OP_SYMBOL_BUILDER("Transpose").SetShapeDepend(kDependShapeValue).SetShapeFunc(BinShapeValue<infershape::Transpose>);
REG_OP_SYMBOL_BUILDER("Tile").SetShapeDepend(kDependShapeValue);
REG_OP_SYMBOL_BUILDER("ExpandDims")
  .SetShapeDepend(kDependShapeValue)
  .SetShapeFunc(BinShapeValue<infershape::ExpandDims>);
REG_OP_SYMBOL_BUILDER("DynamicBroadcastTo")
  .SetShapeDepend({DependOn::kNone, DependOn::kValue})
  .SetShapeFunc(TransparentValue<2>);
REG_OP_SYMBOL_BUILDER("MatMul").SetShapeFunc(MatMulShape<false>);
REG_OP_SYMBOL_BUILDER("BatchMatMul").SetShapeFunc(MatMulShape<true>);
REG_OP_SYMBOL_BUILDER("Dropout").SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
  auto s = b->GetInputShape(1);
  return ListSymbol::Make({s, s});
});
REG_OP_SYMBOL_BUILDER("DropoutGrad").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Assign").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("OnesLike").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Tril").SetShapeFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("LayerNorm")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone, DependOn::kNone})
  .SetShapeFunc([](OperationBuilder *b) -> SymbolPtr {
    auto x = b->GetInputShape(kIndex1);
    auto begin_axis = b->GetAttr(kAttrBeginNormAxis);
    MS_EXCEPTION_IF_NULL(begin_axis);
    return b->Emit(std::make_shared<infershape::LayerNorm>(x, begin_axis));
  });
REG_OP_SYMBOL_BUILDER("LayerNormGrad")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone, DependOn::kNone, DependOn::kNone, DependOn::kShape})
  .SetShapeFunc([](OperationBuilder *b) {
    auto x = b->GetInputShape(kIndex1);
    auto gamma = b->GetInputShape(kIndex5);
    return ListSymbol::Make({x, gamma, gamma});
  });
REG_OP_SYMBOL_BUILDER("TensorShape").SetValueDepend({DependOn::kShape}).SetValueFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Shape").SetValueDepend({DependOn::kShape}).SetValueFunc(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("ShapeCalc").SetValueDepend([](const CNodePtr &cnode) -> std::vector<DependOn> {
  auto p = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(p);
  auto value_depend_attr = p->GetAttr(mindspore::ops::kAttrValueDepend);
  MS_EXCEPTION_IF_NULL(value_depend_attr);
  auto value_depend = GetValue<std::vector<bool>>(value_depend_attr);
  std::vector<DependOn> depends(value_depend.size());
  (void)std::transform(value_depend.cbegin(), value_depend.cend(), depends.begin(),
                       [](bool v) { return v ? DependOn::kValue : DependOn::kShape; });
  return depends;
});
REG_OP_SYMBOL_BUILDER("ScalarMul").SetValueFunc([](OperationBuilder *b) -> SymbolPtr {
  auto x = b->GetInputValue(kIndex1);
  auto y = b->GetInputValue(kIndex2);
  return b->Emit(std::make_shared<ScalarMul>(x, y));
});

// virtual nodes
REG_OP_SYMBOL_BUILDER("MakeTuple").SetShapeFunc(MakeTuple);
REG_OP_SYMBOL_BUILDER("RealMakeTuple").SetShapeFunc(MakeTuple).SetValueFunc(MakeTuple);
REG_OP_SYMBOL_BUILDER("TupleToTensor").SetValueFunc(TupleToTensor);
REG_OP_SYMBOL_BUILDER("TupleGetItem").SetShapeFunc(TupleGetItem).SetValueFunc(TupleGetItem);
REG_OP_SYMBOL_BUILDER("RealTupleGetItem").SetShapeFunc(TupleGetItem).SetValueFunc(TupleGetItem);
REG_OP_SYMBOL_BUILDER("Depend").SetShapeFunc(TransparentShape<1>).SetValueFunc(TransparentValue<1>);
REG_OP_SYMBOL_BUILDER("Load").SetShapeFunc(TransparentShape<1>).SetValueFunc(TransparentValue<1>);
REG_OP_SYMBOL_BUILDER("UpdateState").SetShapeFunc(TransparentShape<1>).SetValueFunc(TransparentValue<1>);
}  // namespace
}  // namespace ops::builders
}  // namespace mindspore::graphkernel::symbol
