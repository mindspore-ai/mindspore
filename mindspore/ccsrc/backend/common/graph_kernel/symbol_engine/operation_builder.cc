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
#include "ops/math_ops.h"
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
    MS_LOG(ERROR) << "The size of inputs should be equal to input symbol size, but got " << inputs.size() << " vs "
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
  } else if (real_node->isa<ValueNode>()) {
    inp_symbol = InputSymbol::Make(real_node->abstract());
  } else {
    // node is a CNode
    MS_LOG(EXCEPTION) << "The shape of " << real_node->DebugString() << " does not exist in symbol cache.";
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
  } else if (real_node->isa<ValueNode>()) {
    inp_symbol = InputSymbol::Make(real_node->abstract());
  } else {
    // node is a CNode
    MS_LOG(EXCEPTION) << "The value of " << real_node->DebugString() << " does not exist in symbol cache.";
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
  if (attr->isa<ValueSequence>()) {
    return IListSymbol::FromShape(GetValue<std::vector<int64_t>>(attr), true);
  }
  MS_LOG(EXCEPTION) << "Only support {bool, int, int-list} attr, but got " << attr->ToString();
}
namespace {
template <int IDX>
SymbolPtr TransparentShape(OperationBuilder *b) {
  return b->RealShape(b->GetInput(IDX));
}
SymbolPtr BinElemwiseShape(OperationBuilder *b) {
  auto lhs = b->RealShape(b->GetInput(1));
  auto rhs = b->RealShape(b->GetInput(2));
  return b->Emit(std::make_shared<infershape::BinElemwise>(lhs, rhs));
}
SymbolPtr BinElemwiseValue(OperationBuilder *b) {
  // todo, call infervalue::BinElemwise
  return nullptr;
}
SymbolPtr ReduceShape(OperationBuilder *b) {
  auto input = b->RealShape(b->GetInput(1));
  auto axis = b->RealValue(b->GetInput(2));
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
  auto x = b->RealShape(b->GetInput(1));
  auto y = b->RealShape(b->GetInput(2));
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

REG_OP_SYMBOL_BUILDER("Abs").SetBuildShape(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Cast").SetBuildShape(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Exp").SetBuildShape(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Log").SetBuildShape(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Neg").SetBuildShape(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Sqrt").SetBuildShape(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("ReLU").SetBuildShape(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Softmax").SetBuildShape(TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("BiasAdd").SetBuildShape({DependOn::kShape, DependOn::kNone}, TransparentShape<1>);
REG_OP_SYMBOL_BUILDER("Pow").SetBuildShape(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Add").SetBuildShape(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Sub").SetBuildShape(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Mul").SetBuildShape(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Maximum").SetBuildShape(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Minimum").SetBuildShape(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("Pow").SetBuildShape(BinElemwiseShape);
REG_OP_SYMBOL_BUILDER("RealDiv").SetBuildShape(BinElemwiseShape).SetBuildValue(BinElemwiseValue);
REG_OP_SYMBOL_BUILDER("Div").SetBuildShape(BinElemwiseShape).SetBuildValue(BinElemwiseValue);
REG_OP_SYMBOL_BUILDER("ReduceSum").SetBuildShape({DependOn::kShape, DependOn::kValue}, ReduceShape);
REG_OP_SYMBOL_BUILDER("ReduceMax").SetBuildShape({DependOn::kShape, DependOn::kValue}, ReduceShape);
REG_OP_SYMBOL_BUILDER("ReduceMin").SetBuildShape({DependOn::kShape, DependOn::kValue}, ReduceShape);
REG_OP_SYMBOL_BUILDER("Reshape").SetBuildShape({DependOn::kShape, DependOn::kValue},
                                               [](OperationBuilder *b) -> SymbolPtr {
                                                 auto input = b->RealShape(b->GetInput(1));
                                                 auto shape = b->RealValue(b->GetInput(2));
                                                 return b->Emit(std::make_shared<infershape::Reshape>(input, shape));
                                               });
REG_OP_SYMBOL_BUILDER("Transpose").SetBuildShape({DependOn::kShape, DependOn::kValue}, [](OperationBuilder *b) {
  auto input = b->RealShape(b->GetInput(1));
  auto perm = b->RealValue(b->GetInput(2));
  return b->Emit(std::make_shared<infershape::Transpose>(input, perm));
});
REG_OP_SYMBOL_BUILDER("MatMul").SetBuildShape(MatMulShape<false>);
REG_OP_SYMBOL_BUILDER("BatchMatMul").SetBuildShape(MatMulShape<true>);
REG_OP_SYMBOL_BUILDER("MakeTuple").SetBuildShape([](OperationBuilder *b) -> SymbolPtr {
  SymbolPtrList result(b->cnode()->size() - 1);
  (void)std::transform(b->cnode()->inputs().cbegin() + 1, b->cnode()->inputs().cend(), result.begin(),
                       [b](const AnfNodePtr &node) { return b->RealShape(node); });
  return ListSymbol::Make(std::move(result));
});
}  // namespace
}  // namespace ops::builders
}  // namespace mindspore::graphkernel::symbol
