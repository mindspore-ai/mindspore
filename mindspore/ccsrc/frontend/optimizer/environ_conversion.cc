/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/environ_conversion.h"

#include <memory>
#include <string>
#include <utility>
#include <unordered_map>

#include "abstract/abstract_function.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "utils/symbolic.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
using SymbolicKeyConversionMap =
  std::unordered_map<SymbolicKeyInstancePtr, int64_t, SymbolicKeyInstanceHash, SymbolicKeyInstanceEqual>;

namespace {
bool IsAbstractEnvType(const abstract::AbstractBasePtr &abs) {
  if (abs != nullptr && abs->isa<abstract::AbstractScalar>() && abs->GetTypeTrack()->isa<EnvType>()) {
    return true;
  }
  return false;
}

abstract::AbstractBasePtr TransformAbstractRecursively(const abstract::AbstractBasePtr &orig_abs,
                                                       const abstract::AbstractBasePtr target_abs) {
  if (orig_abs == nullptr) {
    return nullptr;
  }
  if (IsAbstractEnvType(orig_abs)) {
    return target_abs;
  }
  if (!orig_abs->isa<abstract::AbstractSequence>()) {
    return nullptr;
  }
  const auto &abs_seq = orig_abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(abs_seq);
  const auto &elements = abs_seq->elements();
  abstract::AbstractBasePtrList new_elements;
  bool transformed = false;
  for (auto elem : elements) {
    if (elem->isa<abstract::AbstractSequence>()) {
      auto inner_abs_seq = elem->cast<abstract::AbstractSequencePtr>();
      auto transformed_abs = TransformAbstractRecursively(inner_abs_seq, target_abs);
      if (transformed_abs != nullptr) {
        new_elements.push_back(transformed_abs);
        transformed = true;
      } else {
        new_elements.push_back(elem);
      }
    } else if (IsAbstractEnvType(elem)) {
      new_elements.push_back(target_abs);
      transformed = true;
    } else {
      new_elements.push_back(elem);
    }
  }
  if (transformed) {
    abstract::AbstractBasePtr new_abs;
    if (abs_seq->isa<abstract::AbstractTuple>()) {
      new_abs = std::make_shared<abstract::AbstractTuple>(new_elements);
    } else if (abs_seq->isa<abstract::AbstractList>()) {
      new_abs = std::make_shared<abstract::AbstractList>(new_elements);
    } else {
      MS_LOG(EXCEPTION) << "abs_seq is not AbstractTuple or AbstractList, but: " << abs_seq->ToString();
    }
    return new_abs;
  }
  // No transformation.
  return nullptr;
}

void TransformNodeAbstractIfEnvType(const AnfNodePtr &node, const abstract::AbstractBasePtr &target_abs) {
  auto transformed_abs = TransformAbstractRecursively(node->abstract(), target_abs);
  if (transformed_abs != nullptr) {
    node->set_abstract(transformed_abs);
  }
}

TypeId GetValueType(const CNodePtr &cnode) {
  // (EnvironSet/EnvironGet, environ, key, value/default)
  constexpr size_t environ_input_size = 4;
  if (cnode->size() != environ_input_size) {
    MS_LOG(EXCEPTION) << "EnvrionSet/EnvironGet cnode should have 4 inputs, but: " << cnode->DebugString();
  }
  const auto &value_abstract = cnode->input(3)->abstract();
  if (value_abstract == nullptr) {
    MS_LOG(EXCEPTION) << "4th input of EnvironSet/EnvironGet cnode should abstract, but not set, node: "
                      << cnode->DebugString();
  }
  if (IsAbstractEnvType(value_abstract)) {
    return kObjectTypeEnvType;
  } else if (value_abstract->isa<abstract::AbstractMonad>()) {
    return kObjectTypeMonad;
  } else {
    return kObjectTypeTensorType;
  }
}

AnfNodePtr GetTransformedKeyNode(const AnfNodePtr &old_key_node, SymbolicKeyConversionMap *para_symbol_map) {
  const auto &symbolic_key_inst = GetValueNode<SymbolicKeyInstancePtr>(old_key_node);
  int64_t transformed_key = 0;
  auto &symbolic_key_map = *para_symbol_map;
  auto iter = symbolic_key_map.find(symbolic_key_inst);
  if (iter != symbolic_key_map.end()) {
    transformed_key = iter->second;
  } else {
    static int64_t key_counter = 0;
    transformed_key = ++key_counter;
    (void)symbolic_key_map.emplace(std::make_pair(symbolic_key_inst, transformed_key));
  }
  auto tensor_key = std::make_shared<mindspore::tensor::Tensor>(transformed_key);
  auto transformed_key_node = NewValueNode(tensor_key);
  transformed_key_node->set_abstract(tensor_key->ToAbstract());
  return transformed_key_node;
}

void InsertEnvironDestroyAll(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto output = func_graph->output();
  auto u = NewValueNode(kUMonad);
  u->set_abstract(kUMonad->ToAbstract());
  auto depend1 = func_graph->NewCNode({NewValueNode(prim::kPrimDepend), u, output});
  depend1->set_abstract(kUMonad->ToAbstract());
  auto environ_destroy_all = func_graph->NewCNode({NewValueNode(prim::kPrimEnvironDestroyAll), depend1});
  environ_destroy_all->set_abstract(std::make_shared<abstract::AbstractScalar>(kValueAny, std::make_shared<Bool>()));
  auto depend2 = func_graph->NewCNode({NewValueNode(prim::kPrimDepend), output, environ_destroy_all});
  depend2->set_abstract(output->abstract());
  func_graph->set_output(depend2);
}
}  // namespace

bool EnvironConversion(const pipeline::ResourcePtr &resource) {
  SymbolicKeyConversionMap symbolic_key_map;
  static AbstractBasePtr scalar_abs = std::make_shared<abstract::AbstractScalar>(kValueAny, kInt64);
  static const AbstractBasePtr tensor_abs = std::make_shared<abstract::AbstractTensor>(scalar_abs);
  static const std::string attr_name = "value_type";
  const int kPrimitiveOffset = 0;
  const int kEnvironTypeOffset = 1;
  const int kSymbolicKeyOffset = 2;
  const int kEnvironValueOffset = 3;
  auto mng = resource->manager();
  const auto &all_nodes = mng->all_nodes();
  auto txn = mng->Transact();
  auto destroy_env = false;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimEnvironSet) && !IsPrimitiveCNode(node, prim::kPrimEnvironGet)) {
      continue;
    }
    destroy_env = true;
    const auto &cnode = node->cast<CNodePtr>();
    // Prim
    AnfNodePtr transformed_prim_node;
    const auto &type_id = GetValueType(cnode);
    if (type_id == kObjectTypeMonad) {
      // Eliminate the kPrimEnvironSet node.
      if (IsPrimitiveCNode(node, prim::kPrimEnvironSet)) {
        (void)txn.Replace(cnode, cnode->input(kEnvironTypeOffset));
      } else {
        // Eliminate the kPrimEnvironGet node.
        (void)txn.Replace(cnode, cnode->input(kEnvironValueOffset));
      }
      continue;
    }
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(kPrimitiveOffset));
    MS_EXCEPTION_IF_NULL(prim);
    const auto &old_attr = prim->GetAttr(attr_name);
    if (old_attr != nullptr) {
      const auto old_attr_value = GetValue<int>(old_attr);
      if (old_attr_value != type_id) {
        if (IsPrimitiveCNode(node, prim::kPrimEnvironSet)) {
          prim = std::make_shared<Primitive>(prim::kEnvironSet);
        } else {
          prim = std::make_shared<Primitive>(prim::kEnvironGet);
        }
        MS_EXCEPTION_IF_NULL(prim);
        prim->set_attr(attr_name, MakeValue(static_cast<int>(type_id)));
        transformed_prim_node = NewValueNode(prim);
        txn.SetEdge(node, kPrimitiveOffset, transformed_prim_node);
      }
    } else {
      prim->set_attr(attr_name, MakeValue(static_cast<int>(type_id)));
    }
    // Abstract of Environ & Value will be set by later TransformNodeAbstract function.
    // Key
    if (!IsValueNode<SymbolicKeyInstance>(cnode->input(kSymbolicKeyOffset))) {
      MS_LOG(EXCEPTION) << "should be SymbolicKey, but: " << cnode->input(kSymbolicKeyOffset)->ToString();
    }
    const auto &transformed_key_node = GetTransformedKeyNode(cnode->input(kSymbolicKeyOffset), &symbolic_key_map);
    txn.SetEdge(node, kSymbolicKeyOffset, transformed_key_node);
  }
  txn.Commit();

  // Insert EnvironDestroyAll if env ops exist.
  if (destroy_env) {
    InsertEnvironDestroyAll(resource->func_graph());
  }

  // Previous loop is depending on the AbstractType, so use another loop to modify the AbstractType.
  const auto &new_all_nodes = mng->all_nodes();
  for (const auto &node : new_all_nodes) {
    TransformNodeAbstractIfEnvType(node, tensor_abs);
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
