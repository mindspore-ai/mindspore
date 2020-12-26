/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/mindir_inputs_adjust_pass.h"
#include <vector>
#include <memory>
#include "src/common/log_adapter.h"
#include "src/ops/primitive_c.h"
#include "src/tensor.h"

using mindspore::lite::PrimitiveC;
namespace mindspore {
namespace opt {
namespace {
template <typename T>
void CopyAttrForArgMinMax(T *left, T *right) {
  MS_ASSERT(left != null && right != nullptr);
  left->axis = right->axis;
  left->outMaxValue = right->outMaxValue;
  left->axisType = right->axisType;
  left->keepDims = right->keepDims;
  left->topK = right->topK;
}
}  // namespace

bool MindirInputAdjustOpPass::CheckCNodeIsArgMinMax(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto prim_node = cnode->inputs().at(0);
  MS_ASSERT(prim_node != nullptr);
  auto prim_value_node = prim_node->cast<ValueNodePtr>();
  if (prim_value_node == nullptr) {
    MS_LOG(DEBUG) << "cnode first input is not valueNode.";
    return false;
  }
  auto value = prim_value_node->value();
  MS_ASSERT(value != nullptr);
  auto prim_c = value->cast<PrimitiveCPtr>();
  if (prim_c == nullptr) {
    MS_LOG(DEBUG) << "prim is not primitiveC.";
    return false;
  }
  auto prim = prim_c->primitiveT();
  MS_ASSERT(prim != nullptr);
  return prim->value.type == schema::PrimitiveType_ArgMax || prim->value.type == schema::PrimitiveType_ArgMin;
}

int MindirInputAdjustOpPass::AdjustArgMinMaxInputs(std::vector<AnfNodePtr> *inputs, bool index_or_value) {
  MS_ASSERT(inputs != nullptr);
  auto prim_node = inputs->at(0);
  MS_ASSERT(prim_node != nullptr);
  auto prim_value_node = prim_node->cast<ValueNodePtr>();
  if (prim_value_node == nullptr) {
    MS_LOG(ERROR) << "cnode first input is not valueNode.";
    return lite::RET_ERROR;
  }
  auto prim_value = prim_value_node->value();
  if (prim_value == nullptr) {
    MS_LOG(ERROR) << "valueNode value is nullptr.";
    return lite::RET_ERROR;
  }
  auto prim_c = prim_value->cast<PrimitiveCPtr>();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "value is not primitiveC.";
    return lite::RET_ERROR;
  }
  auto prim = prim_c->primitiveT();
  MS_ASSERT(prim != nullptr && prim->value.value != nullptr);
  auto attr = prim->value.value;
  if (prim->value.type == schema::PrimitiveType_ArgMax) {
    reinterpret_cast<schema::ArgMaxT *>(attr)->outMaxValue = index_or_value;
  } else if (prim->value.type == schema::PrimitiveType_ArgMin) {
    reinterpret_cast<schema::ArgMinT *>(attr)->outMaxValue = index_or_value;
  }
  return lite::RET_OK;
}

int MindirInputAdjustOpPass::CopyPrimitiveCForArgMinMax(std::vector<AnfNodePtr> *inputs) {
  MS_ASSERT(inputs != nullptr);
  auto prim_node = inputs->at(0);
  MS_ASSERT(prim_node != nullptr);
  auto prim_value_node = prim_node->cast<ValueNodePtr>();
  if (prim_value_node == nullptr) {
    MS_LOG(ERROR) << "cnode first input is not valueNode.";
    return lite::RET_ERROR;
  }
  auto prim_value = prim_value_node->value();
  if (prim_value == nullptr) {
    MS_LOG(ERROR) << "valueNode value is nullptr.";
    return lite::RET_ERROR;
  }
  auto prim_c = prim_value->cast<PrimitiveCPtr>();
  if (prim_c == nullptr) {
    MS_LOG(ERROR) << "value is not primitiveC.";
    return lite::RET_ERROR;
  }
  auto prim = prim_c->primitiveT();
  MS_ASSERT(prim != nullptr && prim->value.value != nullptr);
  auto primitive = std::make_unique<schema::PrimitiveT>();
  if (prim->value.type == schema::PrimitiveType_ArgMax) {
    primitive->value.type = schema::PrimitiveType_ArgMax;
    auto attr = std::make_unique<schema::ArgMaxT>();
    CopyAttrForArgMinMax<schema::ArgMaxT>(attr.get(), reinterpret_cast<schema::ArgMaxT *>(prim->value.value));
    primitive->value.value = attr.release();
  } else {
    primitive->value.type = schema::PrimitiveType_ArgMin;
    auto attr = std::make_unique<schema::ArgMinT>();
    CopyAttrForArgMinMax<schema::ArgMinT>(attr.get(), reinterpret_cast<schema::ArgMinT *>(prim->value.value));
    primitive->value.value = attr.release();
  }
  auto primitive_c = PrimitiveC::Create(primitive.release());
  auto value_node = NewValueNode(std::shared_ptr<PrimitiveC>(primitive_c));
  inputs->erase(inputs->begin());
  inputs->insert(inputs->begin(), value_node);
  return lite::RET_OK;
}

int MindirInputAdjustOpPass::BuildCNodeForArgMinMax(const FuncGraphPtr &graph, const CNodePtr &tuple_get_item,
                                                    const CNodePtr &argmin_max) {
  MS_ASSERT(graph != nullptr && tuple_get_item != nullptr && argmin_max != nullptr);
  auto inputs = argmin_max->inputs();
  if (CopyPrimitiveCForArgMinMax(&inputs) != lite::RET_OK) {
    MS_LOG(ERROR) << "copy argmin or argmax failed.";
    return lite::RET_ERROR;
  }
  if (AdjustArgMinMaxInputs(&inputs, false) != lite::RET_OK) {
    MS_LOG(ERROR) << "adjust argmin or argmax attr failed.";
    return lite::RET_ERROR;
  }
  auto new_cnode = graph->NewCNode(inputs);
  new_cnode->set_fullname_with_scope(argmin_max->fullname_with_scope() + "_index");
  auto type_ptr = TypeIdToType(kTypeUnknown);
  std::vector<int64_t> shape_vector;
  new_cnode->set_abstract(std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector));
  auto manager = graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->Replace(tuple_get_item, new_cnode);
  return lite::RET_OK;
}

int MindirInputAdjustOpPass::AdjustArgMinMax(const FuncGraphPtr &graph, const CNodePtr &tuple_get_item,
                                             const CNodePtr &argmin_max) {
  MS_ASSERT(graph != nullptr && tuple_get_item != nullptr && argmin_max != nullptr);
  auto inputs = argmin_max->inputs();
  if (AdjustArgMinMaxInputs(&inputs, true) != lite::RET_OK) {
    MS_LOG(ERROR) << "adjust argmin or argmax attr failed.";
    return lite::RET_ERROR;
  }
  auto type_ptr = TypeIdToType(kTypeUnknown);
  std::vector<int64_t> shape_vector;
  auto abtract_tensor = std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
  argmin_max->set_abstract(abtract_tensor);
  auto manager = graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->Replace(tuple_get_item, argmin_max);
  return lite::RET_OK;
}

int MindirInputAdjustOpPass::AdjustTupleGetItemWithArgMinMax(const FuncGraphPtr &graph, const CNodePtr &cnode) {
  MS_ASSERT(graph != nullptr && cnode != nullptr);
  auto inputs = cnode->inputs();
  if (inputs.size() != 3) {
    MS_LOG(ERROR) << "tupleGetItem inputs size is invalid: " << inputs.size();
    return lite::RET_ERROR;
  }
  auto argmin_max = inputs.at(1);
  MS_ASSERT(argmin_max != nullptr);
  auto argmin_max_cnode = argmin_max->cast<CNodePtr>();
  if (argmin_max_cnode == nullptr) {
    MS_LOG(ERROR) << "the second input is not a cnode.";
    return lite::RET_ERROR;
  }
  if (!CheckCNodeIsArgMinMax(argmin_max_cnode)) {
    MS_LOG(DEBUG) << "tuple_get_item first input is not argmin and argmax.";
    return lite::RET_OK;
  }
  auto index_vnode = inputs.at(2);
  auto value_node = index_vnode->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "TupleGetItem's input 2 is not valuenode";
    return lite::RET_ERROR;
  }
  int index = lite::CastToInt(value_node->value()).front();
  if (index == 0) {
    if (BuildCNodeForArgMinMax(graph, cnode, argmin_max_cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "build new cnode failed.";
      return lite::RET_ERROR;
    }
  } else if (index == 1) {
    if (AdjustArgMinMax(graph, cnode, argmin_max_cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "adjust argmin_max failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

bool MindirInputAdjustOpPass::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto manager = Manage(graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return lite::RET_NULL_PTR;
  }
  auto node_list = TopoSort(graph->get_return());
  int status = lite::RET_OK;
  for (auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      MS_LOG(DEBUG) << "node is not cnode.";
      continue;
    }
    auto type = opt::GetCNodeType(node);
    if (type == schema::PrimitiveType_TupleGetItem) {
      status = AdjustTupleGetItemWithArgMinMax(graph, cnode);
    }
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "adjust input pass is failed.";
      return false;
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
