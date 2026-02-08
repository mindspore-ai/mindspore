/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/strategy_utils.h"

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>

#include "primitive/sequence_ops.h"
#include "primitive/other_ops.h"
#include "primitive/array_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/framework_ops.h"
#include "frontend/operator/ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "include/utils/parallel_context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "include/frontend/parallel/tensor_layout/tensor_transform.h"
#include "frontend/parallel/strategy_loader.h"
#include "mindspore/ccsrc/frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
namespace {
std::pair<std::vector<Shapes>, std::vector<NewShapes>> ObtainShape(const CNodePtr &node) {
  std::vector<Shapes> shape_list;
  std::vector<NewShapes> new_shape_list;
  if (IsSupportNewShapeBaseNode(node)) {
    new_shape_list = ExtractNewShape(node);
  } else {
    shape_list = ExtractShape(node);
  }
  return std::make_pair(shape_list, new_shape_list);
}

ValuePtr ObtainStrategyForNewShapes(const ShapeBasePtr &shape, const int64_t &dev_num) {
  ValuePtr stra_value_ptr;
  if (shape->is_list()) {
    std::vector<ValuePtr> elements;
    for (size_t i = 0; i < shape->size(); ++i) {
      auto value_stra = ObtainStrategyForNewShapes(shape->GetElement(SizeToLong(i)), dev_num);
      elements.emplace_back(value_stra);
    }
    stra_value_ptr = std::make_shared<ValueTuple>(elements);
  } else {
    Dimensions stra;
    stra.push_back(dev_num);
    for (size_t j = 1; j < shape->size(); ++j) {
      stra.push_back(1);
    }
    stra_value_ptr = MakeValue(stra);
  }
  return stra_value_ptr;
}

void ObtainElementsForStrategyNewShape(const std::vector<NewShapes> &new_shape_list, const int64_t &dev_num,
                                       std::vector<ValuePtr> *elements) {
  for (size_t i = 0; i < new_shape_list[0].size(); i++) {
    if (new_shape_list[0][i]->empty()) {
      (void)elements->emplace_back(MakeValue(Dimensions()));
      continue;
    }
    auto input_strategy = ObtainStrategyForNewShapes(new_shape_list[0][i], dev_num);
    (void)elements->emplace_back(MakeValue(input_strategy));
  }
}

void ObtainElementsForStrategy(const std::vector<Shapes> &shape_list, const int64_t &dev_num,
                               std::vector<ValuePtr> *elements) {
  for (size_t i = 0; i < shape_list[0].size(); i++) {
    if (shape_list[0][i].empty()) {
      (void)elements->emplace_back(MakeValue(Dimensions()));
      continue;
    }
    Dimensions input_strategy;
    input_strategy.push_back(dev_num);
    if (shape_list[0][i][0] > 0 && shape_list[0][i][0] % dev_num != 0) {
      MS_LOG(EXCEPTION) << "The shapes of dataset is " << shape_list[0]
                        << ", the batch dim can not be evenly div by dev_num " << dev_num;
    }
    for (size_t j = 1; j < shape_list[0][i].size(); j++) {
      input_strategy.push_back(1);
    }
    (void)elements->emplace_back(MakeValue(input_strategy));
  }
}
}  // namespace

bool StrategyUtils::CheckExtractInformation(const CNodePtr &cnode) {
  if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }

  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  if (!prim || (prim->name() == MAKE_TUPLE) || (prim->name() == MAKE_LIST) || (prim->name() == RECEIVE)) {
    return false;
  }

  return IsParallelCareNode(cnode);
}

template <typename T>
ValuePtr CreateValuePtrFromVector(const std::vector<T> &vec) {
  std::vector<ValuePtr> value_vec;
  std::transform(vec.begin(), vec.end(), std::back_inserter(value_vec),
                 [](const auto &value) { return MakeValue<T>(value); });
  return std::make_shared<ValueTuple>(value_vec);
}

void StrategyUtils::SetDatasetLayout(const CNodePtr &node, const std::string &attrName) {
  auto prim = GetValueNode<PrimitivePtr>(node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto dataset_strategy_tensormap = ParallelContext::GetInstance()->dataset_strategy_tensormap();
  auto dataset_strategy_devmat = ParallelContext::GetInstance()->dataset_strategy_devmat();
  auto dataset_strategy_alias_name = ParallelContext::GetInstance()->dataset_strategy_alias_name();

  auto alias_name_key = MakeValue<std::string>(ALIAS_NAME);
  auto tensormap_key = MakeValue<std::string>(TENSOR_MAP);
  auto devmat_key = MakeValue<std::string>(DEVICE_MATRIX);
  auto interleaved_key = MakeValue<std::string>(INTERLEAVED_PARALLEL);

  std::vector<ValuePtr> layout_vec;
  for (size_t i = 0; i < dataset_strategy_tensormap.size(); ++i) {
    auto interleaved_parallel_kv = std::make_pair(interleaved_key, MakeValue<bool>(false));
    auto alias_name_kv = std::make_pair(alias_name_key, CreateValuePtrFromVector(dataset_strategy_alias_name.at(i)));
    auto devmat_kv = std::make_pair(devmat_key, CreateValuePtrFromVector(dataset_strategy_devmat.at(i)));
    std::vector<ValuePtr> sub_tensor_map_vector;
    for (size_t j = 0; j < dataset_strategy_tensormap.at(i).size(); ++j) {
      sub_tensor_map_vector.push_back(CreateValuePtrFromVector(dataset_strategy_tensormap.at(i).at(j)));
    }
    auto tensormap_kv = std::make_pair(tensormap_key, std::make_shared<ValueTuple>(sub_tensor_map_vector));
    std::vector<std::pair<ValuePtr, ValuePtr>> layout_dict = {alias_name_kv, interleaved_parallel_kv, devmat_kv,
                                                              tensormap_kv};
    layout_vec.emplace_back(std::make_shared<ValueDictionary>(layout_dict));
  }
  auto layout_dict_value = std::make_shared<ValueTuple>(layout_vec);
  prim->set_attr(attrName, layout_dict_value);
  return;
}

void StrategyUtils::SetGetNextLayout(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node->input(0));
  MS_EXCEPTION_IF_NULL(prim);

  if (prim->name() == GET_NEXT && !ParallelContext::GetInstance()->dataset_strategy_tensormap().empty() &&
      !ParallelContext::GetInstance()->dataset_strategy_devmat().empty()) {
    MS_LOG(INFO) << "Set layout to node " << node->DebugString();
    SetDatasetLayout(node, OUT_LAYOUT);
  }
}

void StrategyUtils::SetVirtualDatasetStrategy(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool full_batch = ParallelContext::GetInstance()->full_batch();

  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->name() == VIRTUAL_DATA_SET || prim->name() == VIRTUAL_OUTPUT) {
    if (!ParallelContext::GetInstance()->dataset_strategy_tensormap().empty() &&
        !ParallelContext::GetInstance()->dataset_strategy_devmat().empty() && prim->name() == VIRTUAL_DATA_SET) {
      MS_LOG(INFO) << "Set layout for virtual dataset";
      return SetDatasetLayout(node, IN_LAYOUT);
    }
    CheckGlobalDeviceManager();
    auto attrs_temp = prim->attrs();
    if (!ParallelContext::GetInstance()->dataset_strategy().empty() && prim->name() == VIRTUAL_DATA_SET) {
      std::vector<ValuePtr> elements;
      auto dataset_strategy = ParallelContext::GetInstance()->dataset_strategy();
      (void)std::transform(dataset_strategy.begin(), dataset_strategy.end(), std::back_inserter(elements),
                           [](auto input_stra) { return MakeValue(input_stra); });
      ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
      attrs_temp[IN_STRATEGY] = strategy;
      (void)prim->SetAttrs(attrs_temp);
      if (prim->HasAttr(REPEAT_DIM_DIRECT) && GetValue<std::string>(prim->GetAttr(REPEAT_DIM_DIRECT)) == RIGHT) {
        ParallelContext::GetInstance()->set_dataset_repeat_dim_right(true);
        MS_LOG(INFO) << "dataset repeat dim is right";
      }
      return;
    }
    int64_t dev_num;
    if (full_batch) {
      dev_num = 1;
    } else {
      dev_num = g_device_manager->stage_device_num();
    }
    if (dev_num == 0) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Device Num must be larger than 0, but got 0.";
    }
    std::vector<Shapes> shape_list;
    std::vector<NewShapes> new_shape_list;
    if (IsForwardDynamicShape()) {
      shape_list = ExtractRealDivisor(node);
      MS_LOG(INFO) << "The node is in dynamic shape graph, the real divisor is " << ShapesToString(shape_list[0]);
    } else {
      std::tie(shape_list, new_shape_list) = ObtainShape(node);
    }
    if (shape_list.empty() && new_shape_list.empty()) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Failure:node " << node->ToString() << " failed to extract shape";
    }
    std::vector<ValuePtr> elements;
    if (new_shape_list.empty()) {
      ObtainElementsForStrategy(shape_list, dev_num, &elements);
    } else {
      ObtainElementsForStrategyNewShape(new_shape_list, dev_num, &elements);
    }
    ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
    attrs_temp[IN_STRATEGY] = strategy;
    (void)prim->SetAttrs(attrs_temp);
  }
}

void StrategyUtils::ExtractStrategyAndInit(const CNodePtr &cnode, const PrimitivePtr &prim,
                                           const OperatorInfoPtr &op_info) {
  StrategyPtr in_strategy = nullptr;
  StrategyPtr out_strategy = nullptr;
  auto attrs = prim->attrs();

  // load strategy map from checkpoint
  StrategyMap stra_map;
  if (StrategyCheckpoint::GetInstance().LoadCheckPointOn() &&
      (StrategyCheckpoint::GetInstance().Load(&stra_map) != SUCCESS)) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode)
      << "Load strategy checkpoint failed. Please check if the file exists and the file permission.";
  }

  std::string strategy_key_name = "";
  auto param_names = NodeParameterName(cnode, -1, 0);
  if (!param_names.empty()) {
    strategy_key_name = prim->name() + "_" + param_names[0].first;
  }
  auto is_new_shape_base_node = IsSupportNewShapeBaseNode(cnode);
  std::vector<std::shared_ptr<TensorLayout>> in_tensor_layouts;
  std::vector<std::shared_ptr<TensorLayout>> out_tensor_layouts;
  std::vector<TensorLayoutBasePtr> in_tensor_layouts_new;
  std::vector<TensorLayoutBasePtr> out_tensor_layouts_new;
  if (is_new_shape_base_node) {
    if (ExtractUserConfigLayoutForNewShape(attrs, op_info->inputs_shape_new(), op_info->outputs_shape_new(),
                                           &in_tensor_layouts_new, &out_tensor_layouts_new) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Failure:operator " << prim->name() << " extract configured layout failed"
                        << trace::DumpSourceLines(cnode);
    }
    for (size_t i = 0; i < in_tensor_layouts_new.size(); ++i) {
      auto in_layouts = in_tensor_layouts_new[i]->GetAllElements();
      in_tensor_layouts.insert(in_tensor_layouts.end(), in_layouts.begin(), in_layouts.end());
    }
    for (size_t i = 0; i < out_tensor_layouts_new.size(); ++i) {
      auto out_layouts = out_tensor_layouts_new[i]->GetAllElements();
      out_tensor_layouts.insert(out_tensor_layouts.end(), out_layouts.begin(), out_layouts.end());
    }
  } else {
    if (ExtractUserConfigLayout(attrs, op_info->inputs_shape(), op_info->outputs_shape(), &in_tensor_layouts,
                                &out_tensor_layouts) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Failure:operator " << prim->name() << " extract configured layout failed"
                        << trace::DumpSourceLines(cnode);
    }
  }
  if (in_tensor_layouts.empty() && out_tensor_layouts.empty() && in_tensor_layouts_new.empty() &&
      out_tensor_layouts_new.empty()) {
    ObtainInOutStrategy(stra_map, prim, strategy_key_name, cnode, op_info, is_new_shape_base_node, &in_strategy,
                        &out_strategy);
    PaddingStrategy(op_info, is_new_shape_base_node, &in_strategy);
  } else if (CheckLayoutForDynamicShape(in_tensor_layouts, out_tensor_layouts, op_info) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "check layout for dynamic shape failed";
  }
  if (is_new_shape_base_node) {
    if (op_info->Init(in_strategy, out_strategy, in_tensor_layouts_new, out_tensor_layouts_new) == FAILED) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Failure:operator " << prim->name() << " init failed"
                                         << trace::DumpSourceLines(cnode);
    }
  } else {
    if (op_info->Init(in_strategy, out_strategy, in_tensor_layouts, out_tensor_layouts) == FAILED) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "Failure:operator " << prim->name() << " init failed"
                                         << trace::DumpSourceLines(cnode);
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
