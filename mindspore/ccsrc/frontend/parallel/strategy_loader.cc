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

#include "frontend/parallel/strategy_loader.h"

#include <vector>
#include <string>
#include <memory>
#include "primitive/other_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/framework_ops.h"
#include "include/utils/parallel_context.h"
#include "frontend/operator/ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "include/frontend/parallel/tensor_layout/tensor_transform.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/strategy_utils.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace parallel {
// handle normal layout
Status ProcessInLayout(const std::string &strategy_key_name, const TensorLayoutValueMap &layout_map,
                       const OperatorInfoPtr &operator_info,
                       std::vector<std::shared_ptr<TensorLayout>> *tensor_layouts) {
  auto layout_value_tuple = layout_map.at(strategy_key_name);
  std::vector<ValuePtr> layout_value_vector = layout_value_tuple->value();
  if (operator_info->inputs_shape().size() != layout_value_vector.size()) {
    MS_LOG(ERROR) << "The in_layout configured for node is not equal to its input nums";
    return FAILED;
  }

  for (size_t i = 0; i < layout_value_vector.size(); ++i) {
    auto layout_item = layout_value_vector[i];
    std::vector<std::string> alias_name;
    std::vector<int64_t> device_matrix_vector;
    std::vector<std::vector<int64_t>> tensor_map_vector;
    bool interleaved_parallel;
    if (GetLayoutFromAttrValue(layout_item, &alias_name, &device_matrix_vector, &tensor_map_vector,
                               &interleaved_parallel) != SUCCESS) {
      return FAILED;
    }

    auto layout = std::make_shared<TensorLayout>();
    if (layout->InitFromExtendVector(device_matrix_vector, tensor_map_vector, operator_info->inputs_shape()[i],
                                     interleaved_parallel) != SUCCESS) {
      MS_LOG(ERROR) << "The in_layout configured incorrect, device_matrix:" << device_matrix_vector
                    << ", tensor_map:" << tensor_map_vector;
      return FAILED;
    }
    tensor_layouts->push_back(layout);
  }
  return SUCCESS;
}

Status ProcessOutLayout(const std::string &strategy_key_name, const TensorLayoutValueMap &layout_map,
                        const OperatorInfoPtr &operator_info,
                        std::vector<std::shared_ptr<TensorLayout>> *tensor_layouts) {
  auto layout_value_tuple = layout_map.at(strategy_key_name);
  std::vector<ValuePtr> layout_value_vector = layout_value_tuple->value();
  if (operator_info->outputs_shape().size() != layout_value_vector.size()) {
    MS_LOG(ERROR) << "The in_layout configured for node is not equal to its input nums";
    return FAILED;
  }

  for (size_t i = 0; i < layout_value_vector.size(); ++i) {
    auto layout_item = layout_value_vector[i];
    std::vector<std::string> alias_name;
    std::vector<int64_t> device_matrix_vector;
    std::vector<std::vector<int64_t>> tensor_map_vector;
    bool interleaved_parallel;
    if (GetLayoutFromAttrValue(layout_item, &alias_name, &device_matrix_vector, &tensor_map_vector,
                               &interleaved_parallel) != SUCCESS) {
      return FAILED;
    }

    auto layout = std::make_shared<TensorLayout>();
    if (layout->InitFromExtendVector(device_matrix_vector, tensor_map_vector, operator_info->outputs_shape()[i],
                                     interleaved_parallel) != SUCCESS) {
      MS_LOG(ERROR) << "The in_layout configured incorrect, device_matrix:" << device_matrix_vector
                    << ", tensor_map:" << tensor_map_vector;
      return FAILED;
    }
    tensor_layouts->push_back(layout);
  }
  return SUCCESS;
}

// load normal layout
Status LoadTensorLayouts(const std::string &strategy_key_name, const TensorLayoutValueMap &layout_map,
                         const TensorLayoutValueMap &out_layout_map, const OperatorInfoPtr &operator_info,
                         std::vector<std::shared_ptr<TensorLayout>> *in_tensor_layouts,
                         std::vector<std::shared_ptr<TensorLayout>> *out_tensor_layouts) {
  if (layout_map.find(strategy_key_name) != layout_map.end()) {
    if (ProcessInLayout(strategy_key_name, layout_map, operator_info, in_tensor_layouts) != SUCCESS) {
      return FAILED;
    }

    if (out_layout_map.find(strategy_key_name) != out_layout_map.end()) {
      if (ProcessOutLayout(strategy_key_name, out_layout_map, operator_info, out_tensor_layouts) != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

// load newshape layout
Status LoadNewShapeTensorLayouts(const std::string &strategy_key_name, const TensorLayoutValueMap &layoutnewshape_map,
                                 const TensorLayoutValueMap &out_layoutnewshape_map,
                                 const OperatorInfoPtr &operator_info,
                                 std::vector<TensorLayoutBasePtr> *in_tensor_layouts_new,
                                 std::vector<TensorLayoutBasePtr> *out_tensor_layouts_new) {
  if (layoutnewshape_map.find(strategy_key_name) != layoutnewshape_map.end()) {
    auto layout_value_tuple = layoutnewshape_map.at(strategy_key_name);
    if (ConvertValueTupleToTensorLayoutVector(layout_value_tuple, operator_info->inputs_shape_new(),
                                              in_tensor_layouts_new) != SUCCESS) {
      return FAILED;
    }

    if (out_layoutnewshape_map.find(strategy_key_name) != out_layoutnewshape_map.end()) {
      auto out_layout_value_tuple = out_layoutnewshape_map.at(strategy_key_name);
      if (ConvertValueTupleToTensorLayoutVector(out_layout_value_tuple, operator_info->outputs_shape_new(),
                                                out_tensor_layouts_new) != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

// load strategy
Status LoadStrategies(const std::string &strategy_key_name, const StrategyMap &stra_map,
                      const StrategyMap &out_stra_map, const OperatorInfoPtr &operator_info, StrategyPtr *s_strategy,
                      StrategyPtr *o_strategy) {
  if (stra_map.find(strategy_key_name) != stra_map.end()) {
    *s_strategy = stra_map.at(strategy_key_name);
    operator_info->SetSelectedStrategy(*s_strategy, 0);

    if (out_stra_map.find(strategy_key_name) != out_stra_map.end()) {
      *o_strategy = out_stra_map.at(strategy_key_name);
      operator_info->set_out_strategy(*o_strategy);
    }
  }
  return SUCCESS;
}

// init OperatorInfo with layout and strategy
Status InitOperatorInfo(const std::string &strategy_key_name, const TensorLayoutValueMap &layoutnewshape_map,
                        const OperatorInfoPtr &operator_info,
                        const std::vector<TensorLayoutBasePtr> &in_tensor_layouts_new,
                        const std::vector<TensorLayoutBasePtr> &out_tensor_layouts_new,
                        const std::vector<std::shared_ptr<TensorLayout>> &in_tensor_layouts,
                        const std::vector<std::shared_ptr<TensorLayout>> &out_tensor_layouts, StrategyPtr s_strategy,
                        StrategyPtr o_strategy) {
  operator_info->set_auto_parallel(false);
  Status initRet = FAILED;
  if (layoutnewshape_map.find(strategy_key_name) != layoutnewshape_map.end()) {
    initRet = operator_info->Init(s_strategy, o_strategy, in_tensor_layouts_new, out_tensor_layouts_new);
  } else {
    initRet = operator_info->Init(s_strategy, o_strategy, in_tensor_layouts, out_tensor_layouts);
  }

  if (initRet == SUCCESS) {
    MS_LOG(INFO) << "Init selected strategy succeeded.";
  } else {
    MS_LOG(ERROR) << "Init selected strategy failed.";
  }
  return initRet;
}

Status StrategyLoader::LoadStrategyFromFile(const std::vector<AnfNodePtr> &all_nodes) {
  std::string strategy_search_mode = ParallelContext::GetInstance()->strategy_search_mode();
  if (strategy_search_mode != kShardingPropagation) {
    MS_LOG(EXCEPTION) << "Current mode: " << strategy_search_mode << " doesn't support load strategy.";
  }
  if (!StrategyCheckpoint::GetInstance().LoadAutoOpStrategyOn()) {
    return FAILED;
  }

  StrategyMap stra_map;
  StrategyMap out_stra_map;
  TensorLayoutValueMap layout_map;
  TensorLayoutValueMap out_layout_map;
  TensorLayoutValueMap layoutnewshape_map;
  TensorLayoutValueMap out_layoutnewshape_map;
  if (StrategyCheckpoint::GetInstance().LoadAutoOpStrategy(&stra_map, &out_stra_map, &layout_map, &out_layout_map,
                                                           &layoutnewshape_map, &out_layoutnewshape_map) != SUCCESS) {
    return FAILED;
  }

  MS_LOG(INFO) << "Load strategies map from json successfully";

  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || !StrategyUtils::CheckExtractInformation(cnode) || IsPrimitiveCNode(node, prim::kPrimSend)) {
      continue;
    }

    OperatorInfoPtr operator_info = CreateOperatorInfo(cnode);
    MS_EXCEPTION_IF_NULL(operator_info);

    std::string strategy_key_name = cnode->fullname_with_scope();
    if (stra_map.find(strategy_key_name) == stra_map.end() && layout_map.find(strategy_key_name) == layout_map.end() &&
        layoutnewshape_map.find(strategy_key_name) == layoutnewshape_map.end()) {
      MS_LOG_WITH_NODE(WARNING, node) << "Not found strategy for " << strategy_key_name;
      return FAILED;
    }

    std::vector<std::shared_ptr<TensorLayout>> in_tensor_layouts, out_tensor_layouts;
    std::vector<TensorLayoutBasePtr> in_tensor_layouts_new, out_tensor_layouts_new;

    if (LoadNewShapeTensorLayouts(strategy_key_name, layoutnewshape_map, out_layoutnewshape_map, operator_info,
                                  &in_tensor_layouts_new, &out_tensor_layouts_new) != SUCCESS) {
      return FAILED;
    }

    if (LoadTensorLayouts(strategy_key_name, layout_map, out_layout_map, operator_info, &in_tensor_layouts,
                          &out_tensor_layouts) != SUCCESS) {
      return FAILED;
    }

    StrategyPtr s_strategy = nullptr;
    StrategyPtr o_strategy = nullptr;
    if (LoadStrategies(strategy_key_name, stra_map, out_stra_map, operator_info, &s_strategy, &o_strategy) != SUCCESS) {
      return FAILED;
    }

    if (InitOperatorInfo(strategy_key_name, layoutnewshape_map, operator_info, in_tensor_layouts_new,
                         out_tensor_layouts_new, in_tensor_layouts, out_tensor_layouts, s_strategy,
                         o_strategy) != SUCCESS) {
      return FAILED;
    }

    cnode->set_user_data<OperatorInfo>(operator_info);
    cnode->AddAttr(OP_INFO_CREATED, MakeValue(true));
  }

  MS_LOG(INFO) << "End load strategies from file";
  return SUCCESS;
}

void StrategyLoader::SaveStrategyToFile(const std::vector<AnfNodePtr> &all_nodes) {
  std::string strategy_search_mode = ParallelContext::GetInstance()->strategy_search_mode();
  if (strategy_search_mode != kShardingPropagation) {
    MS_LOG(EXCEPTION) << "Current mode: " << strategy_search_mode << " doesn't support save strategy.";
  }
  if (!StrategyCheckpoint::GetInstance().SaveAutoOpStrategyOn() || GetRank() % DEVICE_NUM_PER_SERVER != 0) {
    return;
  }
  StrategyMap stra_map;
  StrategyMap out_stra_map;
  TensorLayoutValueMap layout_map;
  TensorLayoutValueMap out_layout_map;
  TensorLayoutValueMap layoutnewshape_map;
  TensorLayoutValueMap out_layoutnewshape_map;

  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || (!IsValueNode<Primitive>(cnode->input(0)))) {
      continue;
    }
    OperatorInfoPtr op = cnode->user_data<OperatorInfo>();
    if (op == nullptr) {
      continue;
    }
    std::string strategy_key_name = cnode->fullname_with_scope();
    // save layout_value_vector as value for key
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_attrs = prim->attrs();
    auto is_new_shape_base_node = IsSupportNewShapeBaseNode(cnode);
    // in layout
    if (prim_attrs.count(IN_LAYOUT) > 0) {
      auto layout_value = prim_attrs.at(IN_LAYOUT);
      if (!layout_value->isa<ValueSequence>()) {
        MS_LOG(ERROR) << "The in_layout configured for node is not a tuple";
        return;
      }
      auto layout_value_tuple = layout_value->cast<ValueTuplePtr>();
      if (!is_new_shape_base_node) {
        layout_map[strategy_key_name] = layout_value_tuple;
      } else {
        layoutnewshape_map[strategy_key_name] = layout_value_tuple;
      }
    }

    // out layout
    if (prim_attrs.count(OUT_LAYOUT) > 0) {
      auto layout_value = prim_attrs.at(OUT_LAYOUT);
      if (!layout_value->isa<ValueSequence>()) {
        MS_LOG(EXCEPTION) << "The in_layout configured for node is not a tuple";
      }
      auto layout_value_tuple = layout_value->cast<ValueTuplePtr>();
      if (!is_new_shape_base_node) {
        out_layout_map[strategy_key_name] = layout_value_tuple;
      } else {
        out_layoutnewshape_map[strategy_key_name] = layout_value_tuple;
      }
    }

    StrategyPtr s_strategy = op->selected_strategy();
    if (s_strategy != nullptr) {
      stra_map[strategy_key_name] = s_strategy;
    }
    StrategyPtr o_strategy = op->out_strategy();
    if (o_strategy != nullptr) {
      out_stra_map[strategy_key_name] = o_strategy;
    }
  }
  if (StrategyCheckpoint::GetInstance().SaveAutoOpStrategy(stra_map, out_stra_map, layout_map, out_layout_map,
                                                           layoutnewshape_map, out_layoutnewshape_map) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Save strategy checkpoint failed";
  }
  MS_LOG(INFO) << "Success save strategies to file.";
}
}  // namespace parallel
}  // namespace mindspore
