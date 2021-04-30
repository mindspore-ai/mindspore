/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/optimizer/parallel/operator_info.h"
#include <algorithm>
#include "tools/converter/ops/ops_def.h"
#include "tools/optimizer/parallel/split_strategy.h"
#include "mindspore/core/ops/concat.h"
#include "mindspore/core/ops/addn.h"
#include "mindspore/core/ops/split.h"
#include "include/lite_types.h"
#include "mindspore/ccsrc/utils/utils.h"
#include "base/core_ops.h"
#include "include/errorcode.h"

namespace mindspore {
namespace opt {
bool is_any_none(const std::vector<int64_t> &split) {
  return std::any_of(split.begin(), split.end(), [](int64_t v) { return v == static_cast<int64_t>(NoSplit); });
}

bool is_any_not_none(const std::vector<int64_t> &split) {
  return std::any_of(split.begin(), split.end(), [](int64_t v) { return v != static_cast<int64_t>(NoSplit); });
}

lite::STATUS OperatorInfo::SetCNodeBackend() {
  for (size_t i = 0; i < strategy_.dev_num; ++i) {
    lite::DeviceType dt_type;
    std::string type = strategy_.dev_types[i];
    auto cnode = parallel_output_nodes_[i]->cast<CNodePtr>()->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (type == "GPU") {
      dt_type = lite::DeviceType::DT_GPU;
    } else if (type == "CPU") {
      dt_type = lite::DeviceType::DT_CPU;
    } else if (type == "NPU") {
      dt_type = lite::DeviceType::DT_NPU;
    } else {
      MS_LOG(ERROR) << "SetCnodeBackend: unknown device type.";
      return lite::RET_ERROR;
    }
    cnode->AddAttr(mindspore::ops::kDeviceType, MakeValue(static_cast<int>(dt_type)));
  }
  return lite::RET_OK;
}

lite::STATUS OperatorInfo::CheckStrategyValue() {
  auto strategy_size = strategy_.strategys.size();

  for (size_t index = 0; index < strategy_size; ++index) {
    auto strategy = strategy_.strategys[index];
    for (const auto &s : strategy) {
      if (s.size() != IntToSize(strategy_.dev_num)) {
        MS_LOG(ERROR) << "Strategy split number:" << s.size()
                      << " is not equal to device number: " << strategy_.dev_num;
        return lite::RET_ERROR;
      }
      if (is_any_not_none(s) && is_any_none(s)) {
        MS_LOG(ERROR) << "Strategy split number must be all zero or all non-zero: " << s;
        return lite::RET_ERROR;
      }
    }
  }
  return lite::RET_OK;
}

lite::STATUS OperatorInfo::CreateMultipleOutputsOfAnfNode(const AnfNodePtr &node, size_t output_num,
                                                          std::vector<AnfNodePtr> *outputs) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(outputs);
  AbstractBasePtrList ptr_list;
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(ERROR) << name_ << " : Failed to get CNode.";
    return lite::RET_ERROR;
  }

  for (size_t i = 0; i < output_num; ++i) {
    auto idx = NewValueNode(SizeToInt(i));
    MS_ASSERT(idx);
    auto index = std::make_shared<Int32Imm>(SizeToInt(i));
    auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(index);
    idx->set_abstract(abstract_scalar);
    auto tuple_getitem = func_graph_->NewCNode({NewValueNode(std::make_shared<lite::TupleGetItem>()), node, idx});
    if (tuple_getitem == nullptr) {
      MS_LOG(ERROR) << name_ << " : Failed to create output nodes.";
      return lite::RET_ERROR;
    }
    tuple_getitem->set_fullname_with_scope(cnode->fullname_with_scope() + "_TupleGetItem" + std::to_string(i));
    outputs->push_back(tuple_getitem);
    ptr_list.push_back(abstract_scalar);
  }
  node->set_abstract(std::make_shared<abstract::AbstractTuple>(ptr_list));
  return lite::RET_OK;
}

AnfNodePtr OperatorInfo::CreateOutputsOfSplit(const CNodePtr &orig_node, size_t input_index,
                                              std::vector<AnfNodePtr> *split_outputs, size_t split_dim,
                                              size_t split_num, const std::vector<int64_t> &splits, bool trans_format) {
  MS_EXCEPTION_IF_NULL(orig_node);

  auto split_prim = std::make_shared<ops::Split>();
  split_prim->set_output_num(split_num);
  split_prim->set_size_splits(splits);
  split_prim->set_axis(split_dim);
  auto value_node = NewValueNode(split_prim);
  std::vector<AnfNodePtr> split_inputs = {value_node};
  split_inputs.push_back(orig_node->input(input_index + 1));
  auto split_cnode = func_graph_->NewCNode(split_inputs);
  if (split_cnode == nullptr) {
    MS_LOG(ERROR) << name_ << " : Failed to create split node.";
    return nullptr;
  }
  split_cnode->set_fullname_with_scope("Split_" + name_);
  CreateMultipleOutputsOfAnfNode(split_cnode, split_num, split_outputs);

  return split_cnode;
}

AnfNodePtr OperatorInfo::CreateConcateNode(const CNodePtr &orig_node, const std::vector<AnfNodePtr> &input_nodes,
                                           int32_t concat_dim, size_t input_nodes_num, bool trans_format) {
  MS_EXCEPTION_IF_NULL(orig_node);

  if (input_nodes.size() != input_nodes_num) {
    MS_LOG(ERROR) << name_ << " : Input nodes size of concat is not equal to input nodes number.";
    return nullptr;
  }
  auto concat_prim = std::make_shared<ops::Concat>();
  concat_prim->set_axis(concat_dim);
  auto value_node = NewValueNode(concat_prim);
  std::vector<AnfNodePtr> concat_inputs = {value_node};
  (void)std::transform(input_nodes.begin(), input_nodes.end(), std::back_inserter(concat_inputs),
                       [](const AnfNodePtr &p) { return p->cast<CNodePtr>()->input(1); });
  auto concat_cnode = func_graph_->NewCNode(concat_inputs);
  if (concat_cnode == nullptr) {
    MS_LOG(ERROR) << name_ << " : Failed to create concat node.";
    return nullptr;
  }
  concat_cnode->set_fullname_with_scope("Concat_" + name_);
  concat_cnode->set_scope(orig_node->scope());

  return concat_cnode;
}

AnfNodePtr OperatorInfo::CreateReduceNode(const CNodePtr &orig_node, const std::vector<AnfNodePtr> &input_nodes,
                                          int32_t reduce_dim, size_t input_nodes_num, bool trans_format) {
  MS_EXCEPTION_IF_NULL(orig_node);

  if (input_nodes.size() != input_nodes_num) {
    MS_LOG(ERROR) << name_ << " : Input nodes size of reduce is not equal to input nodes number.";
    return nullptr;
  }
  // addup inputs element-wise
  auto addn_prim = std::make_shared<ops::AddN>();
  auto value_node = NewValueNode(addn_prim);
  std::vector<AnfNodePtr> addn_inputs = {value_node};
  (void)std::transform(input_nodes.begin(), input_nodes.end(), std::back_inserter(addn_inputs),
                       [](const AnfNodePtr &p) { return p->cast<CNodePtr>()->input(1); });
  auto addn_cnode = func_graph_->NewCNode(addn_inputs);
  if (addn_cnode == nullptr) {
    MS_LOG(ERROR) << name_ << " : Failed to create concat node.";
    return nullptr;
  }
  addn_cnode->set_fullname_with_scope("AddN_" + name_);
  addn_cnode->set_scope(orig_node->scope());

  return addn_cnode;
}

lite::STATUS OperatorInfo::Init() {
  if (GetAttrs() != lite::RET_OK) {
    MS_LOG(ERROR) << name_ << ": Parse attrs failed.";
    return lite::RET_ERROR;
  }
  if (CheckStrategyValue() != lite::RET_OK) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy values.";
    return lite::RET_ERROR;
  }
  if (CheckStrategy(strategy_) != lite::RET_OK) {
    MS_LOG(ERROR) << name_ << ": Check strategys failed.";
    return lite::RET_ERROR;
  }
  if (InferParallelCNodes() != lite::RET_OK) {
    MS_LOG(ERROR) << name_ << ": InferReplaceGraph failed.";
    return lite::RET_ERROR;
  }
  if (SetCNodeBackend() != lite::RET_OK) {
    MS_LOG(ERROR) << name_ << ": SetCnodeBackend failed.";
    return lite::RET_ERROR;
  }
  if (InferReplaceOp() != lite::RET_OK) {
    MS_LOG(ERROR) << name_ << ": InferForwardOps failed.";
    return lite::RET_ERROR;
  }

  return lite::RET_OK;
}

}  // namespace opt
}  // namespace mindspore
