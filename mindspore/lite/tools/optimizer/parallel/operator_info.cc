/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/parallel/operator_info.h"
#include <algorithm>
#include "tools/optimizer/parallel/split_strategy.h"
#include "ops/concat.h"
#include "ops/addn.h"
#include "ops/tuple_get_item.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
bool is_any_none(const std::vector<int64_t> &split) {
  return std::any_of(split.begin(), split.end(), [](int64_t v) { return v == static_cast<int64_t>(NoSplit); });
}

bool is_any_not_none(const std::vector<int64_t> &split) {
  return std::any_of(split.begin(), split.end(), [](int64_t v) { return v != static_cast<int64_t>(NoSplit); });
}

std::shared_ptr<abstract::AbstractTensor> OperatorInfo::CreateFakeAbstractTensor() const {
  auto type_ptr = TypeIdToType(operator_type_id_);
  std::vector<int64_t> shape_vector;
  return std::make_shared<abstract::AbstractTensor>(type_ptr, shape_vector);
}

int OperatorInfo::CheckSplitResult(const AnfNodePtr &result_anf_node, const std::vector<AnfNodePtr> &split_results,
                                   int target_output_num) {
  if ((result_anf_node == nullptr) || (split_results.size() != IntToSize(target_output_num))) {
    MS_LOG(ERROR) << name_ << " : Make split cnode failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

void OperatorInfo::Init(const FuncGraphPtr &func_graph, const CNodePtr &cnode, int32_t fmk_type) {
  func_graph_ = func_graph;
  cnode_ = cnode;
  fmk_type_ = fmk_type;
  parallel_output_nodes_.clear();
}

int OperatorInfo::SetCNodeBackend() {
  for (size_t i = 0; i < strategy_.dev_num; ++i) {
    lite::DeviceType dt_type;
    MS_CHECK_LT(i, strategy_.dev_types.size(), lite::RET_ERROR);
    std::string type = strategy_.dev_types[i];
    MS_CHECK_LT(i, parallel_output_nodes_.size(), lite::RET_ERROR);
    auto post_node = parallel_output_nodes_[i];
    MS_CHECK_TRUE_RET(post_node != nullptr, lite::RET_ERROR);
    auto post_cnode = post_node->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(post_cnode != nullptr, lite::RET_ERROR);
    auto cnode = post_cnode->input(1)->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(cnode != nullptr, lite::RET_ERROR);
    auto type_iter = kSupportSplitedDevices.find(type);
    if (type_iter == kSupportSplitedDevices.end()) {
      MS_LOG(ERROR) << "SetCnodeBackend: unknown device type.";
      return lite::RET_ERROR;
    }
    if (type_iter->second == lite::DeviceType::DT_NPU) {
      MS_LOG(ERROR) << "SetCnodeBackend: unsupported device type npu.";
      return lite::RET_ERROR;
    }
    dt_type = type_iter->second;
    cnode->AddAttr(mindspore::ops::kDeviceType, MakeValue(static_cast<int>(dt_type)));
  }
  return lite::RET_OK;
}

int OperatorInfo::CheckStrategyValue() {
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

int OperatorInfo::CreateMultipleOutputsOfAnfNode(const AnfNodePtr &node, size_t output_num,
                                                 std::vector<AnfNodePtr> *outputs) {
  MS_ASSERT(node != nullptr && outputs != nullptr);
  AbstractBasePtrList ptr_list;
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    MS_LOG(ERROR) << name_ << " : Failed to get CNode.";
    return lite::RET_ERROR;
  }
  for (size_t i = 0; i < output_num; ++i) {
    auto idx = NewValueNode(SizeToInt(i));
    auto index = std::make_shared<Int32Imm>(SizeToInt(i));
    MS_CHECK_TRUE_RET(index != nullptr, lite::RET_ERROR);
    auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(index);
    MS_CHECK_TRUE_RET(abstract_scalar != nullptr, lite::RET_ERROR);
    idx->set_abstract(abstract_scalar);
    auto tuple_node = std::make_shared<ops::TupleGetItem>();
    if (tuple_node == nullptr) {
      MS_LOG(ERROR) << name_ << " : Failed to create tuple node.";
      return lite::RET_ERROR;
    }
    auto tuple_prim_c = tuple_node->GetPrim();
    if (tuple_prim_c == nullptr) {
      MS_LOG(ERROR) << name_ << " : Failed to create tuple node primitive.";
      return lite::RET_ERROR;
    }
    auto tuple_getitem = func_graph_->NewCNode({NewValueNode(tuple_prim_c), node, idx});
    if (tuple_getitem == nullptr) {
      MS_LOG(ERROR) << name_ << " : Failed to create output nodes.";
      return lite::RET_ERROR;
    }
    tuple_getitem->set_fullname_with_scope(cnode->fullname_with_scope() + "_TupleGetItem" + std::to_string(i));
    outputs->push_back(tuple_getitem);
    auto abstract_tensor = CreateFakeAbstractTensor();
    ptr_list.push_back(abstract_tensor);
  }
  node->set_abstract(std::make_shared<abstract::AbstractTuple>(ptr_list));
  return lite::RET_OK;
}

AnfNodePtr OperatorInfo::CreateConcateNode(const CNodePtr &orig_node, const std::vector<AnfNodePtr> &input_nodes,
                                           int32_t concat_dim, size_t input_nodes_num) {
  MS_ASSERT(orig_node != nullptr);
  if (input_nodes.size() != input_nodes_num) {
    MS_LOG(ERROR) << name_ << " : Input nodes size of concat is not equal to input nodes number.";
    return nullptr;
  }
  auto concat_prim = std::make_shared<ops::Concat>();
  MS_CHECK_TRUE_RET(concat_prim != nullptr, nullptr);
  auto concat_prim_c = concat_prim->GetPrim();
  MS_CHECK_TRUE_RET(concat_prim_c != nullptr, nullptr);
  concat_prim->set_axis(concat_dim);
  auto value_node = NewValueNode(concat_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
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
  std::vector<AnfNodePtr> outputs;
  (void)CreateMultipleOutputsOfAnfNode(concat_cnode, 1, &outputs);
  return concat_cnode;
}

AnfNodePtr OperatorInfo::CreateReduceNode(const CNodePtr &orig_node, const std::vector<AnfNodePtr> &input_nodes,
                                          size_t input_nodes_num) {
  MS_ASSERT(orig_node != nullptr);
  if (input_nodes.size() != input_nodes_num) {
    MS_LOG(ERROR) << name_ << " : Input nodes size of reduce is not equal to input nodes number.";
    return nullptr;
  }
  // addup inputs element-wise
  auto addn_prim = std::make_shared<ops::AddN>();
  MS_CHECK_TRUE_RET(addn_prim != nullptr, nullptr);
  auto addn_prim_c = addn_prim->GetPrim();
  MS_CHECK_TRUE_RET(addn_prim_c != nullptr, nullptr);
  auto value_node = NewValueNode(addn_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);
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

int OperatorInfo::DoSplit() {
  if (CheckStrategyValue() != lite::RET_OK) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy values.";
    return lite::RET_ERROR;
  }
  if (CheckStrategy(strategy_) != lite::RET_OK) {
    MS_LOG(ERROR) << name_ << ": Check strategys failed.";
    return lite::RET_ERROR;
  }
  if (InferParallelCNodes() != lite::RET_OK) {
    MS_LOG(ERROR) << name_ << ": InferParallelCNodes failed.";
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
