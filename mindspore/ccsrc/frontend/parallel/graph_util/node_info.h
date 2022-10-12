/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_NODE_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_NODE_INFO_H_

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "base/base.h"
#include "ir/anf.h"
#include "frontend/parallel/ops_info/operator_info.h"

namespace mindspore {
namespace parallel {
using OperatorInfoPtr = std::shared_ptr<mindspore::parallel::OperatorInfo>;
std::string ParameterName(const AnfNodePtr &node_ptr);

bool ParameterRequireGrad(const AnfNodePtr &node_ptr);

size_t GetLengthOfDataType(const TypePtr &type);

std::vector<bool> ExtractInputParameterByNode(const CNodePtr &node);

std::vector<size_t> ExtractInputTypeLengthByNode(const CNodePtr &node);

std::vector<TypePtr> ExtractOutputTypeByNode(const CNodePtr &node);

std::vector<AnfNodePtr> FindParameterByRefKeyNode(const AnfNodePtr &node, const FuncGraphPtr &func_graph);

bool AnfNodeIsPrimitive(const AnfNodePtr &anf_node, const std::string &prim_name);

bool FindReshape(const CNodePtr &cnode, mindspore::HashSet<std::string> *op_cache);

bool FindReshapePreNodeStraCosts(const AnfNodePtr &node, OperatorInfoPtr *pre_operator_info, int64_t *out_index,
                                 size_t curr_depth);

void FindReshapeNextNodeStraCosts(const CNodePtr &cnode,
                                  std::vector<std::pair<OperatorInfoPtr, int64_t>> *next_ops_index,
                                  bool *is_next_reshape, size_t curr_depth);

void SetUserAttrs(const mindspore::HashMap<std::string, ValuePtr> &origin_prim_attrs, const PrimitivePtr &self_prim);

Status TransValueSequeueToVector(const ValuePtr &input_value, std::vector<int64_t> *input);

template <typename T>
std::shared_ptr<typename std::enable_if<std::is_base_of<ValueSequeue, T>::value, T>::type> TransVectorToValueSequeue(
  const std::vector<int64_t> &input) {
  std::vector<ValuePtr> elements;
  for (auto dim : input) {
    ValuePtr value_dim = MakeValue<int64_t>(dim);
    elements.push_back(value_dim);
  }
  std::shared_ptr<T> seq_value = std::make_shared<T>(elements);
  return seq_value;
}

const AnfNodePtr RealInputNode(const CNodePtr cnode, size_t index);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_NODE_INFO_H_
