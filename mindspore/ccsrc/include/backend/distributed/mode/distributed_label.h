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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_LABEL_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_LABEL_H_

#include <string>
#include <memory>
#include <unordered_map>
#include "ir/func_graph.h"
#include "include/backend/distributed/constants.h"

namespace mindspore {
namespace distributed {
// For 'AnfNode'(Primitive) and 'FuncGraph'(Cell), they could be marked with 'distributed labels' set by python API
// 'place' by user. DistLabel describes on which processes this 'AnfNode' or 'FuncGraph' is executed. Distributed graph
// partitioner splits DAG according to this label.
class DistLabel {
 public:
  DistLabel() : ms_role_(kEnvRoleOfWorker) {}
  virtual ~DistLabel();

  virtual bool operator==(const DistLabel &label) = 0;
  virtual std::string ToString() const = 0;

 protected:
  std::string ms_role_;
};

// We abstract RankLabel to denote distributed label constructed by one rank id. This means 'AnfNode' or 'FuncGraph' is
// executed on process with this rank id.
class RankLabel : public DistLabel {
 public:
  explicit RankLabel(uint32_t rank_id, const std::string &role) : rank_id_(rank_id), ms_role_(role) {}
  ~RankLabel() override = default;

  std::string ToString() const override;

 private:
  uint32_t rank_id_;
};

// In MPMD execution mode, one 'FuncGraph' will be assigned to multiple processes(with model parallel), so we abstract
// RankListLabel for this case. RankListLabel denotes distributed label constructed by a rank list representing multiple
// processes.
class RankListLabel : public DistLabel {
 public:
  explicit RankListLabel(const RankList &rank_list, const std::string &role) : rank_list_(rank_list), ms_role_(role) {}
  ~RankListLabel() override = default;

  std::string ToString() const override;

 private:
  RankList rank_list_;
};

using DistLabelPtr = std::shared_ptr<DistLabel>;
using RankLabelPtr = std::shared_ptr<RankLabel>;
using RankListLabelPtr = std::shared_ptr<RankListLabel>;

// Maps of node/graph to distributed label. After dyeing graph, 'GraphPartitionLabels' should be generated.
using NodeLabels = std::unordered_map<AnfNodePtr, DistLabelPtr>;
using GraphLabels = std::unordered_map<FuncGraphPtr, DistLabelPtr>;
struct GraphPartitionLabels {
  NodeLabels n_labels_;
  GraphLabels g_labels_;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_LABEL_H_
