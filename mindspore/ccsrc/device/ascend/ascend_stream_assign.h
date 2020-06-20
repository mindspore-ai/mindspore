/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_
#define MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_

#include <functional>
#include <unordered_map>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <unordered_set>
#include "runtime/base.h"
#include "runtime/rt_model.h"
#include "runtime/stream.h"
#include "session/kernel_graph.h"

namespace mindspore {
namespace device {
namespace ascend {
using std::map;
using std::shared_ptr;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using CnodeKey = void *;
const uint32_t kInvalidStreamId = UINT32_MAX;
class AscendStreamMng {
 public:
  static AscendStreamMng &GetInstance() {
    static AscendStreamMng instance;
    return instance;
  }

  void Reset() {
    cur_stream_id = 0;
    cur_stream_num = 0;
  }
  uint32_t ApplyNewStream() {
    if (!cur_stream_num) {
      cur_stream_num++;
      return cur_stream_id;
    }
    cur_stream_num++;
    cur_stream_id++;
    return cur_stream_id;
  }

  uint32_t GetCurAllocStream() { return cur_stream_id; }
  uint32_t GetCurAllocStreamNum() { return cur_stream_num; }

 private:
  uint32_t cur_stream_num{0};
  uint32_t cur_stream_id{0};
};

class AscendStreamAssign {
 public:
  static AscendStreamAssign &GetInstance() {
    static AscendStreamAssign instance;  // Guaranteed to be destroyed.
    return instance;
  }

  AscendStreamAssign(const AscendStreamAssign &) = delete;
  AscendStreamAssign &operator=(const AscendStreamAssign &) = delete;

  uint32_t total_event_num() const { return total_event_num_; }
  void GetHcomStreams(std::vector<uint32_t> *streams);

  void AssignStream(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void GetWaitStreams(vector<uint32_t> *wait_active_stream_list);
  CNodePtr CreateSendApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr, uint32_t event_id,
                                 uint32_t stream_id);
  CNodePtr CreateRecvApplyKernel(const std::shared_ptr<session::KernelGraph> &graph_ptr, uint32_t event_id,
                                 uint32_t stream_id);

 private:
  AscendStreamAssign() = default;
  ~AscendStreamAssign() = default;
  void Reset();
  void CheckStreamAssign(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void AssignAllNodesStream(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void AssignCommonStreamId(const CNodePtr &cur_cnode_ptr, CNodePtr *pre_cnode_ptr, uint32_t *cur_index,
                            uint32_t *cur_stream_id);
  void AssignIndependentStreamId(const CNodePtr &cur_cnode_ptr);
  void UpdateAtomicAddrCleanStreamId(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void FindHcomParallelStreams(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void InsertStreamActive(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void UpdateStreamSwitch(const std::shared_ptr<session::KernelGraph> &graph_ptr, const CNodePtr &switch_ptr,
                          const vector<uint32_t> &independent_stream, vector<CNodePtr> *orders);
  void InsertSendRecvForIndependent(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void InsertSendRecvForHcomParallel(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void InsertSendRecvForDiffHcom(const shared_ptr<mindspore::session::KernelGraph> &graph_ptr);
  void UpdateEventId(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void GetNeedActiveStreams(const std::shared_ptr<session::KernelGraph> &graph_ptr);
  void ReorderIndependentOrders(const std::shared_ptr<session::KernelGraph> &graph_ptr);

  bool IsTaskSink();
  bool IsFusionHcom(const CNodePtr &cur_cnode_ptr);
  bool IsHcom(const CNodePtr &cur_cnode_ptr);
  bool IsIndependentNode(const CNodePtr &node_ptr);
  bool IsProcessedStream(uint32_t stream_id);
  vector<CNodePtr>::iterator FindTargetOp(vector<CNodePtr>::iterator begin, vector<CNodePtr>::iterator end,
                                          const CNodePtr &node);
  void GetParallelStream(uint32_t cur_stream_id, uint32_t stream_acitve_id, std::vector<uint32_t> *parallel_streams);

  uint32_t total_event_num_{0};
  bool independent_stream_activated_{false};
  std::map<uint32_t, uint32_t> independent_stream_map_{};
  std::set<uint32_t> processed_streams_{};
  std::set<uint32_t> hcom_stream_list_{};
  std::vector<uint32_t> need_first_active_streams_{};
  std::vector<std::vector<uint32_t>> inner_parallel_streams_{};

  // new policy end
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_
