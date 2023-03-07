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
#ifndef MINDSPORE_CCSRC_DEBUG_RDR_STREAM_EXEC_ORDER_RECORDER_H_
#define MINDSPORE_CCSRC_DEBUG_RDR_STREAM_EXEC_ORDER_RECORDER_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "nlohmann/json.hpp"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/debug/rdr/base_recorder.h"

using json = nlohmann::json;

namespace mindspore {
class ExecNode {
 public:
  ExecNode() {}
  ExecNode(const size_t index, const std::string &node_name, const uint32_t &logic_id, const uint32_t &stream_id,
           const std::string &node_info)
      : index_(index), node_name_(node_name), logic_id_(logic_id), stream_id_(stream_id), node_info_(node_info) {}
  ~ExecNode() {}
  void SetEventId(const uint32_t &event_id) { event_id_ = event_id; }
  void SetLabelId(const uint32_t &label_id) { label_ids_.push_back(label_id); }
  void SetActiveStreamId(const uint32_t &active_stream_id) { active_stream_ids_.push_back(active_stream_id); }
  json ExecNode2Json() const;

 private:
  size_t index_;
  std::string node_name_;
  uint32_t logic_id_;
  uint32_t stream_id_;
  std::string node_info_;
  uint32_t event_id_{0};
  std::vector<uint32_t> label_ids_;
  std::vector<uint32_t> active_stream_ids_;
};
using ExecNodePtr = std::shared_ptr<ExecNode>;

class StreamExecOrderRecorder : public BaseRecorder {
 public:
  StreamExecOrderRecorder() : BaseRecorder() {}
  StreamExecOrderRecorder(const std::string &module, const std::string &name, const std::vector<CNodePtr> &exec_order)
      : BaseRecorder(module, name) {
    // Extract information from execute order.
    for (size_t i = 0; i < exec_order.size(); i++) {
      CNodePtr cur_cnode_ptr = exec_order[i];
      MS_EXCEPTION_IF_NULL(cur_cnode_ptr);

      ExecNode exec_node =
        ExecNode(i, cur_cnode_ptr->fullname_with_scope(), AnfAlgo::GetStreamDistinctionLabel(cur_cnode_ptr.get()),
                 AnfAlgo::GetStreamId(cur_cnode_ptr), cur_cnode_ptr->DebugString());

      if (common::AnfAlgo::HasNodeAttr(kAttrEventId, cur_cnode_ptr)) {
        exec_node.SetEventId(common::AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrEventId));
      }

      if (common::AnfAlgo::HasNodeAttr(kAttrLabelIndex, cur_cnode_ptr)) {
        exec_node.SetLabelId(common::AnfAlgo::GetNodeAttr<uint32_t>(cur_cnode_ptr, kAttrLabelIndex));
      }

      if (common::AnfAlgo::HasNodeAttr(kAttrLabelSwitchList, cur_cnode_ptr)) {
        auto label_list = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cur_cnode_ptr, kAttrLabelSwitchList);
        for (size_t j = 0; j < label_list.size(); ++j) {
          exec_node.SetLabelId(label_list[j]);
        }
      }

      std::string active_stream_str;
      if (common::AnfAlgo::HasNodeAttr(kAttrActiveStreamList, cur_cnode_ptr)) {
        auto stream_list = common::AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cur_cnode_ptr, kAttrActiveStreamList);
        for (size_t j = 0; j < stream_list.size(); ++j) {
          exec_node.SetActiveStreamId(stream_list[j]);
        }
      }
      ExecNodePtr exec_node_ptr = std::make_shared<ExecNode>(exec_node);
      exec_order_.push_back(std::move(exec_node_ptr));
    }
  }
  ~StreamExecOrderRecorder() {}
  void Export() override;

 private:
  std::vector<ExecNodePtr> exec_order_;
};
using StreamExecOrderRecorderPtr = std::shared_ptr<StreamExecOrderRecorder>;
namespace RDR {
BACKEND_EXPORT bool RecordStreamExecOrder(const SubModuleId module, const std::string &name,
                                          const std::vector<CNodePtr> &exec_order);
}  // namespace RDR
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_STREAM_EXEC_ORDER_RECORDER_H_
