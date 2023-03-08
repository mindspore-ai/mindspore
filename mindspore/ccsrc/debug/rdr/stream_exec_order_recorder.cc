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
#include "debug/rdr/stream_exec_order_recorder.h"
#include <sstream>
#include <fstream>
#include "mindspore/core/ir/anf.h"
#include "mindspore/core/utils/log_adapter.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/rdr/recorder_manager.h"
#include "mindspore/core/utils/file_utils.h"

namespace mindspore {
std::string Vector2String(const std::vector<uint32_t> &v) {
  std::string str = "";
  for (size_t j = 0; j < v.size(); ++j) {
    str += std::to_string(v[j]) + (j + 1 < v.size() ? "," : "");
  }
  return str;
}

json ExecNode::ExecNode2Json() const {
  json exec_node;
  exec_node[kAttrIndex] = index_;
  exec_node[kAttrNodeName] = node_name_;
  exec_node[kAttrLogicId] = logic_id_;
  exec_node[kAttrStreamId] = stream_id_;
  exec_node[kAttrNodeInfo] = node_info_;
  exec_node[kAttrEventId] = event_id_;
  if (!label_ids_.empty()) {
    exec_node[kAttrLabelId] = Vector2String(label_ids_);
  }
  if (!active_stream_ids_.empty()) {
    exec_node[kAttrActiveStreamId] = Vector2String(active_stream_ids_);
  }

  return exec_node;
}

void StreamExecOrderRecorder::Export() {
  auto realpath = GetFileRealPath();
  if (!realpath.has_value()) {
    return;
  }
  std::string real_file_path = realpath.value() + ".json";
  json exec_order_json = json::array();
  for (size_t i = 0; i < exec_order_.size(); ++i) {
    exec_order_json.push_back(exec_order_[i]->ExecNode2Json());
  }
  ChangeFileMode(real_file_path, S_IRWXU);
  std::ofstream fout(real_file_path, std::ofstream::app);
  if (!fout.is_open()) {
    MS_LOG(WARNING) << "Open file for saving stream execute order failed. File path: '" << real_file_path << "'.";
    return;
  }
  const size_t space_num = 2;
  fout << exec_order_json.dump(space_num);
  fout.close();
  ChangeFileMode(real_file_path, S_IRUSR);
}

namespace RDR {
bool RecordStreamExecOrder(const SubModuleId module, const std::string &name, const std::vector<CNodePtr> &exec_order) {
  if (!mindspore::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  StreamExecOrderRecorderPtr stream_exec_order_recorder =
    std::make_shared<StreamExecOrderRecorder>(submodule_name, name, exec_order);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(stream_exec_order_recorder));
  return ans;
}
}  // namespace RDR
}  // namespace mindspore
