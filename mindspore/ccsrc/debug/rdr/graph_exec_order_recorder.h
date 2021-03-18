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
#ifndef MINDSPORE_CCSRC_DEBUG_RDR_GRAPH_EXEC_ORDER_RECORDER_H_
#define MINDSPORE_CCSRC_DEBUG_RDR_GRAPH_EXEC_ORDER_RECORDER_H_
#include <vector>
#include <string>
#include <memory>

#include "debug/rdr/base_recorder.h"

namespace mindspore {
class CNode;
using CNodePtr = std::shared_ptr<CNode>;
class GraphExecOrderRecorder : public BaseRecorder {
 public:
  GraphExecOrderRecorder() : BaseRecorder() {}
  GraphExecOrderRecorder(const std::string &module, const std::string &name,
                         const std::vector<CNodePtr> &final_exec_order)
      : BaseRecorder(module, name), exec_order_(final_exec_order) {}
  ~GraphExecOrderRecorder() {}
  void SetExecOrder(const std::vector<CNodePtr> &final_exec_order) { exec_order_ = final_exec_order; }
  virtual void Export();

 private:
  std::vector<CNodePtr> exec_order_;
};
using GraphExecOrderRecorderPtr = std::shared_ptr<GraphExecOrderRecorder>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_RDR_GRAPH_EXEC_ORDER_RECORDER_H_
