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
#ifndef MINDSPORE_CCSRC_CXX_API_ACL_VM_ACL_MULTI_GRAPH_SESSION_H
#define MINDSPORE_CCSRC_CXX_API_ACL_VM_ACL_MULTI_GRAPH_SESSION_H

#include <deque>
#include <vector>
#include <map>
#include <memory>
#include "include/api/types.h"
#include "include/api/cell.h"
#include "backend/session/session_basic.h"

namespace mindspore {
class AclModelOptions;
namespace session {
class MultiGraphAclSession : public session::SessionBasic {
 public:
  MultiGraphAclSession() = default;
  ~MultiGraphAclSession() override = default;
  void Init(uint32_t device_id) override;
  GraphId CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;
  void RunGraph(GraphId graph_id, const std::vector<MSTensor> &inputs, VectorRef *outputs);
  void SetOptions(const std::shared_ptr<AclModelOptions> &options) { options_ = options; }

 private:
  VectorRef ConstructOutputRef(GraphId graph_id, std::deque<MSTensor> *out_tensors);
  VectorRef ConstructOutputRefByTupleNode(const CNodePtr &tuple_node, std::deque<MSTensor> *out_tensors);

  std::map<GraphId, GraphCell> graphs_ = {};
  std::map<GraphId, KernelGraphPtr> kernel_graphs_ = {};
  std::shared_ptr<AclModelOptions> options_ = nullptr;
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_ACL_VM_ACL_MULTI_GRAPH_SESSION_H
