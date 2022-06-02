/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_GRAPH_ACL_ACL_GRAPH_IMPL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_GRAPH_ACL_ACL_GRAPH_IMPL_H_
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/graph.h"
#include "extendrt/acl/model_process.h"
#include "extendrt/acl/acl_env_guard.h"
#include "extendrt/cxx_api/graph/graph_impl.h"
#include "extendrt/infer_session.h"

namespace mindspore {
class AclInferSession : public InferSession {
 public:
  AclInferSession();
  ~AclInferSession() override;

  Status CompileGraph(FuncGraphPtr graph) override;
  Status RunGraph() override;
  Status Resize(const std::vector<Tensor::TensorPtr> &inputs, const std::vector<std::vector<int64_t>> &dims) override;
  bool CheckDeviceSupport(mindspore::DeviceType device_type) override;
  Status Load(uint32_t device_id);
  Status InitEnv();
  Status FinalizeEnv();
  Status CheckModelInputs(const std::vector<tensor::TensorPtr> &inputs) const;
  Status LoadAclModel(const Buffer om_data);

  bool init_flag_;
  bool load_flag_;
  std::string device_type_;
  int32_t device_id_;
  aclrtContext context_;

  std::shared_ptr<AclEnvGuard> acl_env_;

  ModelProcess model_process_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_GRAPH_ACL_ACL_GRAPH_IMPL_H_
