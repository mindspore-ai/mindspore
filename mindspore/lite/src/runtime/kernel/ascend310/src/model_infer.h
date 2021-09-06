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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_ACL_MODEL_INFER_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_ACL_MODEL_INFER_H_

#include <vector>
#include <memory>
#include <string>
#include "src/runtime/kernel/ascend310/src/model_process.h"
#include "src/runtime/kernel/ascend310/src/acl_env_guard.h"
#include "src/runtime/kernel/ascend310/src/acl_model_options.h"
#include "include/api/types.h"
#include "include/errorcode.h"

namespace mindspore::kernel {
namespace acl {
using mindspore::lite::STATUS;

class ModelInfer {
 public:
  ModelInfer(const Buffer &om_data, const AclModelOptions &options);
  ~ModelInfer() = default;

  STATUS Init();
  STATUS Finalize();
  STATUS Load();
  STATUS Inference(const std::vector<mindspore::MSTensor> &inputs, std::vector<mindspore::MSTensor> *outputs);

 private:
  STATUS LoadAclModel(const Buffer &om_data);

  bool init_flag_;
  bool load_flag_;
  std::string device_type_;
  aclrtContext context_;
  Buffer om_data_;
  AclModelOptions options_;
  ModelProcess model_process_;
  std::shared_ptr<AclEnvGuard> acl_env_;
};
}  // namespace acl
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_ACL_MODEL_INFER_H_
