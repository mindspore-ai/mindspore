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

#ifndef TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACL_MODEL_PROCESS_H
#define TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACL_MODEL_PROCESS_H

#include <vector>
#include "include/errorcode.h"
#include "include/api/types.h"
#include "tools/converter/adapter/acl/common/acl_types.h"
#include "acl/acl_mdl.h"

namespace mindspore {
namespace lite {
class AclModelProcess {
 public:
  AclModelProcess(const Buffer &om_data, const acl::AclModelOptionCfg &options);
  ~AclModelProcess() = default;

  STATUS Load();
  STATUS UnLoad();
  STATUS GetInputsShape(std::vector<std::vector<int64_t>> *inputs_shape);

 private:
  STATUS Init();
  STATUS Finalize();

  Buffer om_data_;
  acl::AclModelOptionCfg options_;
  aclmdlDesc *model_desc_;
  bool is_load_;
  uint32_t model_id_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // TOOLS_CONVERTER_ADAPTER_ACL_SRC_ACL_MODEL_PROCESS_H
