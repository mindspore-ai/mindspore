/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_PROVIDERS_DPICO_MANAGER_ACL_MODEL_MANAGER_H_
#define MINDSPORE_LITE_PROVIDERS_DPICO_MANAGER_ACL_MODEL_MANAGER_H_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include "include/api/types.h"
#include "include/svp_acl_base.h"
#include "include/svp_acl_mdl.h"
#include "manager/acl_context_manager.h"
#include "manager/custom_config_manager.h"
#include "common/custom_enum.h"
#include "include/schema/model_generated.h"
#include "include/lite_types.h"

namespace mindspore {
namespace lite {
class AclModelManager {
 public:
  AclModelManager() = default;
  ~AclModelManager();

  int Init(const std::map<std::string, std::string> &dpico_config,
           const std::map<std::string, std::string> &model_share_config, const schema::Primitive *primitive,
           const std::vector<mindspore::MSTensor> &input_tensors,
           const std::vector<mindspore::MSTensor> &output_tensors);
  int UpdateBatchSize(const std::vector<mindspore::MSTensor> &input_tensors);
  int PrepareAclInputs(std::vector<mindspore::MSTensor> *input_tensors);
  int PrepareAclOutputs(std::vector<mindspore::MSTensor> *output_tensors);
  int UpdateKernelConfig(const std::map<std::string, std::string> &dpico_config);
  int UpdateAclInputs(std::vector<mindspore::MSTensor> *input_tensors);
  int UpdateAclOutputs(std::vector<mindspore::MSTensor> *output_tensors);
  int Execute(const std::vector<mindspore::MSTensor> &input_tensors,
              const std::vector<mindspore::MSTensor> &output_tensors,
              const std::map<std::string, std::string> &model_share_config);

 private:
  int LoadModel(const std::vector<mindspore::MSTensor> &input_tensors);
  int CreateModelDesc();
  int SetDetectParams(void *data);
  int AddDetectParamInput();
  int DetectPostProcess(mindspore::MSTensor *output_tensors);
  int CreateTaskBufAndWorkBuf();
  int CreateNoShareTaskBufAndWorkBuf();
  int GetMaxTaskAndWorkBufSize();
  int CopyTensorDataToAclInputs(const std::vector<mindspore::MSTensor> &input_tensors);
  int CopyAclOutputsToTensorData(const std::vector<mindspore::MSTensor> &output_tensors);
  int FlushAclInputsAndOutputs();
  int AclModelRun(const std::vector<mindspore::MSTensor> &input_tensors);
  int UnloadModel();

 private:
  static AllocatorPtr custom_allocator_;
  static CustomConfigManagerPtr custom_config_manager_ptr_;
  static AclContextManagerPtr acl_context_manager_;

  std::unordered_map<size_t, bool> inputs_mem_managed_by_tensor;   // <acl inputs idx, memory managed flag>
  std::unordered_map<size_t, bool> outputs_mem_managed_by_tensor;  // <acl outputs idx, memory managed flag>

  size_t actual_batch_size_{1};
  /** acl related variables */
  uint32_t acl_model_id_{0};
  int32_t acl_device_id_{0};
  AclModelType acl_model_type_{kCnn};
  void *acl_model_ptr_{nullptr};
  svp_acl_mdl_desc *acl_model_desc_{nullptr};
  svp_acl_mdl_dataset *acl_inputs_{nullptr};
  svp_acl_mdl_dataset *acl_outputs_{nullptr};
};
using AclModelManagerPtr = std::shared_ptr<AclModelManager>;
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_DPICO_MANAGER_ACL_MODEL_MANAGER_H_
