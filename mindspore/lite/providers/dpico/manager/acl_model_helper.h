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

#ifndef MINDSPORE_LITE_PROVIDERS_DPICO_MANAGER_ACL_MODEL_HELPER_H_
#define MINDSPORE_LITE_PROVIDERS_DPICO_MANAGER_ACL_MODEL_HELPER_H_

#include <vector>
#include <unordered_map>
#include "include/svp_acl_mdl.h"
#include "include/api/types.h"
#include "include/schema/model_generated.h"
#include "common/custom_enum.h"
#include "include/lite_types.h"

namespace mindspore {
namespace lite {
struct AclDataInfo {
  enum DataMode { Input = 0, Output = 1 };
  DataMode data_mode{Input};
  size_t stride{0};
  size_t data_size{0};
  svp_acl_mdl_io_dims dim_info{};
  explicit AclDataInfo(DataMode input_mode) : data_mode(input_mode) {}
};
int GetAclModelType(const schema::Primitive *primitive, AclModelType *acl_model_type);
int GetAclDataInfo(struct AclDataInfo *acl_data_info, svp_acl_mdl_desc *acl_mdl_desc, int index);
int AddDatasetBuffer(svp_acl_mdl_dataset *acl_mdl_dataset, size_t data_buffer_size, size_t stride, void *data);
int DestroyAclDataset(svp_acl_mdl_dataset **acl_mdl_dataset,
                      const std::unordered_map<size_t, bool> &mem_managed_by_tensor, const AllocatorPtr &allocator_ptr);
size_t GetDataTypeSize(svp_acl_mdl_desc *acl_mdl_desc, size_t index, AclDataInfo::DataMode data_mode);
int ComputeValidDetectBoxes(svp_acl_mdl_desc *acl_mdl_desc, svp_acl_mdl_dataset *acl_outputs,
                            std::vector<std::vector<float>> *boxes);
int WriteDetBoxesToTensorData(const std::vector<std::vector<float>> &det_boxes,
                              mindspore::MSTensor *detect_boxes_tensor);

int AclMalloc(void **buf, size_t size);
int AclFree(void **buf);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_PROVIDERS_DPICO_MANAGER_ACL_MODEL_HELPER_H_
