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

#ifndef INC_MODEL_PROCESS_ACL
#define INC_MODEL_PROCESS_ACL
#include <vector>
#include <string>
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "include/inference.h"

namespace mindspore {
namespace inference {

struct AclTensorInfo {
  void *device_data;
  size_t buffer_size;
  aclDataType data_type;
  std::vector<int64_t> dims;
};

struct ImagesDvppOutput {
  void *buffer_device = nullptr;
  size_t buffer_size = 0;
  size_t input_index = 0;
};

class ModelProcess {
 public:
  ModelProcess() {}
  ~ModelProcess() {}

  Status LoadModelFromFile(const std::string &file_name, uint32_t &model_id);
  void UnLoad();

  // override this method to avoid request/reply data copy
  Status Execute(const RequestBase &request, ReplyBase &reply);
  Status Execute(const void *dvpp_outputs_buffer_dev, size_t dvpp_outputs_buffer_size, ReplyBase &reply);
  void SetIsDevice(bool is_device) { is_run_on_device_ = is_device; }

  size_t GetBatchSize() const;

 private:
  uint32_t model_id_ = 0xffffffff;
  // if run one device(AICPU), there is no need to alloc device memory and copy inputs to(/outputs from) device
  bool is_run_on_device_ = false;
  aclmdlDesc *model_desc_ = nullptr;
  aclmdlDataset *inputs_ = nullptr;
  aclmdlDataset *outputs_ = nullptr;
  std::vector<AclTensorInfo> input_infos_;
  std::vector<AclTensorInfo> output_infos_;

  Status PreInitModelResource();
  Status CreateDataBuffer(void *&data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset);
  Status CheckAndInitInput(const RequestBase &request);
  Status CheckAndInitDvppInput(const void *dvpp_outputs_buffer_dev, size_t dvpp_outputs_buffer_size,
                               size_t input_index);
  Status BuildOutputs(ReplyBase &reply);

  Status InitInputsBuffer();
  Status InitOutputsBuffer();
  void DestroyInputsDataset();
  void DestroyInputsDataMem();
  void DestroyInputsBuffer();
  void DestroyOutputsBuffer();
};

}  // namespace inference
}  // namespace mindspore

#endif
