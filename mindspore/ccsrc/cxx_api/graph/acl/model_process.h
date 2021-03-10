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

#ifndef MINDSPORE_CCSRC_CXXAPI_GRAPH_ACL_MODEL_PROCESS_H
#define MINDSPORE_CCSRC_CXXAPI_GRAPH_ACL_MODEL_PROCESS_H
#include <vector>
#include <string>
#include <map>
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "include/api/status.h"
#include "include/api/types.h"

namespace mindspore {
struct AclTensorInfo {
  void *cur_device_data;
  void *device_data;
  size_t buffer_size;
  aclDataType data_type;
  std::vector<int64_t> dims;
  std::string name;
};

class ModelProcess {
 public:
  ModelProcess()
      : model_id_(0xffffffff),
        is_run_on_device_(false),
        model_desc_(nullptr),
        inputs_(nullptr),
        outputs_(nullptr),
        input_infos_(),
        output_infos_() {}
  ~ModelProcess() {}

  Status UnLoad();
  Status PredictFromHost(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);
  Status PreInitModelResource();
  std::vector<MSTensor> GetInputs();
  std::vector<MSTensor> GetOutputs();

  // override this method to avoid request/reply data copy
  void SetIsDevice(bool is_device) { is_run_on_device_ = is_device; }

  void set_model_id(uint32_t model_id) { model_id_ = model_id; }
  uint32_t model_id() const { return model_id_; }

 private:
  Status CreateDataBuffer(void **data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset);
  Status CheckAndInitInput(const std::vector<MSTensor> &inputs);
  Status ConstructTensors(const std::vector<AclTensorInfo> &acl_tensor_list, std::vector<MSTensor> *tensor_list);
  Status BuildOutputs(std::vector<MSTensor> *outputs);
  Status SetBatchSize(const std::vector<MSTensor> &inputs);
  Status InitInputsBuffer();
  Status InitOutputsBuffer();
  Status ResetOutputSize();

  void DestroyInputsDataset();
  void DestroyInputsDataMem();
  void DestroyInputsBuffer();
  void DestroyOutputsBuffer();

  uint32_t model_id_;
  // if run one device(AICPU), there is no need to alloc device memory and copy inputs to(/outputs from) device
  bool is_run_on_device_;
  aclmdlDesc *model_desc_;
  aclmdlDataset *inputs_;
  aclmdlDataset *outputs_;
  std::vector<AclTensorInfo> input_infos_;
  std::vector<AclTensorInfo> output_infos_;
  std::vector<MSTensor> input_tensors_;
  std::vector<MSTensor> output_tensors_;
  size_t GetDynamicDims(const std::vector<AclTensorInfo> &);
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_CXXAPI_GRAPH_ACL_MODEL_PROCESS_H
