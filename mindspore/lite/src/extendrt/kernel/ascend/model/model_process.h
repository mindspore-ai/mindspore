/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_PROCESS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_PROCESS_H_

#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <functional>
#include <memory>
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "acl/acl_rt.h"
#include "include/api/types.h"
#include "include/errorcode.h"
#include "kernel/kernel.h"
#include "extendrt/kernel/ascend/options/acl_model_options.h"
#include "extendrt/kernel/ascend/model/dyn_shape_process.h"

namespace mindspore::kernel {
namespace acl {
struct AclTensorInfo {
  void *cur_device_data;
  void *device_data;
  size_t buffer_size;
  size_t malloc_buffer_size;
  aclDataType data_type;
  std::vector<int64_t> dims;
  std::string name;
};

class ModelProcess {
 public:
  explicit ModelProcess(const AclModelOptionsPtr &options) : options_(options) {}
  ~ModelProcess() {}

  bool Load(const void *om_data, size_t om_data_size);
  bool UnLoad();
  bool PredictFromHost(const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs);

  // override this method to avoid request/reply data copy
  void SetIsDevice(bool is_device) { is_run_on_device_ = is_device; }

  std::set<uint64_t> GetDynamicBatch();
  std::set<std::pair<uint64_t, uint64_t>> GetDynamicImage();
  std::vector<Format> GetInputFormat();
  const std::vector<ShapeVector> GetOutputShape();
  const std::vector<ShapeVector> GetInputShape();
  const std::vector<TypeId> GetInputDataType();
  const std::vector<TypeId> GetOutputDataType();
  std::vector<Format> GetOutputFormat();

  bool Resize(const std::vector<ShapeVector> &new_shapes);

 private:
  bool PreInitModelResource();

  bool InitInputsBuffer();
  bool InitOutputsBuffer();
  void DestroyInputsBuffer();
  void DestroyOutputsBuffer();
  bool CreateDataBuffer(void **data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset);

  bool CheckAndInitInput(const std::vector<KernelTensorPtr> &inputs);
  bool CheckAndInitOutput(const std::vector<KernelTensorPtr> &outputs);
  bool CheckInputTensors(const std::vector<KernelTensorPtr> &inputs);
  bool CheckOutputTensors(const std::vector<KernelTensorPtr> &outputs);
  bool GetOutputs(const std::vector<KernelTensorPtr> &outputs);

  bool ResetInputSize(const std::vector<ShapeVector> &new_shapes);
  bool ResetOutputSize();
  bool IsDynamicShape();
  bool IsDynamicBatchSize();
  bool IsDynamicImageSize();

  AclModelOptionsPtr options_;
  uint32_t model_id_ = UINT32_MAX;
  // if run one device(AICPU), there is no need to alloc device memory and copy inputs to(/outputs from) device
  bool is_run_on_device_ = false;
  aclmdlDesc *model_desc_ = nullptr;
  aclmdlDataset *inputs_ = nullptr;
  aclmdlDataset *outputs_ = nullptr;

  bool loaded_ = false;
  size_t data_input_num_ = 0;
  std::vector<AclTensorInfo> input_infos_;
  std::vector<AclTensorInfo> output_infos_;

  AclDynamicShapeOptions dynamic_shape_options_;
  DynShapeProcess dyn_shape_proc_;
  std::vector<ShapeVector> cur_input_shapes_;
};
}  // namespace acl
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_PROCESS_H_
