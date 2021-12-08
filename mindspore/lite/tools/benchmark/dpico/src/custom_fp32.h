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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CUSTOM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CUSTOM_H_

#include <sys/stat.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
#include <map>
#include <sstream>
#include <vector>
#include <string>
#include <thread>
#include "include/api/kernel.h"
#include "include/svp_acl.h"
#include "include/svp_acl_mdl.h"
#include "include/svp_acl_ext.h"
#include "src/common_utils.h"
#include "src/custom_infer.h"

using mindspore::kernel::Kernel;

namespace mindspore {
namespace lite {
class CustomCPUKernel : public Kernel {
 public:
  CustomCPUKernel(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                  const mindspore::schema::Primitive *primitive, const mindspore::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx) {
    std::map<std::string, std::string> attrs;
    ExtractAttrsFromPrimitive(primitive, &attrs);
    for (auto &item : attrs) {
      SetAttr(item.first, item.second);
    }
    num_of_om_model_++;
  }

  ~CustomCPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Execute() override;

 private:
  Result DetermineBatchSize();
  int LoadModelAndInitResource();
  Result LoadModel();
  Result PrepareDevice();
  Result CreateInputs();
  Result CreateOutputs();
  Result SetDetParas();

  Result GetStrideParam(size_t *devSize, int index, size_t *stride, svp_acl_mdl_io_dims *dims);
  Result CreateInput(void *inputDataBuffer, size_t bufferSize, int stride);
  void *GetDeviceBufferOfTensor(const svp_acl_mdl_io_dims &dims, const size_t &stride, size_t dataSize);
  Result CreateTaskBufAndWorkBuf();
  Result CreateBuf(int index);
  Result GetInputDims(int index, svp_acl_mdl_io_dims *dims);
  size_t GetInputDataSize(int index);

  Result PreExecute();
  Result DeviceExecute();
  Result CopyTensorsToNpuWithStride();
  void DumpModelOutputResultToTensor();
  void WriteOutputToTensor(size_t index, size_t output_tensor_index);
  void OutputModelResult();
  void PrintResultToTensor(const std::vector<std::vector<float>> &boxValue);
  void UpdateDetParas();

  void UnloadModel();
  void DestroyInput();
  void DestroyOutput();
  void TerminateDevice();

 private:
  uint32_t model_id_ = 0;
  void *model_mem_ptr_ = nullptr;
  bool load_flag_ = false;  // model load flag
  svp_acl_mdl_desc *model_desc_ = nullptr;
  svp_acl_mdl_dataset *input_ = nullptr;
  svp_acl_mdl_dataset *output_ = nullptr;

  svp_acl_rt_stream stream_;

  std::vector<void *> inputs_data_in_npu_;
  size_t recurrent_total_t = 1;
  bool is_recurrent_net_ = false;  // true: batch is 1, false: not support Total_t
  bool is_detection_net_ = false;
  size_t batch_size_ = 1;
  bool prepared_ = false;
  float *det_param_buf_float_ = nullptr;
  static size_t num_of_om_model_;
  static dpico::CustomInterface custom_infershape_;
  static DpicoConfigParamExtractor dpico_config_param_extractor_;
  static DpicoContextManager dpico_context_manager_;
  static DpicoAicpuThreadManager dpico_aicpu_thread_manager_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_CUSTOM_H_
