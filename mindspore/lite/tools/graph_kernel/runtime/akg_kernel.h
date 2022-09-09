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

#ifndef MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_RUNTIME_AKG_KERNEL_H_
#define MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_RUNTIME_AKG_KERNEL_H_
#include <vector>
#include <string>
#include <unordered_map>
#include "src/litert/lite_kernel.h"
#include "nnacl/custom_parameter.h"

namespace mindspore::kernel {
using AkgParallelLambda = int (*)(int task_id, int num_task, void *cdata);

class AkgKernel : public LiteKernel {
 public:
  AkgKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
            const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    // in PopulateCustomParameter, the primitive is store in attr_data[0]
    params_ = static_cast<void *>(reinterpret_cast<CustomParameter *>(op_parameter_)->attr_data[0]);
    ExtractKernelAttr();
  }
  ~AkgKernel() override;

  int Prepare() override;
  int Run() override;
  int ReSize() override;

  // the real callback function that send to akg
  void AkgParallelLaunchFunc(AkgParallelLambda flambda, void *cdata, int);
  // the callback function that send to thread pool
  int DoTask(int task_id, float, float);

 protected:
  void ExtractKernelAttr();

  void *params_{nullptr};
  void *handle_{nullptr};
  void *kernel_func_{nullptr};
  std::string kernel_name_;
  int nthread_{0};
  int dynamic_batch_size_{0};
  std::vector<std::vector<int8_t>> const_data_align_cache_;
  std::vector<void *> const_inputs_;
  AkgParallelLambda cached_akg_lambda_ = nullptr;
  void *cached_runtimeargs_ = nullptr;
  std::vector<size_t> dynamic_input_index_;
  std::vector<std::vector<int>> origin_inputs_shape_;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_TOOLS_GRAPH_KERNEL_RUNTIME_AKG_KERNEL_H_
