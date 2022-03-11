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

#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_LITE_ADAPTER_RUNTIME_AKG_KERNEL_H_
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_LITE_ADAPTER_RUNTIME_AKG_KERNEL_H_
#include <vector>
#include <string>
#include "src/inner_kernel.h"
#include "common/graph_kernel/lite_adapter/common/graph_kernel_op_parameter.h"

namespace mindspore::kernel {
using AkgParallelLambda = int (*)(int task_id, int num_task, void *cdata);

class AkgKernel : public InnerKernel {
 public:
  AkgKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
            const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    params_ = reinterpret_cast<GraphKernelParameter *>(op_parameter_);
    ExtractKernelName();
  }
  ~AkgKernel() override;

  int Prepare() override;
  int Run() override;
  int ReSize() override {
    // donot support ReSize now.
    return mindspore::lite::RET_ERROR;
  }

  // the real callback function that send to akg
  void AkgParallelLaunchFunc(AkgParallelLambda flambda, void *cdata, int);
  // the callback function that send to thread pool
  int DoTask(int task_id, float, float);

 protected:
  void ExtractKernelName();

  GraphKernelParameter *params_{nullptr};
  void *handle_{nullptr};
  void *kernel_func_{nullptr};
  std::string kernel_name_;
  int nthread_{0};
  AkgParallelLambda cached_akg_lambda_ = nullptr;
  void *cached_runtimeargs_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_LITE_ADAPTER_RUNTIME_AKG_KERNEL_H_
