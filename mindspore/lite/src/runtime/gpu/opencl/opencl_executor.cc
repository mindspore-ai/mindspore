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

#include "src/runtime/gpu/opencl/opencl_executor.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "nnacl/pack.h"
#include "include/errorcode.h"

namespace mindspore::lite::opencl {
int OpenCLExecutor::Run(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                        const std::vector<kernel::LiteKernel *> &kernels, const KernelCallBack &before,
                        const KernelCallBack &after) {
  if (before != nullptr && after != nullptr) {
    ocl_runtime_.GetInstance()->SetProfiling(true);
  }
  return RunOrTune(inputs, outputs, kernels, before, after, false);
}

int OpenCLExecutor::RunOrTune(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                              const std::vector<kernel::LiteKernel *> &kernels, const KernelCallBack &before,
                              const KernelCallBack &after, bool is_tune) {
  int ret{RET_OK};
  auto opencl_runtime_ins = ocl_runtime_.GetInstance();
  auto profiling_tmp = opencl_runtime_ins->isProfiling();
  if (is_tune) {
    opencl_runtime_ins->SetProfiling(true);
  }
  for (auto *kernel : kernels) {
    MS_ASSERT(kernel);
    GPUCallBackParam callbackParam;
    callbackParam.node_name = kernel->name();
    callbackParam.node_type = kernel->type_str();
    if ((before != nullptr) &&
        !before(TensorVectorCast(kernel->in_tensors()), TensorVectorCast(kernel->out_tensors()), callbackParam)) {
      MS_LOG(ERROR) << "run kernel before_callback failed, name: " << kernel->name();
    }
    // Don't support ZeroShape
    for (auto tensor : kernel->out_tensors()) {
      for (size_t i = 0; i < tensor->shape().size(); i++) {
        if (tensor->shape()[i] == 0) {
          MS_LOG(ERROR) << "Opencl don't support ZeroShape.";
          return RET_ERROR;
        }
      }
    }
    if (kernel->IsBuiltin()) {
      auto *op_kernel = reinterpret_cast<kernel::OpenCLKernel *>(kernel->kernel());

      if (is_tune) {
        ret = Tune(op_kernel);
        if (ret != RET_OK) {
          opencl_runtime_ins->SetProfiling(profiling_tmp);
          return RET_OK;
        }
      } else {
        ret = kernel->Execute();
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
          return ret;
        }
        if (profiling_tmp) {
          auto execute_time = op_kernel->GetProfilingTimeMs();
          MS_LOG(INFO) << "OpenCl kernel " << kernel->name() << "(" << kernel->type_str()
                       << ") execute time is: " << op_kernel->GetProfilingTimeMs() << "ms";
          callbackParam.execute_time = execute_time;
        }
      }
    } else {
      if (!is_tune) {
        ret = kernel->Execute();
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
          return ret;
        }
      }
    }

    if ((after != nullptr) &&
        !after(TensorVectorCast(kernel->in_tensors()), TensorVectorCast(kernel->out_tensors()), callbackParam)) {
      MS_LOG(ERROR) << "run kernel after_callback failed, name: " << kernel->name();
    }
  }
  opencl_runtime_ins->SetProfiling(profiling_tmp);
  return ret;
}

int OpenCLExecutor::Tune(kernel::OpenCLKernel *op_kernel) {
  auto ret = op_kernel->PreProcess();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PreProcess kernel failed, name: " << op_kernel->name() << " in tuning";
    return ret;
  }
  ret = op_kernel->Tune();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "tuning kernel failed, name: " << op_kernel->name();
    return ret;
  }
  ret = op_kernel->PostProcess();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PostProcess kernel failed, name: " << op_kernel->name() << " in tuning";
    return ret;
  }
  return RET_OK;
}
}  // namespace mindspore::lite::opencl
