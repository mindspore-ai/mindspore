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

#include "src/runtime/opencl/opencl_executor.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "nnacl/pack.h"
#include "include/errorcode.h"

namespace mindspore::lite::opencl {

int OpenCLExecutor::Run(std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs,
                        std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator, const KernelCallBack &before,
                        const KernelCallBack &after) {
  return RunOrTune(inputs, outputs, kernels, allocator, before, after, false);
}

int OpenCLExecutor::RunOrTune(std::vector<Tensor *> &inputs, std::vector<Tensor *> &outputs,
                              std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator,
                              const KernelCallBack &before, const KernelCallBack &after, bool is_tune) {
  int ret{RET_OK};
  auto opencl_runtime_ins = ocl_runtime.GetInstance();
  auto profiling_tmp = opencl_runtime_ins->isProfiling();
  if (is_tune) {
    opencl_runtime_ins->SetProfiling(true);
  }
  for (auto *kernel : kernels) {
    MS_ASSERT(kernel);
    CallBackParam callbackParam;
    callbackParam.node_name = kernel->name();

    if (before != nullptr) {
      if (!before(TensorVectorCast(kernel->in_tensors()), TensorVectorCast(kernel->out_tensors()), callbackParam)) {
        MS_LOG(ERROR) << "run kernel before_callback failed, name: " << kernel->name();
      }
    }
    auto *op_kernel = reinterpret_cast<kernel::OpenCLKernel *>(kernel);
    auto cur_outputs = kernel->out_tensors();
    for (auto i = 0; i < cur_outputs.size(); ++i) {
      auto *output = cur_outputs.at(i);
      MS_ASSERT(output);
      if (op_kernel->GetMemType() == lite::opencl::MemType::IMG) {
        std::vector<size_t> img_size;
        ret = op_kernel->GetImageSize(i, &img_size);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "GetImageSize failed";
          return ret;
        }
        auto data_ptr = allocator_->Malloc(output->Size(), img_size);
        if (data_ptr == nullptr) {
          MS_LOG(ERROR) << "Malloc data failed";
          return RET_ERROR;
        }
        output->set_data(data_ptr);
      } else {
        ret = output->MallocData(allocator_);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "MallocData failed";
          return ret;
        }
      }
      output->set_allocator(allocator_);
    }
    if (is_tune) {
      ret = op_kernel->Tune();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "tuning kernel failed, name: " << kernel->name();
        return ret;
      }
      ret = kernel->PostProcess();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "PostProcess kernel failed, name: " << kernel->name();
        return ret;
      }
    } else {
      ret = kernel->Run();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
        return ret;
      }
      ret = kernel->PostProcess();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "PostProcess kernel failed, name: " << kernel->name();
        return ret;
      }
      if (profiling_tmp) {
        MS_LOG(INFO) << "OpenCl kernel " << kernel->name() << "(" << kernel->type_str()
                     << ") execute time is: " << op_kernel->GetProfilingTimeMs() << "ms";
      }
    }
    if (after != nullptr) {
      if (!after(TensorVectorCast(kernel->in_tensors()), TensorVectorCast(kernel->out_tensors()), callbackParam)) {
        MS_LOG(ERROR) << "run kernel after_callback failed, name: " << kernel->name();
      }
    }
  }
  opencl_runtime_ins->SetProfiling(profiling_tmp);
  return ret;
}
}  // namespace mindspore::lite::opencl
