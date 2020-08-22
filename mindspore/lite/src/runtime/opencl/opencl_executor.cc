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
#include "src/common/ms_tensor_utils.h"
#include "include/errorcode.h"

namespace mindspore::lite::opencl {

int OpenCLExecutor::Prepare(const std::vector<kernel::LiteKernel *> &kernels) { return RET_OK; }

int OpenCLExecutor::Run(std::vector<tensor::Tensor *> &inputs, std::vector<tensor::Tensor *> &outputs,
                        std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator,
                        const session::KernelCallBack &before, const session::KernelCallBack &after) {
  kernel::LiteKernelUtil::InitTensorRefCount(kernels);
  for (auto *kernel : kernels) {
    MS_ASSERT(nullptr != kernel);
    session::CallBackParam callbackParam;
    callbackParam.name_callback_param = kernel->name();

    if (before != nullptr) {
      if (!before(PackToMSTensors(kernel->in_tensors()), PackToMSTensors(kernel->out_tensors()), callbackParam)) {
        MS_LOG(ERROR) << "run kernel before_callback failed, name: " << kernel->name();
      }
    }
    kernel::OpenCLKernel *op_kernel = reinterpret_cast<kernel::OpenCLKernel *>(kernel);
    auto &cur_outputs = kernel->out_tensors();
    for (auto i = 0; i < cur_outputs.size(); ++i) {
      auto *output = cur_outputs.at(i);
      MS_ASSERT(nullptr != output);
      if (op_kernel->GetMemType() == kernel::OpenCLMemType::IMG) {
        std::vector<size_t> img_size;
        op_kernel->GetImageSize(i, &img_size);
        auto data_ptr = allocator_->Malloc(output->Size(), img_size);
        output->SetData(data_ptr);
      } else {
        output->MallocData(allocator_);
      }
    }

    auto ret = kernel->Run();
    if (0 != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
      return ret;
    }

    if (after != nullptr) {
      if (!after(PackToMSTensors(kernel->in_tensors()), PackToMSTensors(kernel->out_tensors()), callbackParam)) {
        MS_LOG(ERROR) << "run kernel after_callback failed, name: " << kernel->name();
      }
    }
    for (auto input_kernel : kernel->in_kernels()) {
      MS_EXCEPTION_IF_NULL(input_kernel);
      ret = input_kernel->DecOutTensorRefCount();
      if (0 != ret) {
        MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
      }
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::opencl
