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
#include "src/runtime/kernel/arm/opclib/pack.h"
#include "include/errorcode.h"
#include "src/common/ms_tensor_utils.h"

namespace mindspore::lite::opencl {
int OpenCLExecutor::Run(std::vector<tensor::Tensor *> &inputs, std::vector<tensor::Tensor *> &outputs,
                        std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator,
                        const session::KernelCallBack &before, const session::KernelCallBack &after) {
  MS_ASSERT(nullptr != allocator);
  for (auto &inTensor : inputs) {
    if (inTensor == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor is nullptr";
      return RET_ERROR;
    }
    if (inTensor->GetFormat() != schema::Format_NHWC4 && inTensor->GetFormat() != schema::Format_NC4HW4) {
      if (inTensor->GetFormat() != schema::Format_NHWC) {
        MS_LOG(ERROR) << "Model input should be NHWC, actual is " << schema::EnumNameFormat(inTensor->GetFormat());
        return RET_ERROR;
      } else {
        TransformTensorLayout(inTensor, schema::Format_NHWC4);
        // TransformTensorLayout(inTensor, schema::Format_NC4HW4);
      }
    }
  }
  kernel::LiteKernelUtil::InitTensorRefCount(kernels);
  for (auto *kernel : kernels) {
    MS_ASSERT(nullptr != kernel);
    auto &outputs = kernel->GetOutputs();
    for (auto *output : outputs) {
      MS_ASSERT(nullptr != output);
      output->MallocData();
    }
    session::CallBackParam callbackParam;
    callbackParam.name_callback_aram = kernel->Name();

    if (before != nullptr) {
      if (!before(PackToMSTensors(kernel->GetInputs()), PackToMSTensors(kernel->GetOutputs()), callbackParam)) {
        MS_LOG(ERROR) << "run kernel before_callback failed, name: " << kernel->Name();
      }
    }
    auto ret = kernel->Run();
    if (0 != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel->Name();
      return ret;
    }

    if (after != nullptr) {
      if (!after(PackToMSTensors(kernel->GetInputs()), PackToMSTensors(kernel->GetOutputs()), callbackParam)) {
        MS_LOG(ERROR) << "run kernel after_callback failed, name: " << kernel->Name();
      }
    }
    for (auto input_kernel : kernel->GetInKernels()) {
      MS_EXCEPTION_IF_NULL(input_kernel);
      ret = input_kernel->DecOutTensorRefCount();
      if (0 != ret) {
        MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->Name() << " failed";
      }
    }
  }
  // output format transform
  for (auto &outTensor : outputs) {
    if (outTensor == nullptr) {
      MS_LOG(ERROR) << "Graph output tensor is nullptr";
      return RET_ERROR;
    }
    if (outTensor->GetFormat() != schema::Format_NHWC) {
      MS_LOG(ERROR) << "Model output tensor should be NHWC";
    }
  }
  return RET_OK;
}

int OpenCLExecutor::TransformTensorLayout(tensor::Tensor *tensor, schema::Format dst_format) {
  MS_ASSERT(nullptr != tensor);
  MS_ASSERT(4 == tensor->shape().size());
  auto data_type = tensor->data_type();
  switch (data_type) {
    case kNumberTypeInt8:
      return TransformTensorLayoutUint8(tensor, dst_format);
    case kNumberTypeFloat32:
      return TransformTensorLayoutFp32(tensor, dst_format);
    default:
      MS_LOG(ERROR) << "Unsupport layout transform: " << schema::EnumNameFormat(tensor->GetFormat()) << " to "
                    << schema::EnumNameFormat(dst_format);
      return RET_ERROR;
  }
  return RET_OK;
}

int OpenCLExecutor::TransformTensorLayoutFp32(tensor::Tensor *tensor, schema::Format dst_format) {
  MS_ASSERT(nullptr != tensor);
  MS_ASSERT(nullptr != allocator_);
  MS_ASSERT(4 == tensor->shape().size());
  if (dst_format == schema::Format_NHWC4) {
    auto *src_data = tensor->Data();
    auto *dst_data = allocator_->Malloc(tensor->Size());
    if (dst_data == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_ERROR;
    }
    dst_data = reinterpret_cast<FLOAT_t *>(allocator_->MapBuffer(dst_data, CL_MAP_WRITE, nullptr, true));
    PackNHWCToNHWC4Fp32(src_data, dst_data, tensor->Batch(), tensor->Height() * tensor->Width(), tensor->Channel());
    tensor->SetData(dst_data);
    tensor->SetFormat(dst_format);
    allocator_->Free(src_data);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Unsupport layout transform: " << schema::EnumNameFormat(tensor->GetFormat()) << " to "
                  << schema::EnumNameFormat(dst_format) << " in float32";
    return RET_ERROR;
  }
}

int OpenCLExecutor::TransformTensorLayoutUint8(tensor::Tensor *tensor, schema::Format dst_format) {
  MS_ASSERT(nullptr != tensor);
  MS_ASSERT(4 == tensor->shape().size());
  //  auto src_format = tensor->GetFormat();
  // todo
  MS_LOG(ERROR) << "Unsupport layout transform: " << schema::EnumNameFormat(tensor->GetFormat()) << " to "
                << schema::EnumNameFormat(dst_format) << " in uint8";
  return RET_ERROR;
}
}  // namespace mindspore::lite::opencl

