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

#include "mindspore/lite/src/executor.h"
#include "src/runtime/kernel/arm/nnacl/pack.h"
#include "include/errorcode.h"
#include "src/common/ms_tensor_utils.h"

namespace mindspore::lite {
int Executor::Run(std::vector<tensor::Tensor *> &in_tensors, std::vector<tensor::Tensor *> &out_tensors,
                  std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator,
                  const session::KernelCallBack &before, const session::KernelCallBack &after) {
  MS_ASSERT(nullptr != allocator);
  for (auto &inTensor : in_tensors) {
    if (inTensor == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor is nullptr";
      return RET_ERROR;
    }
    if (inTensor->GetFormat() != schema::Format_NHWC) {
      MS_LOG(ERROR) << "Model input tensor should be NHWC";
      return RET_ERROR;
    }
  }
  kernel::LiteKernelUtil::InitTensorRefCount(kernels);
  for (auto *kernel : kernels) {
    MS_ASSERT(nullptr != kernel);

    if (before != nullptr) {
      if (!before(PackToMSTensors(kernel->in_tensors()), PackToMSTensors(kernel->out_tensors()),
                  {kernel->name(), kernel->type_str()})) {
        MS_LOG(ERROR) << "run kernel before_callback failed, name: " << kernel->name();
      }
    }
    auto ret = kernel->Run();
    if (0 != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
      return ret;
    }

    if (after != nullptr) {
      if (!after(PackToMSTensors(kernel->in_tensors()), PackToMSTensors(kernel->out_tensors()),
                 {kernel->name(), kernel->type_str()})) {
        MS_LOG(ERROR) << "run kernel after_callback failed, name: " << kernel->name();
      }
    }
    for (auto input_kernel : kernel->in_kernels()) {
      MS_ASSERT(input_kernel != nullptr);
      if (input_kernel->is_model_output()) {
        continue;
      }
      ret = input_kernel->DecOutTensorRefCount();
      if (0 != ret) {
        MS_LOG(WARNING) << "DecOutTensorRefCount for kernel" << kernel->name() << " failed";
      }
    }
  }
  return RET_OK;
}

int Executor::TransformTensorLayout(tensor::Tensor *tensor, schema::Format dst_format, Allocator *allocator) {
  MS_ASSERT(nullptr != tensor);
  MS_ASSERT(nullptr != allocator);
  MS_ASSERT(4 == tensor->shape().size());
  auto data_type = tensor->data_type();
  switch (data_type) {
    case kNumberTypeInt8:
      return TransformTensorLayoutUint8(tensor, dst_format, allocator);
    case kNumberTypeFloat32:
      return TransformTensorLayoutFp32(tensor, dst_format, allocator);
    default:
      return RET_ERROR;
  }
  return RET_OK;
}

int Executor::TransformTensorLayoutFp32(tensor::Tensor *tensor, schema::Format dst_format, Allocator *allocator) {
  MS_ASSERT(nullptr != tensor);
  MS_ASSERT(nullptr != allocator);
  MS_ASSERT(4 == tensor->shape().size());
  auto src_format = tensor->GetFormat();
  if (src_format == schema::Format_NC4HW4 && dst_format == schema::Format_NHWC) {
    auto *src_data = tensor->Data();
    auto *dst_data = allocator->Malloc(tensor->Size());
    if (dst_data == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_ERROR;
    }
    PackNC4HW4ToNHWCFp32(src_data, dst_data, tensor->Batch(), tensor->Height() * tensor->Width(), tensor->Channel());
    tensor->SetData(dst_data);
    tensor->SetFormat(dst_format);
    allocator->Free(src_data);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Unsupported layout transform: " << schema::EnumNameFormat(tensor->GetFormat()) << " to "
                  << schema::EnumNameFormat(dst_format) << " in float32";
    return RET_ERROR;
  }
}

int Executor::TransformTensorLayoutUint8(tensor::Tensor *tensor, schema::Format dst_format, Allocator *allocator) {
  MS_ASSERT(nullptr != tensor);
  MS_ASSERT(nullptr != allocator);
  MS_ASSERT(4 == tensor->shape().size());
  //  auto src_format = tensor->GetFormat();
  // todo
  MS_LOG(ERROR) << "Unsupported layout transform: " << schema::EnumNameFormat(tensor->GetFormat()) << " to "
                << schema::EnumNameFormat(dst_format) << " in uint8";
  return RET_ERROR;
}
}  // namespace mindspore::lite
