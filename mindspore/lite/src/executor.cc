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
#include "nnacl/pack.h"
#include "include/errorcode.h"

namespace mindspore::lite {
int Executor::CheckInputs(std::vector<Tensor *> &in_tensors) {
  for (auto &inTensor : in_tensors) {
    if (inTensor == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor is nullptr";
      return RET_ERROR;
    }
    if (inTensor->data_c() == nullptr) {
      MS_LOG(ERROR) << "Graph input tensor data is nullptr";
      return RET_ERROR;
    }
    auto shape = inTensor->shape();
    bool valid = all_of(shape.begin(), shape.end(), [](int i) { return i > 0; });
    if (!valid) {
      MS_LOG(ERROR) << "The shape of input tensor contains zero or negative dimension,"
                    << "check the model and assign the input shape with method Resize().";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int Executor::Run(std::vector<Tensor *> &in_tensors, std::vector<Tensor *> &out_tensors,
                  std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator, const KernelCallBack &before,
                  const KernelCallBack &after) {
  MS_ASSERT(nullptr != allocator);
  auto ret = this->CheckInputs(in_tensors);
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "CheckInputs failed";
    return ret;
  }
  kernel::LiteKernelUtil::InitTensorRefCount(kernels);
#ifdef SUPPORT_TRAIN
  for (auto out_tensor : out_tensors) {  // increase RefCount of output tensors, such that Run will not free them
    out_tensor->set_ref_count(out_tensor->ref_count() + 1);
  }
#endif
  for (auto *kernel : kernels) {
    MS_ASSERT(nullptr != kernel);
    ret = kernel->PreProcess();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "PreProcess kernel failed, name: " << kernel->name();
      return ret;
    }
    ret = kernel->Run(before, after);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << kernel->name();
      return ret;
    }
    ret = kernel->PostProcess();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "PostProcess kernel failed, name: " << kernel->name();
      return ret;
    }
  }
  return RET_OK;
}

int Executor::TransformTensorLayout(Tensor *tensor, schema::Format dst_format, Allocator *allocator) {
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

int Executor::TransformTensorLayoutFp32(Tensor *tensor, schema::Format dst_format, Allocator *allocator) {
  MS_ASSERT(nullptr != tensor);
  MS_ASSERT(nullptr != allocator);
  MS_ASSERT(4 == tensor->shape().size());
  auto src_format = tensor->GetFormat();
  if (src_format == schema::Format::Format_NC4HW4 && dst_format == schema::Format::Format_NHWC) {
    auto *src_data = tensor->data_c();
    if (src_data == nullptr) {
      MS_LOG(ERROR) << "data of tensor is nullptr";
      return RET_ERROR;
    }
    auto *dst_data = allocator->Malloc(tensor->Size());
    if (dst_data == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_ERROR;
    }
    PackNC4HW4ToNHWCFp32(src_data, dst_data, tensor->Batch(), tensor->Height() * tensor->Width(), tensor->Channel());
    tensor->set_data(dst_data);
    tensor->SetFormat(dst_format);
    allocator->Free(src_data);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Unsupported layout transform: " << EnumNameFormat(tensor->GetFormat()) << " to "
                  << EnumNameFormat(dst_format) << " in float32";
    return RET_ERROR;
  }
}

int Executor::TransformTensorLayoutUint8(Tensor *tensor, schema::Format dst_format, Allocator *allocator) {
  MS_ASSERT(nullptr != tensor);
  MS_ASSERT(nullptr != allocator);
  MS_ASSERT(4 == tensor->shape().size());
  MS_LOG(ERROR) << "Unsupported layout transform: " << EnumNameFormat(tensor->GetFormat()) << " to "
                << EnumNameFormat(dst_format) << " in uint8";
  return RET_ERROR;
}
}  // namespace mindspore::lite
