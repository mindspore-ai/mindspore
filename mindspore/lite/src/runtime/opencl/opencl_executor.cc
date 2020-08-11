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
#include "src/runtime/kernel/arm/nnacl/pack.h"
#include "src/common/ms_tensor_utils.h"
#include "include/errorcode.h"

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
    if (inTensor->GetFormat() != schema::Format_NHWC4 && inTensor->GetFormat() != schema::Format_NC4HW4 &&
      inTensor->GetFormat() != schema::Format_NHWC) {
      MS_LOG(ERROR) << "input should be NHWC/NHWC4/NC4HW4, actual is " << schema::EnumNameFormat(inTensor->GetFormat());
      return RET_ERROR;
    } else {
      TransformTensorLayout(inTensor, inTensor->GetFormat(), schema::Format_NHWC4, true);
      // TransformTensorLayout(inTensor, inTensor->GetFormat(), schema::Format_NC4HW4, true);
    }
  }
  kernel::LiteKernelUtil::InitTensorRefCount(kernels);
  OpenCLAllocator* op_allocator = reinterpret_cast<OpenCLAllocator*>(allocator);
  for (auto *kernel : kernels) {
    MS_ASSERT(nullptr != kernel);
    kernel::OpenCLKernel *op_kernel = reinterpret_cast<kernel::OpenCLKernel*>(kernel);
    auto &outputs = kernel->out_tensors();
    for (auto i = 0; i < outputs.size(); ++i) {
      auto *output = outputs.at(i);
      MS_ASSERT(nullptr != output);
      if (is_image2d_out_) {
        std::vector<size_t> img_size;
        op_kernel->GetImageSize(i, &img_size);
        auto data_ptr = op_allocator->Malloc(output->Size(), img_size);

        output->SetData(data_ptr);
      } else {
        output->MallocData(allocator);
      }
      output->set_allocator(allocator);
    }
    session::CallBackParam callbackParam;
    callbackParam.name_callback_param = kernel->name();

    if (before != nullptr) {
      if (!before(PackToMSTensors(kernel->in_tensors()), PackToMSTensors(kernel->out_tensors()), callbackParam)) {
        MS_LOG(ERROR) << "run kernel before_callback failed, name: " << kernel->name();
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
  // output format transform
  for (auto &outTensor : outputs) {
    if (outTensor == nullptr) {
      MS_LOG(ERROR) << "Graph output tensor is nullptr";
      return RET_ERROR;
    }
    if (outTensor->GetFormat() != schema::Format_NHWC) {
      TransformTensorLayout(outTensor, outTensor->GetFormat(), schema::Format_NHWC, false);
    }
  }
  return RET_OK;
}

int OpenCLExecutor::TransformTensorLayout(tensor::Tensor *tensor, schema::Format src_format,
    schema::Format dst_format, bool trans_dir) {
  MS_ASSERT(nullptr != tensor);
  MS_ASSERT(4 == tensor->shape().size());
  auto data_type = tensor->data_type();
  switch (data_type) {
    case kNumberTypeInt8:
      return TransformTensorLayoutUint8(tensor, src_format, dst_format, trans_dir);
    case kNumberTypeFloat32:
      return TransformTensorLayoutFp32(tensor, src_format, dst_format, trans_dir);
    default:
      MS_LOG(ERROR) << "Unsupported layout transform: " << schema::EnumNameFormat(tensor->GetFormat()) << " to "
                    << schema::EnumNameFormat(dst_format);
      return RET_ERROR;
  }
  return RET_OK;
}

int OpenCLExecutor::TransformTensorLayoutFp32(tensor::Tensor *tensor, schema::Format src_format,
    schema::Format dst_format, bool trans_dir) {
  MS_ASSERT(nullptr != tensor);
  MS_ASSERT(nullptr != allocator_);
  MS_ASSERT(4 == tensor->shape().size());
  if (trans_dir) {
    if (is_image2d_out_) {
      return TransformTensorLayoutToImage(tensor, src_format, dst_format);
    } else {
      return TransformTensorLayoutToBuffer(tensor, src_format, dst_format);
    }
  } else {
    if (is_image2d_out_) {
      return TransformTensorLayoutFromImage(tensor, src_format, dst_format);
    } else {
      return TransformTensorLayoutToBuffer(tensor, src_format, dst_format);
    }
  }
}

int OpenCLExecutor::TransformTensorLayoutToBuffer(tensor::Tensor *tensor, schema::Format src_format,
    schema::Format dst_format) {
  if (dst_format == schema::Format_NHWC4) {
    auto *src_data = tensor->Data();
    size_t C4 = UP_DIV(tensor->Channel(), C4NUM);
    std::vector <size_t> img_size{tensor->Width() * C4, (size_t) tensor->Height(), CL_FLOAT};
    if (src_format == schema::Format_NHWC) {
      auto *dst_data = allocator_->Malloc(tensor->Size(), img_size);
      if (dst_data == nullptr) {
        MS_LOG(ERROR) << "Malloc data failed";
        return RET_ERROR;
      }
      dst_data = reinterpret_cast<FLOAT_t *>(allocator_->MapBuffer(dst_data, CL_MAP_WRITE, nullptr, true));
      PackNHWCToNHWC4Fp32(src_data, dst_data, tensor->Batch(), tensor->Height() * tensor->Width(), tensor->Channel());
      tensor->SetData(dst_data);
      allocator_->Free(src_data);
      allocator_->UnmapBuffer(dst_data);
    }
    tensor->SetFormat(dst_format);
    return RET_OK;
  } else if (dst_format == schema::Format_NHWC) {
    // TODO(wandongdong): add support !!
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Unsupported layout transform: " << schema::EnumNameFormat(tensor->GetFormat()) << " to "
                  << schema::EnumNameFormat(dst_format) << " in float32";
    return RET_ERROR;
  }
}

int OpenCLExecutor::TransformTensorLayoutToImage(tensor::Tensor *tensor, schema::Format src_format,
    schema::Format dst_format) {
  if (dst_format == schema::Format_NHWC4) {
    tensor->SetFormat(schema::Format_NHWC4);
    // convert to nhwc4
    auto *src_data = tensor->Data();
    auto *dst_data{src_data};
    if (src_format == schema::Format_NHWC) {
      dst_data = allocator_->Malloc(tensor->Size());
      if (dst_data == nullptr) {
        MS_LOG(ERROR) << "Malloc data failed";
        return RET_ERROR;
      }
      dst_data = reinterpret_cast<FLOAT_t *>(allocator_->MapBuffer(dst_data, CL_MAP_WRITE, nullptr, true));
      PackNHWCToNHWC4Fp32(src_data, dst_data, tensor->Batch(), tensor->Height() * tensor->Width(), tensor->Channel());
      tensor->SetData(dst_data);
      allocator_->Free(src_data);
      allocator_->UnmapBuffer(dst_data);
    }
    // copy to image2d
    src_data = dst_data;
    size_t C4 = UP_DIV(tensor->Channel(), C4NUM);
    std::vector<size_t> img_size{tensor->Width() * C4, (size_t)tensor->Height(), CL_FLOAT};
    dst_data = allocator_->CreateImageFromHost(src_data, tensor->Size(), img_size);
    tensor->SetData(dst_data);
    allocator_->Free(src_data);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Unsupported layout transform: " << schema::EnumNameFormat(tensor->GetFormat()) << " to "
                  << schema::EnumNameFormat(dst_format) << " in float32";
    return RET_ERROR;
  }
}

int OpenCLExecutor::TransformTensorLayoutFromImage(tensor::Tensor *tensor, schema::Format src_format,
    schema::Format dst_format) {
  if (dst_format == schema::Format_NHWC) {
    auto src_data = tensor->Data();
    auto dst_data = allocator_->Malloc(tensor->Size());
    cl::Image2D *out_mem = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(src_data));
    std::vector<size_t> img_size;
    allocator_->GetImageSize(src_data, &img_size);
    auto origin = cl::array < cl::size_type, 3U > {0, 0, 0};
    auto region = cl::array < cl::size_type, 3U > {img_size[0], img_size[1], 1};
    auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
    ocl_runtime->GetDefaultCommandQueue()->enqueueReadImage(*out_mem, CL_TRUE, origin, region, 0, 0, dst_data);
    tensor->SetData(dst_data);
    allocator_->Free(src_data);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Unsupported layout transform: " << schema::EnumNameFormat(tensor->GetFormat()) << " to "
                  << schema::EnumNameFormat(dst_format) << " in float32";
    return RET_ERROR;
  }
}

int OpenCLExecutor::TransformTensorLayoutUint8(tensor::Tensor *tensor, schema::Format src_format,
    schema::Format dst_format, bool is_image) {
  MS_ASSERT(nullptr != tensor);
  MS_ASSERT(4 == tensor->shape().size());
  //  auto src_format = tensor->GetFormat();
  // todo
  MS_LOG(ERROR) << "Unsupported layout transform: " << schema::EnumNameFormat(tensor->GetFormat()) << " to "
                << schema::EnumNameFormat(dst_format) << " in uint8";
  return RET_ERROR;
}
}  // namespace mindspore::lite::opencl

