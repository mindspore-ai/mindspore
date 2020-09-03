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

#include "src/runtime/kernel/opencl/kernel/scale.h"
#include <set>
#include <vector>
#include <string>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/common_func.h"
#include "src/runtime/kernel/opencl/utils.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/scale.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Scale;

namespace mindspore::kernel {

ScaleOpenCLKernel::~ScaleOpenCLKernel() {
  auto allocator = ocl_runtime_->GetAllocator();
  if (scale_ptr_ != nullptr) {
    allocator->Free(scale_ptr_);
    scale_ptr_ = nullptr;
  }
  if (offset_ptr_ != nullptr) {
    allocator->Free(offset_ptr_);
    offset_ptr_ = nullptr;
  }
}

std::vector<size_t> ScaleOpenCLKernel::InitGlobalSize() const {
  const size_t global_x = out_tensors_[0]->Width();
  const size_t global_y = out_tensors_[0]->Height();
  const size_t global_z = UP_ROUND_DIV(out_tensors_[0]->Channel(), C4NUM);
  std::vector<size_t> global = {global_x, global_y, global_z};
  return global;
}

void ScaleOpenCLKernel::Image2dGetWorkGroupSize() {
  local_size_ = {16, 16};
  if (out_tensors_[0]->GetFormat() == schema::Format_NC4HW4) {
    size_t H = out_tensors_[0]->Batch() * out_tensors_[0]->Height() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    size_t W = out_tensors_[0]->Width();
    global_size_ = {W, H};
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
    size_t H = out_tensors_[0]->Batch() * out_tensors_[0]->Height();
    size_t W = out_tensors_[0]->Width() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    global_size_ = {W, H};
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NC4) {
    size_t H = out_tensors_[0]->Batch();
    size_t W = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    global_size_ = {W, H};
  } else {
    MS_LOG(ERROR) << "Unsupport data format " << out_tensors_[0]->GetFormat();
  }
}

void ScaleOpenCLKernel::BufferGetWorkGroupSize() {
  uint32_t element_num = out_tensors_[0]->ElementsC4Num();
  global_size_ = {element_num};
}

int ScaleOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  if (out_tensors_[0]->GetFormat() == schema::Format_NC4HW4) {
    im_dst_x = out_tensors_[0]->Width();
    im_dst_y = out_tensors_[0]->Batch() * out_tensors_[0]->Height() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
    im_dst_x = out_tensors_[0]->Width() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    im_dst_y = out_tensors_[0]->Batch() * out_tensors_[0]->Height();
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NC4) {
    im_dst_y = out_tensors_[0]->Batch();
    im_dst_x = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  } else {
    MS_LOG(ERROR) << "Unsupport data format " << out_tensors_[0]->GetFormat();
    return RET_ERROR;
  }

  size_t img_dtype = CL_FLOAT;
  if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
    img_dtype = CL_HALF_FLOAT;
  } else if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    img_dtype = CL_FLOAT;
  } else {
    MS_LOG(ERROR) << "Unsupport data type " << in_tensors_[0]->data_type();
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

int ScaleOpenCLKernel::InitBuffer() {
  if (!element_flag_) {
    return RET_OK;
  }
  if (in_tensors_[1]->TensorType() == schema::NodeType_ValueNode && in_tensors_[1]->Data() != nullptr) {
    auto allocator = ocl_runtime_->GetAllocator();
    std::vector<size_t> img_size;
    GetImageSize(0, &img_size);
    if (in_tensors_[1]->shape().size() == 1 && axis_ == 3) {
      img_size[0] = 1;
      img_size[1] = UP_DIV(in_tensors_[1]->shape()[0], C4NUM);
      scale_ptr_ = allocator->CreateImageFromHost(in_tensors_[1]->Data(), in_tensors_[1]->ElementsNum(), img_size);
      offset_ptr_ = allocator->CreateImageFromHost(in_tensors_[2]->Data(), in_tensors_[2]->ElementsNum(), img_size);
      return RET_OK;
    }
    int pack_weight_size = in_tensors_[1]->ElementsC4Num();
    int plane = in_tensors_[1]->Height() * in_tensors_[1]->Width();
    int channel = in_tensors_[1]->Channel();
    int batch = in_tensors_[1]->Batch();
    if (in_tensors_[0]->GetFormat() == in_tensors_[1]->GetFormat()) {
      if (in_tensors_[0]->data_type() == in_tensors_[1]->data_type()) {
        scale_ptr_ = allocator->CreateImageFromHost(in_tensors_[1]->Data(), in_tensors_[1]->ElementsNum(), img_size);
        offset_ptr_ = allocator->CreateImageFromHost(in_tensors_[2]->Data(), in_tensors_[2]->ElementsNum(), img_size);
      } else {
        MS_LOG(ERROR) << "Unsupport data type transpose from " << in_tensors_[1]->data_type() << "to "
                      << in_tensors_[0]->data_type();
        return RET_ERROR;
      }
    } else if (in_tensors_[0]->GetFormat() == schema::Format_NC4HW4) {
      if (in_tensors_[1]->GetFormat() == schema::Format_NHWC) {
        if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
          float *scale = new (std::nothrow) float[pack_weight_size];
          if (scale == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            return RET_ERROR;
          }
          float *offset = new (std::nothrow) float[pack_weight_size];
          if (offset == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            delete[] scale;
            return RET_ERROR;
          }
          std::function<float(float)> to_dtype = [](float x) -> float { return (float)x; };
          PackNHWCToNC4HW4<float, float>(in_tensors_[1]->Data(), scale, batch, plane, channel, to_dtype);
          PackNHWCToNC4HW4<float, float>(in_tensors_[2]->Data(), offset, batch, plane, channel, to_dtype);
          scale_ptr_ = allocator->CreateImageFromHost(scale, in_tensors_[1]->ElementsNum(), img_size);
          offset_ptr_ = allocator->CreateImageFromHost(offset, in_tensors_[2]->ElementsNum(), img_size);
          delete[] scale;
          delete[] offset;
        } else if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
          int16_t *scale = new (std::nothrow) int16_t[pack_weight_size];
          if (scale == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            return RET_ERROR;
          }
          int16_t *offset = new (std::nothrow) int16_t[pack_weight_size];
          if (offset == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            delete[] scale;
            return RET_ERROR;
          }
          std::function<int16_t(float)> to_dtype = Float32ToShort;
          PackNHWCToNC4HW4<float, int16_t>(in_tensors_[1]->Data(), scale, batch, plane, channel, to_dtype);
          PackNHWCToNC4HW4<float, int16_t>(in_tensors_[2]->Data(), offset, batch, plane, channel, to_dtype);
          scale_ptr_ = allocator->CreateImageFromHost(scale, in_tensors_[1]->ElementsNum(), img_size);
          offset_ptr_ = allocator->CreateImageFromHost(offset, in_tensors_[2]->ElementsNum(), img_size);
          delete[] scale;
          delete[] offset;
        } else {
          MS_LOG(ERROR) << "Unsupport data type transpose from " << in_tensors_[1]->data_type() << "to "
                        << in_tensors_[0]->data_type();
          return RET_ERROR;
        }
      } else {
        MS_LOG(ERROR) << "Unsupport format transpose from " << in_tensors_[1]->GetFormat() << "to "
                      << in_tensors_[0]->GetFormat();
        return RET_ERROR;
      }
    } else if (in_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
      if (in_tensors_[1]->GetFormat() == schema::Format_NHWC) {
        if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
          float *scale = new (std::nothrow) float[pack_weight_size];
          if (scale == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            return RET_ERROR;
          }
          float *offset = new (std::nothrow) float[pack_weight_size];
          if (offset == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            delete[] scale;
            return RET_ERROR;
          }
          std::function<float(float)> to_dtype = [](float x) -> float { return (float)x; };
          PackNHWCToNHWC4<float, float>(in_tensors_[1]->Data(), scale, batch, plane, channel, to_dtype);
          PackNHWCToNHWC4<float, float>(in_tensors_[2]->Data(), offset, batch, plane, channel, to_dtype);
          scale_ptr_ = allocator->CreateImageFromHost(scale, in_tensors_[1]->ElementsNum(), img_size);
          offset_ptr_ = allocator->CreateImageFromHost(offset, in_tensors_[2]->ElementsNum(), img_size);
          delete[] scale;
          delete[] offset;
        } else if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
          int16_t *scale = new (std::nothrow) int16_t[pack_weight_size];
          if (scale == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            return RET_ERROR;
          }
          int16_t *offset = new (std::nothrow) int16_t[pack_weight_size];
          if (offset == nullptr) {
            MS_LOG(ERROR) << "Malloc buffer failed!";
            delete[] scale;
            return RET_ERROR;
          }
          std::function<int16_t(float)> to_dtype = Float32ToShort;
          PackNHWCToNHWC4<float, int16_t>(in_tensors_[1]->Data(), scale, batch, plane, channel, to_dtype);
          PackNHWCToNHWC4<float, int16_t>(in_tensors_[2]->Data(), offset, batch, plane, channel, to_dtype);
          scale_ptr_ = allocator->CreateImageFromHost(scale, in_tensors_[1]->ElementsNum(), img_size);
          offset_ptr_ = allocator->CreateImageFromHost(offset, in_tensors_[2]->ElementsNum(), img_size);
          delete[] scale;
          delete[] offset;
        } else {
          MS_LOG(ERROR) << "Unsupport data type transpose from " << in_tensors_[1]->data_type() << "to "
                        << in_tensors_[0]->data_type();
          return RET_ERROR;
        }
      } else {
        MS_LOG(ERROR) << "Unsupport format transpose from " << in_tensors_[1]->GetFormat() << "to "
                      << in_tensors_[0]->GetFormat();
        return RET_ERROR;
      }
    }
    return RET_OK;
  }
  return RET_OK;
}

int ScaleOpenCLKernel::Init() {
  ocl_runtime_ = lite::opencl::OpenCLRuntime::GetInstance();
  std::string kernel_name;

  const ScaleParameter *scale_param = reinterpret_cast<const ScaleParameter *>(op_parameter_);
  auto in_tensor = in_tensors_.at(0);
  auto in_shape = in_tensor->shape();
  auto scale_tensor = in_tensors_.at(1);
  auto scale_shape = scale_tensor->shape();
  axis_ = scale_param->axis_;
  if (axis_ < 0) {
    axis_ = axis_ + in_shape.size();
  }
  if (scale_shape.size() != in_shape.size()) {
    if (scale_tensor->ElementsNum() == 1) {
      element_flag_ = false;
      kernel_name = "BoardcastScale";
    } else if (axis_ == 3 && scale_shape.size() == 1) {
      element_flag_ = true;
      kernel_name = "Scale_C";
    }
  } else {
    element_flag_ = true;
    kernel_name = "Scale";
  }
  lite::STATUS error_code = RET_OK;
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  if (out_mem_type_ == OpenCLMemType::IMG) {
    kernel_name += "_IMG";
  } else {
    kernel_name += "_BUF";
  }
  std::string program_name = "Scale";
  std::set<std::string> build_options;
  std::string source = scale_source;
  ocl_runtime_->LoadSource(program_name, source);
  error_code = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  if (error_code != RET_OK) {
    return error_code;
  }

  auto format = op_format_;
  if (out_tensors_[0]->shape().size() == 2) {
    format = schema::Format_NC4;
  }
  in_ori_format_ = in_tensors_[0]->GetFormat();
  out_ori_format_ = out_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(format);
  if (element_flag_ && in_tensors_[1]->TensorType() != schema::NodeType_ValueNode) {
    in_tensors_[1]->SetFormat(format);
    in_tensors_[2]->SetFormat(format);
  }
  out_tensors_[0]->SetFormat(format);
  Image2dGetWorkGroupSize();
  InitBuffer();
  return RET_OK;
}

int ScaleOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";

  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->Data());
  if (element_flag_) {
    void *scale = scale_ptr_ == nullptr ? in_tensors_[1]->Data() : scale_ptr_;
    void *offset = offset_ptr_ == nullptr ? in_tensors_[2]->Data() : offset_ptr_;
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, scale);
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, offset);
  } else {
    if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
      float scale = static_cast<float *>(in_tensors_[1]->Data())[0];
      float offset = static_cast<float *>(in_tensors_[2]->Data())[0];
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, scale);
      ocl_runtime_->SetKernelArg(kernel_, arg_idx++, offset);
    } else if (in_tensors_[0]->data_type() == kNumberTypeFloat16) {
      if (in_tensors_[1]->data_type() == kNumberTypeFloat32) {
        float scale = static_cast<float *>(in_tensors_[1]->Data())[0];
        float offset = static_cast<float *>(in_tensors_[2]->Data())[0];
        ocl_runtime_->SetKernelArg(kernel_, arg_idx++, Float32ToShort(scale));
        ocl_runtime_->SetKernelArg(kernel_, arg_idx++, Float32ToShort(offset));
      } else if (in_tensors_[1]->data_type() == kNumberTypeFloat16) {
        int16_t scale = static_cast<int16_t *>(in_tensors_[1]->Data())[0];
        int16_t offset = static_cast<int16_t *>(in_tensors_[2]->Data())[0];
        ocl_runtime_->SetKernelArg(kernel_, arg_idx++, Float32ToShort(scale));
        ocl_runtime_->SetKernelArg(kernel_, arg_idx++, Float32ToShort(offset));
      } else {
        MS_LOG(ERROR) << "Unsupport data type " << in_tensors_[1]->data_type();
        return RET_ERROR;
      }
    }
  }
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->Data());
  int H = 0;
  int W = 0;
  if (out_tensors_[0]->GetFormat() == schema::Format_NC4HW4) {
    H = out_tensors_[0]->Batch() * out_tensors_[0]->Height() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
    W = out_tensors_[0]->Width();
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
    H = out_tensors_[0]->Batch() * out_tensors_[0]->Height();
    W = out_tensors_[0]->Width() * UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  } else if (out_tensors_[0]->GetFormat() == schema::Format_NC4) {
    H = out_tensors_[0]->Batch();
    W = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  } else {
    MS_LOG(ERROR) << "Error output type " << out_tensors_[0]->GetFormat();
    return RET_ERROR;
  }
  cl_int2 output_shape{W, H};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, output_shape);
  if (element_flag_ && axis_ == 3) {
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, UP_DIV(in_tensors_[1]->shape()[0], C4NUM));
  }
  ocl_runtime_->RunKernel(kernel_, global_size_, local_size_, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLScaleKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *opParameter, const lite::Context *ctx,
                                             const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel =
    new (std::nothrow) ScaleOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Scale kernel failed!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: Scale";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Scale, OpenCLScaleKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Scale, OpenCLScaleKernelCreator)
}  // namespace mindspore::kernel
