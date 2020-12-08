/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/opencl/kernel/depthwise_conv2d.h"
#include <float.h>
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include <utility>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "nnacl/fp32/common_func_fp32.h"
#include "nnacl/op_base.h"
#include "include/errorcode.h"

#ifndef PROGRAM_WITH_IL

#include "src/runtime/kernel/opencl/cl/depthwise_conv2d.cl.inc"

#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::MemType;
using mindspore::schema::PrimitiveType_DepthwiseConv2D;

namespace mindspore::kernel {

int DepthwiseConv2dOpenCLKernel::CheckSpecs() {
  if ((in_tensors_.size() != 2 && in_tensors_.size() != 3) || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_[0]->data_type() != kNumberTypeFloat32 && in_tensors_[0]->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Unsupported data type " << in_tensors_[0]->data_type();
    return RET_ERROR;
  }
  return RET_OK;
}
int DepthwiseConv2dOpenCLKernel::Prepare() {
  std::string kernel_name = "DepthwiseConv2d";
  if (out_mem_type_ == MemType::BUF) {
    kernel_name += "_BUF";
  } else {
    kernel_name += "_IMG";
  }
  kernel_name += "_NHWC4";
  auto parameter = reinterpret_cast<ConvParameter *>(op_parameter_);
  if (parameter->kernel_h_ == 1) {
    kernel_name += "_1x1";
  }
  kernel_name += "_b";
  for (auto iv : block_size_) {
    kernel_name += std::to_string(iv);
  }
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::string program_name = "DepthwiseConv2d";
  std::string source = depthwise_conv2d_source;
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
#endif
  auto ret = InitWeights();
  if (ret != RET_OK) {
    return ret;
  }
  SetGlobalLocal();
  SetConstArgs();
  MS_LOG(DEBUG) << kernel_name << " Init Done! mem type=" << static_cast<int>(out_mem_type_);
  return mindspore::lite::RET_OK;
}

int DepthwiseConv2dOpenCLKernel::InitWeights() {
  if (!in_tensors_.at(1)->IsConst()) {
    MS_LOG(ERROR) << "DepthwiseConv2d don't support non-constant filter yet.";
    return RET_ERROR;
  }
  auto ret = DequantWeight();
  if (ret != RET_OK) {
    return ret;
  }
  auto parameter = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto allocator = ocl_runtime_->GetAllocator();
  bool is_fp16 = ocl_runtime_->GetFp16Enable();

  // weight: o, h, w, i; o == group, i == 1
  void *origin_weight = in_tensors_.at(kWeightIndex)->data_c();
  int CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  int pack_weight_size = C4NUM * CO4 * parameter->kernel_h_ * parameter->kernel_w_;

  int plane = parameter->kernel_h_ * parameter->kernel_w_;
  if (is_fp16) {
    packed_weight_ = allocator->Malloc(pack_weight_size * sizeof(int16_t));
    packed_weight_ = allocator->MapBuffer(packed_weight_, CL_MAP_WRITE, nullptr, true);
    if (in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat16) {
      std::function<int16_t(int16_t)> to_dtype = [](int16_t x) -> int16_t { return x; };
      PackNCHWToNC4HW4<int16_t, int16_t>(origin_weight, packed_weight_, 1, plane, out_tensors_[0]->Channel(), to_dtype);
    } else if (in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat32) {
      std::function<float16_t(float)> to_dtype = [](float x) -> float16_t { return static_cast<float16_t>(x); };
      PackNCHWToNC4HW4<float, float16_t>(origin_weight, packed_weight_, 1, plane, out_tensors_[0]->Channel(), to_dtype);
    } else {  // int8 or int16
      std::function<int16_t(int16_t)> to_dtype = [](int16_t x) -> int16_t { return x; };
      PackNCHWToNC4HW4<int16_t, int16_t>(origin_weight, packed_weight_, 1, plane, out_tensors_[0]->Channel(), to_dtype);
      FreeDequantedWeight();
    }
  } else {
    packed_weight_ = allocator->Malloc(pack_weight_size * sizeof(float));
    packed_weight_ = allocator->MapBuffer(packed_weight_, CL_MAP_WRITE, nullptr, true);
    if (in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat32) {
      std::function<float(float)> to_dtype = [](float x) -> float { return x; };
      PackNCHWToNC4HW4<float, float>(origin_weight, packed_weight_, 1, plane, out_tensors_[0]->Channel(), to_dtype);
    } else if (in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat16) {
      std::function<float(float16_t)> to_dtype = [](float16_t x) -> float { return static_cast<float>(x); };
      PackNCHWToNC4HW4<float16_t, float>(origin_weight, packed_weight_, 1, plane, out_tensors_[0]->Channel(), to_dtype);
    } else {  // int8 or int16
      std::function<float(float)> to_dtype = [](float x) -> float { return x; };
      PackNCHWToNC4HW4<float, float>(origin_weight, packed_weight_, 1, plane, out_tensors_[0]->Channel(), to_dtype);
      FreeDequantedWeight();
    }
  }

  allocator->UnmapBuffer(packed_weight_);

  if (in_tensors_.size() == kInputSize2) {
    if (!in_tensors_.at(2)->IsConst()) {
      MS_LOG(ERROR) << "DepthwiseConv2d don't support non-constant bias yet.";
      return RET_ERROR;
    }
    size_t dtype_size = sizeof(float);
    if (is_fp16 && in_tensors_.at(kBiasIndex)->data_type() == kNumberTypeFloat16) {
      dtype_size = sizeof(int16_t);
    }
    bias_data_ = allocator->Malloc(C4NUM * CO4 * dtype_size);
    bias_data_ = allocator->MapBuffer(bias_data_, CL_MAP_WRITE, nullptr, true);
    size_t up_co_size = C4NUM * CO4 * dtype_size;
    memset(bias_data_, 0, up_co_size);
    auto ori_bias = in_tensors_.at(kBiasIndex)->data_c();
    if (is_fp16 && in_tensors_.at(kBiasIndex)->data_type() == kNumberTypeFloat32) {
      float16_t *bias_ptr = static_cast<float16_t *>(bias_data_);
      for (size_t i = 0; i < in_tensors_.at(kBiasIndex)->ElementsNum(); ++i) {
        bias_ptr[i] = static_cast<float16_t>(static_cast<float *>(ori_bias)[i]);
      }
    } else {
      memcpy(bias_data_, ori_bias, out_tensors_[0]->Channel() * dtype_size);
    }
    allocator->UnmapBuffer(bias_data_);
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  return mindspore::lite::RET_OK;
}
void DepthwiseConv2dOpenCLKernel::SetConstArgs() {
  auto parameter = reinterpret_cast<ConvParameter *>(op_parameter_);
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t CI4 = UP_DIV(in_tensors_[0]->Channel(), C4NUM);

  std::map<ActType, std::pair<float, float>> relu_clips{
    {ActType_No, {-FLT_MAX, FLT_MAX}}, {ActType_Relu, {0.0, FLT_MAX}}, {ActType_Relu6, {0, 6.0}}};
  cl_int2 kernel_size = {parameter->kernel_h_, parameter->kernel_w_};
  cl_int2 stride = {parameter->stride_h_, parameter->stride_w_};
  cl_int2 padding = {-parameter->pad_u_, -parameter->pad_l_};
  cl_int2 dilation = {parameter->dilation_h_, parameter->dilation_w_};
  cl_int4 src_size = {in_tensors_[0]->Width(), in_tensors_[0]->Height(), (cl_int)CI4, in_tensors_[0]->Batch()};
  cl_int4 dst_size = {(cl_int)out_tensors_[0]->Width(), (cl_int)out_tensors_[0]->Height(), (cl_int)CO4,
                      (cl_int)out_tensors_[0]->Batch()};

  int arg_cnt = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, packed_weight_, lite::opencl::MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, bias_data_, lite::opencl::MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, kernel_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, stride);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, padding);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, dilation);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, src_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, dst_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, relu_clips[parameter->act_type_].first);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, relu_clips[parameter->act_type_].second);
}
void DepthwiseConv2dOpenCLKernel::SetGlobalLocal() {
  // set global
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM * block_size_[2]);
  global_size_ = {CO4, (size_t)UP_DIV(out_tensors_[0]->Width(), block_size_[1]),
                  (size_t)UP_DIV(out_tensors_[0]->Height(), block_size_[0])};
  // set local
  const int max_group_size = ocl_runtime_->DeviceMaxWorkGroupSize();
  int z = global_size_[0];
  int y = std::min(max_group_size / z, GetMaxDivisorStrategy0(global_size_[2], 8));
  int x = std::max(1, std::min(static_cast<int>(global_size_[1]), max_group_size / (y * z)));
  local_size_ = std::vector<size_t>({static_cast<size_t>(z), static_cast<size_t>(x), static_cast<size_t>(y)});

  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int DepthwiseConv2dOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  ocl_runtime_->SetKernelArg(kernel_, 0, out_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, 1, in_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return mindspore::lite::RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_DepthwiseConv2D, OpenCLKernelCreator<DepthwiseConv2dOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_DepthwiseConv2D, OpenCLKernelCreator<DepthwiseConv2dOpenCLKernel>)
}  // namespace mindspore::kernel
