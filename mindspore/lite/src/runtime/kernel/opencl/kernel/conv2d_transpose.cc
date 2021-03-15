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

#include "src/runtime/kernel/opencl/kernel/conv2d_transpose.h"
#include <string>
#include <set>
#include "nnacl/fp32/common_func_fp32.h"
#include "src/kernel_registry.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/conv2d_transpose.cl.inc"
#endif
#include "src/runtime/kernel/opencl/utils.h"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;

namespace mindspore::kernel {

int Conv2dTransposeOpenCLKernel::CheckSpecs() {
  if ((in_tensors_.size() != 2 && in_tensors_.size() != 3) || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  if (param->act_type_ != ActType_No && param->act_type_ != ActType_Relu && param->act_type_ != ActType_Relu6) {
    MS_LOG(ERROR) << "Unsupported activation type " << param->act_type_;
    return RET_ERROR;
  }
  if (!in_tensors_.at(1)->IsConst()) {
    MS_LOG(ERROR) << "Conv2dTranspose doesn't support non-constant filter yet.";
    return RET_ERROR;
  }
  if (in_tensors_.size() == 3 && !in_tensors_.at(2)->IsConst()) {
    MS_LOG(ERROR) << "Conv2dTranspose doesn't support non-constant bias yet.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Conv2dTransposeOpenCLKernel::Prepare() {
  std::string kernel_name = "conv2d_transpose";
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::string source = GetActDefines() + conv2d_transpose_source;
  std::string program_name = "conv2d_transpose";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
#endif
  auto ret = InitWeights();
  if (ret != RET_OK) {
    return ret;
  }
  SetGlobalLocal();
  SetConstArgs();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return mindspore::lite::RET_OK;
}

void Conv2dTransposeOpenCLKernel::SetGlobalLocal() {
  auto *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  int co = out_tensors_[0]->shape()[3];
  int co4 = UP_DIV(co, C4NUM);
  int stride_h = param->stride_h_;
  int stride_w = param->stride_w_;
  int n = out_tensors_[0]->shape()[0];
  int oh = out_tensors_[0]->shape()[1];
  int ow = out_tensors_[0]->shape()[2];
  local_size_ = {16, 1, 16};
  global_size_ = {(size_t)UP_ROUND(UP_DIV(oh, 2), stride_h), (size_t)UP_ROUND(UP_DIV(ow, 2), stride_w),
                  (size_t)co4 * (size_t)n};
  AlignGlobalLocal(global_size_, local_size_);
}

void Conv2dTransposeOpenCLKernel::SetConstArgs() {
  int arg_cnt = 2;
  auto *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  int ci = in_tensors_[0]->shape()[3];
  int co = out_tensors_[0]->shape()[3];
  int kh = param->kernel_h_;
  int kw = param->kernel_w_;
  int pad_h = param->pad_u_;
  int pad_w = param->pad_l_;
  int stride_h = param->stride_h_;
  int stride_w = param->stride_w_;
  int oh = out_tensors_[0]->shape()[1];
  int ow = out_tensors_[0]->shape()[2];
  int n = in_tensors_[0]->shape()[0];
  int h = in_tensors_[0]->shape()[1];
  int w = in_tensors_[0]->shape()[2];
  cl_int2 kernel_size = {kh, kw};
  cl_int2 stride = {stride_h, stride_w};
  cl_int2 padding = {pad_h, pad_w};
  cl_int4 src_size = {h, w, UP_DIV(ci, C4NUM), n};
  cl_int4 dst_size = {oh, ow, UP_DIV(co, C4NUM), n};
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, padWeight_, lite::opencl::MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, bias_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, kernel_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, stride);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, padding);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, src_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, dst_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt, static_cast<cl_int>(param->act_type_));
}

int Conv2dTransposeOpenCLKernel::InitWeights() {
  auto ret = InitFilter();
  if (ret != RET_OK) {
    return ret;
  }
  return InitBias();
}

int Conv2dTransposeOpenCLKernel::InitFilter() {
  auto ret = DequantWeight();
  if (ret != RET_OK) {
    return ret;
  }
  auto *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  int ci = in_tensors_[0]->shape()[3];
  int co = out_tensors_[0]->shape()[3];
  int kh = param->kernel_h_;
  int kw = param->kernel_w_;
  int div_ci = UP_DIV(ci, C4NUM);
  int div_co = UP_DIV(co, C4NUM);
  auto allocator = ocl_runtime_->GetAllocator();
  auto data_size = enable_fp16_ ? sizeof(int16_t) : sizeof(float);

  // IHWO to OHWI4(I)4(O)(converter format is IHWO)
  // init padWeight_(buffer mem)
  padWeight_ = allocator->Malloc(div_ci * div_co * C4NUM * C4NUM * kh * kw * data_size);
  padWeight_ = allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true);
  memset(padWeight_, 0x00, div_ci * div_co * C4NUM * C4NUM * kh * kw * data_size);
  auto origin_weight = in_tensors_.at(kWeightIndex)->data_c();
  auto weight_dtype = in_tensors_.at(kWeightIndex)->data_type();
  int index = 0;
  for (int co_i = 0; co_i < div_co; co_i++) {
    for (int kh_i = 0; kh_i < kh; kh_i++) {
      for (int kw_i = 0; kw_i < kw; kw_i++) {
        for (int ci_i = 0; ci_i < div_ci; ci_i++) {
          for (int ci4_i = 0; ci4_i < C4NUM; ci4_i++) {
            for (int co4_i = 0; co4_i < C4NUM; co4_i++) {
              int co_offset = co_i * C4NUM + co4_i;
              int ci_offset = ci_i * C4NUM + ci4_i;
              if (co_offset < co && ci_offset < ci) {
                int ori_index = ((ci_offset * kh + kh_i) * kw + kw_i) * co + co_offset;
                if (enable_fp16_) {
                  if (weight_dtype == kNumberTypeFloat32) {
                    reinterpret_cast<float16_t *>(padWeight_)[index++] =
                      reinterpret_cast<float *>(origin_weight)[ori_index];
                  } else {
                    reinterpret_cast<float16_t *>(padWeight_)[index++] =
                      reinterpret_cast<float16_t *>(origin_weight)[ori_index];
                  }
                } else {
                  if (weight_dtype == kNumberTypeFloat32) {
                    reinterpret_cast<float *>(padWeight_)[index++] =
                      reinterpret_cast<float *>(origin_weight)[ori_index];
                  } else {
                    reinterpret_cast<float *>(padWeight_)[index++] =
                      reinterpret_cast<float16_t *>(origin_weight)[ori_index];
                  }
                }
              } else {
                index++;
              }
            }
          }
        }
      }
    }
  }
  allocator->UnmapBuffer(padWeight_);
  FreeDequantedWeight();
  return RET_OK;
}

int Conv2dTransposeOpenCLKernel::InitBias() {
  // init bias_(image2d mem)
  auto allocator = ocl_runtime_->GetAllocator();
  auto data_size = enable_fp16_ ? sizeof(int16_t) : sizeof(float);
  int co = out_tensors_[0]->shape()[3];
  int div_co = UP_DIV(co, C4NUM);
  size_t im_dst_x, im_dst_y;
  im_dst_x = div_co;
  im_dst_y = 1;
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  ImageSize img_size{im_dst_x, im_dst_y, img_dtype};
  bias_ = allocator->Malloc(img_size);
  bias_ = allocator->MapBuffer(bias_, CL_MAP_WRITE, nullptr, true);
  memset(bias_, 0x00, div_co * C4NUM * data_size);
  if (in_tensors_.size() == 3) {
    auto bias_dtype = in_tensors_[2]->data_type();
    if (bias_dtype == kNumberTypeFloat32 && enable_fp16_) {
      for (int i = 0; i < co; i++) {
        reinterpret_cast<float16_t *>(bias_)[i] = reinterpret_cast<float *>(in_tensors_[2]->data_c())[i];
      }
    } else if (bias_dtype == kNumberTypeFloat16 && !enable_fp16_) {
      for (int i = 0; i < co; i++) {
        reinterpret_cast<float *>(bias_)[i] = reinterpret_cast<float16_t *>(in_tensors_[2]->data_c())[i];
      }
    } else {
      memcpy(bias_, in_tensors_[2]->data_c(), co * data_size);
    }
  }
  allocator->UnmapBuffer(bias_);
  return RET_OK;
}

int Conv2dTransposeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  int arg_cnt = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return mindspore::lite::RET_OK;
}

int Conv2dTransposeOpenCLKernel::InferShape() {
  auto ret = OpenCLKernel::InferShape();
  if (ret != RET_OK) {
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *OpenCLConv2dTransposeCreator(const std::vector<lite::Tensor *> &inputs,
                                                 const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                 const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(!inputs.empty());
  MS_ASSERT(!outputs.empty());
  MS_ASSERT(opParameter);
  MS_ASSERT(inputs.front()->shape().size() == 4);
  MS_ASSERT(outputs.front()->shape().size() == 4);
  auto *conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  int input_channel = inputs.front()->shape().at(3);
  int output_channel = outputs.front()->shape().at(3);
  int group = conv_param->group_;

  // case 1: depthwise Conv2dTranspose
  if (group == input_channel && group == output_channel) {
    MS_LOG(ERROR) << "OpenCL doesn't support depthwise Conv2dTranspose.";
    free(conv_param);
    return nullptr;
  }

  // case 2: group Conv2dTranspose
  if (group != 1) {
    MS_LOG(ERROR) << "OpenCL doesn't support group Conv2dTranspose.";
    free(conv_param);
    return nullptr;
  }

  // case 3: common Conv2dTranspose
  return OpenCLKernelCreator<Conv2dTransposeOpenCLKernel>(inputs, outputs, opParameter, ctx, desc);
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Conv2dTransposeFusion, OpenCLConv2dTransposeCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Conv2dTransposeFusion, OpenCLConv2dTransposeCreator)
}  // namespace mindspore::kernel
