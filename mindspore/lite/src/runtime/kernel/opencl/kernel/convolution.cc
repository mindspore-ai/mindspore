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

#include <string>
#include <set>
#include <algorithm>
#include "src/common/utils.h"
#include "src/runtime/kernel/opencl/kernel/convolution.h"
#include "src/runtime/kernel/opencl/kernel/fullconnection.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/opencl/cl/convolution.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {

constexpr size_t CI_TILE = C4NUM;
constexpr size_t CO_TILE = C4NUM;

int ConvolutionOpenCLKernel::Init() {
  auto allocator = ocl_runtime_->GetAllocator();
  auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
  std::set<std::string> build_options;
  use_fp16_ = ocl_runtime_->GetFp16Enable();

  auto input_tensor = in_tensors_[0];
  auto output_tensor = out_tensors_[0];

  batch_size_ = input_tensor->Batch();
  CI_ = input_tensor->Channel();
  IH_ = input_tensor->Height();
  IW_ = input_tensor->Width();
  CO_ = output_tensor->Channel();
  OH_ = output_tensor->Height();
  OW_ = output_tensor->Width();
  CI_SLICES_ = UP_DIV(CI_, C4NUM);
  CO_SLICES_ = UP_DIV(CO_, C4NUM);
  KH_ = param->kernel_h_;
  KW_ = param->kernel_w_;
  has_bias_ = in_tensors_.size() == 3;

  // note: TILES_X TILES_Y TILES_XY is only used when use_winograd_=true
  TILES_X_ = UP_DIV(OW_, 4);
  TILES_Y_ = UP_DIV(OH_, 4);
  TILES_XY_ = TILES_X_ * TILES_Y_;
  use_winograd_ = UseWinograd4x4To6x6();

  // build kernel
  std::string program_name = "Convolution";
  ocl_runtime_->LoadSource(program_name, convolution_source);
  if (use_winograd_) {
    MS_LOG(DEBUG) << "use winograd";
    ocl_runtime_->BuildKernel(kernel_4x4to36_, program_name, "Winograd4x4To36", build_options);
    ocl_runtime_->BuildKernel(kernel_conv_, program_name, "WinogradConvolution", build_options);
    ocl_runtime_->BuildKernel(kernel_36to4x4_, program_name, "Winograd36To4x4", build_options);
  } else {
    ocl_runtime_->BuildKernel(kernel_conv_, program_name, "Convolution", build_options);
  }

  // allocate winograd memory
  if (use_winograd_) {
    size_t img_dtype = use_fp16_ ? CL_HALF_FLOAT : CL_FLOAT;

    size_t size = TILES_XY_ * CI_SLICES_ * 36 * sizeof_FLT();
    size_t width = TILES_XY_;
    size_t height = CI_SLICES_ * 36;
    winograd_mem0_ = allocator->Malloc(size, {width, height, img_dtype});

    size = TILES_XY_ * CO_SLICES_ * 36 * sizeof_FLT();
    width = TILES_XY_;
    height = CO_SLICES_ * 36;
    winograd_mem1_ = allocator->Malloc(size, {width, height, img_dtype});
  }

  InitBuffer();

  MS_LOG(DEBUG) << "Convolution Init Done!";
  return RET_OK;
}

int ConvolutionOpenCLKernel::GenerateWinogradWeight() {
  constexpr float Gt[] = {1.0000000000, 1.0000000000, 1.0000000000,  1.0000000000, 1.0000000000,  0.0000000000,
                          0.0000000000, 0.7071067691, -0.7071067691, 1.4142135382, -1.4142135382, 0.0000000000,
                          0.0000000000, 0.4999999702, 0.4999999702,  1.9999998808, 1.9999998808,  1.0000000000};
  constexpr float G[] = {1.0000000000, 0.0000000000,  0.0000000000, 1.0000000000, 0.7071067691, 0.4999999702,
                         1.0000000000, -0.7071067691, 0.4999999702, 1.0000000000, 1.4142135382, 1.9999998808,
                         1.0000000000, -1.4142135382, 1.9999998808, 0.0000000000, 0.0000000000, 1.0000000000};

  auto weight_tensor = in_tensors_[1];
  auto origin_weight_fp32 = reinterpret_cast<float *>(weight_tensor->data_c());
  auto origin_weight_fp16 = reinterpret_cast<float16_t *>(weight_tensor->data_c());
  std::function<float(int)> access_func;
  if (weight_tensor->data_type() == kNumberTypeFloat32) {
    access_func = [=](int idx) { return origin_weight_fp32[idx]; };
  } else {
    access_func = [=](int idx) { return static_cast<float>(origin_weight_fp16[idx]); };
  }

  // OHWI -> O66I
  std::vector<float> encoded_weight(CO_ * 6 * 6 * CI_);
  for (int co = 0; co < CO_; ++co) {
    for (int ci = 0; ci < CI_; ++ci) {
      float in_vals[9];
      for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
          const int f_index = ((co * 3 + kh) * 3 + kw) * CI_ + ci;
          in_vals[kh * 3 + kw] = access_func(f_index);
        }
      }

      auto temp_vals = MatrixMultiply(G, in_vals, 6, 3, 3);
      auto out_vals = MatrixMultiply(temp_vals.data(), Gt, 6, 3, 6);
      for (int kh = 0; kh < 6; ++kh) {
        for (int kw = 0; kw < 6; ++kw) {
          const int f_index = ((co * 6 + kh) * 6 + kw) * CI_ + ci;
          encoded_weight[f_index] = out_vals[kh * 6 + kw];
        }
      }
    }
  }

  if (use_fp16_) {
    ConvertConvWeight4DTo7D<float, float16_t>(reinterpret_cast<void *>(encoded_weight.data()), packed_weight_, CO_, 6,
                                              6, CI_, 2);
  } else {
    ConvertConvWeight4DTo7D<float, float>(reinterpret_cast<void *>(encoded_weight.data()), packed_weight_, CO_, 6, 6,
                                          CI_, 2);
  }

  return RET_OK;
}

int ConvolutionOpenCLKernel::InitWeight() {
  auto allocator = ocl_runtime_->GetAllocator();

  // allocate memory
  size_t packed_weight_size;
  if (use_winograd_) {
    packed_weight_size = UP_DIV(CO_, 8) * 6 * 6 * CI_SLICES_ * 2 * CI_TILE * CO_TILE * sizeof_FLT();
  } else {
    packed_weight_size = CO_SLICES_ * KH_ * KW_ * CI_SLICES_ * CI_TILE * CO_TILE * sizeof_FLT();
  }
  packed_weight_ = allocator->Malloc(packed_weight_size);
  allocator->MapBuffer(packed_weight_, CL_MAP_WRITE, nullptr, true);
  memset(packed_weight_, 0x00, packed_weight_size);

  // rearrange weight
  if (use_winograd_) {
    GenerateWinogradWeight();
  } else {
    auto weight_tensor = in_tensors_[1];
    if (weight_tensor->data_type() == kNumberTypeFloat16) {
      if (use_fp16_) {
        ConvertConvWeight4DTo7D<float16_t, float16_t>(weight_tensor->data_c(), packed_weight_, CO_, KH_, KW_, CI_);
      } else {
        ConvertConvWeight4DTo7D<float16_t, float>(weight_tensor->data_c(), packed_weight_, CO_, KH_, KW_, CI_);
      }
    } else {
      if (use_fp16_) {
        ConvertConvWeight4DTo7D<float, float16_t>(weight_tensor->data_c(), packed_weight_, CO_, KH_, KW_, CI_);
      } else {
        ConvertConvWeight4DTo7D<float, float>(weight_tensor->data_c(), packed_weight_, CO_, KH_, KW_, CI_);
      }
    }
  }

  allocator->UnmapBuffer(packed_weight_);
  return RET_OK;
}

int ConvolutionOpenCLKernel::InitBias() {
  auto allocator = ocl_runtime_->GetAllocator();

  // align bias from C to C4
  auto bias_tensor = in_tensors_[2];
  size_t packed_bias_size = CO_SLICES_ * CO_TILE * sizeof_FLT();
  packed_bias_ = allocator->Malloc(packed_bias_size);

  allocator->MapBuffer(packed_bias_, CL_MAP_WRITE, nullptr, true);
  memset(packed_bias_, 0x00, packed_bias_size);
  if (bias_tensor->data_type() == kNumberTypeFloat16) {
    if (use_fp16_) {
      memcpy(packed_bias_, bias_tensor->data_c(), CO_ * sizeof_FLT());
    } else {
      auto packed_bias_fp32 = reinterpret_cast<float *>(packed_bias_);
      auto origin_bias_fp16 = reinterpret_cast<float16_t *>(bias_tensor->data_c());
      for (int i = 0; i < CO_; ++i) {
        packed_bias_fp32[i] = static_cast<float>(origin_bias_fp16[i]);
      }
    }
  } else {
    if (use_fp16_) {
      auto packed_bias_fp16 = reinterpret_cast<float16_t *>(packed_bias_);
      auto origin_bias_fp32 = reinterpret_cast<float *>(bias_tensor->data_c());
      for (int i = 0; i < CO_; ++i) {
        packed_bias_fp16[i] = static_cast<float16_t>(origin_bias_fp32[i]);
      }
    } else {
      memcpy(packed_bias_, bias_tensor->data_c(), CO_ * sizeof_FLT());
    }
  }
  allocator->UnmapBuffer(packed_bias_);
  return RET_OK;
}

int ConvolutionOpenCLKernel::InitBuffer() {
  InitWeight();
  if (has_bias_) {
    InitBias();
  }
  return RET_OK;
}

int ConvolutionOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
  cl_int act_type = 0;
  if (param->act_type_ == ActType_Relu) {
    act_type = 1;
  } else if (param->act_type_ == ActType_Relu6) {
    act_type = 3;
  }
  cl_int4 input_shape = {batch_size_, IH_, IW_, CI_SLICES_};
  cl_int4 output_shape = {batch_size_, OH_, OW_, CO_SLICES_};

  int arg_cn;
  if (use_winograd_) {
    arg_cn = 0;
    cl_int4 _4x4to36_out_shape = {1, 36, TILES_XY_, CI_SLICES_};
    ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, in_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, winograd_mem0_, lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, input_shape);
    ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, _4x4to36_out_shape);

    arg_cn = 0;
    cl_int4 conv_in_shape = {1, 36, TILES_XY_, CI_SLICES_};
    cl_int4 conv_out_shape = {1, 36, TILES_XY_, CO_SLICES_};
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, winograd_mem0_, lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, winograd_mem1_, lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, packed_weight_, lite::opencl::MemType::BUF);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, conv_in_shape);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, conv_out_shape);

    arg_cn = 0;
    cl_int4 _36to4x4_in_shape = {1, 16, TILES_XY_, CO_SLICES_};
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, winograd_mem1_, lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, out_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, packed_bias_, lite::opencl::MemType::BUF);
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, _36to4x4_in_shape);
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, output_shape);
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, act_type);
  } else {
    arg_cn = 0;
    cl_int4 kernel_stride = {KH_, KW_, param->stride_h_, param->stride_w_};
    cl_int4 pad = {param->pad_u_, param->pad_d_, param->pad_l_, param->pad_r_};
    cl_int2 dilation = {param->dilation_h_, param->dilation_w_};
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, in_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, out_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, packed_weight_, lite::opencl::MemType::BUF);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, packed_bias_, lite::opencl::MemType::BUF);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, input_shape);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, output_shape);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, kernel_stride);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, pad);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, dilation);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, act_type);
  }

  if (use_winograd_) {
    ocl_runtime_->RunKernel(kernel_4x4to36_, {size_t(TILES_XY_), 6, size_t(CI_SLICES_)}, {8, 6, 4}, nullptr);
    ocl_runtime_->RunKernel(kernel_conv_, {size_t(UP_DIV(TILES_XY_, 2)), 36, size_t(UP_DIV(CO_SLICES_, 2))}, {8, 6, 2},
                            nullptr);
    ocl_runtime_->RunKernel(kernel_36to4x4_, {size_t(TILES_XY_), 4, size_t(CO_SLICES_)}, {32, 4, 2}, nullptr);
  } else {
    std::vector<size_t> global, local;
    SetGlobalLocalConv(&global, &local);
    ocl_runtime_->RunKernel(kernel_conv_, global, local, nullptr);
  }

  return RET_OK;
}

int ConvolutionOpenCLKernel::SetGlobalLocalConv(std::vector<size_t> *global, std::vector<size_t> *local) {
  constexpr size_t work_group_size[] = {4, 4, 1};
  auto max_work_item_sizes = ocl_runtime_->GetWorkItemSize();
  size_t max_work_group_size = ocl_runtime_->GetKernelMaxWorkGroupSize(kernel_conv_(), (*ocl_runtime_->Device())());
  const size_t max_z_size = std::min<size_t>(16, max_work_item_sizes[2]);

  size_t global_nh = UP_DIV(batch_size_ * OH_, work_group_size[0]) * work_group_size[0];
  size_t global_w = UP_DIV(OW_, work_group_size[1]) * work_group_size[1];
  size_t global_c = UP_DIV(CO_SLICES_, work_group_size[2]) * work_group_size[2];

  size_t local_c = GetMaxDivisor(global_c, max_z_size);
  if (local_c == 0) {
    MS_LOG(ERROR) << "Divide by zero";
    return mindspore::lite::RET_ERROR;
  }
  size_t local_hw_size = std::min<size_t>(256, max_work_group_size) / local_c;
  size_t local_w = std::min(global_w, local_hw_size);
  size_t local_nh = std::min(local_hw_size / local_w, global_nh);
  if (local_nh == global_nh && global_nh % 2 == 0) {
    local_nh = global_nh / 2;
  }

  if (OW_ * CO_SLICES_ > MAX_IMAGE2D_SIZE) {
    local_w = 4;
  }

  global->clear();
  global->push_back(UP_DIV(batch_size_ * OH_, local_nh) * local_nh);
  global->push_back(UP_DIV(OW_, local_w) * local_w);
  global->push_back(UP_DIV(CO_SLICES_, local_c) * local_c);
  local->clear();
  local->push_back(local_nh);
  local->push_back(local_w);
  local->push_back(local_c);
  return RET_OK;
}

kernel::LiteKernel *OpenCLConvolutionKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                   const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                   const mindspore::lite::PrimitiveC *primitive) {
  kernel::LiteKernel *kernel;
  bool is_hw1 = inputs[0]->shape().size() == 4 && inputs[0]->shape()[1] == 1 && inputs[0]->shape()[2] == 1 &&
                outputs[0]->shape().size() == 4 && outputs[0]->shape()[1] == 1 && outputs[0]->shape()[2] == 1;
  auto conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  bool is_pad_stride_ok = conv_param->kernel_h_ == 1 && conv_param->kernel_w_ == 1 && conv_param->stride_h_ == 1 &&
                          conv_param->stride_w_ == 1 && conv_param->pad_u_ == 0 && conv_param->pad_d_ == 0 &&
                          conv_param->pad_l_ == 0 && conv_param->pad_r_ == 0 && conv_param->dilation_h_ == 1 &&
                          conv_param->dilation_w_ == 1;
  if (is_hw1 && is_pad_stride_ok) {
    auto param = static_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "Create OpenCL FullConnection kernel param failed!";
      return nullptr;
    }
    param->op_parameter_.type_ = PrimitiveType_FullConnection;
    param->a_transpose_ = false;
    param->b_transpose_ = true;
    param->act_type_ = conv_param->act_type_;
    kernel = new (std::nothrow) FullConnectionOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "Create OpenCL FullConnection kernel failed!";
      free(param);
      free(opParameter);
      return nullptr;
    } else {
      free(opParameter);
    }
  } else {
    kernel = new (std::nothrow) ConvolutionOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "Create OpenCL Convolution kernel failed!";
      free(opParameter);
      return nullptr;
    }
  }
  auto ret = kernel->Init();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: Convolution";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Conv2D, OpenCLConvolutionKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Conv2D, OpenCLConvolutionKernelCreator)
}  // namespace mindspore::kernel
