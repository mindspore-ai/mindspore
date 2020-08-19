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
#include "src/runtime/kernel/opencl/kernel/conv2d.h"
#include "src/runtime/kernel/opencl/kernel/fullconnection.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "schema/ops_generated.h"
#include "src/runtime/kernel/opencl/cl/conv2d.cl.inc"
#include "src/runtime/kernel/opencl/cl/winograd.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::ActivationType_TANH;
using mindspore::schema::PrimitiveType_Conv2D;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {

constexpr size_t CI_TILE = C4NUM;
constexpr size_t CO_TILE = C4NUM;

int Conv2DOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 2 && in_tensors_.size() != 3) {
    MS_LOG(ERROR) << "Conv2D only supports 2 or 3 input Tensor but get " << in_tensors_.size();
    return RET_ERROR;
  }
  if (out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "Conv2D only supports 1 output Tensor but get " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_.front()->shape().size() != 4) {
    MS_LOG(ERROR) << "Conv2D only supports 4D input Tensor but get " << in_tensors_.front()->shape().size() << "D.";
    return RET_ERROR;
  }
  if (in_tensors_.at(1)->shape().size() != 4) {
    MS_LOG(ERROR) << "Conv2D only supports 4D filter Tensor but get " << in_tensors_.at(1)->shape().size() << "D.";
    return RET_ERROR;
  }
  if (out_tensors_.front()->shape().size() != 4) {
    MS_LOG(ERROR) << "Conv2D only supports 4D output Tensor but get " << out_tensors_.front()->shape().size() << "D.";
    return RET_ERROR;
  }
  if (!in_tensors_.at(1)->IsConst()) {
    MS_LOG(ERROR) << "Conv2D don't support non-constant filter yet.";
    return RET_ERROR;
  }
  if (in_tensors_.size() == 3 && !in_tensors_.at(2)->IsConst()) {
    MS_LOG(ERROR) << "Conv2D don't support non-constant bias yet.";
    return RET_ERROR;
  }
  // for fusion: ActivationType_LEAKY_RELU ActivationType_TANH
  switch (static_cast<int>(param_->act_type_)) {
    case ActType_No:
    case ActType_Relu:
    case ActType_Relu6:
    case ActivationType_LEAKY_RELU:
    case ActivationType_TANH:
      break;
    default: {
      MS_LOG(ERROR) << "Unsupported activation type " << param_->act_type_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int Conv2DOpenCLKernel::Prepare() {
  use_fp16_ = ocl_runtime_->GetFp16Enable();
  sizeof_FLT_ = use_fp16_ ? sizeof(float16_t) : sizeof(float);

  auto input_shape = in_tensors_.front()->shape();
  auto output_shape = out_tensors_.front()->shape();
  batch_size_ = input_shape[0];
  IH_ = input_shape[1];
  IW_ = input_shape[2];
  CI_ = input_shape[3];
  // for fusion Conv2D and Reshape(N11C->NC)
  if (output_shape.size() == 2) {
    OH_ = 1;
    OW_ = 1;
    CO_ = output_shape[1];
  } else {  // output_shape.size()==4
    OH_ = output_shape[1];
    OW_ = output_shape[2];
    CO_ = output_shape[3];
  }
  CI_SLICES_ = UP_DIV(CI_, C4NUM);
  CO_SLICES_ = UP_DIV(CO_, C4NUM);
  KH_ = param_->kernel_h_;
  KW_ = param_->kernel_w_;
  has_bias_ = in_tensors_.size() == 3;

  // note: TILES_X TILES_Y TILES_XY is only used when use_winograd_=true
  TILES_X_ = UP_DIV(OW_, 4);
  TILES_Y_ = UP_DIV(OH_, 4);
  TILES_XY_ = TILES_X_ * TILES_Y_;
  use_winograd_ = UseWinograd4x4To6x6();

  // build kernel
  if (use_winograd_) {
    MS_LOG(DEBUG) << "use winograd";
    std::string program_name = "winograd";
    ocl_runtime_->LoadSource(program_name, GetActDefines() + winograd_source);
    ocl_runtime_->BuildKernel(kernel_4x4to36_, program_name, "Winograd4x4To36");
    ocl_runtime_->BuildKernel(kernel_, program_name, "WinogradConvolution");
    ocl_runtime_->BuildKernel(kernel_36to4x4_, program_name, "Winograd36To4x4");
  } else {
    SetBlockSize();
    std::string program_name = "conv2d";
    std::string kernel_name = "Conv2D_H" + std::to_string(block_size_.H) + "W" + std::to_string(block_size_.W) + "C" +
                              std::to_string(block_size_.C);
    ocl_runtime_->LoadSource(program_name, GetActDefines() + conv2d_source);
    ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name);
  }

  // allocate winograd memory
  if (use_winograd_) {
    auto allocator = ocl_runtime_->GetAllocator();
    size_t img_dtype = use_fp16_ ? CL_HALF_FLOAT : CL_FLOAT;

    size_t size = TILES_XY_ * CI_SLICES_ * 36 * sizeof_FLT_;
    size_t width = TILES_XY_;
    size_t height = CI_SLICES_ * 36;
    winograd_mem0_ = allocator->Malloc(size, {width, height, img_dtype});

    size = TILES_XY_ * CO_SLICES_ * 36 * sizeof_FLT_;
    width = TILES_XY_;
    height = CO_SLICES_ * 36;
    winograd_mem1_ = allocator->Malloc(size, {width, height, img_dtype});
  }

  auto ret = InitWeights();
  if (ret != RET_OK) {
    return ret;
  }
  SetGlobalLocal();
  SetConstArgs();
  return RET_OK;
}

int Conv2DOpenCLKernel::GenerateWinogradFilter() {
  constexpr float Gt[] = {1.0000000000, 1.0000000000, 1.0000000000,  1.0000000000, 1.0000000000,  0.0000000000,
                          0.0000000000, 0.7071067691, -0.7071067691, 1.4142135382, -1.4142135382, 0.0000000000,
                          0.0000000000, 0.4999999702, 0.4999999702,  1.9999998808, 1.9999998808,  1.0000000000};
  constexpr float G[] = {1.0000000000, 0.0000000000,  0.0000000000, 1.0000000000, 0.7071067691, 0.4999999702,
                         1.0000000000, -0.7071067691, 0.4999999702, 1.0000000000, 1.4142135382, 1.9999998808,
                         1.0000000000, -1.4142135382, 1.9999998808, 0.0000000000, 0.0000000000, 1.0000000000};

  auto weight_tensor = in_tensors_.at(1);
  auto origin_weight_fp32 = reinterpret_cast<float *>(weight_tensor->data_c());
  MS_ASSERT(origin_weight_fp32);
  auto origin_weight_fp16 = reinterpret_cast<float16_t *>(weight_tensor->data_c());
  MS_ASSERT(origin_weight_fp16);
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

int Conv2DOpenCLKernel::InitFilter() {
  auto allocator = ocl_runtime_->GetAllocator();
  auto ret = DequantWeight();
  if (ret != RET_OK) {
    return ret;
  }

  // allocate memory
  size_t packed_weight_size;
  if (use_winograd_) {
    packed_weight_size = UP_DIV(CO_, 8) * 6 * 6 * CI_SLICES_ * 2 * CI_TILE * CO_TILE * sizeof_FLT_;
  } else {
    packed_weight_size = UP_ROUND(CO_SLICES_, block_size_.C) * KH_ * KW_ * CI_SLICES_ * CI_TILE * CO_TILE * sizeof_FLT_;
  }
  packed_weight_ = allocator->Malloc(packed_weight_size);
  allocator->MapBuffer(packed_weight_, CL_MAP_WRITE, nullptr, true);
  memset(packed_weight_, 0x00, packed_weight_size);

  // rearrange weight
  if (use_winograd_) {
    GenerateWinogradFilter();
  } else {
    auto weight_tensor = in_tensors_.at(1);
    if (weight_tensor->data_type() == kNumberTypeFloat16) {
      if (use_fp16_) {
        ConvertConvWeight4DTo7D<float16_t, float16_t>(weight_tensor->data_c(), packed_weight_, CO_, KH_, KW_, CI_,
                                                      block_size_.C);
      } else {
        ConvertConvWeight4DTo7D<float16_t, float>(weight_tensor->data_c(), packed_weight_, CO_, KH_, KW_, CI_,
                                                  block_size_.C);
      }
    } else if (weight_tensor->data_type() == kNumberTypeFloat32) {
      if (use_fp16_) {
        ConvertConvWeight4DTo7D<float, float16_t>(weight_tensor->data_c(), packed_weight_, CO_, KH_, KW_, CI_,
                                                  block_size_.C);
      } else {
        ConvertConvWeight4DTo7D<float, float>(weight_tensor->data_c(), packed_weight_, CO_, KH_, KW_, CI_,
                                              block_size_.C);
      }
    } else {  // int8 or int16
      if (use_fp16_) {
        ConvertConvWeight4DTo7D<float16_t, float16_t>(weight_tensor->data_c(), packed_weight_, CO_, KH_, KW_, CI_,
                                                      block_size_.C);
      } else {
        ConvertConvWeight4DTo7D<float, float>(weight_tensor->data_c(), packed_weight_, CO_, KH_, KW_, CI_,
                                              block_size_.C);
      }
      FreeDequantedWeight();
    }
  }

  allocator->UnmapBuffer(packed_weight_);
  return RET_OK;
}

int Conv2DOpenCLKernel::InitBias() {
  auto allocator = ocl_runtime_->GetAllocator();

  // align bias from C to C4
  auto bias_tensor = in_tensors_.at(2);
  size_t packed_bias_size = UP_ROUND(CO_SLICES_, block_size_.C) * CO_TILE * sizeof_FLT_;
  packed_bias_ = allocator->Malloc(packed_bias_size);

  allocator->MapBuffer(packed_bias_, CL_MAP_WRITE, nullptr, true);
  memset(packed_bias_, 0x00, packed_bias_size);
  if (bias_tensor->data_type() == kNumberTypeFloat16) {
    if (use_fp16_) {
      memcpy(packed_bias_, bias_tensor->data_c(), CO_ * sizeof_FLT_);
    } else {
      auto packed_bias_fp32 = reinterpret_cast<float *>(packed_bias_);
      auto origin_bias_fp16 = reinterpret_cast<float16_t *>(bias_tensor->data_c());
      MS_ASSERT(origin_bias_fp16);
      for (int i = 0; i < CO_; ++i) {
        packed_bias_fp32[i] = static_cast<float>(origin_bias_fp16[i]);
      }
    }
  } else {
    if (use_fp16_) {
      auto packed_bias_fp16 = reinterpret_cast<float16_t *>(packed_bias_);
      auto origin_bias_fp32 = reinterpret_cast<float *>(bias_tensor->data_c());
      MS_ASSERT(origin_bias_fp32);
      for (int i = 0; i < CO_; ++i) {
        packed_bias_fp16[i] = static_cast<float16_t>(origin_bias_fp32[i]);
      }
    } else {
      memcpy(packed_bias_, bias_tensor->data_c(), CO_ * sizeof_FLT_);
    }
  }
  allocator->UnmapBuffer(packed_bias_);
  return RET_OK;
}

int Conv2DOpenCLKernel::InitWeights() {
  InitFilter();
  if (has_bias_) {
    InitBias();
  }
  return RET_OK;
}

void Conv2DOpenCLKernel::SetBlockSize() {
  auto task_size = static_cast<float>(batch_size_ * OH_ * OW_ * CO_SLICES_);
  auto task_size_per_cu = task_size / ocl_runtime_->DeviceComputeUnits();
  int block_size;
  if (task_size_per_cu <= 256) {
    block_size = 1;
  } else if (task_size_per_cu <= 256 * 4) {
    block_size = 2;
  } else if (task_size_per_cu <= (use_fp16_ ? 256 * 8 : FLT_MAX)) {
    block_size = 4;
  } else {
    block_size = 8;
  }

  bool w_kernel_is_1 =
    KW_ == 1 && param_->stride_w_ == 1 && param_->dilation_w_ == 1 && param_->pad_l_ == 0 && param_->pad_r_ == 0;
  bool h_kernel_is_1 =
    KH_ == 1 && param_->stride_h_ == 1 && param_->dilation_h_ == 1 && param_->pad_u_ == 0 && param_->pad_d_ == 0;
  if (!w_kernel_is_1 || !h_kernel_is_1) {
    block_size = std::min(block_size, 4);
  }

  if (block_size == 8) {
    block_size_ = {2, 2, 2};
  } else if (block_size == 4) {
    block_size_ = {2, 1, 2};
  } else if (block_size == 2) {
    block_size_ = {2, 1, 1};
  } else {
    block_size_ = {1, 1, 1};
  }
}

void AlignWinogradGlobalLocal(const std::vector<int> &global, const std::vector<int> &local, cl::NDRange *global_range,
                              cl::NDRange *local_range) {
  *local_range = cl::NDRange(local[0], local[1], local[2]);
  *global_range =
    cl::NDRange(UP_ROUND(global[0], local[0]), UP_ROUND(global[1], local[1]), UP_ROUND(global[2], local[2]));
}

void Conv2DOpenCLKernel::SetGlobalLocal() {
  if (use_winograd_) {
    AlignWinogradGlobalLocal({TILES_XY_, 6, CI_SLICES_}, {8, 6, 4}, &global_4x4to36_, &local_4x4to36_);
    AlignWinogradGlobalLocal({UP_DIV(TILES_XY_, 2), 36, UP_DIV(CO_SLICES_, 2)}, {8, 6, 2}, &global_conv_, &local_conv_);
    AlignWinogradGlobalLocal({TILES_XY_, 4, CO_SLICES_}, {32, 4, 2}, &global_36to4x4_, &local_36to4x4_);
  } else {
    size_t global_h = batch_size_ * UP_DIV(OH_, block_size_.H);
    size_t global_w = UP_DIV(OW_, block_size_.W);
    size_t global_c = UP_DIV(CO_SLICES_, block_size_.C);
    constexpr int local_c_max = 16;
    constexpr int local_hw_max = 256;
    constexpr int OH_threshold = 100;
    constexpr int OW_threshold = 100;
    constexpr int OC_threshold = 64;
    size_t local_c = GetMaxDivisor(global_c, local_c_max);
    local_c = std::max<size_t>(local_c, 1);
    size_t local_hw = local_hw_max / local_c;
    size_t local_h;
    size_t local_w;
    if (OH_ >= OH_threshold && OW_ >= OW_threshold && CO_ <= OC_threshold) {  // c -> w -> h
      local_w = std::min(global_w, local_hw);
      local_h = std::min(local_hw / local_w, global_h);
    } else {  // c -> h -> w
      local_h = std::min(global_h, local_hw);
      local_w = std::min(local_hw / local_h, global_w);
    }
    global_size_ = {global_h, global_w, global_c};
    local_size_ = {local_h, local_w, local_c};
    AlignGlobalLocal(global_size_, local_size_);
  }
}

std::vector<BaseTuningParameter> Conv2DOpenCLKernel::GenerateTuningParam() {
  // don't need to tune local_c
  std::vector<BaseTuningParameter> tuning_params = {};
  if (use_winograd_) {
    return tuning_params;
  }
  BaseTuningParameter default_tuning_param = BaseTuningParameter();
  default_tuning_param.local_size = local_size_;
  tuning_params.push_back(default_tuning_param);

  std::vector<size_t> max_work_items = ocl_runtime_->GetWorkItemSize();
  size_t max_workgroup_size = ocl_runtime_->GetMaxWorkGroupSize(kernel_);
  std::set<size_t> candidate_x = GenerateLocalByGlobal(global_size_[0]);
  std::set<size_t> candidate_y = GenerateLocalByGlobal(global_size_[1]);
  for (auto x : candidate_x) {
    if (x <= max_work_items[0]) {
      for (auto y : candidate_y) {
        if (y <= max_work_items[1]) {
          auto group_size = x * y * local_size_[2];
          if (group_size <= max_workgroup_size) {
            BaseTuningParameter tuning_param = BaseTuningParameter();
            tuning_param.local_size = {x, y, local_size_[2]};
            tuning_params.push_back(tuning_param);
          }
        }
      }
    }
  }
  return tuning_params;
}

std::string Conv2DOpenCLKernel::Key() {
  auto key = OpenCLKernel::Key();
  key += "_" + std::to_string(KH_) + "_" + std::to_string(KW_) + "_" + std::to_string(param_->stride_h_) + "_" +
         std::to_string(param_->stride_w_) + "_" + std::to_string(param_->dilation_h_) + "_" +
         std::to_string(param_->dilation_w_);
  return key;
}

void Conv2DOpenCLKernel::SetConstArgs() {
  cl_int4 input_shape = {batch_size_, IH_, IW_, CI_SLICES_};
  cl_int4 output_shape = {batch_size_, OH_, OW_, CO_SLICES_};

  int arg_cn;
  if (use_winograd_) {
    arg_cn = 1;
    cl_int4 _4x4to36_out_shape = {1, 36, TILES_XY_, CI_SLICES_};
    ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, winograd_mem0_);
    ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, input_shape);
    ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn, _4x4to36_out_shape);

    arg_cn = 0;
    cl_int4 conv_in_shape = {1, 36, TILES_XY_, CI_SLICES_};
    cl_int4 conv_out_shape = {1, 36, TILES_XY_, CO_SLICES_};
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, winograd_mem0_);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, winograd_mem1_);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, packed_weight_, lite::opencl::MemType::BUF);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, conv_in_shape);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn, conv_out_shape);

    arg_cn = 2;
    cl_int4 _36to4x4_in_shape = {1, 16, TILES_XY_, CO_SLICES_};
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, 0, winograd_mem1_);
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, packed_bias_, lite::opencl::MemType::BUF);
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, _36to4x4_in_shape);
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, output_shape);
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, static_cast<cl_int>(param_->act_type_));
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn, static_cast<cl_float>(alpha_));
  } else {
    arg_cn = 2;
    cl_int4 kernel_stride = {KH_, KW_, param_->stride_h_, param_->stride_w_};
    cl_int4 pad = {param_->pad_u_, param_->pad_d_, param_->pad_l_, param_->pad_r_};
    cl_int2 dilation = {param_->dilation_h_, param_->dilation_w_};
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, packed_weight_, lite::opencl::MemType::BUF);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, packed_bias_, lite::opencl::MemType::BUF);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, output_shape);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, kernel_stride);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, pad);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, dilation);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, static_cast<cl_int>(param_->act_type_));
    ocl_runtime_->SetKernelArg(kernel_, arg_cn, static_cast<cl_float>(alpha_));
  }
}

int Conv2DOpenCLKernel::Tune() {
  if (use_winograd_) {
    return RET_OK;
  }
  return OpenCLKernel::Tune();
}

int Conv2DOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  if (use_winograd_) {
    ocl_runtime_->SetKernelArg(kernel_4x4to36_, 0, in_tensors_.front()->data_c());
    ocl_runtime_->RunKernel(kernel_4x4to36_, global_4x4to36_, local_4x4to36_);

    ocl_runtime_->RunKernel(kernel_, global_conv_, local_conv_);

    ocl_runtime_->SetKernelArg(kernel_36to4x4_, 1, out_tensors_.front()->data_c());
    ocl_runtime_->RunKernel(kernel_36to4x4_, global_36to4x4_, local_36to4x4_);
  } else {
    ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_.front()->data_c());
    ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_.front()->data_c());
    ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  }
  return RET_OK;
}

bool UseFcReplaceConv(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                      ConvParameter *param) {
  MS_ASSERT(param);
  MS_ASSERT(!inputs.empty());
  MS_ASSERT(!outputs.empty());
  auto input_shape = inputs.front()->shape();
  auto output_shape = inputs.front()->shape();
  // IH=1 IW=1 OH=1 OW=1
  bool hw_is_1 = input_shape.size() == 4 && input_shape[1] == 1 && input_shape[2] == 1 && output_shape.size() == 4 &&
                 output_shape[1] == 1 && output_shape[2] == 1;
  bool attr_valid = param->kernel_h_ == 1 && param->kernel_w_ == 1 && param->stride_h_ == 1 && param->stride_w_ == 1 &&
                    param->pad_u_ == 0 && param->pad_d_ == 0 && param->pad_l_ == 0 && param->pad_r_ == 0 &&
                    param->dilation_h_ == 1 && param->dilation_w_ == 1;
  return hw_is_1 && attr_valid;
}

OpParameter *CreateFcParam(const ConvParameter *conv_param) {
  auto fc_param = static_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (fc_param == nullptr) {
    MS_LOG(ERROR) << "Create FullConnection kernel param failed.";
    return nullptr;
  }
  fc_param->op_parameter_.type_ = PrimitiveType_FullConnection;
  fc_param->a_transpose_ = false;
  fc_param->b_transpose_ = true;
  fc_param->act_type_ = conv_param->act_type_;
  return reinterpret_cast<OpParameter *>(fc_param);
}

kernel::LiteKernel *OpenCLConvolutionKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                   const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                   const mindspore::lite::PrimitiveC *primitive) {
  kernel::OpenCLKernel *kernel;
  OpParameter *real_param;
  auto *conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  if (UseFcReplaceConv(inputs, outputs, conv_param)) {
    auto *fc_param = CreateFcParam(conv_param);
    kernel = new (std::nothrow) FullConnectionOpenCLKernel(fc_param, inputs, outputs);
    real_param = fc_param;
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "Create FullConnection kernel failed.";
      free(fc_param);
      free(conv_param);
      return nullptr;
    } else {
      free(conv_param);
      MS_LOG(INFO) << "use FullConnection to replace Convolution.";
    }
  } else {
    kernel = new (std::nothrow) Conv2DOpenCLKernel(reinterpret_cast<OpParameter *>(conv_param), inputs, outputs);
    real_param = reinterpret_cast<OpParameter *>(conv_param);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "Create Convolution kernel failed.";
      free(conv_param);
      return nullptr;
    }
  }

  int ret = kernel->CheckSpecs();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Init Convolution kernel failed.";
    delete kernel;
    free(real_param);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Conv2D, OpenCLConvolutionKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Conv2D, OpenCLConvolutionKernelCreator)
}  // namespace mindspore::kernel
