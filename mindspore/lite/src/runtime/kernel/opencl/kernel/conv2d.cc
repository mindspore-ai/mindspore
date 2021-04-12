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

#include "src/runtime/kernel/opencl/kernel/conv2d.h"
#include <string>
#include <set>
#include <algorithm>
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "schema/ops_generated.h"
#include "src/common/utils.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/kernel/depthwise_conv2d.h"
#include "src/runtime/kernel/opencl/kernel/fullconnection.h"
#include "src/runtime/kernel/opencl/kernel/winograd.h"
#include "src/runtime/kernel/opencl/cl/conv2d.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::ActivationType_TANH;
using mindspore::schema::PrimitiveType_Conv2DFusion;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {

int Conv2DOpenCLKernel::CheckSpecs() {
  int inputs_num = in_tensors_.size();
  if (inputs_num != 2 && inputs_num != 3) {
    MS_LOG(ERROR) << "Conv2D only supports 2 or 3 input Tensor but get " << inputs_num;
    return RET_ERROR;
  }
  int outputs_num = out_tensors_.size();
  if (outputs_num != 1) {
    MS_LOG(ERROR) << "Conv2D only supports 1 output Tensor but get " << outputs_num;
    return RET_ERROR;
  }

  int input_ndim = in_tensors_.at(kInputIndex)->shape().size();
  if (input_ndim != 4) {
    MS_LOG(ERROR) << "Conv2D only supports 4D input Tensor but get " << input_ndim << "D.";
    return RET_ERROR;
  }
  int output_ndim = out_tensors_.at(kOutputIndex)->shape().size();
  if (output_ndim != 4) {
    MS_LOG(ERROR) << "Conv2D only supports 4D output Tensor but get " << output_ndim << "D.";
    return RET_ERROR;
  }

  auto *filter_tensor = in_tensors_.at(kWeightIndex);
  int filter_ndim = filter_tensor->shape().size();
  if (filter_ndim != 4) {
    MS_LOG(ERROR) << "Conv2D only supports 4D filter Tensor but get " << filter_ndim << "D.";
    return RET_ERROR;
  }
  if (!filter_tensor->IsConst()) {
    MS_LOG(ERROR) << "Conv2D don't support non-constant filter yet.";
    return RET_ERROR;
  }

  auto *bias_tensor = in_tensors_.size() >= 3 ? in_tensors_.at(kBiasIndex) : nullptr;
  if (bias_tensor != nullptr && !bias_tensor->IsConst()) {
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
  InitAttrs();
  BuildKernel();
  InitWeights();
  SetGlobalLocal();
  SetConstArgs();
  return RET_OK;
}

void Conv2DOpenCLKernel::InitAttrs() {
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
  CI_SLICES_ = UP_DIV(CI_, CI_TILE);
  CO_SLICES_ = UP_DIV(CO_, CO_TILE);
  KH_ = param_->kernel_h_;
  KW_ = param_->kernel_w_;
  has_bias_ = in_tensors_.size() == 3;
  // note: TILE_HW_ is only used when use_winograd_=true
  TILE_HW_ = UP_DIV(OW_, 4) * UP_DIV(OH_, 4);
}

void Conv2DOpenCLKernel::BuildKernel() {
  SetBlockSize();
  std::string program_name = "conv2d";
  std::stringstream kernel_name;
  kernel_name << "Conv2D_H" << block_size_.H << "W" << block_size_.W << "C" << block_size_.C;
  if (filter_type_ == MemType::IMG) {
    kernel_name << "_Img";
  }
  ocl_runtime_->LoadSource(program_name, GetActDefines() + conv2d_source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name.str());
}

void Conv2DOpenCLKernel::SetBlockSize() {
  if (filter_type_ == MemType::IMG) {
    block_size_ = {2, 2, 2};
    return;
  }
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

int Conv2DOpenCLKernel::InitWeights() {
  InitFilter();
  if (has_bias_) {
    InitBias();
  }
  return RET_OK;
}

void ConvertFilter(void *src, void *dst, TypeId src_dtype, TypeId dst_dtype, FilterFormat src_format,
                   FilterFormat dst_format, size_t CO, size_t KH, size_t KW, size_t CI, size_t OGroup) {
  MS_ASSERT(src);
  MS_ASSERT(dst);
  MS_ASSERT(src_dtype == kNumberTypeFloat16 || src_dtype == kNumberTypeFloat32);
  MS_ASSERT(dst_dtype == kNumberTypeFloat16 || dst_dtype == kNumberTypeFloat32);
  MS_ASSERT(src_format == OHWI);
  MS_ASSERT(dst_format == HWII4OO4 || dst_format == OHWIOgroupI4O4);
  auto src_fp16 = reinterpret_cast<float16_t *>(src);
  auto src_fp32 = reinterpret_cast<float32_t *>(src);
  auto dst_fp16 = reinterpret_cast<float16_t *>(dst);
  auto dst_fp32 = reinterpret_cast<float32_t *>(dst);
  bool src_is_fp16 = src_dtype == kNumberTypeFloat16;
  bool dst_is_fp16 = dst_dtype == kNumberTypeFloat16;
  auto CI_SLICES = UP_DIV(CI, CI_TILE);
  auto CO_SLICES = UP_DIV(CO, CO_TILE);
  for (size_t co = 0, src_idx = 0; co < CO; ++co) {
    for (size_t kh = 0; kh < KH; ++kh) {
      for (size_t kw = 0; kw < KW; ++kw) {
        for (size_t ci = 0; ci < CI; ++ci, ++src_idx) {
          size_t dst_idx = 0;
          size_t co_inner = co % CO_TILE;
          size_t ci_slice = ci / CI_TILE;
          size_t ci_inner = ci % CI_TILE;
          if (dst_format == OHWIOgroupI4O4) {
            size_t co_slice = co / (CO_TILE * OGroup);
            size_t group_idx = co % (CO_TILE * OGroup) / CO_TILE;
            dst_idx =
              (((((co_slice * KH + kh) * KW + kw) * CI_SLICES + ci_slice) * OGroup + group_idx) * CI_TILE + ci_inner) *
                CO_TILE +
              co_inner;
          } else {  // if(dst_format==HWII4OO4)
            size_t co_slice = co / CO_TILE;
            dst_idx =
              ((((kh * KW + kw) * CI_SLICES + ci_slice) * CI_TILE + ci_inner) * CO_SLICES + co_slice) * CO_TILE +
              co_inner;
          }
          if (dst_is_fp16) {
            dst_fp16[dst_idx] = src_is_fp16 ? src_fp16[src_idx] : static_cast<float16_t>(src_fp32[src_idx]);
          } else {
            dst_fp32[dst_idx] = src_is_fp16 ? static_cast<float32_t>(src_fp16[src_idx]) : src_fp32[src_idx];
          }
        }
      }
    }
  }
}

void Conv2DOpenCLKernel::InitFilter() {
  auto allocator = ocl_runtime_->GetAllocator();

  auto ret = DequantWeight();
  if (ret != RET_OK) {
    return;
  }

  // allocate opencl memory: buffer or image2d
  size_t size = 0;
  int Ogroup = block_size_.C;
  if (filter_type_ == MemType::IMG) {
    size_t width = CO_SLICES_;
    size_t height = KH_ * KW_ * UP_ROUND(CI_, CI_TILE);
    size_t dtype = use_fp16_ ? CL_HALF_FLOAT : CL_FLOAT;
    size = width * height * CO_TILE * sizeof_FLT_;
    packed_filter_ = allocator->Malloc({width, height, dtype});
  } else {
    size = UP_DIV(CO_SLICES_, Ogroup) * KH_ * KW_ * CI_SLICES_ * Ogroup * CI_TILE * CO_TILE * sizeof_FLT_;
    packed_filter_ = allocator->Malloc(size);
  }

  // rearrange filter
  auto filter_tensor = in_tensors_.at(1);
  void *src_data = filter_tensor->data_c();
  auto src_dtype = filter_tensor->data_type();
  auto dst_dtype = use_fp16_ ? kNumberTypeFloat16 : kNumberTypeFloat32;
  std::vector<char> tmp(size, 0);
  if (filter_type_ == MemType::IMG) {
    ConvertFilter(src_data, tmp.data(), src_dtype, dst_dtype, OHWI, HWII4OO4, CO_, KH_, KW_, CI_);
  } else {
    ConvertFilter(src_data, tmp.data(), src_dtype, dst_dtype, OHWI, OHWIOgroupI4O4, CO_, KH_, KW_, CI_, Ogroup);
  }

  // unmap
  if (filter_type_ == MemType::IMG) {
    ocl_runtime_->WriteImage(packed_filter_, tmp.data());
  } else {
    allocator->MapBuffer(packed_filter_, CL_MAP_WRITE, nullptr, true);
    memcpy(packed_filter_, tmp.data(), size);
    allocator->UnmapBuffer(packed_filter_);
  }

  FreeDequantedWeight();
  FreeTmpWeight(in_tensors_.at(kWeightIndex)->data_c());
}

void Conv2DOpenCLKernel::InitBias() {
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
  FreeTmpWeight(in_tensors_.at(kBiasIndex)->data_c());
}

void Conv2DOpenCLKernel::SetConstArgs() {
  cl_int4 input_shape = {batch_size_, IH_, IW_, CI_SLICES_};
  cl_int4 output_shape = {batch_size_, OH_, OW_, CO_SLICES_};
  cl_int4 kernel_stride = {KH_, KW_, param_->stride_h_, param_->stride_w_};
  cl_int4 pad = {param_->pad_u_, param_->pad_d_, param_->pad_l_, param_->pad_r_};
  cl_int2 dilation = {param_->dilation_h_, param_->dilation_w_};

  int arg_cn = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, packed_filter_, filter_type_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, packed_bias_, MemType::BUF);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, output_shape);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, kernel_stride);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, pad);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, dilation);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, param_->act_type_);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn, alpha_);
}

void Conv2DOpenCLKernel::SetGlobalLocal() {
  size_t global_h = batch_size_ * UP_DIV(OH_, block_size_.H);
  size_t global_w = UP_DIV(OW_, block_size_.W);
  size_t global_c = UP_DIV(CO_SLICES_, block_size_.C);
  int local_max = filter_type_ == MemType::IMG ? 64 : 128;
  if (ocl_runtime_->DeviceComputeUnits() > 16) {
    local_max = 256;
  }
  const int local_c_max = 16;
  const int OH_threshold = 100;
  const int OW_threshold = 100;
  const int OC_threshold = 64;
  size_t local_c = GetMaxDivisor(global_c, local_c_max);
  local_c = std::max<size_t>(local_c, 1);
  size_t local_hw = local_max / local_c;
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

int Conv2DOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_.front()->data_c());
  ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_.front()->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);
  return RET_OK;
}

std::vector<BaseTuningParameter> Conv2DOpenCLKernel::GenerateTuningParam() {
  // don't need to tune local_c
  std::vector<BaseTuningParameter> tuning_params = {};
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

OpParameter *CreateFcParam(const ConvParameter *conv_param, const std::vector<lite::Tensor *> &inputs) {
  auto fc_param = static_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  if (fc_param == nullptr) {
    MS_LOG(ERROR) << "Create FullConnection kernel param failed.";
    return nullptr;
  }
  fc_param->op_parameter_.type_ = PrimitiveType_FullConnection;
  fc_param->op_parameter_.infer_flag_ = true;
  fc_param->a_transpose_ = false;
  fc_param->b_transpose_ = true;
  fc_param->act_type_ = conv_param->act_type_;
  fc_param->has_bias_ = inputs.size() == 3;
  return reinterpret_cast<OpParameter *>(fc_param);
}

bool UseWinograd4x4To6x6(const ConvParameter *param, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs) {
  if (!(inputs.size() == 2 || inputs.size() == 3) || outputs.empty()) {
    return false;
  }
  auto input_shape = inputs.front()->shape();
  auto output_shape = outputs.front()->shape();
  if (input_shape.size() != 4 || (output_shape.size() != 2 && output_shape.size() != 4)) {
    return false;
  }
  int batch_size = input_shape[0];
  int IH = input_shape[1];
  int IW = input_shape[2];
  int CI = input_shape[3];
  int OH = output_shape.size() == 2 ? 1 : output_shape[1];
  int OW = output_shape.size() == 2 ? 1 : output_shape[2];
  int CO = output_shape.size() == 2 ? output_shape[1] : output_shape[3];
  int CI_SLICES = UP_DIV(CI, CI_TILE);
  int CO_SLICES = UP_DIV(CO, CO_TILE);
  int TILE_HW_ = UP_DIV(OH, 4) * UP_DIV(OW, 4);

  bool pad_is_all_0 = param->pad_u_ == 0 && param->pad_d_ == 0 && param->pad_l_ == 0 && param->pad_r_ == 0;
  bool pad_is_all_1 = param->pad_u_ == 1 && param->pad_d_ == 1 && param->pad_l_ == 1 && param->pad_r_ == 1;
  bool attr_valid = param->kernel_h_ == 3 && param->kernel_w_ == 3 && param->stride_h_ == 1 && param->stride_w_ == 1 &&
                    param->dilation_h_ == 1 && param->dilation_w_ == 1 && (pad_is_all_0 || pad_is_all_1);

  bool shape_valid = false;
  if (pad_is_all_1) {
    shape_valid = batch_size == 1 && IH == OH && IW == OW;
  } else if (pad_is_all_0) {
    shape_valid = batch_size == 1 && IH - 2 == OH && IW - 2 == OW;
  }

  bool channel_good = CI_SLICES >= 8 && CO_SLICES >= 8;
  bool hw_good = TILE_HW_ >= 16;
  return attr_valid && shape_valid && channel_good && hw_good;
}

kernel::LiteKernel *OpenCLConv2DCreator(const std::vector<lite::Tensor *> &inputs,
                                        const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                        const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(!inputs.empty());
  MS_ASSERT(!outputs.empty());
  MS_ASSERT(opParameter);
  auto *conv_param = reinterpret_cast<ConvParameter *>(opParameter);
  int input_channel = conv_param->input_channel_;
  int output_channel = conv_param->output_channel_;
  int group = conv_param->group_;

  // case 1: depthwise conv2d
  if (group == input_channel && group == output_channel) {
    return OpenCLKernelCreator<DepthwiseConv2dOpenCLKernel>(inputs, outputs, opParameter, ctx, desc);
  }

  // case 2: group conv2d
  if (group != 1) {
    MS_LOG(ERROR) << "OpenCL doesn't support group conv2d.";
    free(conv_param);
    return nullptr;
  }

  // case 3: common conv2d
  kernel::OpenCLKernel *kernel = nullptr;
  bool infer_shape_done = opParameter->infer_flag_;
  if (infer_shape_done && UseFcReplaceConv(inputs, outputs, conv_param)) {
    auto *fc_param = CreateFcParam(conv_param, inputs);
    kernel = new (std::nothrow) FullConnectionOpenCLKernel(fc_param, inputs, outputs, ctx);
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
    if (infer_shape_done && UseWinograd4x4To6x6(conv_param, inputs, outputs)) {
      MS_LOG(DEBUG) << "use Winograd algorithm.";
      kernel =
        new (std::nothrow) WinogradOpenCLKernel(reinterpret_cast<OpParameter *>(conv_param), inputs, outputs, ctx);
    } else {
      kernel = new (std::nothrow) Conv2DOpenCLKernel(reinterpret_cast<OpParameter *>(conv_param), inputs, outputs, ctx);
    }
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "Create Convolution kernel failed.";
      free(conv_param);
      return nullptr;
    }
  }
  if (!infer_shape_done) {
    StoreTmpWeight(inputs.at(kWeightIndex));
    if (inputs.size() > kBiasIndex) {
      StoreTmpWeight(inputs.at(kBiasIndex));
    }
    MS_LOG(WARNING) << "kernel don't infer shape yet!";
    return kernel;
  }
  if (kernel->CheckSpecs() != RET_OK || kernel->OpenCLKernel::CheckSpecs() != RET_OK) {
    MS_LOG(ERROR) << "Init Convolution kernel failed.";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Conv2DFusion, OpenCLConv2DCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Conv2DFusion, OpenCLConv2DCreator)
}  // namespace mindspore::kernel
