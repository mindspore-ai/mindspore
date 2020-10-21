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
#include "src/runtime/kernel/opencl/utils.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;
using mindspore::schema::Format::Format_NC4HW4;
using mindspore::schema::Format::Format_NCHW;
using mindspore::schema::Format::Format_NHWC;
using mindspore::schema::Format::Format_NHWC4;

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
  in_ori_format_ = input_tensor->GetFormat();
  out_ori_format_ = output_tensor->GetFormat();
  if (op_format_ != Format_NHWC4 && op_format_ != Format_NC4HW4) {
    MS_LOG(ERROR) << "op_format_ " << op_format_ << " not support!";
    return RET_ERROR;
  }
  input_tensor->SetFormat(op_format_);
  output_tensor->SetFormat(op_format_);

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
  auto code_id = get_code_id();
  std::string program_name;
  if (use_winograd_) {
    MS_LOG(DEBUG) << "use winograd";
    program_name = "Winograd4x4To36" + code_id;
    ocl_runtime_->LoadSource(program_name, CodeGenWinograd4x4To36());
    ocl_runtime_->BuildKernel(kernel_4x4to36_, program_name, "Winograd4x4To36", build_options);

    program_name = "WinogradConvolution" + code_id;
    ocl_runtime_->LoadSource(program_name, CodeGenWinogradConvolution());
    ocl_runtime_->BuildKernel(kernel_conv_, program_name, "WinogradConvolution", build_options);

    program_name = "Winograd36To4x4" + code_id;
    ocl_runtime_->LoadSource(program_name, CodeGenWinograd36To4x4());
    ocl_runtime_->BuildKernel(kernel_36to4x4_, program_name, "Winograd36To4x4", build_options);
  } else {
    program_name = "Convolution" + code_id;
    std::string source = op_format_ == Format_NHWC4 ? CodeGenConvolutionNHWC4() : CodeGenConvolutionNC4HW4();
    ocl_runtime_->LoadSource(program_name, source);
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

  this->InitBuffer();

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

int ConvolutionOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  if (in_tensors_[0]->GetFormat() == Format_NHWC4) {
    if (OW_ * CO_SLICES_ <= MAX_IMAGE2D_SIZE) {
      {
        im_dst_y = batch_size_ * OH_;
        im_dst_x = OW_ * CO_SLICES_;
      }
    } else {
      im_dst_y = OW_;
      im_dst_x = batch_size_ * OH_ * CO_SLICES_;
    }
  } else {
    im_dst_y = batch_size_ * CO_SLICES_ * OH_;
    im_dst_x = OW_;
  }
  size_t img_dtype = use_fp16_ ? CL_HALF_FLOAT : CL_FLOAT;
  img_size->clear();
  img_size->push_back(im_dst_x);
  img_size->push_back(im_dst_y);
  img_size->push_back(img_dtype);
  return RET_OK;
}

int ConvolutionOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";

  int arg_cn = 0;
  if (use_winograd_) {
    arg_cn = 0;
    cl_int4 _4x4to36_in_shape = {1, IH_, IW_, CI_SLICES_};
    cl_int4 _4x4to36_out_shape = {1, 36, TILES_XY_, CI_SLICES_};
    ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, in_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, winograd_mem0_, lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_4x4to36_, arg_cn++, _4x4to36_in_shape);
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
    cl_int4 _36to4x4_out_shape = {1, OH_, OW_, CO_SLICES_};
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, winograd_mem1_, lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, out_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
    if (has_bias_) {
      ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, packed_bias_, lite::opencl::MemType::BUF);
    }
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, _36to4x4_in_shape);
    ocl_runtime_->SetKernelArg(kernel_36to4x4_, arg_cn++, _36to4x4_out_shape);
  } else {
    arg_cn = 0;
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, in_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, out_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
    ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, packed_weight_, lite::opencl::MemType::BUF);
    if (has_bias_) {
      ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, packed_bias_, lite::opencl::MemType::BUF);
    }
    if (op_format_ == Format_NC4HW4) {
      cl_int4 input_shape = {1, IH_, IW_, CI_SLICES_};
      cl_int4 output_shape = {1, OH_, OW_, CO_SLICES_};
      ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, input_shape);
      ocl_runtime_->SetKernelArg(kernel_conv_, arg_cn++, output_shape);
    }
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

std::string ConvolutionOpenCLKernel::CodeGenConvolutionNHWC4() {
  auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
  const size_t CI_ALIGN = CI_SLICES_ * C4NUM;
  const size_t CO_ALIGN = CO_SLICES_ * C4NUM;
  const size_t strideH = param->stride_h_;
  const size_t strideW = param->stride_w_;
  const size_t padTop = param->pad_u_;
  const size_t padBottom = param->pad_d_;
  const size_t padLeft = param->pad_l_;
  const size_t padRight = param->pad_r_;

  std::string code;
  code += "#define CI_TILE 4\n";
  code += "#define CO_TILE 4\n\n";
  code += "#define N " + std::to_string(batch_size_) + "\n";
  code += "#define N_OH " + std::to_string(batch_size_ * OH_) + "\n";
  code += "#define CI " + std::to_string(CI_ALIGN) + "\n";
  code += "#define IH " + std::to_string(IH_) + "\n";
  code += "#define IW " + std::to_string(IW_) + "\n";
  code += "#define CO " + std::to_string(CO_ALIGN) + "\n";
  code += "#define OH " + std::to_string(OH_) + "\n";
  code += "#define OW " + std::to_string(OW_) + "\n";
  code += "#define KH " + std::to_string(KH_) + "\n";
  code += "#define KW " + std::to_string(KW_) + "\n";
  code += "#define strideH " + std::to_string(strideH) + "\n";
  code += "#define strideW " + std::to_string(strideW) + "\n";
  code += "#define padTop " + std::to_string(padTop) + "\n";
  code += "#define padBottom " + std::to_string(padBottom) + "\n";
  code += "#define padLeft " + std::to_string(padLeft) + "\n";
  code += "#define padRight " + std::to_string(padRight) + "\n";
  code += "#define dilationH " + std::to_string(param->dilation_h_) + "\n";
  code += "#define dilationW " + std::to_string(param->dilation_w_) + "\n";
  code += "#define CI_SLICES " + std::to_string(CI_SLICES_) + "\n";
  code += "#define CO_SLICES " + std::to_string(CO_SLICES_) + "\n\n";

  if (use_fp16_) {
    code += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
  }

  code += "__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n\n";

  code +=
    "__kernel void Convolution(__read_only image2d_t input,\n"
    "                          __write_only image2d_t output,\n";
  if (has_bias_) {
    code +=
      "                          __global FLT4 *weight,\n"
      "                          __global FLT4 *bias) {\n";
  } else {
    code += "                          __global FLT4 *weight) {\n";
  }

  code += "    int n_oh = get_global_id(0);  // [0, N*OH)\n";
  if (batch_size_ == 1) {
    code += "    #define n 0\n";
    code += "    int oh = n_oh;\n";
  } else {
    code += "    int n = n_oh / " + std::to_string(OH_) + ";\n";
    code += "    int oh = n_oh % " + std::to_string(OH_) + ";\n";
  }

  code +=
    "    int ow = get_global_id(1);  // [0, OW)\n"
    "    int co_slice = get_global_id(2); // [0, UP_DIV(CO, CO_TILE) )\n"
    "\n"
    "    if (n_oh >= N_OH || ow >= OW || co_slice >= CO_SLICES) {\n"
    "        return;\n"
    "    }\n"
    "\n"
    "    FLT4 out0_c4 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
    "    __global FLT4 *w0_ic1_oc4 = weight + co_slice * KH * KW * CI_SLICES * CI_TILE;\n";

  code +=
    "    for (int kh = 0; kh < KH; ++kh)\n"
    "    {\n"
    "        int ih = kh * dilationH + oh * strideH - padTop;\n"
    "        for (int kw = 0; kw < KW; ++kw)\n"
    "        {\n"
    "            int iw = kw * dilationW + ow * strideW - padLeft;\n"
    "            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW)\n"
    "            {\n"
    "                for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++)\n"
    "                {\n";

  code +=
    "FLT4 in_c4 = READ_IMAGE(input, smp_zero, (int2)(iw * CI_SLICES + ci_slice, n * IH + ih)); // NHWC4: NH WC\n\n";

  code +=
    "                    out0_c4 += w0_ic1_oc4[0] * in_c4.x;\n"
    "                    out0_c4 += w0_ic1_oc4[1] * in_c4.y;\n"
    "                    out0_c4 += w0_ic1_oc4[2] * in_c4.z;\n"
    "                    out0_c4 += w0_ic1_oc4[3] * in_c4.w;\n"
    "                    w0_ic1_oc4 += 4;\n"
    "                }\n"
    "            }\n"
    "            else\n"
    "            {\n"
    "                w0_ic1_oc4 += 4 * CI_SLICES;\n"
    "            }\n"
    "        }\n"
    "    }\n\n";

  if (has_bias_) {
    code += "    out0_c4 = out0_c4 + bias[co_slice];\n";
  }

  if (param->act_type_ == ActType_Relu) {
    code += "    out0_c4 = max(out0_c4, (FLT4)(0.0f));\n";
  } else if (param->act_type_ == ActType_Relu6) {
    code += "    out0_c4 = clamp(out0_c4, (FLT4)(0.0f), (FLT4)(6.0f));\n";
  }
  if (OW_ * CO_SLICES_ <= MAX_IMAGE2D_SIZE) {
    code += "    WRITE_IMAGE(output, (int2)(ow * CO_SLICES + co_slice, n_oh), out0_c4);// NHWC4: NH WC\n}";
  } else {
    code += "    WRITE_IMAGE(output, (int2)(n_oh * CO_SLICES + co_slice, ow), out0_c4);\n}";
  }
  return code;
}

std::string ConvolutionOpenCLKernel::CodeGenConvolutionNC4HW4() {
  auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
  const size_t strideH = param->stride_h_;
  const size_t strideW = param->stride_w_;
  const size_t padTop = param->pad_u_;
  const size_t padBottom = param->pad_d_;
  const size_t padLeft = param->pad_l_;

  std::string code;

  if (use_fp16_) {
    code += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
  }

  code +=
    "__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
    "\n"
    "__kernel void Convolution(__read_only image2d_t input,\n"
    "                          __write_only image2d_t output,\n"
    "                          __global FLT4 *weight,\n";
  if (has_bias_) {
    code += "                          __global FLT4 *bias,\n";
  }
  code +=
    "                          const int4 input_shape,\n"
    "                          const int4 output_shape)\n"
    "{\n";

  code += "    int n_oh = get_global_id(0);  // [0, N*OH)\n";
  if (batch_size_ == 1) {
    code += "    #define n 0\n";
    code += "    int oh = n_oh;\n";
  } else {
    code += "    int n = n_oh / " + std::to_string(OH_) + ";\n";
    code += "    int oh = n_oh % " + std::to_string(OH_) + ";\n";
  }

  code +=
    "    int ow = get_global_id(1) * 2;\n"
    "    int co_slice = get_global_id(2);\n"
    "\n"
    "    int CI_SLICES = input_shape.w;\n"
    "    int CO_SLICES = output_shape.w;\n\n";

  code += "    #define N " + std::to_string(batch_size_) + "\n";
  code += "    #define N_OH " + std::to_string(batch_size_ * OH_) + "\n";
  code += "    #define IH " + std::to_string(IH_) + "\n";
  code += "    #define IW " + std::to_string(IW_) + "\n";
  code += "    #define OH " + std::to_string(OH_) + "\n";
  code += "    #define OW " + std::to_string(OW_) + "\n";
  code += "    #define KH " + std::to_string(KH_) + "\n";
  code += "    #define KW " + std::to_string(KW_) + "\n";
  code += "    #define strideH " + std::to_string(strideH) + "\n";
  code += "    #define strideW " + std::to_string(strideW) + "\n";
  code += "    #define padTop " + std::to_string(padTop) + "\n";
  code += "    #define padLeft " + std::to_string(padLeft) + "\n";
  code += "    #define dilationH " + std::to_string(param->dilation_h_) + "\n";
  code += "    #define dilationW " + std::to_string(param->dilation_w_) + "\n";

  code +=
    "    if (n_oh >= N_OH || ow >= OW || co_slice >= CO_SLICES) {\n"
    "        return;\n"
    "    }\n";

  bool check_ow = (OW_ % 2) == 1;
  if (check_ow) {
    code +=
      "    int last_is_double = 1;\n"
      "    if (ow + 1 >= OW)\n"
      "        last_is_double = 0;\n\n";
  }

  code +=
    "    FLT4 out0 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
    "    FLT4 out1 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
    "    __global FLT4 *w = weight + co_slice * KH * KW * CI_SLICES * 4;\n"
    "\n"
    "    for (int kh = 0; kh < KH; ++kh)\n"
    "    {\n"
    "        int ih = kh * dilationH + oh * strideH - padTop;\n"
    "        for (int kw = 0; kw < KW; ++kw)\n"
    "        {\n";

  if (padTop || padBottom) {
    code +=
      "if (ih >= 0 && ih < IH)\n"
      "{\n";
  }

  code += "            int iw0 = kw * dilationW + (ow + 0) * strideW - padLeft;\n";
  if (check_ow) {
    code +=
      "            if (last_is_double)\n"
      "            {\n";
  }

  code +=
    "                int iw1 = kw * dilationW + (ow + 1) * strideW - padLeft;\n"
    "                for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++)\n"
    "                {\n"
    "                    FLT4 in0 = READ_IMAGE(input, smp_zero, (int2)(iw0, (n * CI_SLICES + ci_slice) * IH + ih));\n"
    "                    out0 += w[0] * in0.x;\n"
    "                    out0 += w[1] * in0.y;\n"
    "                    out0 += w[2] * in0.z;\n"
    "                    out0 += w[3] * in0.w;\n"
    "                    FLT4 in1 = READ_IMAGE(input, smp_zero, (int2)(iw1, (n * CI_SLICES + ci_slice) * IH + ih));\n"
    "                    out1 += w[0] * in1.x;\n"
    "                    out1 += w[1] * in1.y;\n"
    "                    out1 += w[2] * in1.z;\n"
    "                    out1 += w[3] * in1.w;\n"
    "                    w += 4;\n"
    "                }\n";
  if (check_ow) {
    code +=
      "            }\n"
      "            else\n"
      "            {\n"
      "                for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++)\n"
      "                {\n"
      "                    FLT4 in0 = READ_IMAGE(input, smp_zero, (int2)(iw0, (n * CI_SLICES + ci_slice) * IH + ih));\n"
      "                    out0 += w[0] * in0.x;\n"
      "                    out0 += w[1] * in0.y;\n"
      "                    out0 += w[2] * in0.z;\n"
      "                    out0 += w[3] * in0.w;\n"
      "                    w += 4;\n"
      "                }\n"
      "            }\n";
  }
  if (padTop || padBottom) {
    code +=
      "}\n"
      "else\n"
      "{\n"
      "    w += CI_SLICES * 4;\n"
      "}\n";
  }
  code +=
    "        }\n"
    "    }\n\n";

  if (has_bias_) {
    code += "    out0 = out0 + bias[co_slice];\n";
  }

  if (param->act_type_ == ActType_Relu) {
    code += "    out0 = max(out0, (FLT4)(0.0f));\n";
  } else if (param->act_type_ == ActType_Relu6) {
    code += "    out0 = clamp(out0, (FLT4)(0.0f), (FLT4)(6.0f));\n";
  }
  code += "    WRITE_IMAGE(output, (int2)(ow + 0, (n * CO_SLICES + co_slice) * OH + oh), out0);\n";

  if (check_ow) {
    code +=
      "    if (last_is_double)"
      "    {\n";
  }
  if (has_bias_) {
    code += "    out1 = out1 + bias[co_slice];\n";
  }
  if (param->act_type_ == ActType_Relu) {
    code += "    out1 = max(out1, (FLT4)(0.0f));\n";
  } else if (param->act_type_ == ActType_Relu6) {
    code += "    out1 = clamp(out1, (FLT4)(0.0f), (FLT4)(6.0f));\n";
  }
  code += "    WRITE_IMAGE(output, (int2)(ow + 1, (n * CO_SLICES + co_slice) * OH + oh), out1);\n";
  if (check_ow) {
    code += "}\n";
  }
  code += "}\n";

  return code;
}

std::string ConvolutionOpenCLKernel::CodeGenWinograd4x4To36() {
  std::string code;
  code +=
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n"
    "#define PAD 1\n"
    "\n"
    "__constant sampler_t\n"
    "smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
    "\n"
    "constant FLT Bt[36] = {\n"
    "        1.0000000000f, 0.0000000000f, -2.5000004768f, -0.0000001192f, 1.0000001192f, 0.0000000000f,\n"
    "        0.0000000000f, 0.9428091049f, 1.3333333731f, -0.4714044929f, -0.6666667461f, 0.0000000000f,\n"
    "        0.0000000000f, -0.9428089857f, 1.3333334923f, 0.4714045525f, -0.6666667461f, 0.0000000000f,\n"
    "        0.0000000000f, -0.1178511307f, -0.0833333358f, 0.2357022613f, 0.1666666865f, 0.0000000000f,\n"
    "        0.0000000000f, 0.1178511307f, -0.0833333507f, -0.2357022911f, 0.1666666865f, 0.0000000000f,\n"
    "        0.0000000000f, 0.9999998808f, -0.0000000596f, -2.5000000000f, 0.0000000000f, 1.0000000000f,\n"
    "};\n"
    "\n"
    "__kernel void Winograd4x4To36(__read_only image2d_t input,\n"
    "                              __write_only image2d_t output,\n"
    "                              int4 input_shape,     // N H W CI_SLICES\n"
    "                              int4 output_shape)    // N 36 H/4*W/4 CI_SLICES\n"
    "{\n"
    "    int tile_xy = get_global_id(0);\n"
    "    int row = get_global_id(1);\n"
    "    int slice = get_global_id(2);\n"
    "\n"
    "    int TILE_XY = output_shape.z;\n"
    "    int SLICES = input_shape.w;\n"
    "    if (tile_xy >= TILE_XY || row >= 6 || slice >= SLICES)\n"
    "    {\n"
    "        return;\n"
    "    }\n"
    "\n"
    "    int IH = input_shape.y, IW = input_shape.z;\n"
    "    int TILE_X = UP_DIV(IW, 4);\n"
    "    int tile_x = tile_xy % TILE_X;\n"
    "    int tile_y = tile_xy / TILE_X;\n"
    "\n"
    "    constant FLT *Bt_row = Bt + row * 6;\n"
    "    FLT4 BtD_row[6] = {0};\n"
    "    for (int y = 0; y < 6; y++)\n"
    "    {\n"
    "        int ih = tile_y * 4 - PAD + y;\n";

  if (op_format_ == Format_NHWC4) {
    code += "        int y_idx = ih;\n";
  } else if (op_format_ == Format_NC4HW4) {
    code +=
      "        if(ih < 0 || ih >= IH) {continue;}\n"
      "        int y_idx = slice * IH + ih;\n";
  }

  code +=
    "        for (int x = 0; x < 6; x++)\n"
    "        {\n"
    "            int iw = tile_x * 4 - PAD + x;\n";

  if (op_format_ == Format_NHWC4) {
    code +=
      "            if(iw < 0 || iw >= IW) {continue;}\n"
      "            int x_idx = iw * SLICES + slice;\n";
  } else if (op_format_ == Format_NC4HW4) {
    code += "            int x_idx = iw;\n";
  }

  code +=
    "            BtD_row[x] += Bt_row[y] * READ_IMAGE(input, smp_none, (int2)(x_idx, y_idx));\n"
    "        }\n"
    "    }\n"
    "\n"
    "    for (int y = 0; y < 6; y++)\n"
    "    {\n"
    "        FLT4 acc = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
    "        for (int x = 0; x < 6; x++)\n"
    "        {\n"
    "            acc += BtD_row[x] * Bt[y * 6 + x];\n"
    "        }\n"
    "        WRITE_IMAGE(output, (int2)(tile_xy, slice * 36 + (row * 6 + y)), acc); // CH W  H=36\n"
    "    }\n"
    "}";
  return code;
}

std::string ConvolutionOpenCLKernel::CodeGenWinogradConvolution() {
  return "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
         "#define CI_TILE 4\n"
         "#define H 36\n"
         "__constant sampler_t\n"
         "smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
         "\n"
         "__kernel void WinogradConvolution(__read_only image2d_t input,\n"
         "                                  __write_only image2d_t output,\n"
         "                                  __global FLT16 *weight,\n"
         "                                  int4 input_shape,         // N 36 H/4*W/4 CI_SLICES\n"
         "                                  int4 output_shape)        // N 36 H/4*W/4 CO_SLICES\n"
         "{\n"
         "    int w = get_global_id(0) * 2;\n"
         "    int h = get_global_id(1);\n"
         "    int co_slice = get_global_id(2) * 2;\n"
         "\n"
         "    int CI_SLICES = input_shape.w;\n"
         "    int W = input_shape.z;\n"
         "    int CO_SLICES = output_shape.w;\n"
         "\n"
         "    if (h >= H || w >= W || co_slice >= CO_SLICES)\n"
         "    {\n"
         "        return;\n"
         "    }\n"
         "\n"
         "    FLT4 out00 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
         "    FLT4 out01 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
         "    FLT4 out10 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
         "    FLT4 out11 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
         "\n"
         "    int y_idx = h;\n"
         "    __global FLT16 *weight_ptr = weight + (co_slice / 2 * 36 + h) * CI_SLICES * 2;\n"
         "    for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++)\n"
         "    {\n"
         "        FLT4 in0 = READ_IMAGE(input, smp_none, (int2)(w + 0, y_idx));\n"
         "        FLT4 in1 = READ_IMAGE(input, smp_none, (int2)(w + 1, y_idx));\n"
         "        y_idx += 36;\n"
         "\n"
         "        FLT16 weight0 = weight_ptr[0], weight1 = weight_ptr[1];\n"
         "        weight_ptr += 2;\n"
         "\n"
         "\n"
         "        out00 += in0.x * weight0.s0123;\n"
         "        out00 += in0.y * weight0.s4567;\n"
         "        out00 += in0.z * weight0.s89ab;\n"
         "        out00 += in0.w * weight0.scdef;\n"
         "\n"
         "        out01 += in1.x * weight0.s0123;\n"
         "        out01 += in1.y * weight0.s4567;\n"
         "        out01 += in1.z * weight0.s89ab;\n"
         "        out01 += in1.w * weight0.scdef;\n"
         "\n"
         "        out10 += in0.x * weight1.s0123;\n"
         "        out10 += in0.y * weight1.s4567;\n"
         "        out10 += in0.z * weight1.s89ab;\n"
         "        out10 += in0.w * weight1.scdef;\n"
         "\n"
         "        out11 += in1.x * weight1.s0123;\n"
         "        out11 += in1.y * weight1.s4567;\n"
         "        out11 += in1.z * weight1.s89ab;\n"
         "        out11 += in1.w * weight1.scdef;\n"
         "    }\n"
         "\n"
         "    WRITE_IMAGE(output, (int2)(w + 0, (co_slice + 0) * H + h), out00);\n"
         "    if (w + 1 < W)\n"
         "    {\n"
         "        WRITE_IMAGE(output, (int2)(w + 1, (co_slice + 0) * H + h), out01);\n"
         "    }\n"
         "\n"
         "    if (co_slice + 1 < CO_SLICES)\n"
         "    {\n"
         "        WRITE_IMAGE(output, (int2)(w + 0, (co_slice + 1) * H + h), out10);\n"
         "        if (w + 1 < W)\n"
         "        {\n"
         "            WRITE_IMAGE(output, (int2)(w + 1, (co_slice + 1) * H + h), out11);\n"
         "        }\n"
         "    }\n"
         "}";
}

std::string ConvolutionOpenCLKernel::CodeGenWinograd36To4x4() {
  std::string code =
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n"
    "\n"
    "__constant sampler_t\n"
    "smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
    "\n"
    "constant FLT At[24] = {\n"
    "        1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 0.0000000000f,\n"
    "        0.0000000000f, 0.7071067691f, -0.7071067691f, 1.4142135382f, -1.4142135382f, 0.0000000000f,\n"
    "        0.0000000000f, 0.4999999702f, 0.4999999702f, 1.9999998808f, 1.9999998808f, 0.0000000000f,\n"
    "        0.0000000000f, 0.3535533845f, -0.3535533845f, 2.8284270763f, -2.8284270763f, 1.0000000000f\n"
    "};\n"
    "\n"
    "__kernel void Winograd36To4x4(__read_only image2d_t input,\n"
    "                              __write_only image2d_t output,\n";
  if (has_bias_) {
    code += "                              __global FLT4 *bias,\n";
  }
  code +=
    "                              int4 input_shape,      // N 36 H/4*W/4 CO_SLICES\n"
    "                              int4 output_shape)     // N H W CO_SLICES\n"
    "{\n"
    "    int tile_xy = get_global_id(0);\n"
    "    int row = get_global_id(1);\n"
    "    int slice = get_global_id(2);\n"
    "\n"
    "    int TILE_XY = input_shape.z;\n"
    "    int SLICES = input_shape.w;\n"
    "    int OH = output_shape.y;\n"
    "    int OW = output_shape.z;\n"
    "\n"
    "    if (tile_xy >= TILE_XY || row >= 4 || slice >= SLICES)\n"
    "    {\n"
    "        return;\n"
    "    }\n"
    "\n"
    "    constant FLT *At_row = At + row * 6;\n"
    "    FLT4 AtM_row[6] = {0};\n"
    "    for (int y = 0; y < 6; y++)\n"
    "    {\n"
    "        for (int x = 0; x < 6; x++)\n"
    "        {\n"
    "            AtM_row[x] += At_row[y] * READ_IMAGE(input, smp_none, (int2)(tile_xy, slice * 36 + y * 6 + x));\n"
    "        }\n"
    "    }\n"
    "\n"
    "    int TILE_X = UP_DIV(OW, 4);\n"
    "    for (int x = 0; x < 4; x++)\n"
    "    {\n"
    "        FLT4 acc = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
    "        for (int y = 0; y < 6; y++)\n"
    "        {\n"
    "            acc += AtM_row[y] * At[x * 6 + y];\n"
    "        }\n";
  if (has_bias_) {
    code += "        acc += bias[slice];\n";
  }

  auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
  if (param->act_type_ == ActType_Relu) {
    code += "        acc = max(acc, (FLT4)(0.0f));\n\n";
  } else if (param->act_type_ == ActType_Relu6) {
    code += "        acc = clamp(acc, (FLT4)(0.0f), (FLT4)(6.0f));\n\n";
  }

  code +=
    "        int tile_x = tile_xy % TILE_X;\n"
    "        int tile_y = tile_xy / TILE_X;\n"
    "        int ow = tile_x * 4 + x;\n"
    "        int oh = tile_y * 4 + row;\n";

  if (op_format_ == Format_NHWC4) {
    code += "        if(ow < OW) { WRITE_IMAGE(output, (int2)(ow * SLICES + slice, oh), acc);}\n";
  } else if (op_format_ == Format_NC4HW4) {
    code += "        if(oh < OH) { WRITE_IMAGE(output, (int2)(ow, slice * OH + oh), acc);}\n";
  }

  code +=
    "    }\n"
    "}";
  return code;
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

  if (op_format_ == Format_NHWC4) {
    if (OW_ * CO_SLICES_ > MAX_IMAGE2D_SIZE) {
      local_w = 4;
    }
  }

  global->clear();
  global->push_back(UP_DIV(batch_size_ * OH_, local_nh) * local_nh);
  global->push_back(UP_DIV(OW_, local_w) * local_w);
  global->push_back(UP_DIV(CO_SLICES_, local_c) * local_c);
  local->clear();
  local->push_back(local_nh);
  local->push_back(local_w);
  local->push_back(local_c);

  if (op_format_ == Format_NC4HW4) {
    // calculate 2 FLT4 along width per work-item
    global->at(1) = UP_DIV(global->at(1), 2);
    if (local->at(1) > global->at(1)) {
      local->at(1) = global->at(1);
    }
  }
  return RET_OK;
}

kernel::LiteKernel *OpenCLConvolutionKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                   const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                   const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel =
    new (std::nothrow) ConvolutionOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Convolution kernel failed!";
    free(opParameter);
    return nullptr;
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
