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
#include "src/runtime/kernel/opencl/kernel/convolution.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;

namespace mindspore::kernel {

int ConvolutionOpenCLKernel::Init() {
  static int count = 0;
  std::set<std::string> build_options;
  std::string source = CodeGen();
  std::string program_name = "convolution" + std::to_string(count);
  count++;
  std::string kernel_name = "convolution";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();

  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
  this->InitBuffer();
  out_tensors_[0]->SetFormat(schema::Format_NHWC4);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

std::string ConvolutionOpenCLKernel::CodeGen() {
  auto param = reinterpret_cast<ConvParameter *>(op_parameter_);

  auto input_tensor = in_tensors_[0];
  auto output_tensor = out_tensors_[0];
  const size_t CI = input_tensor->Channel();
  const size_t CI_SLICES = UP_DIV(CI, C4NUM);
  const size_t CI_ALIGN = UP_DIV(CI, C4NUM) * C4NUM;
  const size_t IH = input_tensor->Height();
  const size_t IW = input_tensor->Width();
  const size_t CO = output_tensor->Channel();
  const size_t CO_SLICES = UP_DIV(CO, C4NUM);
  const size_t CO_ALIGN = UP_DIV(CO, C4NUM) * C4NUM;
  const size_t OH = output_tensor->Height();
  const size_t OW = output_tensor->Width();
  const size_t KH = param->kernel_h_;
  const size_t KW = param->kernel_w_;
  const size_t strideH = param->stride_h_;
  const size_t strideW = param->stride_w_;
  const size_t padTop = param->pad_u_;
  const size_t padBottom = param->pad_d_;
  const size_t padLeft = param->pad_l_;
  const size_t padRight = param->pad_r_;

  std::string code;
  code += "#define CI_TILE 4\n";
  code += "#define CO_TILE 4\n\n";
  code += "#define CI " + std::to_string(CI_ALIGN) + "\n";
  code += "#define IH " + std::to_string(IH) + "\n";
  code += "#define IW " + std::to_string(IW) + "\n";
  code += "#define CO " + std::to_string(CO_ALIGN) + "\n";
  code += "#define OH " + std::to_string(OH) + "\n";
  code += "#define OW " + std::to_string(OW) + "\n";
  code += "#define KH " + std::to_string(KH) + "\n";
  code += "#define KW " + std::to_string(KW) + "\n";
  code += "#define strideH " + std::to_string(strideH) + "\n";
  code += "#define strideW " + std::to_string(strideW) + "\n";
  code += "#define padTop " + std::to_string(padTop) + "\n";
  code += "#define padBottom " + std::to_string(padBottom) + "\n";
  code += "#define padLeft " + std::to_string(padLeft) + "\n";
  code += "#define padRight " + std::to_string(padRight) + "\n";
  code += "#define CI_SLICES " + std::to_string(CI_SLICES) + "\n";
  code += "#define CO_SLICES " + std::to_string(CO_SLICES) + "\n\n";

#ifdef ENABLE_FP16
  code +=
    "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    "#define FLT4 half4\n"
    "#define READ_FLT4 read_imageh\n"
    "#define WRITE_FLT4 write_imageh\n\n";
#else
  code +=
    "#define FLT4 float4\n"
    "#define READ_FLT4 read_imagef\n"
    "#define WRITE_FLT4 write_imagef\n\n";
#endif

  code += "__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n\n";

  code +=
    "__kernel void convolution(__read_only image2d_t input,\n"
    "                                  __global FLT4 *weight,\n"
    "                                  __global FLT4 *bias,\n"
    "                                  __write_only image2d_t output)\n"
    "{\n";

  code +=
    "    int oh = get_global_id(0);  // [0, OH)\n"
    "    int ow = get_global_id(1);  // [0, OW)\n"
    "    int co_slice = get_global_id(2); // [0, UP_DIV(CO, CO_TILE) )\n"
    "\n"
    "    if (oh >= OH || ow >= OW || co_slice >= CO_SLICES)\n"
    "        return;\n"
    "\n"
    "    FLT4 out0_c4 = (FLT4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
    "    __global FLT4 *w0_ic1_oc4 = weight + co_slice * KH * KW * CI_SLICES * CI_TILE;\n";

  code +=
    "    for (int kh = 0; kh < KH; ++kh)\n"
    "    {\n"
    "        int ih = kh + oh * strideH - padTop;\n"
    "        for (int kw = 0; kw < KW; ++kw)\n"
    "        {\n"
    "            int iw = kw + ow * strideW - padLeft;\n"
    "            if (ih >= 0 && ih < IH && iw >= 0 && iw < IW)\n"
    "            {\n"
    "                for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++)\n"
    "                {\n";

  //  NHWC4 NHC4W4 NC4HW4
  code += "FLT4 in_c4 = READ_FLT4(input, smp_zero, (int2)(iw * CI_SLICES + ci_slice, ih)); // NHWC4: H WC\n\n";
  //  code += "FLT4 in_c4 = READ_FLT4(input, smp_zero, (int2)(iw, ih * CI_SLICES + ci_slice)); // NHC4W4: HC W\n\n";
  //  code += "FLT4 in_c4 = READ_FLT4(input, smp_zero, (int2)(iw, ci_slice * IH + ih)); // NC4HW4: CH W\n\n";

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
  code += "    FLT4 out0_c4_bias = out0_c4 + bias[co_slice];\n";
  if (param->is_relu_) {
    code += "    out0_c4_bias = max(out0_c4_bias, (FLT4)(0.0f));\n";
  } else if (param->is_relu6_) {
    code += "    out0_c4_bias = clamp(out0_c4_bias, (FLT4)(0.0f), (FLT4)(6.0f));\n";
  }
  //  NHWC4 NHC4W4 NC4HW4
  if (OW * CO_SLICES < 65536) {
    code += "    WRITE_FLT4(output, (int2)(ow * CO_SLICES + co_slice, oh), out0_c4_bias);// NHWC4: H WC\n}";
  } else {
    code += "    WRITE_FLT4(output, (int2)(oh * CO_SLICES + co_slice, ow), out0_c4_bias);// NHWC4: H WC\n}";
  }
  //  code += "    WRITE_FLT4(output, (int2)(ow, oh * CO_SLICES + co_slice), out0_c4_bias);// NHC4W4: HC W\n}";
  //  code += "    WRITE_FLT4(output, (int2)(ow ,co_slice * OH +  oh), out0_c4_bias);// NC4HW4: CH W\n}";

  //  std::cout << code << std::endl;
  return code;
}

int ConvolutionOpenCLKernel::InitBuffer() {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto allocator = ocl_runtime->GetAllocator();

  // weight: OHWI -> OHWIIO
  auto weight_tensor = in_tensors_[1];
  auto weight_shape = weight_tensor->shape();
  size_t CO = weight_shape[0];
  size_t KH = weight_shape[1];
  size_t KW = weight_shape[2];
  size_t CI = weight_shape[3];
  size_t CI_SLICES = UP_DIV(CI, C4NUM);
  size_t CO_SLICES = UP_DIV(CO, C4NUM);
  constexpr size_t CI_TILE = C4NUM;
  constexpr size_t CO_TILE = C4NUM;
  size_t packed_weight_size = CO_SLICES * KH * KW * CI_SLICES * CI_TILE * CO_TILE * sizeof(float);
  packed_weight_ = reinterpret_cast<float *>(allocator->Malloc(packed_weight_size));
  packed_weight_ = reinterpret_cast<float *>(allocator->MapBuffer(packed_weight_, CL_MAP_WRITE, nullptr, true));
  memset(packed_weight_, 0x00, packed_weight_size);
  auto weight_data = reinterpret_cast<float *>(weight_tensor->Data());
  for (int co = 0; co < CO; ++co) {
    for (int kh = 0; kh < KH; ++kh) {
      for (int kw = 0; kw < KW; ++kw) {
        for (int ci = 0; ci < CI; ++ci) {
          auto co_outer = co / CO_TILE;
          auto co_inner = co % CO_TILE;
          auto ci_outer = ci / CI_TILE;
          auto ci_inner = ci % CI_TILE;
          packed_weight_[((((co_outer * KH + kh) * KW + kw) * CI_SLICES + ci_outer) * CI_TILE + ci_inner) * CO_TILE +
                         co_inner] = *(weight_data++);
        }
      }
    }
  }
  allocator->UnmapBuffer(packed_weight_);

  // align bias
  auto bias_tensor = in_tensors_[2];
  size_t packed_bias_size = CO_SLICES * CO_TILE * sizeof(float);
  packed_bias_ = reinterpret_cast<float *>(allocator->Malloc(packed_bias_size));
  packed_bias_ = reinterpret_cast<float *>(allocator->MapBuffer(packed_bias_, CL_MAP_WRITE, nullptr, true));
  memset(packed_bias_, 0x00, packed_bias_size);
  auto bias_data = reinterpret_cast<float *>(bias_tensor->Data());
  for (int co = 0; co < CO; ++co) {
    packed_bias_[co] = bias_data[co];
  }
  allocator->UnmapBuffer(packed_bias_);

  return RET_OK;
}  // namespace mindspore::kernel

static int GetBiggestDivider(int x, int y) {
  for (int i = y; i != 0; i--) {
    if (x % i == 0) {
      return i;
    }
  }
  return 1;
}

int ConvolutionOpenCLKernel::GetGlobalLocal(std::vector<size_t> *global, std::vector<size_t> *local) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
  param->output_h_ = out_tensors_[0]->Height();
  param->output_w_ = out_tensors_[0]->Width();
  param->output_channel_ = out_tensors_[0]->Channel();

  constexpr size_t work_group_size[] = {4, 4, 1};
  auto max_work_item_sizes = ocl_runtime->GetWorkItemSize();
  size_t max_work_group_size = ocl_runtime->GetKernelMaxWorkGroupSize(kernel_(), (*ocl_runtime->Device())());
  const size_t max_z_size = std::min<size_t>(16, max_work_item_sizes[2]);

  size_t global_h = UP_DIV(param->output_h_, work_group_size[0]) * work_group_size[0];
  size_t global_w = UP_DIV(param->output_w_, work_group_size[1]) * work_group_size[1];
  size_t global_c = UP_DIV(UP_DIV(param->output_channel_, C4NUM), work_group_size[2]) * work_group_size[2];

  size_t local_c = GetBiggestDivider(global_c, max_z_size);
  if (local_c == 0) {
    MS_LOG(ERROR) << "Divide by zero";
    return RET_ERROR;
  }
  size_t local_hw_size = std::min<size_t>(256, max_work_group_size) / local_c;
  size_t local_w = std::min(global_w, local_hw_size);
  size_t local_h = std::min(local_hw_size / local_w, global_h);
  if (local_h == global_h && global_h % 2 == 0) {
    local_h = global_h / 2;
  }

  auto output_tensor = out_tensors_[0];
  const size_t CO = output_tensor->Channel();
  const size_t CO_SLICES = UP_DIV(CO, C4NUM);
  const size_t OW = output_tensor->Width();
  if (OW * CO_SLICES > 65536) {
    local_w = 4;
  }

  global->clear();
  global->push_back(UP_DIV(param->output_h_, local_h) * local_h);
  global->push_back(UP_DIV(param->output_w_, local_w) * local_w);
  global->push_back(UP_DIV(UP_DIV(param->output_channel_, C4NUM), local_c) * local_c);
  local->clear();
  local->push_back(local_h);
  local->push_back(local_w);
  local->push_back(local_c);
  return RET_OK;
}

int ConvolutionOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t CO_SLICES = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t im_dst_x, im_dst_y;
  if (in_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
    if (out_tensors_[0]->Width() * CO_SLICES < 65536) {
      {
        im_dst_x = out_tensors_[0]->Width() * CO_SLICES;
        im_dst_y = out_tensors_[0]->Height();
      }
    } else {
      im_dst_x = out_tensors_[0]->Height() * CO_SLICES;
      im_dst_y = out_tensors_[0]->Width();
    }
  } else {
    im_dst_y = out_tensors_[0]->Height() * CO_SLICES;
    im_dst_x = out_tensors_[0]->Width();
  }
#ifdef ENABLE_FP16
  size_t img_dtype = CL_HALF_FLOAT;
#else
  size_t img_dtype = CL_FLOAT;
#endif
  img_size->clear();
  img_size->push_back(im_dst_x);
  img_size->push_back(im_dst_y);
  img_size->push_back(img_dtype);
  return RET_OK;
}

int ConvolutionOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();

  int arg_cn = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, packed_weight_);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, packed_bias_);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->Data());

  std::vector<size_t> global;
  std::vector<size_t> local;
  GetGlobalLocal(&global, &local);
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLConvolutionKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *opParameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  auto *kernel = new ConvolutionOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Convolution kernel failed!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: Convolution";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Conv2D, OpenCLConvolutionKernelCreator)
}  // namespace mindspore::kernel
