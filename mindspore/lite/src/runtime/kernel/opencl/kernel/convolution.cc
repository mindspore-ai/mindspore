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
  static int init_count = 0;
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto allocator = ocl_runtime->GetAllocator();
  std::set<std::string> build_options;
  init_count++;

  CI = in_tensors_[0]->Channel();
  IH = in_tensors_[0]->Height();
  IW = in_tensors_[0]->Width();
  CO = out_tensors_[0]->Channel();
  OH = out_tensors_[0]->Height();
  OW = out_tensors_[0]->Width();
  CI_SLICES = UP_DIV(CI, C4NUM);
  CO_SLICES = UP_DIV(CO, C4NUM);

  // note: TILES_X TILES_Y TILES_XY is only used when use_winograd_=true
  TILES_X = UP_DIV(OW, 4);
  TILES_Y = UP_DIV(OH, 4);
  TILES_XY = TILES_X * TILES_Y;
  use_winograd_ = UseWinograd4x4To6x6();

  // build kernel
  if (use_winograd_) {
    MS_LOG(DEBUG) << "use winograd";
    std::string program_name;
    program_name = "Winograd4x4To36" + std::to_string(init_count);
    ocl_runtime->LoadSource(program_name, CodeGenWinograd4x4To36());
    ocl_runtime->BuildKernel(kernel_4x4to36, program_name, "Winograd4x4To36", build_options);

    program_name = "WinogradConvolution" + std::to_string(init_count);
    ocl_runtime->LoadSource(program_name, CodeGenWinogradConvolution());
    ocl_runtime->BuildKernel(kernel_conv, program_name, "WinogradConvolution", build_options);

    program_name = "Winograd36To4x4" + std::to_string(init_count);
    ocl_runtime->LoadSource(program_name, CodeGenWinograd36To4x4());
    ocl_runtime->BuildKernel(kernel_36to4x4, program_name, "Winograd36To4x4", build_options);
  } else {
    std::string program_name = "convolution" + std::to_string(init_count);
    ocl_runtime->LoadSource(program_name, CodeGenConvolution());
    ocl_runtime->BuildKernel(kernel_conv, program_name, "Convolution", build_options);
  }

  // allocate winograd memory
  if (use_winograd_) {
#ifdef ENABLE_FP16
    size_t img_dtype = CL_HALF_FLOAT;
    size_t sizeof_datatype = 2;
#else
    size_t img_dtype = CL_FLOAT;
    size_t sizeof_datatype = 4;
#endif
    size_t size = TILES_XY * CI_SLICES * 36 * sizeof_datatype;
    size_t width = TILES_XY;
    size_t height = CI_SLICES * 36;
    winograd_mem0_ = allocator->Malloc(size, {width, height, img_dtype});

    size = TILES_XY * CO_SLICES * 36 * sizeof_datatype;
    width = TILES_XY;
    height = CO_SLICES * 36;
    winograd_mem1_ = allocator->Malloc(size, {width, height, img_dtype});
  }

  this->InitBuffer();
  in_ori_format_ = in_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(schema::Format_NHWC4);
  out_ori_format_ = out_tensors_[0]->GetFormat();
  out_tensors_[0]->SetFormat(schema::Format_NHWC4);
  MS_LOG(DEBUG) << "Convolution Init Done!";
  return RET_OK;
}

int ConvolutionOpenCLKernel::InitBuffer() {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto allocator = ocl_runtime->GetAllocator();

  auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
  size_t KH = param->kernel_h_;
  size_t KW = param->kernel_w_;
  constexpr size_t CI_TILE = C4NUM;
  constexpr size_t CO_TILE = C4NUM;
  size_t packed_weight_size;
  if (use_winograd_) {
    packed_weight_size = UP_DIV(CO, 8) * 6 * 6 * CI_SLICES * 2 * CI_TILE * CO_TILE * sizeof(float);
  } else {
    packed_weight_size = CO_SLICES * KH * KW * CI_SLICES * CI_TILE * CO_TILE * sizeof(float);
  }
  packed_weight_ = reinterpret_cast<float *>(allocator->Malloc(packed_weight_size));
  allocator->MapBuffer(packed_weight_, CL_MAP_WRITE, nullptr, true);
  memset(packed_weight_, 0x00, packed_weight_size);
  auto weight_tensor = in_tensors_[1];
  auto origin_weight = reinterpret_cast<float *>(weight_tensor->Data());

  if (use_winograd_) {
    // weight: OHWI -> O66I -> O/8 6 6 I/4 O2 I4 O4
    std::vector<float> encoded_weight(CO * 6 * 6 * CI);
    std::vector<float> Gt = {1.0000000000, 1.0000000000, 1.0000000000,  1.0000000000, 1.0000000000,  0.0000000000,
                             0.0000000000, 0.7071067691, -0.7071067691, 1.4142135382, -1.4142135382, 0.0000000000,
                             0.0000000000, 0.4999999702, 0.4999999702,  1.9999998808, 1.9999998808,  1.0000000000};

    std::vector<float> G(Gt.size());
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 6; ++x) {
        G[x * 3 + y] = Gt[y * 6 + x];
      }
    }

    for (int co = 0; co < CO; ++co) {
      for (int ci = 0; ci < CI; ++ci) {
        std::vector<float> in_vals(9);
        for (int kh = 0; kh < 3; ++kh) {
          for (int kw = 0; kw < 3; ++kw) {
            const int f_index = ((co * 3 + kh) * 3 + kw) * CI + ci;
            in_vals[kh * 3 + kw] = origin_weight[f_index];
          }
        }

        auto temp_vals = MatrixMultiply(G, in_vals, 6, 3, 3);
        auto out_vals = MatrixMultiply(temp_vals, Gt, 6, 3, 6);
        for (int kh = 0; kh < 6; ++kh) {
          for (int kw = 0; kw < 6; ++kw) {
            const int f_index = ((co * 6 + kh) * 6 + kw) * CI + ci;
            encoded_weight[f_index] = out_vals[kh * 6 + kw];
          }
        }
      }
    }

    for (int co = 0, src_idx = 0; co < CO; ++co) {
      for (int kh = 0; kh < 6; ++kh) {
        for (int kw = 0; kw < 6; ++kw) {
          for (int ci = 0; ci < CI; ++ci) {
            int co_outer = co / 8;
            int co_inner_group = co % 8 / 4;
            int co_inner = co % 8 % 4;
            int ci_outer = ci / 4;
            int ci_inner = ci % 4;
            size_t dst_idx =
              (((((co_outer * 6 + kh) * 6 + kw) * CI_SLICES + ci_outer) * 2 + co_inner_group) * CI_TILE + ci_inner) *
                CO_TILE +
              co_inner;
            packed_weight_[dst_idx] = encoded_weight[src_idx++];
          }
        }
      }
    }
  } else {
    // weight: OHWI -> O/4 H W I/4 I4 O4
    for (int co = 0; co < CO; ++co) {
      for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
          for (int ci = 0; ci < CI; ++ci) {
            auto co_outer = co / CO_TILE;
            auto co_inner = co % CO_TILE;
            auto ci_outer = ci / CI_TILE;
            auto ci_inner = ci % CI_TILE;
            packed_weight_[((((co_outer * KH + kh) * KW + kw) * CI_SLICES + ci_outer) * CI_TILE + ci_inner) * CO_TILE +
                           co_inner] = *(origin_weight++);
          }
        }
      }
    }
  }
  allocator->UnmapBuffer(packed_weight_);

  // align bias from C to C4
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
}

int ConvolutionOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
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
  if (use_winograd_) {
    arg_cn = 0;
    cl_int4 _4x4to36_in_shape = {1, IH, IW, CI_SLICES};
    cl_int4 _4x4to36_out_shape = {1, 36, TILES_XY, CI_SLICES};
    ocl_runtime->SetKernelArg(kernel_4x4to36, arg_cn++, in_tensors_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_4x4to36, arg_cn++, winograd_mem0_);
    ocl_runtime->SetKernelArg(kernel_4x4to36, arg_cn++, _4x4to36_in_shape);
    ocl_runtime->SetKernelArg(kernel_4x4to36, arg_cn++, _4x4to36_out_shape);

    arg_cn = 0;
    cl_int4 conv_in_shape = {1, 36, TILES_XY, CI_SLICES};
    cl_int4 conv_out_shape = {1, 36, TILES_XY, CO_SLICES};
    ocl_runtime->SetKernelArg(kernel_conv, arg_cn++, winograd_mem0_);
    ocl_runtime->SetKernelArg(kernel_conv, arg_cn++, winograd_mem1_);
    ocl_runtime->SetKernelArg(kernel_conv, arg_cn++, packed_weight_);
    ocl_runtime->SetKernelArg(kernel_conv, arg_cn++, conv_in_shape);
    ocl_runtime->SetKernelArg(kernel_conv, arg_cn++, conv_out_shape);

    arg_cn = 0;
    cl_int4 _36to4x4_in_shape = {1, 16, TILES_XY, CO_SLICES};
    cl_int4 _36to4x4_out_shape = {1, OH, OW, CO_SLICES};
    ocl_runtime->SetKernelArg(kernel_36to4x4, arg_cn++, winograd_mem1_);
    ocl_runtime->SetKernelArg(kernel_36to4x4, arg_cn++, out_tensors_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_36to4x4, arg_cn++, packed_bias_);
    ocl_runtime->SetKernelArg(kernel_36to4x4, arg_cn++, _36to4x4_in_shape);
    ocl_runtime->SetKernelArg(kernel_36to4x4, arg_cn++, _36to4x4_out_shape);
  } else {
    arg_cn = 0;
    ocl_runtime->SetKernelArg(kernel_conv, arg_cn++, in_tensors_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_conv, arg_cn++, out_tensors_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_conv, arg_cn++, packed_weight_);
    ocl_runtime->SetKernelArg(kernel_conv, arg_cn++, packed_bias_);
  }

  if (use_winograd_) {
    ocl_runtime->RunKernel(kernel_4x4to36, {size_t(TILES_XY), 6, size_t(CI_SLICES)}, {8, 6, 4}, nullptr);
    ocl_runtime->RunKernel(kernel_conv, {size_t(TILES_XY / 2), 36, size_t(CO_SLICES / 2)}, {8, 6, 2}, nullptr);
    ocl_runtime->RunKernel(kernel_36to4x4, {size_t(TILES_XY), 4, size_t(CO_SLICES)}, {32, 4, 2}, nullptr);
  } else {
    std::vector<size_t> global, local;
    SetGlobalLocalConv(&global, &local);
    ocl_runtime->RunKernel(kernel_conv, global, local, nullptr);
  }

  return RET_OK;
}

std::string ConvolutionOpenCLKernel::CodeGenConvolution() {
  auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
  const size_t CI_ALIGN = CI_SLICES * C4NUM;
  const size_t CO_ALIGN = CO_SLICES * C4NUM;
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
    "__kernel void Convolution(__read_only image2d_t input,\n"
    "                          __write_only image2d_t output,\n"
    "                          __global FLT4 *weight,\n"
    "                          __global FLT4 *bias)"
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
  code += "FLT4 in_c4 = READ_FLT4(input, smp_zero, (int2)(iw * CI_SLICES + ci_slice, ih)); // NHWC4: H WC\n\n";
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

  if (OW * CO_SLICES < 65536) {
    code += "    WRITE_FLT4(output, (int2)(ow * CO_SLICES + co_slice, oh), out0_c4_bias);// NHWC4: H WC\n}";
  } else {
    code += "    WRITE_FLT4(output, (int2)(oh * CO_SLICES + co_slice, ow), out0_c4_bias);// NHWC4: H WC\n}";
  }
  return code;
}

std::string ConvolutionOpenCLKernel::CodeGenWinograd4x4To36() {
  return "#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))\n"
         "#define PAD 1\n"
         "\n"
         "__constant sampler_t\n"
         "smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
         "\n"
         "constant float Bt[36] = {\n"
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
         "    int TILE_X = IW / 4;\n"
         "    int tile_x = tile_xy % TILE_X;\n"
         "    int tile_y = tile_xy / TILE_X;\n"
         "\n"
         "    constant float *Bt_row = Bt + row * 6;\n"
         "    float4 BtD_row[6] = {0};\n"
         "    for (int y = 0; y < 6; y++)\n"
         "    {\n"
         "        int y_idx = tile_y * 4 - PAD + y;\n"
         "        for (int x = 0; x < 6; x++)\n"
         "        {\n"
         "            int x_idx = (tile_x * 4 - PAD + x) * SLICES + slice;\n"
         "            BtD_row[x] += Bt_row[y] * read_imagef(input, smp_none, (int2)(x_idx, y_idx));\n"
         "        }\n"
         "    }\n"
         "\n"
         "    for (int y = 0; y < 6; y++)\n"
         "    {\n"
         "        float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
         "        for (int x = 0; x < 6; x++)\n"
         "        {\n"
         "            acc += BtD_row[x] * Bt[y * 6 + x];\n"
         "        }\n"
         "//        write_imagef(output, (int2)((row * 6 + y) * SLICES + slice, tile_xy), acc); // H WC  W=36\n"
         "        write_imagef(output, (int2)(tile_xy, slice * 36 + (row * 6 + y)), acc); // CH W  H=36\n"
         "    }\n"
         "}";
}

std::string ConvolutionOpenCLKernel::CodeGenWinogradConvolution() {
  return "#define CI_TILE 4\n"
         "#define H 36\n"
         "//#define W 256\n"
         "//#define CI 96\n"
         "//#define CO 80s\n"
         "//#define CI_SLICES 24\n"
         "//#define CO_SLICES 20\n"
         "\n"
         "#define FLT4 float4\n"
         "#define READ_FLT4 read_imagef\n"
         "#define WRITE_FLT4 write_imagef\n"
         "\n"
         "//#define __global\n"
         "\n"
         "__constant sampler_t\n"
         "smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
         "\n"
         "__kernel void WinogradConvolution(__read_only image2d_t input,\n"
         "                                  __write_only image2d_t output,\n"
         "                                  __global float16 *weight,\n"
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
         "    __global float16 *weight_ptr = weight + (co_slice / 2 * 36 + h) * CI_SLICES * 2;\n"
         "    for (int ci_slice = 0; ci_slice < CI_SLICES; ci_slice++)\n"
         "    {\n"
         "        FLT4 in0 = READ_FLT4(input, smp_none, (int2)(w + 0, y_idx));\n"
         "        FLT4 in1 = READ_FLT4(input, smp_none, (int2)(w + 1, y_idx));\n"
         "        y_idx += 36;\n"
         "\n"
         "        float16 weight0 = weight_ptr[0], weight1 = weight_ptr[1];\n"
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
         "    WRITE_FLT4(output, (int2)(w + 0, (co_slice + 0) * H + h), out00);\n"
         "    if (w + 1 < W)\n"
         "    {\n"
         "        WRITE_FLT4(output, (int2)(w + 1, (co_slice + 0) * H + h), out01);\n"
         "    }\n"
         "\n"
         "    if (co_slice + 1 < CO_SLICES)\n"
         "    {\n"
         "        WRITE_FLT4(output, (int2)(w + 0, (co_slice + 1) * H + h), out10);\n"
         "        if (w + 1 < W)\n"
         "        {\n"
         "            WRITE_FLT4(output, (int2)(w + 1, (co_slice + 1) * H + h), out11);\n"
         "        }\n"
         "    }\n"
         "}";
}

std::string ConvolutionOpenCLKernel::CodeGenWinograd36To4x4() {
  std::string code =
    "//#define TILE_XY 256\n"
    "//#define SLICES 20\n"
    "//#define OH 16\n"
    "//#define OW 256\n"
    "\n"
    "//#define __global\n"
    "__constant sampler_t\n"
    "smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
    "\n"
    "constant float At[24] = {\n"
    "        1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 1.0000000000f, 0.0000000000f,\n"
    "        0.0000000000f, 0.7071067691f, -0.7071067691f, 1.4142135382f, -1.4142135382f, 0.0000000000f,\n"
    "        0.0000000000f, 0.4999999702f, 0.4999999702f, 1.9999998808f, 1.9999998808f, 0.0000000000f,\n"
    "        0.0000000000f, 0.3535533845f, -0.3535533845f, 2.8284270763f, -2.8284270763f, 1.0000000000f\n"
    "};\n"
    "\n"
    "__kernel void Winograd36To4x4(__read_only image2d_t input,\n"
    "                              __write_only image2d_t output,\n"
    "                              __global float4 *bias,\n"
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
    "    constant float *At_row = At + row * 6;\n"
    "    float4 AtM_row[6] = {0};\n"
    "    for (int y = 0; y < 6; y++)\n"
    "    {\n"
    "        for (int x = 0; x < 6; x++)\n"
    "        {\n"
    "            AtM_row[x] += At_row[y] * read_imagef(input, smp_none, (int2)(tile_xy, slice * 36 + y * 6 + "
    "x));\n"
    "        }\n"
    "    }\n"
    "\n"
    "    for (int x = 0; x < 4; x++)\n"
    "    {\n"
    "        float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);\n"
    "        for (int y = 0; y < 6; y++)\n"
    "        {\n"
    "            acc += AtM_row[y] * At[x * 6 + y];\n"
    "        }\n"
    "        acc += bias[slice];\n";

  auto param = reinterpret_cast<ConvParameter *>(op_parameter_);
  if (param->is_relu_) {
    code += "    acc = max(acc, (float4)(0.0f));\n";
  } else if (param->is_relu6_) {
    code += "    acc = clamp(acc, (float4)(0.0f), (float4)(6.0f));\n";
  }

  code +=
    "        int TILE_X = OW / 4;\n"
    "        int tile_x = tile_xy % TILE_X * 4;\n"
    "        int tile_y = tile_xy / TILE_X * 4;\n"
    "//        write_imagef(output, (int2)(tile_x + x, slice * OH + tile_y + row), acc); // height=CH width=W\n"
    "        write_imagef(output, (int2)((tile_x + x) * SLICES + slice, tile_y + row), acc); // height=H "
    "width=WC\n"
    "    }\n"
    "}";
  return code;
}

int ConvolutionOpenCLKernel::SetGlobalLocalConv(std::vector<size_t> *global, std::vector<size_t> *local) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  constexpr size_t work_group_size[] = {4, 4, 1};
  auto max_work_item_sizes = ocl_runtime->GetWorkItemSize();
  size_t max_work_group_size = ocl_runtime->GetKernelMaxWorkGroupSize(kernel_conv(), (*ocl_runtime->Device())());
  const size_t max_z_size = std::min<size_t>(16, max_work_item_sizes[2]);

  size_t global_h = UP_DIV(OH, work_group_size[0]) * work_group_size[0];
  size_t global_w = UP_DIV(OW, work_group_size[1]) * work_group_size[1];
  size_t global_c = UP_DIV(CO_SLICES, work_group_size[2]) * work_group_size[2];

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

  if (OW * CO_SLICES > 65536) {
    local_w = 4;
  }

  global->clear();
  global->push_back(UP_DIV(OH, local_h) * local_h);
  global->push_back(UP_DIV(OW, local_w) * local_w);
  global->push_back(UP_DIV(CO_SLICES, local_c) * local_c);
  local->clear();
  local->push_back(local_h);
  local->push_back(local_w);
  local->push_back(local_c);
  return RET_OK;
}

kernel::LiteKernel *OpenCLConvolutionKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *opParameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc,
                                                   const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel =
    new (std::nothrow) ConvolutionOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Convolution kernel failed!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: Convolution";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Conv2D, OpenCLConvolutionKernelCreator)
}  // namespace mindspore::kernel
