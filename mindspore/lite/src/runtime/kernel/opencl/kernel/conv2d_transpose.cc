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
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/kernel/conv2d_transpose.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/fp16/conv2d_transpose2x2.cl.inc"
#include "src/runtime/kernel/opencl/cl/fp32/conv2d_transpose2x2.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_DeConv2D;

namespace mindspore::kernel {

int Conv2dTransposeOpenCLKernel::Init() {
  ConvParameter *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  if (param->kernel_h_ != 2 || param->kernel_w_ != 2 || param->stride_h_ != 2 || param->stride_w_ != 2) {
    MS_LOG(ERROR) << "only support kh=kw=2 and stride_h=stride_w=2.";
    return RET_ERROR;
  }
  if (param->pad_h_ != 0 || param->pad_w_ != 0) {
    MS_LOG(ERROR) << "only support pad =0.";
    return RET_ERROR;
  }
  std::string kernel_name = "conv2d_transpose2x2";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
#ifdef PROGRAM_WITH_IL
  ocl_runtime->CreateKernelFromIL(kernel_(), kernel_name);
#else
#ifdef ENABLE_FP16
  std::string source = conv2d_transpose2x2_source_fp16;
#else
  std::string source = conv2d_transpose2x2_source_fp32;
#endif
  std::set<std::string> build_options;
  std::string program_name = "conv2d_transpose2x2";
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  PadWeight();
  out_tensors_[0]->SetFormat(schema::Format_NHWC4);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int Conv2dTransposeOpenCLKernel::ReSize() { return 0; }

void Conv2dTransposeOpenCLKernel::PadWeight() {
  ConvParameter *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  int ci = in_tensors_[0]->Channel();
  int co = out_tensors_[0]->Channel();
  int kh = param->kernel_h_;
  int kw = param->kernel_w_;
  int div_ci = UP_DIV(ci, C4NUM);
  int div_co = UP_DIV(co, C4NUM);
  auto allocator = lite::opencl::OpenCLRuntime::GetInstance()->GetAllocator();

  // IHWO to OHWI4(I)4(O)(converter format is IHWO)
  // init padWeight_(buffer mem)
  padWeight_ =
    reinterpret_cast<FLOAT_t *>(allocator->Malloc(div_ci * div_co * C4NUM * C4NUM * kh * kw * sizeof(FLOAT_t)));
  padWeight_ = reinterpret_cast<FLOAT_t *>(allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true));
  auto origin_weight = reinterpret_cast<FLOAT_t *>(in_tensors_.at(kWeightIndex)->Data());
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
                int ori_index = ((ci_offset * kh + kh_i) * kw + kw_i) * ci + co_offset;
                padWeight_[index++] = origin_weight[ori_index];
              } else {
                padWeight_[index++] = 0.;
              }
            }
          }
        }
      }
    }
  }
  allocator->UnmapBuffer(padWeight_);

  // init bias_(image2d mem)
  size_t im_dst_x, im_dst_y;
  im_dst_x = div_co;
  im_dst_y = 1;
#ifdef ENABLE_FP16
  size_t img_dtype = CL_HALF_FLOAT;
#else
  size_t img_dtype = CL_FLOAT;
#endif
  std::vector<size_t> img_size{im_dst_x, im_dst_y, img_dtype};
  bias_ = reinterpret_cast<FLOAT_t *>(allocator->Malloc(im_dst_x * im_dst_y * C4NUM * sizeof(FLOAT_t), img_size));
  bias_ = reinterpret_cast<FLOAT_t *>(allocator->MapBuffer(bias_, CL_MAP_WRITE, nullptr, true));
  memset(bias_, 0x00, div_co * C4NUM * sizeof(FLOAT_t));
  if (in_tensors_.size() >= 3) {
    memcpy(bias_, in_tensors_[2]->Data(), co * sizeof(FLOAT_t));
  }
  allocator->UnmapBuffer(bias_);
}

int Conv2dTransposeOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  im_dst_x = UP_DIV(out_tensors_[0]->Channel() * out_tensors_[0]->Width(), C4NUM);
  im_dst_y = out_tensors_[0]->Height();
#ifdef ENABLE_FP16
  size_t img_dtype = CL_HALF_FLOAT;
#else
  size_t img_dtype = CL_FLOAT;
#endif
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

int Conv2dTransposeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  std::vector<int> shapex = in_tensors_[0]->shape();
  int n = shapex[0];
  if (n > 1) {
    MS_LOG(ERROR) << " n > 1 not supported!";
    return RET_ERROR;
  }
  ConvParameter *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  int ci = in_tensors_[0]->Channel();
  int co = out_tensors_[0]->Channel();
  int kh = param->kernel_h_;
  int kw = param->kernel_w_;
  int pad = param->pad_h_;
  int oh = out_tensors_[0]->Height();
  int ow = out_tensors_[0]->Width();
  int h = in_tensors_[0]->Height();
  int w = in_tensors_[0]->Width();
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  // local size should less than MAX_GROUP_SIZE
  std::vector<size_t> local = {16, 1, 16};
  std::vector<size_t> global = {UP_ROUND((size_t)UP_ROUND(oh / 2, 2), local[0]),
                                UP_ROUND((size_t)UP_ROUND(ow / 2, 2), local[1]), UP_ROUND((size_t)co / 4, local[2])};

  cl_int2 kernel_size = {kh, kw};
  cl_int2 stride = {2, 2};
  cl_int2 padding = {pad, pad};
  cl_int4 src_size = {h, w, UP_DIV(ci, C4NUM), 1};
  cl_int4 dst_size = {oh, ow, UP_DIV(co, C4NUM), 1};
  int arg_cnt = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_cnt++, in_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_cnt++, padWeight_);
  ocl_runtime->SetKernelArg(kernel_, arg_cnt++, bias_);
  ocl_runtime->SetKernelArg(kernel_, arg_cnt++, out_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_cnt++, kernel_size);
  ocl_runtime->SetKernelArg(kernel_, arg_cnt++, stride);
  ocl_runtime->SetKernelArg(kernel_, arg_cnt++, padding);
  ocl_runtime->SetKernelArg(kernel_, arg_cnt++, src_size);
  ocl_runtime->SetKernelArg(kernel_, arg_cnt++, dst_size);
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLConv2dTransposeKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                       const std::vector<lite::tensor::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::Context *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const lite::Primitive *primitive) {
  auto *kernel =
    new (std::nothrow) Conv2dTransposeOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << "is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (0 != ret) {
    // MS_LOG(ERROR) << "Init kernel failed, name: " << opDef.name()->str()
    //               << ", type: " << lite::EnumNameOpT(opDef.attr_type());
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_DeConv2D, OpenCLConv2dTransposeKernelCreator)
}  // namespace mindspore::kernel
