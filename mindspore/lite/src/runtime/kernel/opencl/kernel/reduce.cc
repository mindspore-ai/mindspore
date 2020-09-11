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

#include <set>
#include <string>
#include <map>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/kernel/reduce.h"
#include "src/runtime/kernel/opencl/cl/reduce.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_Mean;
using mindspore::schema::PrimitiveType_Reduce;
using mindspore::schema::ReduceMode;
using mindspore::schema::ReduceMode_ReduceMax;
using mindspore::schema::ReduceMode_ReduceMean;
using mindspore::schema::ReduceMode_ReduceMin;
using mindspore::schema::ReduceMode_ReduceProd;
using mindspore::schema::ReduceMode_ReduceSum;
using mindspore::schema::ReduceMode_ReduceSumSquare;

namespace mindspore::kernel {

int ReduceOpenCLKernel::Init() {
  InitNHWCShape();
  auto reduce_param = reinterpret_cast<ReduceParameter *>(op_parameter_);
  if (reduce_param == nullptr) {
    return RET_NULL_PTR;
  }
  std::map<int, std::string> reduce_type2str{{ReduceMode_ReduceMean, "mean"}, {ReduceMode_ReduceSum, "sum"}};
  if (reduce_type2str.find(reduce_param->mode_) == reduce_type2str.end()) {
    MS_LOG(ERROR) << "not supported reduce type:" << reduce_param->mode_;
    return RET_PARAM_INVALID;
  }
  if (reduce_param->num_axes_ != 2 || ((reduce_param->axes_[0] != 1 || reduce_param->axes_[1] != 2) &&
                                       (reduce_param->axes_[0] != 2 || reduce_param->axes_[1] != 1))) {
    MS_LOG(ERROR) << "reduce op only support axes HW";
    return RET_PARAM_INVALID;
  }
  std::string kernel_name = reduce_type2str.at(reduce_param->mode_);
  kernel_name += "_" + std::string(EnumNameFormat(op_format_));
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  enable_fp16_ = ocl_runtime->GetFp16Enable();

  if (in_tensors_[0]->shape().back() != out_tensors_[0]->shape().back()) {
    MS_LOG(ERROR) << "Reduce input channel " << in_tensors_[0]->shape().back() << " should equal output channel"
                  << out_tensors_[0]->shape().back();
    return RET_ERROR;
  }
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = reduce_source;
  ocl_runtime->LoadSource(kernel_name, source);
  ocl_runtime->BuildKernel(kernel_, kernel_name, kernel_name, build_options);
#endif
  in_ori_format_ = in_tensors_[0]->GetFormat();
  out_ori_format_ = out_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(op_format_);
  out_tensors_[0]->SetFormat(op_format_);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

void ReduceOpenCLKernel::InitNHWCShape() {
  std::vector<int> shapex = out_tensors_[0]->shape();
  size_t n = 1, h = 1, w = 1, c = 1;
  if (shapex.size() == 2) {
    n = shapex[0];
    c = shapex[1];
  } else if (shapex.size() == 4) {
    n = shapex[0];
    h = shapex[1];
    w = shapex[2];
    c = shapex[3];
  }
  nhwc_shape_ = {n, h, w, c};
}

int ReduceOpenCLKernel::ReSize() { return RET_OK; }

int ReduceOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;

  if (op_format_ == schema::Format_NHWC4) {
    im_dst_x = nhwc_shape_[2] * UP_DIV(nhwc_shape_[3], C4NUM);
    im_dst_y = nhwc_shape_[0] * nhwc_shape_[1];
  } else if (op_format_ == schema::Format_NC4HW4) {
    im_dst_x = nhwc_shape_[2];
    im_dst_y = nhwc_shape_[0] * UP_DIV(nhwc_shape_[3], C4NUM) * nhwc_shape_[1];
  } else {
    MS_LOG(ERROR) << "not support op format:" << EnumNameFormat(op_format_);
    return RET_ERROR;
  }
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

int ReduceOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  std::vector<int> shapex = in_tensors_[0]->shape();
  int h = shapex[1];
  int w = shapex[2];
  int c = shapex[3];
  int c4 = UP_DIV(c, C4NUM);
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  std::vector<size_t> local = {};
  std::vector<size_t> global = {static_cast<size_t>(c4)};
  cl_int4 size = {h, w, c4, 1};
  int arg_idx = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->MutableData());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->MutableData());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, size);
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLReduceKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::Context *ctx, const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ReduceOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << " create failed.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Mean, OpenCLReduceKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Mean, OpenCLReduceKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Reduce, OpenCLReduceKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Reduce, OpenCLReduceKernelCreator)
}  // namespace mindspore::kernel
