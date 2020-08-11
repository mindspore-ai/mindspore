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
#include "src/runtime/kernel/arm/base/depth_to_space_base.h"
#include "src/runtime/kernel/arm/nnacl/depth_to_space.h"
#include "src/runtime/kernel/arm/fp32/depth_to_space.h"
#include "src/runtime/kernel/arm/int8/depth_to_space_int8.h"
#include "src/runtime/kernel/arm/nnacl/arithmetic_common.h"
#include "schema/model_generated.h"
#include "src/kernel_factory.h"
#include "include/errorcode.h"
#include "include/context.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_DepthToSpace;

namespace mindspore::kernel {
int DepthToSpaceBaseCPUKernel::Init() { return RET_OK; }

int DepthToSpaceBaseCPUKernel::ReSize() {
  if (in_tensors_[0]->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "depth_to_space only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  DepthToSpaceParameter *param = reinterpret_cast<DepthToSpaceParameter *>(op_parameter_);
  if (param->block_size_ <= 0) {
    MS_LOG(ERROR) << "Input block_size should > 0!";
    return RET_PARAM_INVALID;
  }
  auto shape_size = in_tensors_[0]->shape().size();
  if (shape_size != DIMENSION_4D) {
    MS_LOG(ERROR) << "Input shape size should be " << DIMENSION_4D;
    return RET_PARAM_INVALID;
  }
  int32_t in_strides[DIMENSION_4D];
  ComputeStrides(const_cast<int *>(in_tensors_[0]->shape().data()), in_strides, shape_size);
  param->in_stride_dim0_ = in_strides[0];
  param->in_stride_dim1_ = in_strides[1];
  param->in_stride_dim2_ = in_strides[2];
  int32_t out_strides[DIMENSION_4D];
  ComputeStrides(const_cast<int *>(out_tensors_[0]->shape().data()), out_strides, shape_size);
  param->out_stride_dim0_ = out_strides[0];
  param->out_stride_dim1_ = out_strides[1];
  param->out_stride_dim2_ = out_strides[2];
  return RET_OK;
}

kernel::LiteKernel *CpuDepthToSpaceInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                     const std::vector<lite::tensor::Tensor *> &outputs,
                                                     OpParameter *op_parameter, const lite::Context *ctx,
                                                     const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_DepthToSpace);
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) DepthToSpaceInt8CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BatchToSpaceInt8CPUKernel fail!";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuDepthToSpaceFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                     const std::vector<lite::tensor::Tensor *> &outputs,
                                                     OpParameter *op_parameter, const lite::Context *ctx,
                                                     const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_DepthToSpace);
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) DepthToSpaceCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new DepthToSpaceCPUKernel fail!";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DepthToSpace, CpuDepthToSpaceFp32KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_DepthToSpace, CpuDepthToSpaceInt8KernelCreator)
}  // namespace mindspore::kernel
