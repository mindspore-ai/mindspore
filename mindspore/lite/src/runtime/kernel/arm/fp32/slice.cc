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
#include "src/runtime/kernel/arm/fp32/slice.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/fp32/slice.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Slice;

namespace mindspore::kernel {
namespace {
int SliceLaunch(int thread_id, LiteParallelGroupEnv *penv, void *cdata) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "Input cdata is nullptr!";
    return RET_NULL_PTR;
  }
  auto kernel = reinterpret_cast<SliceCPUKernel *>(cdata);
  return kernel->SliceParallelRun(thread_id);
}
}  // namespace

int SliceCPUKernel::ReSize() {
  auto *param = reinterpret_cast<SliceParameter *>(op_parameter_);
  auto input_shape = in_tensors_[0]->shape();
  if (input_shape.size() != param->param_length_) {
    MS_LOG(ERROR) << "Input begin's lenth " << param->param_length_ << "is not equal to input shape size "
                  << input_shape.size();
    return RET_ERROR;
  }
  if (input_shape.size() > DIMENSION_4D) {
    MS_LOG(ERROR) << "input dimension num should <= " << DIMENSION_4D;
    return RET_ERROR;
  }

  for (size_t i = 0; i < input_shape.size(); ++i) {
    param->shape_[i] = input_shape[i];
  }
  return RET_OK;
}

int SliceCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SliceCPUKernel::SliceParallelRun(int thread_id) {
  const float *input_data = reinterpret_cast<const float *>(in_tensors_[0]->Data());
  float *output_data = reinterpret_cast<float *>(out_tensors_[0]->Data());
  SliceParameter *param = reinterpret_cast<SliceParameter *>(op_parameter_);
  DoSlice(input_data, output_data, param);
  return RET_OK;
}

int SliceCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  SliceParameter *param = reinterpret_cast<SliceParameter *>(op_parameter_);
  for (int i = 0; i < param->param_length_; ++i) {
    if (param->size_[i] < 0) {
      param->size_[i] = param->shape_[i] - param->begin_[i];
    }
    param->end_[i] = param->begin_[i] + param->size_[i];
  }

  if (param->param_length_ < DIMENSION_4D) {
    PadSliceParameterTo4D(param);
  }

  const float *input_data = reinterpret_cast<const float *>(in_tensors_[0]->Data());
  float *output_data = reinterpret_cast<float *>(out_tensors_[0]->Data());
  if (param->size_[1] < param->op_parameter_.thread_num_) {
    DoSliceNoParallel(input_data, output_data, param);
    return RET_OK;
  }
  ret = LiteBackendParallelLaunch(SliceLaunch, this, param->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "slice launch fail!ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuSliceFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *op_parameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }

  MS_ASSERT(desc.type == schema::PrimitiveType_Slice);
  auto *kernel = new (std::nothrow) SliceCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SliceCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Slice, CpuSliceFp32KernelCreator)
}  // namespace mindspore::kernel
