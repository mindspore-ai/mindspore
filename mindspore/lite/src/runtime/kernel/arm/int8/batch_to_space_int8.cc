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
#include "src/runtime/kernel/arm/int8/batch_to_space_int8.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/batch_to_space.h"
#include "nnacl/int8/batch_to_space_int8.h"
#include "include/errorcode.h"

using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::schema::PrimitiveType_BatchToSpace;
using mindspore::schema::PrimitiveType_BatchToSpaceND;

namespace mindspore::kernel {
int BatchToSpaceInt8CPUKernel::Init() {
  auto ret = BatchToSpaceBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->GetQuantParams();
  in_quant_arg_.scale_ = in_quant_args.front().scale;
  in_quant_arg_.zp_ = in_quant_args.front().zeroPoint;

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->GetQuantParams();
  out_quant_arg_.scale_ = out_quant_args.front().scale;
  out_quant_arg_.zp_ = out_quant_args.front().zeroPoint;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int BatchToSpaceInt8CPUKernel::ReSize() { return BatchToSpaceBaseCPUKernel::ReSize(); }

int BatchToSpaceInt8CPUKernel::Run() {
  auto input = in_tensors_[0];
  auto output = out_tensors_[0];
  const int8_t *input_data = reinterpret_cast<const int8_t *>(input->MutableData());
  int8_t *output_data = reinterpret_cast<int8_t *>(output->MutableData());
  auto in_shape = input->shape();
  auto out_shape = output->shape();
  BatchToSpaceParameter *param = reinterpret_cast<BatchToSpaceParameter *>(this->op_parameter_);

  if (in_quant_arg_.scale_ == out_quant_arg_.scale_ && in_quant_arg_.zp_ == out_quant_arg_.zp_) {
    if (IsNoCrop()) {
      BatchToSpaceNoCropForNHWC(input_data, output_data, in_shape.data(), out_shape[0], param->block_shape_,
                                sizeof(int8_t));
    } else {
      BatchToSpaceForNHWC(input_data, output_data, in_shape.data(), out_shape[0], param->block_shape_, param->crops_,
                          sizeof(int8_t));
    }
  } else {
    if (IsNoCrop()) {
      BatchToSpaceNoCropForNHWCInt8(input_data, output_data, in_shape.data(), out_shape[0], param->block_shape_,
                                    &in_quant_arg_, &out_quant_arg_);
    } else {
      BatchToSpaceForNHWCInt8(input_data, output_data, in_shape.data(), out_shape[0], param->block_shape_,
                              param->crops_, &in_quant_arg_, &out_quant_arg_);
    }
  }

  return RET_OK;
}

kernel::LiteKernel *CpuBatchToSpaceInt8KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                     const std::vector<lite::Tensor *> &outputs,
                                                     OpParameter *op_parameter, const lite::InnerContext *ctx,
                                                     const kernel::KernelKey &desc,
                                                     const mindspore::lite::PrimitiveC *primitive) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) BatchToSpaceInt8CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
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
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_BatchToSpace, CpuBatchToSpaceInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_BatchToSpaceND, CpuBatchToSpaceInt8KernelCreator)
}  // namespace mindspore::kernel
