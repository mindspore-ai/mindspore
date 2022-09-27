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

#include "src/litert/kernel/cpu/int8/matmul_int8.h"
#include "src/litert/kernel/cpu/int8/matmul_dynamic_int8.h"
#include "src/litert/kernel/cpu/int8/matmul_dynamic_sdot_int8.h"
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/common_func.h"
#include "include/errorcode.h"
#include "src/litert/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore::kernel {
namespace {
constexpr int min_matmul_input = 2;
constexpr int min_matmul_output = 1;
}  // namespace
int MatmulInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), min_matmul_input);
  CHECK_LESS_RETURN(out_tensors_.size(), min_matmul_output);
  InitParameter();

  auto ret = MatmulBaseInt8CPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParallelLaunch failed";
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int MatmulInt8CPUKernel::ReSize() {
  int batch = 1;
  auto x_shape = in_tensors_.at(0)->shape();
  auto o_shape = out_tensors_.at(0)->shape();
  const size_t min_size = 2;
  MS_ASSERT(x_shape.size() >= min_size);
  for (size_t i = 0; i < x_shape.size() - min_size; ++i) {
    batch *= x_shape[i];
  }
  param_->batch = batch;
  MS_ASSERT(o_shape.size() >= min_size);
  const size_t row_offset = 2;
  const size_t col_offset = 1;
  param_->row_ = o_shape[o_shape.size() - row_offset];
  param_->col_ = o_shape[o_shape.size() - col_offset];
  param_->deep_ = param_->a_transpose_ ? x_shape[x_shape.size() - row_offset] : x_shape[x_shape.size() - col_offset];

  auto ret = MatmulBaseInt8CPUKernel::MatmulReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "MatmulBaseInt8CPUKernel failed";
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *MatmulInt8CPUKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "parameter is nullptr.";
    return nullptr;
  }

  LiteKernel *kernel = nullptr;
  if (parameter->quant_type_ == schema::QuantType_QUANT_ALL) {
    kernel = new (std::nothrow) MatmulInt8CPUKernel(parameter, inputs, outputs, ctx);
  } else if (parameter->quant_type_ == schema::QuantType_QUANT_DYNAMIC) {
    if (inputs.front()->IsConst()) {
      MS_LOG(ERROR) << "kernel: " << parameter->name_ << " is unsupported A is const.";
      return nullptr;
    }
    if (lite::IsSupportSDot()) {
      kernel = new (std::nothrow) MatMulDynamicSdotInt8Kernel(parameter, inputs, outputs, ctx);
    } else {
      kernel = new (std::nothrow) MatmulDynamicInt8CPUKernel(parameter, inputs, outputs, ctx);
    }
  } else {
    MS_LOG(ERROR) << "kernel: " << parameter->name_ << " is unsupported quant type:" << parameter->quant_type_;
    free(parameter);
    return nullptr;
  }
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel: " << parameter->name_ << "is nullptr.";
    free(parameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_MatMulFusion, MatmulInt8CPUKernelCreator)
}  // namespace mindspore::kernel
