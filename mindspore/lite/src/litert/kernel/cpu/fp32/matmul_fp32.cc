/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/matmul_fp32.h"
#include <algorithm>
#include "include/errorcode.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/intrinsics/ms_simd_cpu_info.h"
#if defined(ENABLE_AVX512)
#include "src/litert/kernel/cpu/fp32/matmul_fp32_avx512.h"
#endif

#if defined(ENABLE_AVX)
#include "src/litert/kernel/cpu/fp32/matmul_fp32_avx.h"
#endif

#if defined(ENABLE_SSE)
#include "src/litert/kernel/cpu/fp32/matmul_fp32_sse.h"
#endif

#if defined(ENABLE_ARM32)
#include "src/litert/kernel/cpu/fp32/matmul_fp32_arm32.h"
#endif

#if defined(ENABLE_ARM64)
#include "src/litert/kernel/cpu/fp32/matmul_fp32_arm64.h"
#endif

using mindspore::lite::kCHWDimNumber;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::kHWDimNumber;
using mindspore::lite::kNCHWDimNumber;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_MatMulFusion;

namespace mindspore::kernel {
int MatmulCPUKernel::Prepare() {
  CHECK_NULL_RETURN(matmul_base_);
  matmul_base_->set_name(name_);
  matmul_base_->set_workspace(workspace());
  return matmul_base_->MatmulPrepare();
}

int MatmulCPUKernel::ReSize() {
  CHECK_NULL_RETURN(matmul_base_);
  matmul_base_->set_workspace(workspace());
  return matmul_base_->MatmulReSize();
}

int MatmulCPUKernel::Run() {
  CHECK_NULL_RETURN(matmul_base_);
  matmul_base_->set_workspace(workspace());
  return matmul_base_->Run();
}

MatmulFp32BaseCPUKernel *CreateMatmulFp32CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs,
                                                   const lite::InnerContext *ctx) {
  MatmulFp32BaseCPUKernel *kernel = nullptr;
#if defined(ENABLE_AVX512)
  AVX512_HARDWARE_SELF_AWARENESS_BEGIN
  kernel = new (std::nothrow) MatmulFp32AVX512CPUKernel(parameter, inputs, outputs, ctx);
  if (kernel != nullptr) {
    return kernel;
  }
  AVX512_HARDWARE_SELF_AWARENESS_END
#endif

#if defined(ENABLE_AVX)
  kernel = new (std::nothrow) MatmulFp32AVXCPUKernel(parameter, inputs, outputs, ctx);
  if (kernel != nullptr) {
    return kernel;
  }
#endif

#if defined(ENABLE_SSE)
  kernel = new (std::nothrow) MatmulFp32SSECPUKernel(parameter, inputs, outputs, ctx);
  if (kernel != nullptr) {
    return kernel;
  }
#endif

#if defined(ENABLE_ARM64)
  kernel = new (std::nothrow) MatmulFp32ARM64CPUKernel(parameter, inputs, outputs, ctx);
  if (kernel != nullptr) {
    return kernel;
  }
#elif defined(ENABLE_ARM32)
  kernel = new (std::nothrow) MatmulFp32ARM32CPUKernel(parameter, inputs, outputs, ctx);
  if (kernel != nullptr) {
    return kernel;
  }
#endif

  kernel = new (std::nothrow) MatmulFp32BaseCPUKernel(parameter, inputs, outputs, ctx);
  return kernel;
}

int MatmulCPUKernel::PreparePackedWeight(const lite::Tensor *tensor) {
  matmul_base_->SetWeightIsPacked(true);
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MatMulFusion, LiteKernelCreator<MatmulCPUKernel>)
}  // namespace mindspore::kernel
