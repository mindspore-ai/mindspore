/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/pass/infershape_pass.h"
#include "src/litert/infer_manager.h"
#include "src/litert/kernel_exec_util.h"

namespace mindspore::lite::pass {
int Infershape::Run(kernel::SubGraphKernel *subgraph, std::vector<Tensor *> *) {
  auto kernels = &(subgraph->nodes());
  for (const auto &kernel : *kernels) {
    CHECK_NULL_RETURN(kernel);
    auto ret = KernelInferShape(kernel->in_tensors(), kernel->out_tensors(), kernel->op_parameter());
    // infer invalid, maybe resized when executing
    if (ret == RET_INFER_INVALID) {
      return RET_OK;
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "InferShape failed, op name: " << kernel->name();
      return RET_ERROR;
    }

    // SetKernelTensorDataType for fp16 subgraph.
    // The model input tensor must be fp32, other kernels output tensor data type will be mistaken after infershape.
    ret = kernel::KernelExecUtil::SetKernelTensorDataType(kernel);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set tensor data type for kernel " << kernel->name() << std::endl;
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::lite::pass
