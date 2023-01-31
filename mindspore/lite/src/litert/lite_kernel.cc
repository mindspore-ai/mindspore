/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/litert/lite_kernel.h"
#include <algorithm>
#include "src/tensor.h"
#include "src/common/utils.h"
#include "src/litert/infer_manager.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

void LiteKernel::AllocWorkspace() {
  workspace_ = malloc(workspace_size());
  if (workspace_ == nullptr) {
    MS_LOG(ERROR) << "fail to alloc " << workspace_size() << "in kernel" << name();
    return;
  }
  ws_allocated_ = true;
}

void LiteKernel::FreeWorkspace() {
  if (ws_allocated_) {
    free(workspace_);
  }
  workspace_ = nullptr;
  ws_allocated_ = false;
}

int LiteKernel::PreProcess() {
  if (!InferShapeDone()) {
    auto ret = lite::KernelInferShape(in_tensors_, out_tensors_, op_parameter_, ms_context_->allocator);
    if (ret != 0) {
      MS_LOG(ERROR) << "InferShape fail!";
      return ret;
    }
    ret = ReSize();
    if (ret != 0) {
      MS_LOG(ERROR) << "ReSize fail!ret: " << ret;
      return ret;
    }
  }

  // check if inputs are valid
  if (!CheckInputsValid()) {
    MS_LOG(ERROR) << "The input is not valid.";
    return RET_ERROR;
  }
  // check if parameters are valid
  if (!CheckParamsValid()) {
    MS_LOG(ERROR) << "The parameter is not valid.";
    return RET_ERROR;
  }
  for (auto *output : this->out_tensors()) {
    MS_ASSERT(output != nullptr);
    if (registry_data_type_ == kNumberTypeFloat16 && output->data_type() == kNumberTypeFloat32) {
      output->set_data_type(kNumberTypeFloat16);
    }
    auto ret = output->MallocData();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "MallocData failed";
      return ret;
    }
    output->ResetRefCount();
  }
  return RET_OK;
}

int LiteKernel::UpdateThreadNumProcess(int32_t kernel_type, int64_t per_unit_load_num, int64_t per_unit_store_num,
                                       int64_t unit_num) {
  thread_num_ =
    lite::UpdateThreadNum(kernel_type, per_unit_load_num, per_unit_store_num, unit_num, op_parameter_->thread_num_);
  return lite::RET_OK;
}

int LiteKernel::UpdateThreadNumPass(int32_t kernel_type, int64_t per_unit_load_num, int64_t per_unit_store_num,
                                    int64_t unit_num) {
#ifdef DYNAMIC_THREAD_DISTRIBUTE
  if (UpdateThreadNumProcess(kernel_type, per_unit_load_num, per_unit_store_num, unit_num) != lite::RET_OK) {
    MS_LOG(ERROR) << "update thread num failed";
    return lite::RET_ERROR;
  }
#else
  thread_num_ = op_parameter_->thread_num_ > 0 ? op_parameter_->thread_num_ : 1;
#endif

  return lite::RET_OK;
}

int LiteKernel::Execute() {
  auto ret = PreProcess();
  if (lite::RET_OK != ret) {
    MS_LOG(ERROR) << "run kernel PreProcess failed, name: " << this->name();
    return ret;
  }

  /* op_parameter_ is null : run in kernel mod */
  if (op_parameter_ == nullptr || op_parameter_->is_zero_shape_ == false) {
    ret = Run();
    if (lite::RET_OK != ret) {
      MS_LOG(ERROR) << "run kernel failed, name: " << this->name();
      return ret;
    }
  }

  ret = PostProcess();
  if (lite::RET_OK != ret) {
    MS_LOG(ERROR) << "run kernel PostProcess failed, name: " << this->name();
    return ret;
  }
  return lite::RET_OK;
}
}  // namespace mindspore::kernel
