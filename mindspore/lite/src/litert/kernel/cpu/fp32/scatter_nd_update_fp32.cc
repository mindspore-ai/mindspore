/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/scatter_nd_update_fp32.h"
#include <cstring>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ScatterNdUpdate;

namespace mindspore::kernel {
namespace {
int ScatterNdUpdateRun(void *cdata, int task_id, float, float) {
  auto kernel = static_cast<ScatterNdUpdateCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  return kernel->ScatterNdUpdate(task_id);
}
}  // namespace

int ScatterNdUpdateCPUKernel::ScatterNdUpdate(int task_id) {
  void *update_data = in_tensors_[kScatterUpdateIndex]->data();
  auto output_tensor = out_tensors_[kOutputIndex];
  void *output_data = output_tensor->data();
  CHECK_NULL_RETURN(update_data);
  CHECK_NULL_RETURN(output_data);
  param_->data_type_len = output_tensor->data_type() == kNumberTypeFloat16 ? FP16_DATA_TYPE_LEN : sizeof(float);
  auto ret = ScatterNDUpdate(output_data, update_data, output_unit_offsets_.data(), param_, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Execute ScatterNDUpdate failed, ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

int ScatterNdUpdateCPUKernel::Run() {
  auto in_tensor = in_tensors().front();
  auto out_tensor = out_tensors().front();
  if (in_tensor->allocator() == nullptr || in_tensor->allocator() != out_tensor->allocator() ||
      in_tensor->own_data() == false || in_tensor->IsConst() || op_parameter_->is_train_session_) {
    (void)memcpy(out_tensor->data(), in_tensor->data(), in_tensor->Size());
  } else {
    out_tensor->FreeData();
    out_tensor->ResetRefCount();
    out_tensor->set_data(in_tensor->data());
    out_tensor->set_own_data(in_tensor->own_data());
  }
  auto indices = in_tensors_.at(kScatterIndicesIndex);
  if (!indices->IsConst() && ReSize() != RET_OK) {
    MS_LOG(ERROR) << "ScatterNdUpdate resize failed.";
    return RET_ERROR;
  }

  auto ret = ParallelLaunch(ms_context_, ScatterNdUpdateRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterNdUpdate error error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_ScatterNdUpdate, LiteKernelCreator<ScatterNdUpdateCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ScatterNdUpdate, LiteKernelCreator<ScatterNdUpdateCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ScatterNdUpdate, LiteKernelCreator<ScatterNdUpdateCPUKernel>)
#endif
}  // namespace mindspore::kernel
