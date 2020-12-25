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
#include <vector>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/tensorlist_reserve_fp32.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorListReserve;

namespace mindspore::kernel {

int TensorListReserveCPUKernel::Init() { return RET_OK; }

int TensorListReserveCPUKernel::Run() {
  auto output = reinterpret_cast<lite::TensorList *>(out_tensors_[0]);
  output->set_tensors_data_type(element_dtype_);
  return RET_OK;
}

int TensorListReserveCPUKernel::ReSize() { return RET_OK; }

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorListReserve, LiteKernelCreator<TensorListReserveCPUKernel>)
}  // namespace mindspore::kernel
