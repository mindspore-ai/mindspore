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
#include "src/runtime/kernel/arm/control/tensorlist_reserve.h"
#include <vector>
#include "include/errorcode.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TensorListReserve;
namespace {
constexpr int kNumInputSize = 2;
}
namespace mindspore::kernel {
int TensorListReserveCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), kNumInputSize);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  return RET_OK;
}

int TensorListReserveCPUKernel::Run() {
  auto input0 = in_tensors_.at(0);
  auto input1 = in_tensors_.at(1);
  MS_ASSERT(input1->data() != nullptr);
  int num_elements = reinterpret_cast<int *>(input1->data())[0];
  auto output = reinterpret_cast<lite::TensorList *>(out_tensors_[0]);
  CHECK_NULL_RETURN(output);
  if (output->tensors().size() < static_cast<uint32_t>(num_elements)) {
    auto ele_shape_ptr = reinterpret_cast<int *>(input0->data());
    if (ele_shape_ptr == nullptr) {
      return RET_NULL_PTR;
    }
    std::vector<std::vector<int> > tmp_shape(num_elements, std::vector<int>());
    output->set_element_shape(std::vector<int>(ele_shape_ptr, ele_shape_ptr + input0->ElementsNum()));
    output->set_shape(std::vector<int>(1, num_elements));
    auto ret = output->MallocTensorListData(kTypeUnknown, tmp_shape);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Failed to MallocTensorListData";
      return ret;
    }
  }
  output->set_tensors_data_type(element_dtype_);
  return RET_OK;
}

int TensorListReserveCPUKernel::ReSize() { return RET_OK; }

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_TensorListReserve, LiteKernelCreator<TensorListReserveCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_TensorListReserve, LiteKernelCreator<TensorListReserveCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TensorListReserve, LiteKernelCreator<TensorListReserveCPUKernel>)
}  // namespace mindspore::kernel
