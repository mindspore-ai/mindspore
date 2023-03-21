/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "nnacl/reshape.h"
#include "nnacl/nnacl_manager.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ExpandDims;
using mindspore::schema::PrimitiveType_Flatten;
using mindspore::schema::PrimitiveType_FlattenGrad;
using mindspore::schema::PrimitiveType_Reshape;
using mindspore::schema::PrimitiveType_Squeeze;
using mindspore::schema::PrimitiveType_Unsqueeze;

namespace mindspore::nnacl {
int ReshapeKernel::Run() {
  auto in_tensor = in_tensors().front();
  CHECK_NULL_RETURN(in_tensor);
  auto out_tensor = out_tensors().front();
  CHECK_NULL_RETURN(out_tensor);
  auto in_shape = in_tensor->shape();
  // element number is 0, no need to copy data
  if (std::any_of(in_shape.begin(), in_shape.end(), [](auto dim) { return dim == 0; })) {
    return RET_OK;
  }

  if (in_tensor->data_type() != out_tensor->data_type() || in_tensor->data() == nullptr ||
      in_tensor->Size() != out_tensor->Size()) {
    MS_LOG(ERROR) << "Invalid param in reshape";
    return RET_ERROR;
  }

  if (in_tensor->allocator() == nullptr || in_tensor->allocator() != out_tensor->allocator() ||
      in_tensor->allocator() != ms_context_->allocator || /* runtime allocator */
      op_parameter_->is_train_session_ || out_tensor->IsGraphOutput()) {
    UpdateTensorData();
    auto ret = kernel_->compute(kernel_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Reshape compute failed.";
    }
    return ret;
  }

  out_tensor->FreeData();
  out_tensor->ResetRefCount();
  out_tensor->set_data(in_tensor->data());
  if (in_tensor->IsConst()) {
    out_tensor->set_own_data(false);
  } else {
    out_tensor->set_own_data(in_tensor->own_data());
  }
  return RET_OK;
}

NNACL_KERNEL(PrimitiveType_Reshape, kNumberTypeInt32, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Reshape, kNumberTypeFloat32, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Reshape, kNumberTypeFloat16, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Reshape, kNumberTypeBool, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Flatten, kNumberTypeInt32, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Flatten, kNumberTypeFloat16, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Flatten, kNumberTypeFloat32, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_FlattenGrad, kNumberTypeFloat16, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_FlattenGrad, kNumberTypeFloat32, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_ExpandDims, kNumberTypeInt32, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_ExpandDims, kNumberTypeFloat16, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_ExpandDims, kNumberTypeFloat32, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_ExpandDims, kNumberTypeBool, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_ExpandDims, kNumberTypeInt8, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Squeeze, kNumberTypeFloat32, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Squeeze, kNumberTypeFloat16, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Squeeze, kNumberTypeInt32, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Squeeze, kNumberTypeBool, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Unsqueeze, kNumberTypeFloat16, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Unsqueeze, kNumberTypeFloat32, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Unsqueeze, kNumberTypeInt32, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Unsqueeze, kNumberTypeInt64, NnaclOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Unsqueeze, kNumberTypeBool, NnaclOpt<ReshapeKernel>)
}  // namespace mindspore::nnacl
