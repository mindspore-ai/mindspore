/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "nnacl/nnacl_reshape.h"
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

  if (in_tensor->data_type() != out_tensor->data_type()) {
    MS_LOG(ERROR) << "NNACL check in_tensor and out_tensor dtype failed. Kernel: " << name();
    return RET_ERROR;
  }

  if (in_tensor->data() == nullptr || in_tensor->Size() != out_tensor->Size()) {
    MS_LOG(ERROR) << "NNACL check in_tensor and out_tensor size failed, Kernel: " << name();
    return RET_ERROR;
  }
  return NNACLKernel::OptimizeDataCopy();
}

NNACL_KERNEL(PrimitiveType_Reshape, kNumberTypeInt32, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Reshape, kNumberTypeFloat32, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Reshape, kNumberTypeFloat16, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Reshape, kNumberTypeBool, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Flatten, kNumberTypeInt32, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Flatten, kNumberTypeFloat16, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Flatten, kNumberTypeFloat32, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_FlattenGrad, kNumberTypeFloat16, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_FlattenGrad, kNumberTypeFloat32, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_ExpandDims, kNumberTypeInt32, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_ExpandDims, kNumberTypeFloat16, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_ExpandDims, kNumberTypeFloat32, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_ExpandDims, kNumberTypeBool, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_ExpandDims, kNumberTypeInt8, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Squeeze, kNumberTypeFloat32, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Squeeze, kNumberTypeFloat16, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Squeeze, kNumberTypeInt32, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Squeeze, kNumberTypeBool, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Unsqueeze, kNumberTypeFloat16, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Unsqueeze, kNumberTypeFloat32, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Unsqueeze, kNumberTypeInt32, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Unsqueeze, kNumberTypeInt64, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Unsqueeze, kNumberTypeBool, NNACLOpt<ReshapeKernel>)
NNACL_KERNEL(PrimitiveType_Unsqueeze, kNumberTypeUInt8, NNACLOpt<ReshapeKernel>)
}  // namespace mindspore::nnacl
