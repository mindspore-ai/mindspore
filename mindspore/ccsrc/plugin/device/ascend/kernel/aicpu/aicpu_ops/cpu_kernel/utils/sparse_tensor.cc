/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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

#include "cpu_kernel/utils/sparse_tensor.h"
#include <vector>
#include "cpu_types.h"

namespace aicpu {
uint32_t SparseTensor::CreateSparseTensor(CpuKernelContext &ctx, Tensor *ix, Tensor *tensorvals,
                                          std::vector<int64_t> shape, std::vector<int64_t> order) {
  CUST_KERNEL_LOG_INFO(ctx, "Start to execute CreateSparseTensor.");
  if (ix == nullptr || ix->GetData() == nullptr) {
    CUST_KERNEL_LOG_ERROR(ctx, "Ix is nullptr.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (tensorvals == nullptr || tensorvals->GetData() == nullptr) {
    CUST_KERNEL_LOG_ERROR(ctx, "Vals is nullptr.");
    return KERNEL_STATUS_INNER_ERROR;
  }

  if (ix->GetTensorShape()->GetDims() > 2) {
    CUST_KERNEL_LOG_ERROR(ctx, "Index tensor dim size less than 2 or equal to 2, got size [%d] ",
                          ix->GetTensorShape()->GetDims());
    return KERNEL_STATUS_INNER_ERROR;
  }

  int64_t dims = (ix->GetTensorShape()->GetDims() == 0) ? 1 : ix->GetTensorShape()->GetDimSize(0);
  int64_t vals_dim0 = (tensorvals->GetTensorShape()->GetDims() == 0) ? 1 : tensorvals->GetTensorShape()->GetDimSize(0);
  if (dims != vals_dim0) {
    CUST_KERNEL_LOG_ERROR(ctx, "Ix dim_size_0 [%ld] != tensorvals dim_size_0 [%ld]", dims, vals_dim0);
    return KERNEL_STATUS_INNER_ERROR;
  }
  dims = ix->GetTensorShape()->GetDims() == 2 ? ix->GetTensorShape()->GetDimSize(1) : 1;
  int64_t orderSize = static_cast<int64_t>(order.size());
  int64_t shapeSize = static_cast<int64_t>(shape.size());
  if (orderSize != dims) {
    CUST_KERNEL_LOG_ERROR(ctx, "orderSize [%ld] != dims [%ld]", orderSize, dims);
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (shapeSize != dims) {
    CUST_KERNEL_LOG_ERROR(ctx, "shapeSize [%ld] != dims [%ld]", shapeSize, dims);
    return KERNEL_STATUS_INNER_ERROR;
  }
  ix_ = std::make_shared<EigenTensor>(ix, ix->GetData());
  vals_ = std::make_shared<EigenTensor>(tensorvals, tensorvals->GetData());
  if (ix_ == nullptr || vals_ == nullptr) {
    CUST_KERNEL_LOG_ERROR(ctx, "Indices or values create eigen tensor failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }

  shape_.assign(shape.begin(), shape.end());
  order_.assign(order.begin(), order.end());
  dims_ = static_cast<int32_t>(dims);
  CUST_KERNEL_LOG_INFO(ctx, "Execute CreateSparseTensor end");
  return KERNEL_STATUS_OK;
}

uint32_t SparseTensor::IndicesValid(CpuKernelContext &ctx) {
  if (std::any_of(order_.begin(), order_.end(), [](int64_t ord) { return ord < 0; })) {
    CUST_KERNEL_LOG_ERROR(ctx, "Order was not provided.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  if (ix_->GetTensor()->GetDataType() == DT_INT32) {
    if (EigenTensorIndicesValid<int32_t>(ctx) != KERNEL_STATUS_OK) {
      CUST_KERNEL_LOG_ERROR(ctx, "Indices valid failed.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else {
    if (EigenTensorIndicesValid<int64_t>(ctx) != KERNEL_STATUS_OK) {
      CUST_KERNEL_LOG_ERROR(ctx, "Indices valid failed.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

bool SparseTensor::ValidateToDense(CpuKernelContext &ctx, const Tensor *out) const {
  CUST_KERNEL_LOG_INFO(ctx, "Start execute ValidateToDense.");
  if (out->GetDataType() != vals_->GetTensor()->GetDataType()) {
    CUST_KERNEL_LOG_ERROR(ctx, "Output data type must match vals, got out [%d], vals [%d].", out->GetDataType(),
                          vals_->GetTensor()->GetDataType());
    return false;
  }
  if (out->GetTensorShape()->GetDims() != dims_) {
    CUST_KERNEL_LOG_ERROR(ctx, "Output dims must match idx, got output dims [%d], idx dims [%d].",
                          out->GetTensorShape()->GetDims(), dims_);
    return false;
  }
  const auto out_shape = out->GetTensorShape();
  int32_t shapeSize = static_cast<int32_t>(shape_.size());
  if (shapeSize != out_shape->GetDims()) {
    CUST_KERNEL_LOG_ERROR(ctx, "output dims must match shape dims, got output dim [%d], shape dim [%d].",
                          out_shape->GetDims(), shapeSize);
    return false;
  }
  for (size_t d = 0; d < shape_.size(); ++d) {
    if (shape_[d] > out_shape->GetDimSize(static_cast<int32_t>(d))) {
      CUST_KERNEL_LOG_ERROR(ctx,
                            "Valid output shape dims value failed, index [%zu], shape value [%ld], "
                            "greater than output shape value [%d].",
                            d, shape_[d], out_shape->GetDimSize(static_cast<int32_t>(d)));
      return false;
    }
  }
  CUST_KERNEL_LOG_INFO(ctx, "Execute Validate dense end.");
  return true;
}

GroupIterable SparseTensor::group(CpuKernelContext &ctx, const std::vector<int64_t> &group_ix) const {
  if (group_ix.size() > static_cast<size_t>(dims_)) {
    CUST_KERNEL_LOG_WARN(ctx, "Grop_ix.size:%zu > dims_:%d", group_ix.size(), dims_);
  }
  return GroupIterable(const_cast<Tensor *>(ix_->GetTensor()), const_cast<Tensor *>(vals_->GetTensor()), dims_,
                       group_ix);
}
}  // namespace aicpu
