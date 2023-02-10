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
#include "cpu_kernel/cpu_proto/tensor_shape_impl.h"

#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"

namespace aicpu {
/*
 * get dims value of tensor shape.
 */
std::vector<int64_t> TensorShapeImpl::GetDimSizes() const {
  std::vector<int64_t> ret;
  for (int32_t i = 0; i < tensor_shape_->dim_size(); i++) {
    ret.emplace_back(tensor_shape_->dim(i).size());
  }
  return ret;
}

/*
 * set dims value to tensor shape.
 */
void TensorShapeImpl::SetDimSizes(const std::vector<int64_t> &dims) {
  tensor_shape_->clear_dim();
  for (size_t i = 0; i < dims.size(); ++i) {
    aicpuops::TensorShape_Dim *aicpu_dims = tensor_shape_->add_dim();
    KERNEL_CHECK_NULLPTR_VOID(aicpu_dims, "Protobuf add dim is null")
    aicpu_dims->set_size(dims[i]);
  }
}

/*
 * get format value of tensor shape.
 */
Format TensorShapeImpl::GetFormat() const { return static_cast<Format>(tensor_shape_->data_format()); }

/*
 * set format value to tensor shape.
 */
void TensorShapeImpl::SetFormat(Format format) { tensor_shape_->set_data_format(format); }

/*
 * get unknown rank value of tensor shape.
 */
bool TensorShapeImpl::GetUnknownRank() const { return tensor_shape_->unknown_rank(); }

/*
 * set unknown rank value to tensor shape.
 */
void TensorShapeImpl::SetUnknownRank(bool unknown_rank) { tensor_shape_->set_unknown_rank(unknown_rank); }

/*
 * get dims size of tensor shape.
 */
int32_t TensorShapeImpl::GetDims() const { return tensor_shape_->dim_size(); }

/*
 * get dim value of tensor shape index dim.
 */
int64_t TensorShapeImpl::GetDimSize(int32_t index) const {
  if ((index >= GetDims()) || (index < 0)) {
    KERNEL_LOG_ERROR(
      "Dim index[%d] must be not less than 0 and not greater than dims "
      "size[%d]",
      index, GetDims());
    return 0;
  }

  return tensor_shape_->dim(index).size();
}

/*
 * get data elements number.
 */
int64_t TensorShapeImpl::NumElements() const {
  int64_t num_elements = 1;
  for (int32_t i = 0; i < tensor_shape_->dim_size(); i++) {
    int64_t dim_size = tensor_shape_->dim(i).size();
    if (dim_size < 0) {
      return -1;
    }

    KERNEL_CHECK_ASSIGN_64S_MULTI(num_elements, dim_size, num_elements, -1);
  }
  return num_elements;
}

/*
 * get tensor proto.
 * @return shared_ptr<TensorShapeProto>:tensor shape proto ptr
 */

aicpuops::TensorShape *TensorShapeImpl::GetProto() const { return tensor_shape_.get(); }
}  // namespace aicpu