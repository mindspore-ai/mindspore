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

#include "src/runtime/kernel/arm/fp32/scatter_nd_fp32.h"
#include <cstring>
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ScatterNd;

namespace mindspore::kernel {
namespace {
constexpr int kScatterShapeIndex = 0;
constexpr int kScatterIndicesIndex = 1;
constexpr int kScatterUpdateIndex = 2;
}  // namespace
int ScatterNDCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ScatterNDCPUKernel::ReSize() {
  auto indices = in_tensors_[kScatterIndicesIndex];
  auto update = in_tensors_[kScatterUpdateIndex];
  auto shape = in_tensors_[kScatterShapeIndex];
  CHECK_NULL_RETURN(indices);
  CHECK_NULL_RETURN(update);
  CHECK_NULL_RETURN(shape);

  // check indices shape
  auto shape_rank = shape->ElementsNum();
  auto shape_data = reinterpret_cast<int *>(shape->data());
  CHECK_NULL_RETURN(shape_data);
  auto indice_unit_rank = indices->shape().back();
  if (indice_unit_rank > shape_rank) {
    MS_LOG(ERROR) << "Value of last dimension of indices is greater than shape rank.";
    return RET_ERROR;
  }

  if (indices->shape().size() < 2) {
    MS_LOG(ERROR) << "Indices dimension smaller than 2.";
    return RET_ERROR;
  }

  // check consistency of the shape indices and shape
  auto update_rank = static_cast<int>(update->shape().size());
  auto indices_shape = indices->shape();
  if (update_rank != static_cast<int>(indices->shape().size() - 1 + shape_rank - indice_unit_rank)) {
    MS_LOG(ERROR) << "Update, shape rank and indices rank inconsistent.";
    return RET_ERROR;
  }
  // check update shape
  auto update_shape = update->shape();
  for (size_t i = 0; i < indices_shape.size() - 1; i++) {
    if (update_shape.at(i) != indices_shape.at(i)) {
      MS_LOG(ERROR) << "Value of " << i << " th dimension of indices is not equal to that of update.";
      return RET_ERROR;
    }
  }
  for (size_t i = 0; i < shape_rank - (indices_shape.size() - 1); i++) {
    if (update_shape.at(i + indices_shape.size() - 1) != shape_data[i + indices_shape.size() - 1]) {
      MS_LOG(ERROR) << "Value of " << i + indices_shape.size() - 1
                    << " th dimension of indices is not equal to the corresbonding dimension of shape.";
      return RET_ERROR;
    }
  }

  // calculate unit_size
  param_->unit_size = 1;
  for (int i = indices_shape.size() - 1; i < update_rank; i++) {
    param_->unit_size *= update_shape.at(i);
  }

  // calculate offsets
  int out_stride = 1;
  std::vector<int> out_strides;
  out_strides.push_back(1);
  for (int i = indice_unit_rank - 2; i >= 0; i--) {
    out_stride *= shape_data[i + 1];
    out_strides.push_back(out_stride);
  }

  param_->num_unit = 1;
  param_->num_unit *= update_shape.at(indices_shape.size() - C2NUM);
  for (int i = indices_shape.size() - 3; i >= 0; i--) {
    param_->num_unit *= update_shape.at(i);
  }

  int *indices_ptr = reinterpret_cast<int *>(indices->data());
  CHECK_NULL_RETURN(indices_ptr);
  output_unit_offsets_.clear();
  for (int i = 0; i < param_->num_unit; i++) {
    int tmp_stride = 0;
    for (int j = 0; j < indice_unit_rank; j++) {
      tmp_stride += indices_ptr[i * indice_unit_rank + j] * out_strides.at(j) * param_->unit_size;
    }
    output_unit_offsets_.push_back(tmp_stride);
  }
  return RET_OK;
}

int ScatterNDCPUKernel::ScatterND(int task_id) {
  void *update_data = in_tensors_[kScatterUpdateIndex]->data();
  auto output_tensor = out_tensors_[kOutputIndex];
  void *output_data = output_tensor->data();
  CHECK_NULL_RETURN(update_data);
  CHECK_NULL_RETURN(output_data);
  param_->data_type_len = output_tensor->data_type() == kNumberTypeFloat16 ? FP16_DATA_TYPE_LEN : sizeof(float);
  return DoScatterND(output_data, update_data, output_unit_offsets_.data(), param_, task_id);
}

int ScatterNDRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = static_cast<ScatterNDCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  return kernel->ScatterND(task_id);
}

int ScatterNDCPUKernel::Run() {
  auto ret = ParallelLaunch(ms_context_, ScatterNDRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterNDRun failed, ret: " << ret;
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ScatterNd, LiteKernelCreator<ScatterNDCPUKernel>)
#ifdef ENABLE_FP16
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_ScatterNd, LiteKernelCreator<ScatterNDCPUKernel>)
#endif
}  // namespace mindspore::kernel
