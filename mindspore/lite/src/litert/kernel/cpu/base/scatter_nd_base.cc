/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/base/scatter_nd_base.h"
#include <cstring>
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ScatterNd;

namespace mindspore::kernel {
namespace {
constexpr int kScatterIndicesIndex = 0;
constexpr int kScatterUpdateIndex = 1;
constexpr int kScatterShapeIndex = 2;
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
  auto indices_shape = indices->shape();
  auto update_shape = update->shape();
  auto shape_data = reinterpret_cast<int *>(shape->data());
  CHECK_NULL_RETURN(shape_data);
  int indices_rank = static_cast<int>(indices->shape().size());
  int update_rank = static_cast<int>(update->shape().size());
  auto shape_rank = shape->ElementsNum();
  int indice_unit_rank = indices->shape().back();

  // check indices shape
  MS_CHECK_TRUE_MSG(indices_rank >= DIMENSION_2D, RET_ERROR, "The rank of indices must be greater equal than 2.");
  MS_CHECK_TRUE_MSG(indice_unit_rank <= shape_rank, RET_ERROR,
                    "The value of indices' last dimension must be less equal than the input rank.");
  MS_CHECK_TRUE_MSG(update_rank == indices_rank - 1 + shape_rank - indice_unit_rank, RET_ERROR,
                    "The rank of update is illegal.");
  // check consistency of the shape indices and shape
  for (int i = 0; i < update_rank; i++) {
    if (i < indices_rank - 1) {
      MS_CHECK_TRUE_MSG(update_shape[i] == indices_shape[i], RET_ERROR, "the shape of update tensor is illegal.");
    }
    if (i >= indice_unit_rank) {
      MS_CHECK_TRUE_MSG(update_shape[i] == shape_data[i], RET_ERROR, "the shape of update tensor is illegal.");
    }
  }

  // calculate unit_size
  param_->unit_size = 1;
  for (int i = indices_shape.size() - 1; i < update_rank; i++) {
    param_->unit_size *= update_shape.at(i);
  }

  // calculate offsets
  int out_stride = 1;
  out_strides_.push_back(1);
  for (int i = indice_unit_rank - C2NUM; i >= 0; i--) {
    out_stride *= shape_data[i + 1];
    out_strides_.push_back(out_stride);
  }

  param_->num_unit = 1;
  param_->num_unit *= update_shape.at(indices_shape.size() - C2NUM);
  for (int i = indices_shape.size() - C3NUM; i >= 0; i--) {
    param_->num_unit *= update_shape.at(i);
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
  return ScatterNDUpdate(output_data, update_data, output_unit_offsets_.data(), param_, task_id);
}

int ScatterNDRun(void *cdata, int task_id, float, float) {
  auto kernel = static_cast<ScatterNDCPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  return kernel->ScatterND(task_id);
}

int ScatterNDCPUKernel::Run() {
  auto indices = in_tensors_[kScatterIndicesIndex];
  auto indice_unit_rank = indices->shape().back();
  int *indices_ptr = reinterpret_cast<int *>(indices->data());
  output_unit_offsets_.clear();
  for (int i = 0; i < param_->num_unit; i++) {
    int tmp_stride = 0;
    for (int j = 0; j < indice_unit_rank; j++) {
      tmp_stride += indices_ptr[i * indice_unit_rank + j] * out_strides_.at(j) * param_->unit_size;
    }
    output_unit_offsets_.push_back(tmp_stride);
  }
  auto ret = ParallelLaunch(ms_context_, ScatterNDRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScatterNDRun failed, ret: " << ret;
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_ScatterNd, LiteKernelCreator<ScatterNDCPUKernel>)
}  // namespace mindspore::kernel
