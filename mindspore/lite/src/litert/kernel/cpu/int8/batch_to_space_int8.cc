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
#include "src/litert/kernel/cpu/int8/batch_to_space_int8.h"
#include <cfloat>
#include <cmath>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchToSpace;
using mindspore::schema::PrimitiveType_BatchToSpaceND;

namespace mindspore::kernel {
BatchToSpaceInt8CPUKernel::~BatchToSpaceInt8CPUKernel() {
  if (in_quant_arg_ != nullptr) {
    free(in_quant_arg_);
    in_quant_arg_ = nullptr;
  }
  if (out_quant_arg_ != nullptr) {
    free(out_quant_arg_);
    out_quant_arg_ = nullptr;
  }
}

int BatchToSpaceInt8CPUKernel::Processinput() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_NULL_RETURN(in_tensors_[DIMENSION_1D]);
  CHECK_NULL_RETURN(in_tensors_[DIMENSION_2D]);
  auto block_shape_data = in_tensors_[DIMENSION_1D]->data();
  auto crops_data = in_tensors_[DIMENSION_2D]->data();
  CHECK_NULL_RETURN(block_shape_data);
  CHECK_NULL_RETURN(crops_data);
  auto block_shape = static_cast<int *>(block_shape_data);
  auto crops = static_cast<int *>(crops_data);
  CHECK_LESS_RETURN(in_tensors_[DIMENSION_1D]->ElementsNum(), BATCH_TO_SPACE_BLOCK_SHAPE_SIZE);
  CHECK_LESS_RETURN(in_tensors_[DIMENSION_2D]->ElementsNum(), COMM_SHAPE_SIZE);
  for (int i = 0; i < BATCH_TO_SPACE_BLOCK_SHAPE_SIZE; ++i) {
    block_shape_[i] = block_shape[i];
  }
  no_crop_ = true;
  for (int i = 0; i < COMM_SHAPE_SIZE; ++i) {
    crops_[i] = crops[i];
    if (crops_[i] != 0) {
      no_crop_ = false;
    }
  }
  return RET_OK;
}

int BatchToSpaceInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_1D);
  CHECK_LESS_RETURN(out_tensors_.size(), DIMENSION_1D);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  MS_ASSERT(in_tensors_.front()->format() == mindspore::NHWC);
  in_quant_arg_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (in_quant_arg_ == nullptr) {
    MS_LOG(ERROR) << "Malloc QuantArg for BatchToSpace int8 op failed!";
    return RET_ERROR;
  }
  auto *input_tensor = in_tensors_[kInputIndex];
  CHECK_NULL_RETURN(input_tensor);
  auto in_quant_args = input_tensor->quant_params();
  in_quant_arg_->scale_ = in_quant_args.front().scale;
  in_quant_arg_->zp_ = in_quant_args.front().zeroPoint;

  out_quant_arg_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (out_quant_arg_ == nullptr) {
    MS_LOG(ERROR) << "Malloc QuantArg for BatchToSpace int8 op failed!";
    return RET_ERROR;
  }
  auto *out_tensor = out_tensors_[kOutputIndex];
  CHECK_NULL_RETURN(out_tensor);
  auto out_quant_args = out_tensor->quant_params();
  out_quant_arg_->scale_ = out_quant_args.front().scale;
  out_quant_arg_->zp_ = out_quant_args.front().zeroPoint;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int BatchToSpaceInt8CPUKernel::ReSize() {
  MS_ASSERT(in_tensors_.front()->shape().size() == DIMENSION_4D);
  return RET_OK;
}

int BatchToSpaceInt8CPUKernel::Run() {
  auto input = in_tensors_[kInputIndex];
  auto output = out_tensors_[kOutputIndex];
  CHECK_NULL_RETURN(input);
  CHECK_NULL_RETURN(output);
  const int8_t *input_data = reinterpret_cast<const int8_t *>(input->data());
  int8_t *output_data = reinterpret_cast<int8_t *>(output->data());
  auto in_shape = input->shape();
  auto out_shape = output->shape();

  if (in_tensors_.size() == 1) {
    BatchToSpaceParameter *param = reinterpret_cast<BatchToSpaceParameter *>(this->op_parameter_);
    CHECK_NULL_RETURN(param);
    block_shape_[DIMENSION_0D] = param->block_shape_[DIMENSION_0D];
    block_shape_[DIMENSION_1D] = param->block_shape_[DIMENSION_1D];
    for (int i = 0; i < COMM_SHAPE_SIZE; ++i) {
      crops_[i] = param->crops_[i];
      if (crops_[i] != 0) {
        no_crop_ = false;
      }
    }
    no_crop_ = param->no_crop_;
  } else if (in_tensors_.size() == DIMENSION_3D) {
    auto ret = Processinput();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Processinput failed in BatchToSpace.";
      return ret;
    }
  }

  if (in_tensors_.size() == DIMENSION_1D || in_tensors_.size() == DIMENSION_3D) {
    if (std::abs(in_quant_arg_->scale_ - out_quant_arg_->scale_) < FLT_EPSILON &&
        in_quant_arg_->zp_ == out_quant_arg_->zp_) {
      if (no_crop_) {
        BatchToSpaceNoCropForNHWC(input_data, output_data, in_shape.data(), out_shape[0], block_shape_, sizeof(int8_t));
      } else {
        BatchToSpaceForNHWC(input_data, output_data, in_shape.data(), out_shape[0], block_shape_, crops_,
                            sizeof(int8_t));
      }
    } else {
      if (no_crop_) {
        BatchToSpaceNoCropForNHWCInt8(input_data, output_data, in_shape.data(), out_shape[0], block_shape_,
                                      in_quant_arg_, out_quant_arg_);
      } else {
        BatchToSpaceForNHWCInt8(input_data, output_data, in_shape.data(), out_shape[0], block_shape_, crops_,
                                in_quant_arg_, out_quant_arg_);
      }
    }
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_BatchToSpace, LiteKernelCreator<BatchToSpaceInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_BatchToSpaceND, LiteKernelCreator<BatchToSpaceInt8CPUKernel>)
}  // namespace mindspore::kernel
