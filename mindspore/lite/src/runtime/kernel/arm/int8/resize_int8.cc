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
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/int8/resize.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/int8/resize_int8.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INVALID_OP_ATTR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
namespace {
constexpr int kInputNum = 1;
constexpr int kOutputNum = 1;
constexpr size_t kRank = 4;
}  // namespace

int ResizeInt8CPUKernel::Init() {
  auto ret = ResizeBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  quant_in_ = new (std::nothrow) QuantArg;
  MS_ASSERT(quant_in_);
  quant_out_ = new (std::nothrow) QuantArg;
  MS_ASSERT(quant_out_);
  auto input = in_tensors_.at(0);
  quant_in_->zp_ = input->GetQuantParams().front().zeroPoint;
  quant_in_->scale_ = input->GetQuantParams().front().scale;
  auto output = out_tensors_.at(0);
  quant_out_->zp_ = output->GetQuantParams().front().zeroPoint;
  quant_out_->scale_ = output->GetQuantParams().front().scale;

  multiplier_ = new (std::nothrow) QuantMulArg;
  MS_ASSERT(multiplier_);
  QuantizeRoundParameter(quant_in_->scale_ / quant_out_->scale_, &multiplier_->multiplier_, &multiplier_->left_shift_,
                         &multiplier_->right_shift_);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ResizeInt8Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto resize = reinterpret_cast<ResizeInt8CPUKernel *>(cdata);
  auto error_code = resize->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ResizeInt8CPUKernel::RunImpl(int task_id) {
  auto input = in_tensors_.at(0);
  auto input_data = reinterpret_cast<const int8_t *>(input->Data());
  if (input_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->Data());
  if (output_data == nullptr) {
    return RET_NULL_PTR;
  }
  auto input_shape = input->shape();

  if (context_ == nullptr) {
    return RET_NULL_PTR;
  }

  int ret = 0;
  switch (method_) {
    case static_cast<int>(schema::ResizeMethod_BILINEAR): {
      ret = ResizeBilinearInt8(input_data, output_data, input_shape.data(), out_tensors_[0]->shape().data(),
                               align_corners_, quant_in_, quant_out_, multiplier_, task_id, context_->thread_num_);

      break;
    }
    case static_cast<int>(schema::ResizeMethod_NEAREST_NEIGHBOR): {
      bool same_zp = quant_in_->zp_ == quant_out_->zp_;
      bool same_scale = abs(quant_out_->scale_ - quant_in_->scale_) < 1e-6;
      if (same_zp && same_scale) {
        ret =
          ResizeNearestNeighborInt8Simple(input_data, output_data, input_shape.data(), out_tensors_[0]->shape().data(),
                                          align_corners_, task_id, context_->thread_num_);
      } else {
        ret =
          ResizeNearestNeighborInt8(input_data, output_data, input_shape.data(), out_tensors_[0]->shape().data(),
                                    align_corners_, multiplier_, quant_in_, quant_out_, task_id, context_->thread_num_);
      }
      break;
    }
    case schema::ResizeMethod_UNKNOW:
    default: {
      MS_LOG(ERROR) << "Resize unknown method " << method_;
      ret = RET_ERROR;
    }
  }
  return ret;
}

int ResizeInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  int error_code = LiteBackendParallelLaunch(ResizeInt8Impl, this, context_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Resize run error, error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
