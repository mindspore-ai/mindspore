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

#include "src/runtime/kernel/arm/fp16/pad_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PadFusion;

namespace mindspore::kernel {
namespace {
constexpr size_t kPadCommonInputSize = 2;
}  // namespace
int PadFp16CPUKernel::RunImpl(int task_id) {
  PadFp16(input_, output_, in_, out_, pad_param_->paddings_, task_id, op_parameter_->thread_num_);
  return RET_OK;
}

int PadFp16CPUKernel::RunMirrorPadImpl(int task_id) {
  auto input = in_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  auto output = out_tensors_.at(0);
  CHECK_NULL_RETURN(output);
  auto input_data = reinterpret_cast<float16_t *>(input->data());
  CHECK_NULL_RETURN(input_data);
  auto output_data = reinterpret_cast<float16_t *>(output->data());
  CHECK_NULL_RETURN(output_data);

  /* Fast Mirror pad */
  if (mirror_pad_block_.size() != 0) {
    /* copy center part */
    PadFp16(input_data, output_data, in_, out_, pad_param_->paddings_, task_id, op_parameter_->thread_num_);

    /* calculate region part */
    for (size_t i = task_id; i < mirror_pad_block_.size(); i += op_parameter_->thread_num_) {
      auto block = mirror_pad_block_[i];

      for (int a = 0; a < block.size_[0]; a++) {
        int out_a_index = block.out_offset_ + a * block.out_stride_[0];
        for (int b = 0; b < block.size_[1]; b++) {
          int out_b_index = out_a_index + b * block.out_stride_[1];
          for (int c = 0; c < block.size_[2]; ++c) {
            int out_c_index = out_b_index + c * block.out_stride_[2];
            for (int d = 0; d < block.size_[3]; ++d) {
              int out_d_index = out_c_index + d * block.out_stride_[3];
              for (int e = 0; e < block.size_[4]; ++e) {
                int output_index = out_d_index + e * block.out_stride_[4];
                MirrorPadFp16(input_data, output_data, in_, pad_param_, output_index, output_index + block.size_[5]);
              }
            }
          }
        }
      }
    }
    return RET_OK;
  }
  MS_CHECK_FALSE(op_parameter_->thread_num_ == 0, RET_ERROR);
  int unit = UP_DIV(out_tensors_.at(0)->ElementsNum(), op_parameter_->thread_num_);
  int begin = unit * task_id;
  int end = MSMIN(begin + unit, out_tensors_.at(0)->ElementsNum());
  MirrorPadFp16(input_, output_, in_, pad_param_, begin, end);
  return RET_OK;
}

int PadFp16CPUKernel::Run() {
  if (in_tensors_.size() == kInputSize2) {
    auto pad_value = in_tensors_.at(2);
    auto value_num = pad_value->ElementsNum();
    if (value_num != 1) {
      MS_LOG(ERROR) << "The number of padding value should be only one, but got " << value_num;
      return RET_ERROR;
    }
    CHECK_NULL_RETURN(pad_value->data());
    pad_param_->constant_value_ = *(reinterpret_cast<float16_t *>(pad_value->data()));
  }

  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);
  input_ = reinterpret_cast<float16_t *>(input_tensor->data());
  output_ = reinterpret_cast<float16_t *>(output_tensor->data());
  CHECK_NULL_RETURN(input_);
  CHECK_NULL_RETURN(output_);
  int ret = 0;
  if (pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_CONSTANT)) {
    if (in_tensors_.size() >= kPadCommonInputSize) {
      ret = CopyPaddingFromInput();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "PadFp16CPUKernel CopyPaddingFromInput failed";
        return RET_ERROR;
      }
    }
    if (pad_param_->constant_value_ - 0.0f < 1e-5) {
      memset(output_, 0, output_tensor->ElementsNum() * sizeof(float16_t));
    } else {
      for (int i = 0; i < output_tensor->ElementsNum(); ++i) {
        output_[i] = pad_param_->constant_value_;
      }
    }
    ret = ParallelLaunch(this->ms_context_, PadImpl, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
    }
  } else {
    // mirror pad case
    ret = HandleMirrorPad();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Handle mirror pad failed, error_code[" << ret << "]";
      return ret;
    }

    ret = ParallelLaunch(this->ms_context_, MirrorPadImpl, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Pad Reflect or Symmetric mode run error, error_code[" << ret << "]";
    }
  }

  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_PadFusion, LiteKernelCreator<PadFp16CPUKernel>)
}  // namespace mindspore::kernel
