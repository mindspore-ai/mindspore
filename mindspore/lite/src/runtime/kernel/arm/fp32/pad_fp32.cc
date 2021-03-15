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

#include "src/runtime/kernel/arm/fp32/pad_fp32.h"
#include "src/kernel_registry.h"
#include "schema/model_generated.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PadFusion;

namespace mindspore::kernel {
namespace {
constexpr size_t kMirrorPadInputSize = 2;
constexpr size_t kPadMaxInputSize = 2;
}  // namespace
int PadCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PadCPUKernel::ReSize() {
  auto input = in_tensors_.at(0);
  auto rank = input->shape().size();
  if (rank > COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Pad input rank should <= " << COMM_SHAPE_SIZE << ", got " << rank;
    return RET_ERROR;
  }
  auto output = out_tensors_.at(0);
  if (pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_CONSTANT)) {
    auto ret = ExtendShape(in_, COMM_SHAPE_SIZE, input->shape().data(), rank);
    if (ret != RET_OK) {
      return ret;
    }
    ret = ExtendShape(out_, COMM_SHAPE_SIZE, output->shape().data(), rank);
    if (ret != RET_OK) {
      return ret;
    }
    if (pad_param_->padding_length < MAX_SHAPE_SIZE) {
      int ori_paddings[MAX_SHAPE_SIZE];
      for (auto i = 0; i < pad_param_->padding_length; ++i) {
        ori_paddings[i] = pad_param_->paddings_[i];
      }
      ret = ExtendPaddings(pad_param_->paddings_, MAX_SHAPE_SIZE, ori_paddings, pad_param_->padding_length);
      if (ret != RET_OK) {
        return ret;
      }
      pad_param_->padding_length = MAX_SHAPE_SIZE;
    }
  }
  return RET_OK;
}

void PadCPUKernel::InitMirrorPadBlock() {
  mirror_pad_block_.clear();
  std::vector<int> left_pads(COMM_SHAPE_SIZE);
  for (size_t i = 0; i < COMM_SHAPE_SIZE; ++i) {
    left_pads[i] = pad_param_->paddings_[2 * i];
  }

  std::vector<int> input_separate_dims;
  std::vector<int> output_separate_dims;
  std::vector<int> separate_offset;

  /* init separate dims */
  int cur_input = 1;
  int cur_output = 1;
  for (size_t i = 0; i < COMM_SHAPE_SIZE; ++i) {
    if (1 < cur_input) {
      input_separate_dims.emplace_back(cur_input);
      output_separate_dims.emplace_back(cur_output);
      separate_offset.emplace_back(0);
    }
    input_separate_dims.emplace_back(in_[i]);
    output_separate_dims.emplace_back(out_[i]);
    separate_offset.emplace_back(left_pads[i]);
    cur_input = 1;
    cur_output = 1;
  }
  if (cur_input != 1 || cur_output != 1) {
    input_separate_dims.emplace_back(cur_input);
    output_separate_dims.emplace_back(cur_output);
    separate_offset.emplace_back(0);
  }

  /* init separate stride */
  std::vector<int> output_separate_stride;
  output_separate_stride.resize(output_separate_dims.size());
  GetStride(output_separate_stride.data(), output_separate_dims.data(), output_separate_dims.size());

  /* init separate stride */
  std::vector<int> remain_stride;
  remain_stride.resize(0);
  int remain_size = GetStride(remain_stride.data(), output_separate_dims.data(), remain_stride.size());

  std::vector<int> right_pads(separate_offset.size());
  for (size_t i = 0; i < right_pads.size(); ++i) {
    right_pads[i] = output_separate_dims[i] - input_separate_dims[i] - separate_offset[i];
  }

  /* init pad region */
  std::vector<int> pad_region;
  for (size_t i = remain_stride.size(); i < output_separate_stride.size(); ++i) {
    // 0: center, 1: left, 2: right
    int r = 1;
    if (separate_offset[i] > 0) {
      r++;
    }
    if (right_pads[i] > 0) {
      r++;
    }
    pad_region.emplace_back(r);
  }

  std::vector<int> pad_region_stride(pad_region.size());
  int region_size = GetStride(pad_region_stride.data(), pad_region.data(), pad_region.size());
  int remain_dim_offset = remain_stride.size();

  std::vector<int> pad_cord(pad_region.size());

  for (int pos = 0; pos < remain_size; ++pos) {
    const int dst_basic_offset = 0;

    for (int index = 1; index < region_size; ++index) {
      int dst_offset = dst_basic_offset;

      int value = index;
      for (size_t i = 0; i < pad_region.size(); ++i) {
        pad_cord[i] = value / pad_region_stride[i];
        value = value % pad_region_stride[i];
      }

      MirrorPadBlock block;
      const int size_offset = COMM_SHAPE_SIZE - static_cast<int>(pad_region.size());
      for (size_t i = 0; i < pad_region.size(); ++i) {
        int di = size_offset + i;
        int si = remain_dim_offset + i;
        switch (pad_cord[i]) {
          case 0:
            dst_offset += separate_offset[si] * output_separate_stride[si];
            block.size_[di] = input_separate_dims[si];
            block.out_stride_[di] = output_separate_stride[si];
            break;
          case 2:
            dst_offset += (separate_offset[si] + input_separate_dims[si]) * output_separate_stride[si];
            block.size_[di] = right_pads[si];
            block.out_stride_[di] = output_separate_stride[si];
            break;
          case 1:
            if (separate_offset[si] > 0) {
              block.size_[di] = separate_offset[si];
              block.out_stride_[di] = output_separate_stride[si];
            } else {
              dst_offset += (separate_offset[si] + input_separate_dims[si]) * output_separate_stride[si];
              block.size_[di] = right_pads[si];
              block.out_stride_[di] = output_separate_stride[si];
            }
            break;
          default:
            break;
        }
      }
      block.out_offset_ = dst_offset;
      mirror_pad_block_.push_back(std::move(block));
    }
  }
  return;
}

int PadCPUKernel::ExtendShape(int *shape, int length, const int *ori_shape, int rank) {
  if (shape == nullptr || ori_shape == nullptr) {
    return RET_NULL_PTR;
  }
  for (auto i = 0; i < length - rank; ++i) {
    shape[i] = 1;
  }
  for (auto i = length - rank; i < length; ++i) {
    shape[i] = ori_shape[i - (length - rank)];
  }
  return RET_OK;
}

int PadCPUKernel::ExtendPaddings(int *paddings, int length, const int *ori_paddings, int ori_length) {
  if (paddings == nullptr || ori_paddings == nullptr) {
    return RET_NULL_PTR;
  }
  for (auto i = 0; i < length - ori_length; ++i) {
    paddings[i] = 0;
  }
  for (auto i = length - ori_length; i < length; ++i) {
    paddings[i] = ori_paddings[i - (length - ori_length)];
  }
  return RET_OK;
}

int PadImpl(void *cdata, int task_id) {
  auto padKernel = reinterpret_cast<PadCPUKernel *>(cdata);
  int error_code = padKernel->RunImpl(task_id);
  if (error_code != NNACL_OK) {
    MS_LOG(ERROR) << "Pad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadCPUKernel::RunImpl(int task_id) {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);
  auto input_data = reinterpret_cast<float *>(input->data_c());
  auto output_data = reinterpret_cast<float *>(output->data_c());
  MS_ASSERT(input_data);
  MS_ASSERT(output_data);
  Pad(input_data, output_data, in_, out_, pad_param_->paddings_, task_id, context_->thread_num_);

  return RET_OK;
}

int MirrorPadImpl(void *cdata, int task_id) {
  auto padKernel = reinterpret_cast<PadCPUKernel *>(cdata);
  int error_code = padKernel->RunMirrorPadImpl(task_id);
  if (error_code != NNACL_OK) {
    MS_LOG(ERROR) << "Pad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadCPUKernel::RunMirrorPadImpl(int task_id) {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  auto input_data = reinterpret_cast<float *>(input->data_c());
  auto output_data = reinterpret_cast<float *>(output->data_c());

  /* Fast Mirror pad */
  if (mirror_pad_block_.size() != 0) {
    /* copy center part */
    Pad(input_data, output_data, in_, out_, pad_param_->paddings_, task_id, context_->thread_num_);

    /* calculate region part */
    for (size_t i = task_id; i < mirror_pad_block_.size(); i += context_->thread_num_) {
      auto block = mirror_pad_block_[i];

      for (int a = 0; a < block.size_[0]; a++) {
        int out_a_index = block.out_offset_ + a * block.out_stride_[0];
        for (int b = 0; b < block.size_[1]; b++) {
          int out_b_index = out_a_index + b * block.out_stride_[1];
          for (int c = 0; c < block.size_[2]; ++c) {
            int output_index = out_b_index + c * block.out_stride_[2];
            MirrorPad(input_data, output_data, in_, pad_param_, output_index, output_index + block.size_[3]);
          }
        }
      }
    }
    return RET_OK;
  }

  /* Common Mirror pad */
  int unit = UP_DIV(output->ElementsNum(), context_->thread_num_);
  int begin = unit * task_id;
  int end = MSMIN(begin + unit, output->ElementsNum());
  MirrorPad(input_data, output_data, in_, pad_param_, begin, end);
  return RET_OK;
}

int PadCPUKernel::CheckPaddings(int *paddings, int length, int *input_shape, int mode) {
  if (paddings == nullptr || input_shape == nullptr) {
    return RET_NULL_PTR;
  }
  std::string prefix;
  int offset;
  if (mode == static_cast<int>(schema::PaddingMode_SYMMETRIC)) {
    prefix = "For Pad SYMMETRIC ";
    offset = 0;
  } else {
    prefix = "For Pad REFLECT ";
    offset = 1;
  }
  for (auto i = 0; i < length; ++i) {
    int max_valid = input_shape[i] - offset;
    if (paddings[i * 2] > max_valid) {
      MS_LOG(WARNING) << prefix << "paddings " << paddings[i * 2] << " should be less than " << max_valid + 1;
      MS_LOG(WARNING) << "Running mirror pad with padding bigger than shape.";
    }
    if (paddings[i * 2 + 1] > max_valid) {
      MS_LOG(WARNING) << prefix << "paddings " << paddings[i * 2 + 1] << " should be less than " << max_valid + 1;
      MS_LOG(WARNING) << "Running mirror pad with padding bigger than shape.";
    }
  }
  return RET_OK;
}

int PadCPUKernel::CopyPaddingFromInput() {
  if (in_tensors_.size() != kMirrorPadInputSize) {
    MS_LOG(ERROR) << "Pad Reflect or Symmetric mode need 2 inputs, got " << in_tensors_.size();
    return RET_ERROR;
  }
  auto padding_tensor = in_tensors_.at(1);
  auto paddings = reinterpret_cast<int *>(padding_tensor->data_c());
  if (paddings == nullptr) {
    MS_LOG(ERROR) << "Pad second input data nullptr";
    return RET_ERROR;
  }
  auto input_shape = in_tensors_.at(0)->shape();
  int rank = static_cast<int>(input_shape.size());
  if (padding_tensor->ElementsNum() != rank * 2) {
    MS_LOG(ERROR) << "Pad second input elements num" << padding_tensor->ElementsNum() << ", should be " << rank * 2;
    return RET_ERROR;
  }

  auto ret = ExtendShape(in_, COMM_SHAPE_SIZE, input_shape.data(), rank);
  if (ret != RET_OK) {
    return ret;
  }
  ret = ExtendPaddings(pad_param_->paddings_, MAX_SHAPE_SIZE, paddings, padding_tensor->ElementsNum());
  if (ret != RET_OK) {
    return ret;
  }
  pad_param_->padding_length = MAX_SHAPE_SIZE;
  return RET_OK;
}

void PadCPUKernel::CalculateStrides() {
  pad_param_->in_strides[COMM_SHAPE_SIZE - 1] = 1;
  for (auto i = COMM_SHAPE_SIZE - 2; i >= 0; --i) {
    pad_param_->in_strides[i] = in_[i + 1] * pad_param_->in_strides[i + 1];
  }
  for (auto i = 0; i < COMM_SHAPE_SIZE; ++i) {
    out_[i] = in_[i] + pad_param_->paddings_[i * 2] + pad_param_->paddings_[i * 2 + 1];
  }
  pad_param_->out_strides[COMM_SHAPE_SIZE - 1] = 1;
  for (auto i = COMM_SHAPE_SIZE - 2; i >= 0; --i) {
    pad_param_->out_strides[i] = out_[i + 1] * pad_param_->out_strides[i + 1];
  }
}

int PadCPUKernel::HandleMirrorPad() {
  if (in_tensors_.size() == 1) {
    auto input_shape = in_tensors_.at(0)->shape();
    int rank = static_cast<int>(input_shape.size());
    auto ret = ExtendShape(in_, COMM_SHAPE_SIZE, input_shape.data(), rank);
    if (ret != RET_OK) {
      return ret;
    }
  } else {
    auto ret = CopyPaddingFromInput();
    if (ret != RET_OK) {
      return ret;
    }
  }
  auto ret = CheckPaddings(pad_param_->paddings_, COMM_SHAPE_SIZE, in_, pad_param_->pad_mode_);
  if (ret != RET_OK) {
    return ret;
  }
  CalculateStrides();
  pad_param_->mirror_offset_ = pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_REFLECT) ? 1 : 0;

  InitMirrorPadBlock();
  return RET_OK;
}

int PadCPUKernel::Run() {
  int error_code;
  if (pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_CONSTANT)) {
    if (in_tensors_.size() == kPadMaxInputSize) {
      CopyPaddingFromInput();
    }
    auto output = out_tensors_.at(0);
    int output_size = output->ElementsNum();
    auto output_data = reinterpret_cast<float *>(output->data_c());
    if (abs(pad_param_->constant_value_ - 0.0f) < 1e-5) {
      memset(output_data, 0, output_size * sizeof(float));
    } else {
      for (auto i = 0; i < output_size; ++i) {
        output_data[i] = pad_param_->constant_value_;
      }
    }
    error_code = ParallelLaunch(this->context_->thread_pool_, PadImpl, this, context_->thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Pad run error, error_code[" << error_code << "]";
      return RET_ERROR;
    }
  } else {
    // mirror pad case
    error_code = HandleMirrorPad();
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Handle mirror pad failed, error_code[" << error_code << "]";
      return error_code;
    }

    error_code = ParallelLaunch(this->context_->thread_pool_, MirrorPadImpl, this, context_->thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "Pad Reflect or Symmetric mode run error, error_code[" << error_code << "]";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PadFusion, LiteKernelCreator<PadCPUKernel>)
}  // namespace mindspore::kernel
