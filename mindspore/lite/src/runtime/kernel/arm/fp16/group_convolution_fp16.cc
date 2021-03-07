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

#include "src/runtime/kernel/arm/fp16/group_convolution_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/infer_manager.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int GroupConvolutionFP16CPUKernel::Init() {
  for (int i = 0; i < group_num_; ++i) {
    auto ret = group_convs_.at(i)->Init();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Sub kernel init failed.";
      return ret;
    }
  }
  // if infer shape is done, resize func will be invoked in sub kernels
  return RET_OK;
}

int GroupConvolutionFP16CPUKernel::ReSize() {
  for (int i = 0; i < group_num_; ++i) {
    auto ret = group_convs_.at(i)->ReSize();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Sub kernel resize failed.";
      return RET_ERROR;
    }
  }
  conv_param_->input_channel_ /= group_num_;
  conv_param_->output_channel_ /= group_num_;
  return RET_OK;
}

void GroupConvolutionFP16CPUKernel::FreeSubKernel() {
  for (auto &sub_conv : group_convs_) {
    // free sub conv input tensors / output tensors manually
    auto sub_in_tensors = sub_conv->in_tensors();
    auto sub_in_tensor_num = sub_in_tensors.size();
    for (size_t i = 0; i < sub_in_tensor_num; ++i) {
      delete sub_in_tensors[i];
      sub_in_tensors[i] = nullptr;
    }
    auto sub_out_tensors = sub_conv->out_tensors();
    auto sub_out_tensor_num = sub_out_tensors.size();
    for (size_t i = 0; i < sub_out_tensor_num; ++i) {
      delete sub_out_tensors[i];
      sub_out_tensors[i] = nullptr;
    }
    delete sub_conv;
    sub_conv = nullptr;
  }
}

int GroupConvolutionFP16CPUKernel::PreProcess() {
  if (!InferShapeDone()) {
    op_parameter_->infer_flag_ = true;

    auto ret = lite::KernelInferShape(in_tensors_, &out_tensors_, op_parameter_);
    if (ret != 0) {
      op_parameter_->infer_flag_ = false;
      MS_LOG(ERROR) << "InferShape fail!";
      return ret;
    }

    // if infershape func is called in runtime stage, we should malloc memory and set shape info for outputs of sub
    // kernels here.
    std::vector<int> in_shape;
    std::vector<int> out_shape;
    for (int i = 0; i < group_num_; ++i) {
      // in
      auto in_tensor = in_tensors_.front();
      in_shape = {in_tensor->Batch(), in_tensor->Height(), in_tensor->Width(), conv_param_->input_channel_};
      auto sub_kernel_in_tensor = group_convs_.at(i)->in_tensors().front();
      sub_kernel_in_tensor->set_shape(in_shape);
      ret = sub_kernel_in_tensor->MallocData();
      if (ret != RET_OK) {
        FreeSubKernel();
        MS_LOG(ERROR) << "sub kernel in tensor malloc data failed.";
        return ret;
      }
      // out
      auto out_tensor = out_tensors_.front();
      out_shape = {out_tensor->Batch(), out_tensor->Height(), out_tensor->Width(), conv_param_->output_channel_};
      auto sub_kernel_out_tensors = group_convs_[i]->out_tensors();
      for (auto tensor : sub_kernel_out_tensors) {
        tensor->set_shape(out_shape);
        ret = tensor->MallocData();
        if (ret != RET_OK) {
          FreeSubKernel();
          MS_LOG(ERROR) << "sub kernel out tensor malloc data failed.";
          return ret;
        }
      }
    }
    ret = ReSize();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ReSize fail!ret: " << ret;
      return ret;
    }
  }

  auto outputs = this->out_tensors();
  for (auto *output : outputs) {
    MS_ASSERT(output != nullptr);
    auto ret = output->MallocData();
    if (ret != RET_OK) {
      FreeSubKernel();
      MS_LOG(ERROR) << "fp16 group conv out tensor malloc data failed.";
      return ret;
    }
  }
  return RET_OK;
}

int GroupConvolutionFP16CPUKernel::SeparateInput(int group_id) {
  // input may either be float32 or float16
  auto in_tensor = in_tensors_.front();
  int in_plane = in_tensor->Height() * in_tensor->Width() * in_tensor->Batch();
  int sub_in_channel = conv_param_->input_channel_;
  int ori_in_channel = sub_in_channel * group_num_;
  auto sub_in_data = group_convs_.at(group_id)->in_tensors().front()->data_c();
  auto in_data_type = in_tensors_.front()->data_type();
  auto sub_in_data_type = group_convs_.at(group_id)->in_tensors().front()->data_type();
  if (in_data_type != sub_in_data_type) {
    MS_LOG(ERROR) << "data type of sub conv kernel input should be the same as origin input's.";
    return RET_ERROR;
  }
  if (!(in_data_type == kNumberTypeFloat32 || in_data_type == kNumberTypeFloat16)) {
    MS_LOG(ERROR) << "Invalid data type.";
    return RET_ERROR;
  }
  if (in_tensors_.front()->data_type() == kNumberTypeFloat16) {
    float16_t *src_ptr = reinterpret_cast<float16_t *>(ori_in_data_) + group_id * sub_in_channel;
    float16_t *dst_ptr = reinterpret_cast<float16_t *>(sub_in_data);
    MS_ASSERT(src_ptr);
    MS_ASSERT(dst_ptr);
    for (int i = 0; i < in_plane; ++i) {
      memcpy(dst_ptr, src_ptr, sub_in_channel * sizeof(float16_t));
      src_ptr += ori_in_channel;
      dst_ptr += sub_in_channel;
    }
  } else {
    float *src_ptr = reinterpret_cast<float *>(ori_in_data_) + group_id * sub_in_channel;
    float *dst_ptr = reinterpret_cast<float *>(sub_in_data);
    MS_ASSERT(src_ptr);
    MS_ASSERT(dst_ptr);
    for (int i = 0; i < in_plane; ++i) {
      memcpy(dst_ptr, src_ptr, sub_in_channel * sizeof(float));
      src_ptr += ori_in_channel;
      dst_ptr += sub_in_channel;
    }
  }
  return RET_OK;
}

void GroupConvolutionFP16CPUKernel::PostConcat(int group_id) {
  // output is must float16 data type
  auto out_tensor = out_tensors_.front();
  int out_plane = out_tensor->Height() * out_tensor->Width() * out_tensor->Batch();
  int sub_out_channel = conv_param_->output_channel_;
  int ori_out_channel = sub_out_channel * group_num_;
  auto sub_out_data = reinterpret_cast<float16_t *>(group_convs_.at(group_id)->out_tensors().front()->data_c());
  MS_ASSERT(sub_out_data);
  float16_t *src_ptr = sub_out_data;
  float16_t *dst_ptr = ori_out_data_ + group_id * sub_out_channel;
  for (int i = 0; i < out_plane; ++i) {
    memcpy(dst_ptr, src_ptr, sub_out_channel * sizeof(float16_t));
    src_ptr += sub_out_channel;
    dst_ptr += ori_out_channel;
  }
}

int GroupConvolutionFP16CPUKernel::Run() {
  ori_in_data_ = in_tensors().front()->data_c();
  ori_out_data_ = reinterpret_cast<float16_t *>(out_tensors().front()->data_c());
  MS_ASSERT(ori_out_data_);
  for (int i = 0; i < group_num_; ++i) {
    // first, separate group conv input into several parts. This step must be in runtime stage.
    auto ret = SeparateInput(i);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Separate input failed.";
      return ret;
    }
    // sun kernels run
    ret = group_convs_.at(i)->Run();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "sub kernel " << i << " execute failed.";
      return ret;
    }
    // post process, concat all outputs of sub-kernels into one output
    PostConcat(i);
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
