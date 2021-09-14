/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/base/group_convolution_base.h"
#include "src/runtime/infer_manager.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int GroupConvolutionBaseCPUKernel::Init() {
  for (int i = 0; i < group_num_; ++i) {
    auto sub_conv = group_convs_.at(i);
    if (sub_conv == nullptr) {
      MS_LOG(ERROR) << "sub con " << i << " is null.";
      return RET_ERROR;
    }
    auto ret = group_convs_.at(i)->Init();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Sub kernel init failed.";
      return ret;
    }
  }
  // if infer shape is done, resize func will be invoked in sub kernels
  return RET_OK;
}

int GroupConvolutionBaseCPUKernel::ReSize() {
  for (int i = 0; i < group_num_; ++i) {
    auto ret = group_convs_.at(i)->ReSize();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Sub kernel resize failed.";
      return RET_ERROR;
    }
  }
  if (group_num_ == 0) {
    return RET_ERROR;
  }
  conv_param_->input_channel_ /= group_num_;
  conv_param_->output_channel_ /= group_num_;
  return RET_OK;
}

void GroupConvolutionBaseCPUKernel::FreeSubKernel() {
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
  group_convs_.clear();
  if (group_conv_creator_ != nullptr) {
    delete group_conv_creator_;
    group_conv_creator_ = nullptr;
  }
}

int GroupConvolutionBaseCPUKernel::PreProcess() {
  if (!InferShapeDone()) {
    auto ret = lite::KernelInferShape(in_tensors_, out_tensors_, op_parameter_);
    if (ret != 0) {
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
      auto sub_kernel_out_tensors = group_convs_.at(i)->out_tensors();
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

  auto outputs = this->out_tensors_;
  for (auto *output : outputs) {
    MS_ASSERT(output != nullptr);
    auto ret = output->MallocData();
    if (ret != RET_OK) {
      FreeSubKernel();
      MS_LOG(ERROR) << "group conv out tensor malloc data failed.";
      return ret;
    }
    output->ResetRefCount();
  }
  return RET_OK;
}

int GroupConvolutionBaseCPUKernel::Run() {
  ori_in_data_ = in_tensors_[0]->data();
  ori_out_data_ = out_tensors_[0]->data();
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
    ret = PostConcat(i);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Concat output failed.";
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
