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

#include "src/runtime/agent/npu/npu_executor.h"
#include "include/errorcode.h"
#include "src/runtime/agent/npu/npu_manager.h"
#include "nnacl/pack.h"
namespace mindspore::lite {
NPUExecutor::~NPUExecutor() {
  npu_input_tensors_.clear();
  npu_output_tensors_.clear();
}

int NPUExecutor::Prepare(const std::vector<kernel::LiteKernel *> &kernels) {
  this->client_ = mindspore::lite::NPUManager::GetInstance()->GetClient(model_name_);
  if (this->client_ == nullptr) {
    MS_LOG(ERROR) << "client is nullptr.";
    return RET_ERROR;
  }
  if (GetIOTensorVec() != RET_OK) {
    MS_LOG(ERROR) << "Load model failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

bool IsSameShapeTensor(Tensor *tensor, std::shared_ptr<hiai::AiTensor> npu_tensor) {
  return tensor->Batch() == npu_tensor->GetTensorDimension().GetNumber() &&
         tensor->Channel() == npu_tensor->GetTensorDimension().GetChannel() &&
         tensor->Height() == npu_tensor->GetTensorDimension().GetHeight() &&
         tensor->Width() == npu_tensor->GetTensorDimension().GetWidth();
}

int NPUExecutor::Run(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                     const std::vector<kernel::LiteKernel *> &out_kernels,
                     const std::vector<kernel::LiteKernel *> &kernels, Allocator *allocator,
                     const KernelCallBack &before, const KernelCallBack &after) {
  hiai::AiContext context;
  for (int i = 0; i < npu_input_tensors_.size(); ++i) {
    int index = 0;
    for (; index < in_tensors.size(); index++) {
      if (IsSameShapeTensor(in_tensors[index], npu_input_tensors_[i])) {
        void *data = in_tensors[index]->data_c();
        if (data == nullptr) {
          MS_LOG(ERROR) << model_name_ << " Inputs data is nullptr";
          return RET_ERROR;
        }

        memcpy(npu_input_tensors_[i]->GetBuffer(), data, in_tensors[index]->Size());
        in_tensors[index]->set_ref_count(in_tensors[index]->ref_count() - 1);
        if (in_tensors[index]->ref_count() <= 0) {
          auto ret = in_tensors[index]->FreeData();
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Free tensor data failed";
            return RET_ERROR;
          }
        }
        break;
      }
      if (index == in_tensors.size()) {
        MS_LOG(ERROR) << "Can't find corresponding ms lite tensor of " << i << " input tensor for npu executor "
                      << model_name_;
        return RET_ERROR;
      }
    }
  }
  context.AddPara("model_name", model_name_);
  if (this->client_ == nullptr) {
    MS_LOG(ERROR) << "NPU client is nullptr";
    return RET_ERROR;
  }
  int stamp;
  int ret = this->client_->Process(context, this->npu_input_tensors_, this->npu_output_tensors_, 1000, stamp);
  if (ret != hiai::AI_SUCCESS) {
    MS_LOG(ERROR) << "NPU Process failed. code is " << ret;
    return RET_ERROR;
  }

  // For the output kernel of the entire model, and the format is nchw, the output tensor needs to be nchw TO nhwc.
  std::vector<Tensor *> trans_tensors;
  for (auto kernel : out_kernels) {
    if (kernel->out_kernels().empty() && npu_trans_nodes.find(kernel->Type()) != npu_trans_nodes.end()) {
      for (int i = 0; i < kernel->out_tensors().size(); ++i) {
        trans_tensors.push_back(kernel->out_tensors()[i]);
      }
    }
  }
  for (int i = 0; i < npu_output_tensors_.size(); ++i) {
    void *data = out_tensors[i]->MutableData();
    if (data == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }

    if (std::find(trans_tensors.begin(), trans_tensors.end(), out_tensors[i]) != trans_tensors.end()) {
      // Change data&tensor shape nc->nh
      PackNCHWToNHWCFp32(npu_output_tensors_[i]->GetBuffer(), data,
                         npu_output_tensors_[i]->GetTensorDimension().GetNumber(),
                         npu_output_tensors_[i]->GetTensorDimension().GetWidth() *
                           npu_output_tensors_[i]->GetTensorDimension().GetHeight(),
                         npu_output_tensors_[i]->GetTensorDimension().GetChannel());
    } else {
      memcpy(data, npu_output_tensors_[i]->GetBuffer(), npu_output_tensors_[i]->GetSize());
      out_tensors[i]->ResetRefCount();
    }
  }
  return RET_OK;
}

int NPUExecutor::GetIOTensorVec() {
  std::vector<hiai::TensorDimension> input_dimension;
  std::vector<hiai::TensorDimension> output_dimension;
  input_dimension.clear();
  output_dimension.clear();
  if (this->client_ == nullptr) {
    MS_LOG(ERROR) << "client is nullptr.";
    return RET_ERROR;
  }
  auto ret = this->client_->GetModelIOTensorDim(model_name_, input_dimension, output_dimension);
  if (ret != hiai::AI_SUCCESS) {
    MS_LOG(ERROR) << "Get model input and output tensor dims failed." << ret;
    return RET_ERROR;
  }
  ret = UpdateInputTensorVec(input_dimension);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Update input tensor vector failed. " << ret;
    return RET_ERROR;
  }
  ret = UpdateOutputTensorVec(output_dimension);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Update output tensor vector failed. " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

int NPUExecutor::UpdateInputTensorVec(const std::vector<hiai::TensorDimension> &input_dimension) {
  if (input_dimension.empty()) {
    MS_LOG(ERROR) << "npu input tensor dimension is empty.";
    return RET_ERROR;
  }
  npu_input_tensors_.resize(input_dimension.size());
  npu_input_tensors_.clear();
  for (const auto &inDim : input_dimension) {
    std::shared_ptr<hiai::AiTensor> input = std::make_shared<hiai::AiTensor>();
    if (input->Init(&inDim) != hiai::AI_SUCCESS) {
      MS_LOG(ERROR) << "Input AiTensor init failed.";
      return RET_ERROR;
    }
    npu_input_tensors_.push_back(input);
  }
  if (npu_input_tensors_.empty()) {
    MS_LOG(ERROR) << "NPU input tensor is empty.";
    return RET_ERROR;
  }
  return RET_OK;
}

int NPUExecutor::UpdateOutputTensorVec(const std::vector<hiai::TensorDimension> &output_dimension) {
  if (output_dimension.empty()) {
    MS_LOG(ERROR) << "output_dimension_ is empty.";
    return RET_ERROR;
  }
  npu_output_tensors_.resize(output_dimension.size());
  npu_output_tensors_.clear();
  for (const auto &outDim : output_dimension) {
    std::shared_ptr<hiai::AiTensor> output = std::make_shared<hiai::AiTensor>();
    int ret = output->Init(&outDim);
    if (ret != hiai::AI_SUCCESS) {
      return RET_ERROR;
    }
    npu_output_tensors_.push_back(output);
  }
  if (npu_output_tensors_.empty()) {
    MS_LOG(ERROR) << "NPU output tensor is empty.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
