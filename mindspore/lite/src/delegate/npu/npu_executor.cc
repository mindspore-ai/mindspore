/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/delegate/npu/npu_executor.h"
#include <unordered_map>
#include "include/errorcode.h"
#include "src/delegate/npu/npu_manager.h"
#include "src/common/log_adapter.h"

namespace mindspore {
NPUExecutor::~NPUExecutor() {
  client_.reset();
  for (auto t : npu_input_tensors_) {
    t.reset();
  }
  npu_input_tensors_.clear();
  for (auto t : npu_output_tensors_) {
    t.reset();
  }
  npu_output_tensors_.clear();
}

int NPUExecutor::Prepare() {
  MS_ASSERT(npu_manager_ != nullptr);
  this->client_ = npu_manager_->GetClient(model_name_);
  if (this->client_ == nullptr) {
    MS_LOG(ERROR) << "client is nullptr.";
    return RET_ERROR;
  }
  if (GetIOTensorVec() != RET_OK) {
    MS_LOG(ERROR) << "NPUExecutor GetIOTensorVec failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

std::vector<int64_t> GetNpuTensorShape(int dim, std::shared_ptr<hiai::AiTensor> npu_tensor) {
  std::vector<int64_t> npu_shape;
  if (dim > 0) {
    npu_shape.push_back(npu_tensor->GetTensorDimension().GetNumber());
  }
  if (dim > 1) {
    npu_shape.push_back(npu_tensor->GetTensorDimension().GetChannel());
  }
  if (dim > 2) {
    npu_shape.push_back(npu_tensor->GetTensorDimension().GetHeight());
  }
  if (dim > 3) {
    npu_shape.push_back(npu_tensor->GetTensorDimension().GetWidth());
  }
  return npu_shape;
}

std::vector<int> ExpandShapeTo4d(const std::vector<int> &shape) {
  if (shape.size() == 0 || shape.size() >= NPU_SHAPE_SIZE) {
    return shape;
  }
  std::vector<int> ret{shape};
  for (auto i = shape.size(); i < NPU_SHAPE_SIZE; ++i) {
    ret.push_back(1);
  }
  return ret;
}

bool IsSameShapeTensor(mindspore::MSTensor tensor, std::shared_ptr<hiai::AiTensor> npu_tensor) {
  if (tensor.Shape().size() > NPU_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Npu does not support output tensor dims greater than 4";
    return false;
  }
  return GetNpuTensorShape(tensor.Shape().size(), npu_tensor) == tensor.Shape();
}

int NPUExecutor::Run(const std::vector<mindspore::MSTensor> &in_tensors,
                     const std::vector<mindspore::MSTensor> &out_tensors, const std::vector<NPUOp *> &in_ops) {
  hiai::AiContext context;
  std::unordered_map<std::string, int> tensor_uses;
  for (const auto op : in_ops) {
    for (const auto op_input : op->inputs()) {
      if (tensor_uses.find(op_input.Name()) == tensor_uses.end()) {
        tensor_uses.insert({op_input.Name(), 1});
      } else {
        tensor_uses[op_input.Name()]++;
      }
    }
  }
  for (int i = 0; i < npu_input_tensors_.size(); ++i) {
    int index = 0;
    for (; index < in_tensors.size(); index++) {
      if (tensor_uses[in_tensors[index].Name()] > 0 && IsSameShapeTensor(in_tensors[index], npu_input_tensors_[i])) {
        auto data = in_tensors[index].Data();
        if (data == nullptr) {
          MS_LOG(ERROR) << "For " << model_name_ << ", the input tensor " << in_tensors[index].Name()
                        << " data is nullptr";
          return RET_ERROR;
        }

        memcpy(npu_input_tensors_[i]->GetBuffer(), data.get(), in_tensors[index].DataSize());
        tensor_uses[in_tensors[index].Name()]--;
        break;
      }
    }
    if (index == in_tensors.size()) {
      MS_LOG(ERROR) << "Can't find corresponding ms lite tensor of the " << i << "th input tensor for npu executor "
                    << model_name_;
      return RET_ERROR;
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

  std::vector<bool> outputs_visited(out_tensors.size(), false);
  for (int i = 0; i < npu_output_tensors_.size(); ++i) {
    int index = 0;
    for (; index < out_tensors.size(); index++) {
      if (!outputs_visited[index] && IsSameShapeTensor(out_tensors[index], npu_output_tensors_[i])) {
        mindspore::MSTensor out_tensor = out_tensors[index];
        auto data = out_tensor.MutableData();
        if (data == nullptr) {
          MS_LOG(ERROR) << "For " << model_name_ << ", the output tensor " << out_tensors[index].Name()
                        << " data is nullptr";
          return RET_ERROR;
        }

        memcpy(data, npu_output_tensors_[i]->GetBuffer(), npu_output_tensors_[i]->GetSize());
        outputs_visited[index] = true;
        break;
      }
    }
    if (index == out_tensors.size()) {
      MS_LOG(ERROR) << "Can't find corresponding ms lite tensor of the " << i << "th output tensor for npu executor "
                    << model_name_;
      return RET_ERROR;
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
}  // namespace mindspore
