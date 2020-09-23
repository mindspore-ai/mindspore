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

#include "src/lite_kernel.h"
#include <algorithm>

namespace mindspore::kernel {
void LiteKernel::InitOutTensorRefCount() {
  for (auto *tensor : this->out_tensors_) {
    tensor->SetRefCount(this->out_kernels_.size());
  }
}

int LiteKernel::DecOutTensorRefCount() {
  for (auto *tensor : this->out_tensors_) {
    tensor->decRefCount();
    if (0 >= tensor->RefCount()) {
      auto ret = tensor->FreeData();
      if (0 != ret) {
        MS_LOG(ERROR) << "Free tensor data failed";
        return ret;
      }
    }
  }
  return 0;
}

int LiteKernel::Prepare() {
  if (!InferShapeDone()) {
    (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->SetInferFlag(true);
    auto ret = (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->InferShape(in_tensors_, out_tensors_);
    if (ret != 0) {
      (const_cast<mindspore::lite::PrimitiveC *>(primitive_))->SetInferFlag(false);
      MS_LOG(ERROR) << "InferShape fail!";
      return ret;
    }
    ret = ReSize();
    if (ret != 0) {
      MS_LOG(ERROR) << "ReSize fail!ret: " << ret;
      return ret;
    }
  }

  auto &outputs = this->out_tensors();
  for (auto *output : outputs) {
    MS_ASSERT(output != nullptr);
    output->MallocData();
  }
  return RET_OK;
}

std::vector<kernel::LiteKernel *> LiteKernelUtil::SubgraphInputKernels(
  const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<kernel::LiteKernel *> input_kernels;
  for (const auto &kernel : kernels) {
    if (kernel->in_kernels().empty() && !kernel->in_tensors().empty()) {
      input_kernels.emplace_back(kernel);
      continue;
    }
    for (const auto &input : kernel->in_kernels()) {
      auto iter = std::find(kernels.begin(), kernels.end(), input);
      auto item = std::find(input_kernels.begin(), input_kernels.end(), kernel);
      if (iter == kernels.end() && item == input_kernels.end()) {
        input_kernels.emplace_back(kernel);
      }
    }
  }
  return input_kernels;
}

std::vector<kernel::LiteKernel *> LiteKernelUtil::SubgraphOutputKernels(
  const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<kernel::LiteKernel *> output_kernels;
  for (const auto &kernel : kernels) {
    if (kernel->out_kernels().empty() && !kernel->out_tensors().empty()) {
      output_kernels.emplace_back(kernel);
      continue;
    }
    for (const auto &output : kernel->out_kernels()) {
      auto iter = std::find(kernels.begin(), kernels.end(), output);
      auto item = std::find(output_kernels.begin(), output_kernels.end(), kernel);
      if (iter == kernels.end() && item == output_kernels.end()) {
        output_kernels.emplace_back(kernel);
      }
    }
  }
  return output_kernels;
}

std::vector<lite::Tensor *> LiteKernelUtil::SubgraphInputTensors(const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<lite::Tensor *> input_tensors;
  std::vector<lite::Tensor *> all_output_tensors;
  for (const auto &kernel : kernels) {
    all_output_tensors.insert(all_output_tensors.end(), kernel->out_tensors().begin(), kernel->out_tensors().end());
  }
  std::vector<kernel::LiteKernel *> input_kernels = SubgraphInputKernels(kernels);
  for (const auto &kernel : input_kernels) {
    for (const auto &tensor : kernel->in_tensors()) {
      auto iter = std::find(all_output_tensors.begin(), all_output_tensors.end(), tensor);
      if (iter == all_output_tensors.end() && tensor->data_c() == nullptr) {
        input_tensors.emplace_back(tensor);
      }
    }
  }
  return input_tensors;
}

std::vector<lite::Tensor *> LiteKernelUtil::SubgraphOutputTensors(const std::vector<kernel::LiteKernel *> &kernels) {
  std::vector<lite::Tensor *> output_tensors;
  std::vector<lite::Tensor *> all_input_tensors;
  for (const auto &kernel : kernels) {
    all_input_tensors.insert(all_input_tensors.end(), kernel->in_tensors().begin(), kernel->in_tensors().end());
  }
  std::vector<kernel::LiteKernel *> output_kernels = SubgraphOutputKernels(kernels);
  for (const auto &kernel : output_kernels) {
    for (const auto &tensor : kernel->out_tensors()) {
      auto iter = std::find(all_input_tensors.begin(), all_input_tensors.end(), tensor);
      if (iter == all_input_tensors.end()) {
        output_tensors.emplace_back(tensor);
      }
    }
  }
  return output_tensors;
}

void LiteKernelUtil::TopologicalSortKernels(std::vector<kernel::LiteKernel *> &kernels) {
  for (auto *kernel : kernels) {
    for (auto *search_kernel : kernels) {
      if (search_kernel == kernel) {
        continue;
      }
      for (auto *tensor : kernel->in_tensors()) {
        if (lite::IsContain(search_kernel->out_tensors(), tensor)) {
          kernel->AddInKernel(search_kernel);
        }
      }
      for (auto *tensor : kernel->out_tensors()) {
        if (lite::IsContain(search_kernel->in_tensors(), tensor)) {
          kernel->AddOutKernel(search_kernel);
        }
      }
    }
  }
}

void LiteKernelUtil::InitTensorRefCount(std::vector<kernel::LiteKernel *> &kernels) {
  for (auto *kernel : kernels) {
    kernel->InitOutTensorRefCount();
  }
}

int LiteKernelUtil::SetInput(LiteKernel &kernelMod, std::vector<lite::Tensor *> inputs) { return -1; }

float *LiteKernelUtil::DequantWeight(lite::Tensor *input_tensor) {
  MS_ASSERT(input_tensor != nullptr);
  if (input_tensor->data_type() != kNumberTypeInt8) {
    MS_LOG(ERROR) << "conv weight input type error" << input_tensor->data_type();
    return nullptr;
  }
  if (input_tensor->GetQuantParams().empty()) {
    MS_LOG(ERROR) << "no quant param";
    return nullptr;
  }
  const auto *quant_datas = static_cast<const int8_t *>(input_tensor->MutableData());
  auto *dequant_datas = static_cast<float *>(malloc(input_tensor->ElementsNum() * sizeof(float)));
  if (dequant_datas == nullptr) {
    MS_LOG(ERROR) << "malloc faile";
    return nullptr;
  }

  if (input_tensor->GetQuantParams().size() != kPerTensor) {
    size_t channels = static_cast<size_t>(input_tensor->Batch());
    if (input_tensor->GetQuantParams().size() != channels) {
      MS_LOG(ERROR) << "Quant param not equal channel num " << input_tensor->GetQuantParams().size() << channels;
      free(dequant_datas);
      return nullptr;
    }
    size_t per_channel_size = input_tensor->ElementsNum() / channels;
    auto quant_param = input_tensor->GetQuantParams();
    for (size_t i = 0; i < channels; i++) {
      auto param = quant_param.at(i);
      auto scale = param.scale;
      auto zero_point = param.zeroPoint;
      auto var_corr = param.var_corr;
      auto mean_corr = param.mean_corr;
      if (var_corr < 0 || var_corr > 10) {
        MS_LOG(WARNING) << "unexpeted var_corr: " << var_corr;
        var_corr = 1;
      }
      for (size_t j = 0; j < per_channel_size; j++) {
        auto dequant_data = (quant_datas[per_channel_size * i + j] - zero_point) * scale;
        dequant_datas[per_channel_size * i + j] = static_cast<float>(dequant_data * var_corr + mean_corr);
      }
    }
  } else {
    auto quant_param = input_tensor->GetQuantParams();
    auto param = quant_param.front();
    auto scale = param.scale;
    auto zero_point = param.zeroPoint;
    for (int64_t j = 0; j < input_tensor->ElementsNum(); j++) {
      dequant_datas[j] = static_cast<float>((quant_datas[j] - zero_point) * scale);
    }
  }
  return dequant_datas;
}
}  // namespace mindspore::kernel
