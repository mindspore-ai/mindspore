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
#include "src/runtime/kernel/arm/string/predict.h"
#include <string>
#include <algorithm>
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_CustomPredict;

namespace mindspore::kernel {
int PredictCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PredictCPUKernel::ReSize() { return RET_OK; }

std::vector<LabelInfo> PredictCPUKernel::GetLabelInfo() {
  std::vector<LabelInfo> label_info_vec;
  auto input_tensor = in_tensors_.at(0);
  auto keys_tensor = in_tensors_.at(1);
  auto labels_tensor = in_tensors_.at(2);
  auto weights_tensor = in_tensors_.at(3);

  int32_t *input = reinterpret_cast<int32_t *>(input_tensor->MutableData());
  int32_t *key_begin = reinterpret_cast<int32_t *>(keys_tensor->MutableData());
  int32_t *key_end = key_begin + keys_tensor->ElementsNum();
  int32_t *labels = reinterpret_cast<int32_t *>(labels_tensor->MutableData());
  float *weights = reinterpret_cast<float *>(weights_tensor->MutableData());

  int32_t input_elements_num = input_tensor->ElementsNum();
  int32_t items = labels_tensor->shape().at(1);

  for (int i = 0; i < input_elements_num; i++) {
    int *p = std::lower_bound(key_begin, key_end, input[i]);
    if (p == nullptr || p == key_end || *p != input[i]) {
      continue;
    }
    int index = p - key_begin;
    for (int j = 0; j < items; j++) {
      int offset = index * items + j;
      auto it = std::find_if(label_info_vec.begin(), label_info_vec.end(),
                             [&](const LabelInfo &element) { return element.label == labels[offset]; });
      if (it != label_info_vec.end()) {
        it->weight += weights[offset] / input_elements_num;
      } else {
        LabelInfo tmp = {labels[offset], weights[offset] / input_elements_num};
        label_info_vec.push_back(tmp);
      }
    }
  }
  return label_info_vec;
}

static bool LabelInfoCmp(const LabelInfo &lhs, const LabelInfo &rhs) { return lhs.weight > rhs.weight; }

int PredictCPUKernel::Run() {
  std::vector<LabelInfo> label_info_vec = GetLabelInfo();
  std::sort(label_info_vec.begin(), label_info_vec.end(), LabelInfoCmp);

  auto output_label_tensor = out_tensors_.at(0);
  auto output_weight_tensor = out_tensors_.at(1);
  auto output_label = reinterpret_cast<int32_t *>(output_label_tensor->MutableData());
  auto output_weight = reinterpret_cast<float *>(output_weight_tensor->MutableData());
  auto param = reinterpret_cast<PredictParameter *>(op_parameter_);
  for (int i = 0; i < output_label_tensor->ElementsNum(); i++) {
    if (static_cast<size_t>(i) >= label_info_vec.size() || label_info_vec[i].weight < param->weight_threshold) {
      output_label[i] = -1;
      output_weight[i] = 0.0f;
    } else {
      output_label[i] = label_info_vec[i].label;
      output_weight[i] = label_info_vec[i].weight;
    }
  }
  return RET_OK;
}

kernel::LiteKernel *CpuPredictKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                            const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                            const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  auto *kernel = new (std::nothrow) PredictCPUKernel(parameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PredictCPUKernel fail!";
    free(parameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << parameter->name_
                  << ", type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_CustomPredict, CpuPredictKernelCreator)
}  // namespace mindspore::kernel
