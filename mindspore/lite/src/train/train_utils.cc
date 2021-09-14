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

#include "src/train/train_utils.h"
#include <vector>
#include "include/errorcode.h"
#include "include/ms_tensor.h"
#include "src/common/utils.h"
#include "src/lite_kernel.h"
#ifdef ENABLE_FP16
#include "src/runtime/kernel/arm/fp16/fp16_op_handler.h"
#endif

namespace mindspore {
namespace lite {
size_t TSFindTensor(const std::vector<lite::Tensor *> &where, const lite::Tensor *searchParameter) {
  for (size_t i = 0; i < where.size(); i++) {
    if (where[i] == searchParameter) {
      return i;
    }
  }
  return where.size();
}

size_t TSFindTensorByName(const std::vector<lite::Tensor *> &where, const std::string &searchParameter) {
  for (size_t i = 0; i < where.size(); i++) {
    if (where[i]->tensor_name() == searchParameter) {
      return i;
    }
  }
  return where.size();
}

kernel::LiteKernel *TSFindKernel(const std::vector<kernel::LiteKernel *> &where, const std::string &searchParameter) {
  auto it = std::find_if(where.begin(), where.end(),
                         [&searchParameter](const kernel::LiteKernel *k) { return (k->name() == searchParameter); });
  if (it == where.end()) {
    return nullptr;
  }
  return *it;
}

template <typename T>
float CalcSparseClassificationAccuracy(T *predictions, int *labels, int batch_size, int num_of_classes) {
  float accuracy = 0.0;
  for (int b = 0; b < batch_size; b++) {
    int max_idx = 0;
    T max_score = predictions[num_of_classes * b];
    for (int c = 1; c < num_of_classes; c++) {
      if (predictions[num_of_classes * b + c] > max_score) {
        max_score = predictions[num_of_classes * b + c];
        max_idx = c;
      }
    }
    if (labels[b] == max_idx) accuracy += 1.0;
  }
  return accuracy / (static_cast<float>(batch_size));
}

float CalculateSparseClassification(tensor::MSTensor *input, tensor::MSTensor *output) {
  if ((input->shape().size() != 1) || (input->data_type() != kNumberTypeInt32) || (output->shape().size() != 2)) {
    MS_LOG(WARNING) << "SparseClassification got a " << input->shape() << "-D input tensor, " << output->shape()
                    << "-D output tensor";
    return 0.0;
  }

  int batch = input->shape().at(0);
  int num_classes = output->shape().at(1);
  auto labels = reinterpret_cast<int *>(input->data());
  float acc = 0.0f;
  if (output->data_type() == kNumberTypeFloat32) {
    acc = CalcSparseClassificationAccuracy(reinterpret_cast<float *>(output->data()), labels, batch, num_classes);
#ifdef ENABLE_FP16
  } else if (output->data_type() == kNumberTypeFloat16) {
    acc = CalcSparseClassificationAccuracy(reinterpret_cast<float16_t *>(output->data()), labels, batch, num_classes);
#endif
  }
  return acc;
}

template <typename T>
float CalcOneHotClassificationAccuracy(T *predictions, float *labels, int batch_size, int num_of_classes) {
  float accuracy = 0.0;
  for (int b = 0; b < batch_size; b++) {
    int label = 0;
    int max_idx = 0;
    float max_label_score = labels[num_of_classes * b];
    T max_score = predictions[num_of_classes * b];
    for (int c = 1; c < num_of_classes; c++) {
      if (predictions[num_of_classes * b + c] > max_score) {
        max_score = predictions[num_of_classes * b + c];
        max_idx = c;
      }
      if (labels[num_of_classes * b + c] > max_label_score) {
        max_label_score = labels[num_of_classes * b + c];
        label = c;
      }
    }
    if (label == max_idx) accuracy += 1.0;
  }
  return accuracy / (static_cast<float>(batch_size));
}

float CalculateOneHotClassification(tensor::MSTensor *input, tensor::MSTensor *output) {
  if ((input->shape().size() != 2) || (output->shape().size() != 2)) {
    MS_LOG(WARNING) << "OneHotClassification got a " << input->shape() << "-D input tensor, " << output->shape()
                    << "-D output tensor";
    return 0.0;
  }

  int batch = input->shape().at(0);
  int num_classes = input->shape().at(1);
  auto labels = reinterpret_cast<float *>(input->data());
  float acc = 0.0f;
  if (output->data_type() == kNumberTypeFloat32) {
    acc = CalcOneHotClassificationAccuracy(reinterpret_cast<float *>(output->data()), labels, batch, num_classes);
#ifdef ENABLE_FP16
  } else if (output->data_type() == kNumberTypeFloat16) {
    acc = CalcOneHotClassificationAccuracy(reinterpret_cast<float16_t *>(output->data()), labels, batch, num_classes);
#endif
  }
  return acc;
}

Tensor *CastTensor(Tensor *tensor, TypeId dst_data_type, bool support_fp16) {
#ifdef ENABLE_FP16
  MS_ASSERT(tensor != nullptr);
  std::vector<TypeId> valid_type = {kNumberTypeFloat32, kNumberTypeFloat16, kNumberTypeFloat};
  std::vector<TypeId> fp32_type = {kNumberTypeFloat32, kNumberTypeFloat};
  if (!IsContain(valid_type, tensor->data_type())) {
    MS_LOG(ERROR) << "source data type must be fp32 or fp16,cur is " << tensor->data_type();
    return nullptr;
  }

  if (!IsContain(valid_type, dst_data_type)) {
    MS_LOG(ERROR) << "destination data type must be fp32 or fp16";
    return nullptr;
  }

  auto origin_data = tensor->data();
  MS_ASSERT(origin_data != nullptr);
  auto restore_tensor = Tensor::CopyTensor(*tensor, false);
  restore_tensor->set_data(origin_data);
  restore_tensor->set_own_data(tensor->own_data());
  restore_tensor->set_allocator(tensor->allocator());
  restore_tensor->set_scale(tensor->get_scale());
  if (IsContain(fp32_type, tensor->data_type()) && dst_data_type == kNumberTypeFloat16) {
    tensor->set_data(nullptr);
    tensor->set_data_type(kNumberTypeFloat16);
    auto ret = tensor->MallocData();
    auto new_tensor_data = tensor->data();
    MS_ASSERT(new_tensor_data != nullptr);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "malloc data failed";
      delete restore_tensor;
      return nullptr;
    }
    MS_LOG(DEBUG) << "Convert tensor to fp16 " << tensor->tensor_name();
    Float32ToFloat16_fp16_handler(origin_data, new_tensor_data, tensor->ElementsNum(), support_fp16);
  } else {
    tensor->set_data(nullptr);
    tensor->set_data_type(kNumberTypeFloat32);
    auto ret = tensor->MallocData();
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "malloc data failed";
      delete restore_tensor;
      return nullptr;
    }
    auto new_tensor_data = tensor->data();
    MS_ASSERT(new_tensor_data != nullptr);
    MS_LOG(DEBUG) << "Convert tensor to fp32 " << tensor->tensor_name();
    Float16ToFloat32_fp16_handler(origin_data, new_tensor_data, tensor->ElementsNum(), support_fp16);
  }
  return restore_tensor;
#else
  return nullptr;
#endif
}

int ScaleTensor(Tensor *tensor, float scale) {
  MS_ASSERT(tensor != nullptr);
  std::vector<TypeId> valid_type = {kNumberTypeFloat32, kNumberTypeFloat};
  if (!IsContain(valid_type, tensor->data_type())) {
    MS_LOG(DEBUG) << "Tensor: " << tensor->tensor_name() << " type is " << tensor->data_type();
    return RET_OK;
  }

  MS_LOG(DEBUG) << "Scale tensor: " << tensor->tensor_name() << " " << scale;
  return tensor->Scale<float>(scale);
}
}  // namespace lite
}  // namespace mindspore
