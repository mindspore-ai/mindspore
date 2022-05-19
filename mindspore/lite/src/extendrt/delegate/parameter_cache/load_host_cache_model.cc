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

#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "src/extendrt/delegate/parameter_cache/load_host_cache_model.h"
#include "src/common/log_adapter.h"
#include "src/common/common.h"
#include "include/errorcode.h"
#include "src/common/file_utils.h"

namespace {
constexpr size_t kGatherInputsSize = 3;
}
namespace mindspore {
namespace cache {
HostCacheModel::~HostCacheModel() {
  if (cache_model_ != nullptr) {
    delete cache_model_;
    cache_model_ = nullptr;
  }
}
MSTensor *SchemaTensorToMSTensor(lite::SchemaTensorWrapper *schema_tensor_wrapper,
                                 mindspore::schema::Tensor *schema_tensor) {
  std::vector<int64_t> shape;
  for (size_t j = 0; j < schema_tensor->dims()->size(); j++) {
    shape.push_back(schema_tensor->dims()->data()[j]);
  }
  std::string tensor_name;
  if (schema_tensor->name() != nullptr) {
    tensor_name = schema_tensor->name()->str();
  }
  return MSTensor::CreateRefTensor(tensor_name, (DataType)schema_tensor->dataType(), shape,
                                   schema_tensor_wrapper->data(), schema_tensor_wrapper->length());
}

Status HostCacheModel::LoadCache(const std::string &model_path) {
  cache_model_ = lite::LiteImportFromPath(model_path.c_str());
  if (cache_model_ == nullptr) {
    MS_LOG(ERROR) << "Import model failed";
    return kLiteGraphFileError;
  }

  auto allTensors = cache_model_->graph_.all_tensors_;
  for (auto node : cache_model_->graph_.all_nodes_) {
    // only support embedding cache
    if (node == nullptr || node->node_type_ != schema::PrimitiveType_Gather) {
      continue;
    }

    auto input_index = node->input_indices_[0];
    if (input_index > allTensors.size() - 1) {
      MS_LOG(ERROR) << "invalid kernel input, input_index " << input_index << ",allTensors.size() "
                    << allTensors.size();
      return kLiteOutOfTensorRange;
    }
    auto schema_tensor_wrapper = cache_model_->GetSchemaTensor(input_index);
    if (schema_tensor_wrapper == nullptr) {
      MS_LOG(ERROR) << "invalid kernel input, input_index " << input_index;
      return kLiteOutOfTensorRange;
    }

    auto schema_tensor = allTensors[input_index];
    if (schema_tensor != nullptr && schema_tensor_wrapper->data() != nullptr) {
      auto tensor = SchemaTensorToMSTensor(schema_tensor_wrapper, schema_tensor);
      if (tensor == nullptr) {
        return kLiteMemoryFailed;
      }
      cache_tensor_[tensor->Name()] = *tensor;
      MS_LOG(INFO) << tensor->Name() << " is cache tensor, and the node is [" << node->name_ << "]";
      delete tensor;
    }
  }
  return kSuccess;
}

size_t GetVocabSize(kernel::Kernel *kernel) {
  size_t vocab_size = 0;
  auto cache_config = kernel->GetConfig(lite::kMSCache);
  auto vocab_size_iter = cache_config.find(lite::kMSCacheVocabSize);
  if (vocab_size_iter == cache_config.end()) {
    return vocab_size;
  }

  auto vocab_size_opt = lite::GenericParseValue<size_t>(vocab_size_iter->second);
  if (!vocab_size_opt.IsNone()) {
    vocab_size = vocab_size_opt.Get();
  }
  return vocab_size;
}

Status HostCacheModel::LoadCache(DelegateModel<schema::Primitive> *model) {
  KernelIter from, end;
  for (KernelIter iter = model->BeginKernelIterator(); iter != model->EndKernelIterator(); iter++) {
    kernel::Kernel *kernel = *iter;
    // only support embedding cache
    if (kernel->type() != schema::PrimitiveType_Gather) {
      continue;
    }
    MS_ASSERT(kernel->inputs().size() == kGatherInputsSize);
    auto tensor = kernel->inputs()[0];
    if (tensor.Data() == nullptr) {
      continue;
    }

    size_t vocab_size = GetVocabSize(kernel);
    if (vocab_size == 0) {
      continue;
    }

    cache_tensor_[tensor.Name()] = tensor;
  }
  return mindspore::kSuccess;
}

bool HostCacheModel::CheckIsCacheKernel(kernel::Kernel *kernel) {
  if (GetHostCacheTensor(kernel) == nullptr) {
    return false;
  }
  return true;
}

MSTensor HostCacheModel::GetHostCacheTensor(kernel::Kernel *kernel) {
  if (kernel != nullptr && kernel->inputs().size() > 0) {
    auto iter = cache_tensor_.find(kernel->inputs()[0].Name());
    if (iter != cache_tensor_.end()) {
      return iter->second;
    }
  }
  return MSTensor(nullptr);
}
}  // namespace cache
}  // namespace mindspore
