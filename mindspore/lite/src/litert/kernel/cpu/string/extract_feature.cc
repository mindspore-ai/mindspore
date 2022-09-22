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
#include "src/litert/kernel/cpu/string/extract_feature.h"
#include <string>
#include "src/litert/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_CustomExtractFeatures;

namespace mindspore::kernel {
int ExtractFeatureCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), C2NUM);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[1]);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ExtractFeatureCPUKernel::ReSize() { return RET_OK; }

bool ExtractFeatureCPUKernel::IsInBlacklist(const lite::StringPack &str) {
  std::vector<std::string> kBlacklist = {"<S>", "<E>", "<S> <E>"};
  for (const auto &s : kBlacklist) {
    if (str.len != static_cast<int>(s.length())) {
      continue;
    }
    if (memcmp(str.data, s.data(), str.len) == 0) {
      return true;
    }
  }
  return false;
}

int ExtractFeatureCPUKernel::Run() {
  const int kMaxDimension = 1000000;
  auto input_tensor = in_tensors_.at(0);
  auto label_data = reinterpret_cast<int32_t *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(label_data);
  auto weight_data = out_tensors_.at(1)->MutableData();
  CHECK_NULL_RETURN(weight_data);
  int string_num = lite::GetStringCount(input_tensor);
  std::vector<lite::StringPack> all_string_pack = ParseTensorBuffer(input_tensor);
  CHECK_LESS_RETURN(all_string_pack.size(), static_cast<uint32_t>(string_num));
  for (int i = 0; i < string_num; i++) {
    lite::StringPack str = all_string_pack[i];
    if (IsInBlacklist(str)) {
      label_data[i] = 0;
      reinterpret_cast<int32_t *>(weight_data)[i] = 0;
      continue;
    }
    int64_t hash_value = lite::StringHash64(str.data, str.len) % kMaxDimension;
    label_data[i] = hash_value;
    reinterpret_cast<float *>(weight_data)[i] = std::count(str.data, str.data + str.len, ' ') + 1;
  }
  if (string_num == 0) {
    label_data[0] = 0;
    reinterpret_cast<int32_t *>(weight_data)[0] = 0;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuExtractFeatureKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs, OpParameter *parameter,
                                                   const lite::Context *ctx, const kernel::KernelKey &desc) {
  auto *kernel = new (std::nothrow)
    ExtractFeatureCPUKernel(parameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ExtractFeatureCPUKernel fail!";
    free(parameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kObjectTypeString, PrimitiveType_CustomExtractFeatures, CpuExtractFeatureKernelCreator)
}  // namespace mindspore::kernel
