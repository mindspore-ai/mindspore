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
#include "src/runtime/kernel/arm/string/extract_feature.h"
#include <string>
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_CustomExtractFeatures;

namespace mindspore::kernel {
int ExtractFeatureCPUKernel::Init() {
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
  auto weight_data = out_tensors_.at(1)->MutableData();
  int string_num = lite::GetStringCount(input_tensor);
  std::vector<lite::StringPack> all_string_pack = ParseTensorBuffer(input_tensor);

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
                                                   const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  auto *kernel = new (std::nothrow) ExtractFeatureCPUKernel(parameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ExtractFeatureCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_CustomExtractFeatures, CpuExtractFeatureKernelCreator)
}  // namespace mindspore::kernel
