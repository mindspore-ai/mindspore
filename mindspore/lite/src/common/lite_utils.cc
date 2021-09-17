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

#include "include/lite_utils.h"
#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include "src/tensor.h"
#include "src/common/log_adapter.h"
#ifndef STRING_KERNEL_CLIP
#include "src/common/string_util.h"
#endif
#include "tools/common/option.h"
#include "include/errorcode.h"
#include "include/ms_tensor.h"

namespace mindspore {
namespace lite {
int StringsToMSTensor(const std::vector<std::string> &inputs, tensor::MSTensor *tensor) {
#ifndef STRING_KERNEL_CLIP
  if (tensor == nullptr) {
    return RET_PARAM_INVALID;
  }

  std::vector<StringPack> all_pack;
  for (auto &input : inputs) {
    StringPack pack = {static_cast<int>(input.length()), input.data()};
    all_pack.push_back(pack);
  }
  return WriteStringsToTensor(static_cast<Tensor *>(tensor), all_pack);
#else
  MS_LOG(ERROR) << unsupport_string_tensor_log;
  return RET_ERROR;
#endif
}

std::vector<std::string> MSTensorToStrings(const tensor::MSTensor *tensor) {
#ifndef STRING_KERNEL_CLIP
  if (tensor == nullptr) {
    return {""};
  }
  const void *ptr = const_cast<Tensor *>(static_cast<const Tensor *>(tensor))->data();
  std::vector<StringPack> all_pack = ParseStringBuffer(ptr);
  std::vector<std::string> result(all_pack.size());
  std::transform(all_pack.begin(), all_pack.end(), result.begin(), [](StringPack &pack) {
    std::string str(pack.data, pack.len);
    return str;
  });
  return result;
#else
  MS_LOG(ERROR) << unsupport_string_tensor_log;
  return {""};
#endif
}
}  // namespace lite
}  // namespace mindspore
