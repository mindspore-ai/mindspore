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

#include <vector>
#include <memory>
#include "utils/base_ref_utils.h"
#include "include/ms_tensor.h"
#include "ir/tensor.h"

namespace mindspore {
std::vector<std::shared_ptr<inference::MSTensor>> TransformBaseRefToMSTensor(const BaseRef &base_ref) {
  std::vector<std::shared_ptr<inference::MSTensor>> msTensors;
  if (utils::isa<VectorRef>(base_ref)) {
    auto ref_list = utils::cast<VectorRef>(base_ref);
    for (size_t i = 0; i < ref_list.size(); ++i) {
      if (utils::isa<tensor::Tensor>(ref_list[i])) {
        auto tensor_ptr = utils::cast<std::shared_ptr<tensor::Tensor>>(ref_list[i]);
        MS_EXCEPTION_IF_NULL(tensor_ptr);
        auto tensor = new inference::Tensor(tensor_ptr);
        msTensors.emplace_back(std::shared_ptr<inference::MSTensor>(tensor));
      } else {
        MS_LOG(EXCEPTION) << "The output is not a tensor!";
      }
    }
  } else if (utils::isa<tensor::Tensor>(base_ref)) {
    auto tensor_ptr = utils::cast<std::shared_ptr<tensor::Tensor>>(base_ref);
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    auto tensor = new inference::Tensor(tensor_ptr);
    msTensors.emplace_back(std::shared_ptr<inference::MSTensor>(tensor));
  } else {
    MS_LOG(EXCEPTION) << "The output is not a base ref list or a tensor!";
  }
  return msTensors;
}

std::vector<std::vector<std::shared_ptr<inference::MSTensor>>> TransformVectorRefToMultiTensor(
  const VectorRef &vector_ref) {
  std::vector<std::vector<std::shared_ptr<inference::MSTensor>>> multiTensor;
  for (size_t i = 0; i < vector_ref.size(); ++i) {
    auto tensors = TransformBaseRefToMSTensor(vector_ref[i]);
    multiTensor.emplace_back(tensors);
  }
  return multiTensor;
}
}  // namespace mindspore
