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
#include "base/base_ref_utils.h"
#include <vector>
#include <memory>
#include "ir/tensor.h"

namespace mindspore {
void IterateFindTensor(std::vector<tensor::TensorPtr> *msTensors, const VectorRef &ref_list) {
  for (size_t i = 0; i < ref_list.size(); ++i) {
    if (utils::isa<tensor::TensorPtr>(ref_list[i])) {
      auto tensor_ptr = utils::cast<std::shared_ptr<tensor::Tensor>>(ref_list[i]);
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      msTensors->emplace_back(tensor_ptr);
    } else if (utils::isa<VectorRef>(ref_list[i])) {
      auto ref_iter = utils::cast<VectorRef>(ref_list[i]);
      IterateFindTensor(msTensors, ref_iter);
    } else {
      MS_LOG(EXCEPTION) << "The output is not a tensor";
    }
  }
}

std::vector<tensor::TensorPtr> TransformVectorRefToMultiTensor(const VectorRef &base_ref) {
  std::vector<tensor::TensorPtr> msTensors;
  if (utils::isa<VectorRef>(base_ref)) {
    auto ref_list = utils::cast<VectorRef>(base_ref);
    IterateFindTensor(&msTensors, ref_list);
  } else if (utils::isa<tensor::Tensor>(base_ref)) {
    auto tensor_ptr = utils::cast<std::shared_ptr<tensor::Tensor>>(base_ref);
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    msTensors.emplace_back(tensor_ptr);
  } else {
    MS_LOG(EXCEPTION) << "The output is not a base ref list or a tensor!";
  }
  return msTensors;
}
}  // namespace mindspore
