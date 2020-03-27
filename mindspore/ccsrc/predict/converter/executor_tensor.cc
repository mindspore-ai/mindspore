/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "predict/converter/executor_tensor.h"

namespace mindspore {
namespace executor {
int TensorCache::addExTensor(int tensor_key, const TensorPtr &tensor, int refCount, const std::vector<int> &host_shape,
                             ExTensorType stable, bool inc) {
  MS_EXCEPTION_IF_NULL(tensor);
  TensorPtr tmp_tensor = tensor;
  ExTensorPtr ex_tensor_ptr =
    std::make_shared<ExTensor>(tensor_key, tmp_tensor, refCount, nodeIndex, host_shape, stable);
  int pre_index = ex_tensor_ptr->index_;
  if (inc) {
    nodeIndex++;
  }
  // no need to judge,just add to map directly
  tensors[tensor_key].push_back(ex_tensor_ptr);
  return pre_index;
}

std::vector<ExTensorPtr> TensorCache::findTensor(int key) {
  std::vector<ExTensorPtr> ex_tensors;
  auto iter = tensors.find(key);
  if (iter != tensors.end()) {
    return iter->second;
  } else {
    MS_LOG(INFO) << "can not find any tensorlist";
    return ex_tensors;
  }
}

void TensorCache::deleteTensor(int key) { (void)tensors.erase(key); }
}  // namespace executor
}  // namespace mindspore
