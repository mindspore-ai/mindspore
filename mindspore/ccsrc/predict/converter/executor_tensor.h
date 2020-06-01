/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_EXECUTOR_TENSOR_H_
#define MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_EXECUTOR_TENSOR_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <utility>
#include "ir/tensor.h"

namespace mindspore {
namespace executor {
using TensorPtr = tensor::TensorPtr;
static constexpr int MS_MAX_REFCOUNT = 999;
enum ExTensorType { INPUTDATA, WEIGHTS, CONSTANT, KERNEL, OUTPUT };
class ExTensor {
 public:
  int key_;
  TensorPtr device_tensor_ptr_;
  int ref_count_;
  int index_;
  std::vector<int> host_shape_;
  ExTensorType stable_;
  ExTensor(int key, TensorPtr tensor_ptr, int ref_count, int index, std::vector<int> host_shape,
           ExTensorType ex_tensor_type)
      : key_(key),
        device_tensor_ptr_(std::move(tensor_ptr)),
        ref_count_(ref_count),
        index_(index),
        host_shape_(std::move(host_shape)),
        stable_(ex_tensor_type) {}
  ~ExTensor() { host_shape_.clear(); }
};
using ExTensorPtr = std::shared_ptr<ExTensor>;
class TensorCache {
 public:
  TensorCache() = default;

  ~TensorCache() { tensors.clear(); }

  int addExTensor(int tensor_key, const TensorPtr &tensor, int refCount, const std::vector<int> &host_shape,
                  ExTensorType stable, bool inc = true);
  // just adjust for dynamic tensor
  std::vector<ExTensorPtr> findTensor(int key);
  void deleteTensor(int key);
  const std::unordered_map<int, std::vector<ExTensorPtr>> &GetCachedTensor() const { return tensors; }

 private:
  std::unordered_map<int, std::vector<ExTensorPtr>> tensors;
  int nodeIndex = 0;
};
using TensorCachePtr = std::shared_ptr<TensorCache>;
}  // namespace executor
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_EXECUTOR_TENSOR_H_
