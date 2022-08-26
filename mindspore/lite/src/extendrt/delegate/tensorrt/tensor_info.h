/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSOR_INFO_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSOR_INFO_H_

#include <utility>
#include <string>
#include <vector>
#include <memory>
#include "include/api/kernel.h"
#include "core/ir/tensor.h"

namespace mindspore::lite {
class TensorInfoImpl;
class TensorInfo {
 public:
  TensorInfo() = default;
  TensorInfo(const std::string &name, mindspore::DataType type, const std::vector<int64_t> &shape,
             mindspore::Format format, const void *data, size_t data_len,
             const mindspore::tensor::TensorPtr &tensor_val);
  ~TensorInfo() = default;

  std::string Name() const;
  mindspore::DataType DataType() const;
  mindspore::Format format() const;
  const std::vector<int64_t> &Shape() const;
  int64_t ElementNum() const;
  const void *Data() const;
  void *MutableData();
  size_t DataSize() const;

  bool IsConst() const;

  void SetShape(const std::vector<int64_t> &shape);
  void SetData(const void *data, size_t data_len);

  size_t item_size() const;

  TensorInfo &operator=(const TensorInfo &other);
  bool operator==(const TensorInfo &other) const;
  bool operator!=(const TensorInfo &other) const;
  bool operator<(const TensorInfo &other) const;

 private:
  std::shared_ptr<TensorInfoImpl> impl_ = nullptr;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_TENSORRT_TENSOR_INFO_H_
