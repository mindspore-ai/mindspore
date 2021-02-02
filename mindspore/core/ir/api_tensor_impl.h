/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CORE_IR_API_TENSOR_IMPL_H_
#define MINDSPORE_CORE_IR_API_TENSOR_IMPL_H_

#include <string>
#include <vector>
#include <memory>
#include "include/api/types.h"

namespace mindspore {
class MSTensor::Impl {
 public:
  Impl() = default;
  virtual ~Impl() = default;

  virtual const std::string &Name() const = 0;
  virtual enum DataType DataType() const = 0;
  virtual const std::vector<int64_t> &Shape() const = 0;

  virtual std::shared_ptr<const void> Data() const = 0;
  virtual void *MutableData() = 0;
  virtual size_t DataSize() const = 0;

  virtual bool IsDevice() const = 0;

  virtual std::shared_ptr<Impl> Clone() const = 0;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_IR_API_TENSOR_IMPL_H_
