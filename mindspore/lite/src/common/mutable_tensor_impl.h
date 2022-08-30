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

#ifndef MINDSPORE_LITE_SRC_COMMON_MUTABLE_TESNOR_IMPL_H_
#define MINDSPORE_LITE_SRC_COMMON_MUTABLE_TESNOR_IMPL_H_

#include <string>
#include <memory>
#include <vector>
#include "ir/api_tensor_impl.h"

namespace mindspore {
class MutableTensorImpl : public MSTensor::Impl {
 public:
  virtual void SetName(const std::string &name) = 0;
  virtual void SetDataType(mindspore::DataType data_type) = 0;
  virtual void SetShape(const std::vector<int64_t> &shape) = 0;
  virtual mindspore::Format Format() const = 0;
  virtual void SetFormat(mindspore::Format format) = 0;
  virtual void SetData(void *data, bool own_data) = 0;
  virtual bool IsConst() const = 0;
  virtual void SetAllocator(const std::shared_ptr<Allocator> &allocator) = 0;
  virtual std::shared_ptr<Allocator> GetAllocator() const = 0;
  virtual std::vector<QuantParam> GetQuantParams() const = 0;
  virtual void SetQuantParams(const std::vector<QuantParam> &quant_param) = 0;
  virtual void SetDeviceData(void *data) = 0;
  virtual void *GetDeviceData() = 0;
};
using MutableTensorImplPtr = std::shared_ptr<MutableTensorImpl>;
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_COMMON_MUTABLE_TESNOR_IMPL_H_
