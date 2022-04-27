
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

#include <numeric>
#include <algorithm>
#include "src/expression/ops.h"
#include "src/expression/ops_utils.h"
#include "src/expression/param.h"
#include "include/api/cfg.h"
#include "src/expression/sequential.h"

namespace mindspore {
namespace lite {
void InputM::SetUp(const std::vector<int> &dims, TypeId data_type, int fmt) {
  expr()->SetSize(C0NUM);
  expr()->SetDims(dims);
  expr()->set_data_type(data_type);
  expr()->set_format(fmt);
  set_primitive(schema::PrimitiveType_NONE);
}

InputM::InputM(const std::vector<int> &dims, TypeId data_type, int fmt) : Node() { SetUp(dims, data_type, fmt); }

InputM::InputM(const schema::Tensor *tensor) : Node() {
  std::vector<int> dims(tensor->dims()->size());
  (void)std::transform(tensor->dims()->begin(), tensor->dims()->end(), dims.begin(), [](int32_t x) { return x; });
  SetUp(dims, static_cast<TypeId>(tensor->dataType()), tensor->format());
  if (tensor->name()) set_name(tensor->name()->str());
  if (tensor->data() != nullptr) data_.Copy(tensor->data()->data(), tensor->data()->size());
}

namespace NN {
Node *Input(const std::vector<int> &dims, TypeId data_type, int fmt) {
  auto i = new (std::nothrow) InputM(dims, data_type, fmt);
  if (i == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate input expression ";
    return nullptr;
  }
  return i;
}

Net *Sequential() {
  auto s = new (std::nothrow) mindspore::lite::Sequential();
  if (s == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate sequential expression";
    return nullptr;
  }
  return s;
}
};  // namespace NN
}  // namespace lite
}  // namespace mindspore
