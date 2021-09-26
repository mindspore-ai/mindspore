/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "cxx_api/model/acl/acl_vm/ms_tensor_ref.h"
#include <algorithm>

namespace mindspore {
VectorRef MSTensorRef::Convert(const std::vector<MSTensor> &tensors) {
  VectorRef res;
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(res),
                 [](const MSTensor &t) { return MSTensorRef(t); });
  return res;
}

std::vector<MSTensor> MSTensorRef::Convert(const BaseRef &args) {
  std::vector<MSTensor> res;
  if (utils::isa<VectorRef>(args)) {
    VectorRef args_vec = utils::cast<VectorRef>(args);
    res = ConvertTuple(args_vec);
  } else if (utils::isa<MSTensorRef>(args)) {
    auto wrapper = utils::cast<MSTensorRef>(args);
    res.push_back(wrapper.ms_tensor_);
  } else {
    MS_LOG(EXCEPTION) << "Invalid BaseRef " << args.ToString() << " must be MSTensorRef or VectorRef{MSTensorRef...}";
  }

  return res;
}

std::shared_ptr<Base> MSTensorRef::copy() const {
  MSTensor *tensor = ms_tensor_.Clone();
  auto res = std::make_shared<MSTensorRef>(static_cast<const MSTensor &>(*tensor));
  MSTensor::DestroyTensorPtr(tensor);
  return res;
}

bool MSTensorRef::operator==(const BaseRef &other) const {
  if (!utils::isa<MSTensorRef>(other)) {
    return false;
  }
  auto other_ms_tensor = utils::cast<MSTensorRef>(other).ms_tensor_;
  auto this_ms_tensor = ms_tensor_;
  return (this_ms_tensor.Name() == other_ms_tensor.Name()) && (this_ms_tensor.Shape() == other_ms_tensor.Shape()) &&
         (this_ms_tensor.MutableData() == other_ms_tensor.MutableData()) &&
         (this_ms_tensor.DataSize() == other_ms_tensor.DataSize()) &&
         (this_ms_tensor.DataType() == other_ms_tensor.DataType());
}

std::vector<MSTensor> MSTensorRef::ConvertTuple(const VectorRef &args) {
  std::vector<MSTensor> outs;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &item = args[i];
    if (utils::isa<VectorRef>(item)) {
      VectorRef args_vec = utils::cast<VectorRef>(args);
      auto ret = ConvertTuple(args_vec);
      outs.insert(outs.end(), ret.begin(), ret.end());
    } else if (utils::isa<MSTensorRef>(item)) {
      auto wrapper = utils::cast<MSTensorRef>(item);
      outs.push_back(wrapper.ms_tensor_);
    } else {
      MS_LOG(EXCEPTION) << "Invalid BaseRef " << args.ToString() << " must be MSTensorRef or VectorRef{MSTensorRef...}";
    }
  }
  return outs;
}
}  // namespace mindspore
