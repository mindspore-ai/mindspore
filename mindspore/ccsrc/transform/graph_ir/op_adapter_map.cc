/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "transform/graph_ir/op_adapter_map.h"
#include <memory>
#include "graph/operator.h"
#include "transform/graph_ir/op_adapter_desc.h"

namespace mindspore {
namespace transform {
namespace {
mindspore::HashMap<std::string, OpAdapterDescPtr> adpt_map_ = {
  {kNameCustomOp, std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<Operator>>())}};
}  // namespace

template <>
mindspore::HashMap<std::string, mindspore::HashMap<int, std::string>> OpAdapter<::ge::Operator>::cus_input_map_{};
template <>
mindspore::HashMap<std::string, std::map<int, std::string>> OpAdapter<::ge::Operator>::cus_output_map_{};

mindspore::HashMap<std::string, OpAdapterDescPtr> &OpAdapterMap::get() { return adpt_map_; }
}  // namespace transform
}  // namespace mindspore
