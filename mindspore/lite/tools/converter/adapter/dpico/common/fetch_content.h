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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_FETCH_CONTENT_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_FETCH_CONTENT_H_

#include <string>
#include <vector>
#include "mindapi/ir/primitive.h"
#include "mindapi/ir/func_graph.h"

namespace mindspore {
namespace dpico {
struct DataInfo {
  int data_type_;
  std::vector<int> shape_;
  std::vector<uint8_t> data_;
  DataInfo() : data_type_(0) {}
};

int FetchFromDefaultParam(const api::ParameterPtr &param_node, DataInfo *data_info);

int FetchDataFromParameterNode(const api::CNodePtr &cnode, size_t index, DataInfo *data_info);

int GetDataSizeFromTensor(DataInfo *data_info, int *data_size);
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_COMMON_FETCH_CONTENT_H_
