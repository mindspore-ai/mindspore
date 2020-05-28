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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_ATTR_UTILS_CONVERT_UTIL_H_
#define MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_ATTR_UTILS_CONVERT_UTIL_H_

#include <vector>
#include <utility>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <string>
#include <fstream>
#include "ir/tensor.h"
#include "session/anf_runtime_algorithm.h"
#include "predict/schema/inner/ms_generated.h"

using TensorPtr = mindspore::tensor::TensorPtr;
using TensorPtrList = std::vector<mindspore::tensor::TensorPtr>;
using AllOutputTensors = std::unordered_map<int, TensorPtrList>;
using OpDefT = mindspore::predict::OpDefT;
using GraphDefT = mindspore::predict::GraphDefT;
using TensorDefT = mindspore::predict::TensorDefT;
using SubGraphDefT = mindspore::predict::SubGraphDefT;
using SubGraphPtr = std::unique_ptr<mindspore::predict::SubGraphDefT>;
using NodeDef = mindspore::predict::NodeDefT;
using MsDataType = mindspore::predict::DataType;
using MsFormat = mindspore::predict::Format;
using MsKernelKey = void *;
namespace mindspore {
namespace predict {
namespace utils {
TypePtr GetTypePtr(const AnfNodePtr &anf_node);
MsDataType GetMSDataType(TypeId ori_data_type);
MsFormat GetMsFormat(const std::string &format_str);
TensorPtr GetParaAscendTensor(const AnfNodePtr &anf_node);
TensorPtr GetParaCpuTensor(const AnfNodePtr &anf_node);
TensorPtr GetValueTensor(const ValueNodePtr &const_node);
TensorPtr GetKernelCpuTensor(const CNodePtr &c_node_ptr, size_t inx);
TensorPtr GetKernelAscendTensor(const CNodePtr &c_node_ptr, size_t inx);
TensorPtr GetOutputTensor(const AnfNodePtr &out_node, size_t inx);
bool FindNodeInMap(const std::unordered_map<MsKernelKey, int> &Nodemap, const AnfNodePtr &node);
bool SaveDeviceModelUtil(const std::shared_ptr<GraphDefT> &new_ms_graph_ptr, const std::string &save_path_name,
                         SubGraphDefT *sub_graph_def_t);
}  // namespace utils
}  // namespace predict
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PREDICT_CONVERTER_ATTR_UTILS_CONVERT_UTIL_H_
