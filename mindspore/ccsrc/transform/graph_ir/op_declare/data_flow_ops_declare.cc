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

#include "transform/graph_ir/op_declare/data_flow_ops_declare.h"
#include <vector>
#include <string>

namespace mindspore::transform {
INPUT_MAP(TensorArray) = {{1, INPUT_DESC(size)}};
ATTR_MAP(TensorArray) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())},
                         {"element_shape", ATTR_DESC(element_shape, AnyTraits<std::vector<int64_t>>())},
                         {"dynamic_size", ATTR_DESC(dynamic_size, AnyTraits<bool>())},
                         {"clear_after_read", ATTR_DESC(clear_after_read, AnyTraits<bool>())},
                         {"identical_element_shapes", ATTR_DESC(identical_element_shapes, AnyTraits<bool>())},
                         {"tensor_array_name", ATTR_DESC(tensor_array_name, AnyTraits<std::string>())}};
OUTPUT_MAP(TensorArray) = {{0, OUTPUT_DESC(handle)}, {1, OUTPUT_DESC(flow)}};
REG_ADPT_DESC(TensorArray, kNameTensorArray, ADPT_DESC(TensorArray))

INPUT_MAP(TensorArrayWrite) = {
  {1, INPUT_DESC(handle)}, {2, INPUT_DESC(index)}, {3, INPUT_DESC(value)}, {4, INPUT_DESC(flow_in)}};
ATTR_MAP(TensorArrayWrite) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TensorArrayWrite) = {{0, OUTPUT_DESC(flow_out)}};
REG_ADPT_DESC(TensorArrayWrite, kNameTensorArrayWrite, ADPT_DESC(TensorArrayWrite))

INPUT_MAP(TensorArrayGather) = {{1, INPUT_DESC(handle)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(flow_in)}};
ATTR_MAP(TensorArrayGather) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())},
                               {"element_shape", ATTR_DESC(element_shape, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(TensorArrayGather) = {{0, OUTPUT_DESC(value)}};
REG_ADPT_DESC(TensorArrayGather, kNameTensorArrayGather, ADPT_DESC(TensorArrayGather))

// DynamicStitch
INPUT_MAP(DynamicStitch) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(DynamicStitch) = {{1, DYN_INPUT_DESC(indices)}, {2, DYN_INPUT_DESC(x)}};
ATTR_MAP(DynamicStitch) = EMPTY_ATTR_MAP;
OUTPUT_MAP(DynamicStitch) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(DynamicStitch, kNameDynamicStitch, ADPT_DESC(DynamicStitch))
}  // namespace mindspore::transform
