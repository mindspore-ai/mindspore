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
#include "tools/converter/parser/tf/tf_split_parser.h"
#include <functional>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <algorithm>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/split.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFSplitParser::Parse(const tensorflow::NodeDef &tf_op,
                                   const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                   std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Split>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(tf_op, "num_split", &attr_value)) {
    MS_LOG(ERROR) << "The attribute num_split should be specified";
    return nullptr;
  }
  auto number_split = attr_value.i();
  prim->set_output_num(number_split);

  int split_dim_index;
  int input_index;
  if (tf_op.op() == "Split") {
    split_dim_index = 0;
    input_index = 1;
  } else {
    split_dim_index = 2;
    input_index = 0;
  }

  auto split_dim_node = GetConstInputNode(tf_node_map, tf_op.input(split_dim_index));
  if (split_dim_node == nullptr) {
    MS_LOG(ERROR) << "Find Split input split_dim node failed";
    return nullptr;
  }
  if (!TensorFlowUtils::FindAttrValue(*split_dim_node, "value", &attr_value)) {
    MS_LOG(ERROR) << "The attribute splitDim should be specified";
    return nullptr;
  }
  auto splitDim = attr_value.tensor().int_val(0);
  prim->set_axis(splitDim);

  if (tf_op.op() == "SplitV") {
    auto size_splits_node = GetConstInputNode(tf_node_map, tf_op.input(SECOND_INPUT));
    if (size_splits_node == nullptr) {
      MS_LOG(ERROR) << "Find Split input size_splits failed";
      return nullptr;
    }
    if (!TensorFlowUtils::FindAttrValue(*size_splits_node, "value", &attr_value)) {
      MS_LOG(ERROR) << "The attribute size splits should be specified";
      return nullptr;
    }
    auto size_splits_tensor = attr_value.tensor();
    auto size = size_splits_tensor.tensor_content().size() / sizeof(int32_t);

    std::vector<int32_t> size_splits_int32;
    if (size > 0) {
      size_splits_int32.resize(size);
      auto ret = memcpy_s(size_splits_int32.data(), size * sizeof(int32_t), size_splits_tensor.tensor_content().data(),
                          size * sizeof(int32_t));
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed";
        return nullptr;
      }
    }
    std::vector<int64_t> size_splits;
    std::transform(size_splits_int32.begin(), size_splits_int32.end(), std::back_inserter(size_splits),
                   [](int32_t val) { return static_cast<int64_t>(val); });
    prim->set_size_splits(size_splits);
  }

  *output_size = number_split;
  if (AddOpInput(tf_op, input_index, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

TFNodeRegistrar g_tfSplitParser("Split", new TFSplitParser());
TFNodeRegistrar g_tfSplitVParser("SplitV", new TFSplitParser());
}  // namespace lite
}  // namespace mindspore
