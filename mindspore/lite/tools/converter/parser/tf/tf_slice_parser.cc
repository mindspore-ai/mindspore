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
#include "tools/converter/parser/tf/tf_slice_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/fusion/slice_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *TFSliceParser::Parse(const tensorflow::NodeDef &tf_op,
                                      const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                      std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::SliceFusion>();

  // begin
  tensorflow::AttrValue attr_value;
  auto begin_node = GetConstInputNode(tf_node_map, tf_op.input(1));
  if (begin_node == nullptr) {
    MS_LOG(ERROR) << "Find StridedSlice input begin failed";
    return nullptr;
  }
  if (!TensorFlowUtils::FindAttrValue(*begin_node, "value", &attr_value)) {
    MS_LOG(ERROR) << "The value attr should be specified";
    return nullptr;
  }
  auto tensor_proto = attr_value.tensor();

  std::vector<int32_t> begin;
  if (tensor_proto.int_val_size() > 0) {
    for (int i = 0; i < tensor_proto.int_val_size(); ++i) {
      begin.push_back(tensor_proto.int_val(i));
    }
  } else {
    auto data_num = tensor_proto.tensor_content().size() / sizeof(int32_t);
    auto data = reinterpret_cast<const int32_t *>(tensor_proto.tensor_content().data());
    for (size_t i = 0; i < data_num; ++i) {
      begin.push_back(data[i]);
    }
  }

  // axes
  std::vector<int64_t> axes;
  axes.clear();
  for (size_t i = 0; i < begin.size(); ++i) {
    axes.push_back(i);
  }
  prim->set_axes(axes);

  *output_size = 1;
  for (int i = 0; i < 3; i++) {
    if (AddOpInput(tf_op, i, inputs) != RET_OK) {
      MS_LOG(ERROR) << "Add Op input " << i << " failed.";
      return nullptr;
    }
  }

  return prim.release();
}
TFNodeRegistrar g_tfSliceParser("Slice", new TFSliceParser());
}  // namespace lite
}  // namespace mindspore
