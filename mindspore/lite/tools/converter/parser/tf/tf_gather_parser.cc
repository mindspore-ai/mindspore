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
#include "tools/converter/parser/tf/tf_gather_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/gather.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFGatherParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Gather>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  int batchDims = 0;
  tensorflow::AttrValue attr_value;
  if (TensorFlowUtils::FindAttrValue(tf_op, "batch_dims", &attr_value)) {
    batchDims = attr_value.i();
  }

  int32_t axis = 1;
  bool axis_is_set = false;
  if (tf_op.input_size() == 3) {
    axis_is_set = true;
    auto axis_node = GetConstInputNode(tf_node_map, tf_op.input(THIRD_INPUT));
    if (axis_node == nullptr) {
      MS_LOG(ERROR) << "Find Gather input axis failed";
      return nullptr;
    }
    if (!TensorFlowUtils::FindAttrValue(*axis_node, "value", &attr_value)) {
      MS_LOG(ERROR) << "The value attr should be specified";
      return nullptr;
    }
    auto tensor_proto = attr_value.tensor();
    if (tensor_proto.dtype() == tensorflow::DT_INT32) {
      if (tensor_proto.int_val_size() > 0) {
        axis = tensor_proto.int_val(0);
      } else {
        MS_CHECK_GE(tensor_proto.tensor_content().size(), sizeof(int32_t), nullptr);
        axis = (reinterpret_cast<const int32_t *>(tensor_proto.tensor_content().data()))[0];
      }
    } else if (tensor_proto.dtype() == tensorflow::DT_INT64) {
      if (tensor_proto.int64_val_size() > 0) {
        axis = tensor_proto.int64_val(0);
      } else {
        MS_CHECK_GE(tensor_proto.tensor_content().size(), sizeof(int64_t), nullptr);
        axis = (reinterpret_cast<const int64_t *>(tensor_proto.tensor_content().data()))[0];
      }
    } else {
      MS_LOG(ERROR) << "axis must be int32 or int64";
      return nullptr;
    }
  }
  if (batchDims != 0 && !axis_is_set) {
    axis = batchDims;
  }
  (void)prim_c->AddAttr("axis", MakeValue(axis));

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

TFNodeRegistrar g_tfGatherV2Parser("GatherV2", new TFGatherParser());
}  // namespace lite
}  // namespace mindspore
