/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "parser/onnx/onnx_lstm_parser.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/op_attr.h"
#include "ops/custom.h"
#include "./onnx.pb.h"
#include "include/registry/node_parser_registry.h"
#include "mindapi/base/logging.h"
#include "third_party/securec/include/securec.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
constexpr int kNums1 = 1;
constexpr int kNums2 = 2;
ops::BaseOperatorPtr OnnxLSTMParser::Parse(const onnx::GraphProto &, const onnx::NodeProto &onnx_node) {
  auto prim = api::MakeShared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Custom prim failed.";
    return nullptr;
  }
  prim->set_type("LSTM");
  int32_t hidden_size;
  int32_t directions = 1;
  float clip;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "direction") {
      const auto &direction = onnx_node_attr.s();
      if (direction == "forward" || direction == "reverse") {
        directions = kNums1;
      } else if (direction == "bidirectional") {
        directions = kNums2;
      }
      (void)prim->AddAttr(dpico::kDirection, api::MakeValue(direction));
    } else if (onnx_node_attr.name() == "hidden_size") {
      hidden_size = onnx_node_attr.i();
      (void)prim->AddAttr(dpico::kHiddenSize, api::MakeValue(hidden_size));
    } else if (onnx_node_attr.name() == "clip") {
      clip = onnx_node_attr.f();
      (void)prim->AddAttr(dpico::kClip, api::MakeValue(clip));
    }
  }
  // set attr for infershape
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  std::vector<uint8_t> hidden_size_attr(sizeof(int32_t));
  std::vector<uint8_t> direction_attr(sizeof(int32_t));

  if (memcpy_s(hidden_size_attr.data(), hidden_size_attr.size() * sizeof(uint8_t), &hidden_size, sizeof(int32_t)) !=
      EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return nullptr;
  }
  custom_attrs["hidden_size"] = hidden_size_attr;

  if (memcpy_s(direction_attr.data(), direction_attr.size() * sizeof(uint8_t), &directions, sizeof(int32_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return nullptr;
  }
  custom_attrs["direction"] = direction_attr;
  prim->set_attr(custom_attrs);

  return prim;
}
}  // namespace lite
}  // namespace mindspore
using mindspore::converter::kFmkTypeOnnx;
namespace mindspore::registry {
REG_NODE_PARSER(kFmkTypeOnnx, LSTM, std::make_shared<lite::OnnxLSTMParser>())
}  // namespace mindspore::registry
