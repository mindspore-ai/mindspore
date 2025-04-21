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

#include "parser/onnx/onnx_maxunpool_parser.h"
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
constexpr int kNums2 = 2;
constexpr int kNums3 = 3;
constexpr int kNums4 = 4;
ops::BaseOperatorPtr OnnxMaxUnpoolParser::Parse(const onnx::GraphProto &, const onnx::NodeProto &onnx_node) {
  auto prim = api::MakeShared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "new Custom prim failed.";
    return nullptr;
  }
  prim->set_type("MaxUnpool");
  std::vector<int32_t> kernels;
  std::vector<int32_t> strides;
  std::vector<int32_t> pads;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    if (onnx_node_attr.name() == "kernel_shape") {
      if (onnx_node_attr.ints_size() == kNums2) {
        kernels.push_back(onnx_node_attr.ints(0));
        kernels.push_back(onnx_node_attr.ints(1));
      }
    }
    if (onnx_node_attr.name() == "strides") {
      if (onnx_node_attr.ints_size() == kNums2) {
        strides.push_back(onnx_node_attr.ints(0));
        strides.push_back(onnx_node_attr.ints(1));
      }
    }
    if (onnx_node_attr.name() == "pads") {
      if (onnx_node_attr.ints_size() == kNums4) {
        pads.push_back(onnx_node_attr.ints(0));
        pads.push_back(onnx_node_attr.ints(kNums2));
        pads.push_back(onnx_node_attr.ints(1));
        pads.push_back(onnx_node_attr.ints(kNums3));
      }
    }
  }
  if (pads.empty()) {
    pads = {0, 0, 0, 0};
  }
  if (strides.empty()) {
    strides.push_back(1);
    strides.push_back(1);
  }
  // set attr for mapper
  (void)prim->AddAttr(ops::kStrides, api::MakeValue(strides.front()));
  (void)prim->AddAttr(dpico::kKernelShape, api::MakeValue(kernels.front()));
  (void)prim->AddAttr(dpico::kPads, api::MakeValue(pads.front()));
  // set attr for infershape
  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  std::vector<uint8_t> kernel_shape_attr(sizeof(int32_t));
  if (memcpy_s(kernel_shape_attr.data(), kernel_shape_attr.size() * sizeof(uint8_t), &kernels.front(),
               sizeof(int32_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return nullptr;
  }
  custom_attrs[dpico::kKernelShape] = kernel_shape_attr;

  std::vector<uint8_t> strides_attr(sizeof(int32_t));
  if (memcpy_s(strides_attr.data(), strides_attr.size() * sizeof(uint8_t), &strides.front(), sizeof(int32_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return nullptr;
  }
  custom_attrs[ops::kStrides] = strides_attr;
  prim->set_attr(custom_attrs);

  return prim;
}
}  // namespace lite
}  // namespace mindspore
using mindspore::converter::kFmkTypeOnnx;
namespace mindspore::registry {
REG_NODE_PARSER(kFmkTypeOnnx, MaxUnpool, std::make_shared<lite::OnnxMaxUnpoolParser>())
}  // namespace mindspore::registry
