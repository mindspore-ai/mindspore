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

#include "parser/caffe/caffe_pass_through_parser.h"
#include <memory>
#include <map>
#include <string>
#include "ops/custom.h"
#include "common/op_attr.h"
#include "third_party/securec/include/securec.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffePassThroughParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("PassThrough");

  std::map<std::string, std::vector<uint8_t>> custom_attrs;
  if (proto.has_pass_through_param()) {
    const auto &pass_through_param = proto.pass_through_param();
    if (pass_through_param.has_num_output()) {
      uint32_t num_output = pass_through_param.num_output();
      (void)prim->AddAttr(dpico::kNumOutput, api::MakeValue<int64_t>(num_output));  // for mapper

      std::vector<uint8_t> num_output_attr(sizeof(uint32_t));
      if (memcpy_s(num_output_attr.data(), num_output_attr.size() * sizeof(uint8_t), &num_output, sizeof(uint32_t)) !=
          EOK) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return nullptr;
      }
      custom_attrs[dpico::kNumOutput] = num_output_attr;
    }
    if (pass_through_param.has_block_height()) {
      uint32_t block_height = pass_through_param.block_height();
      (void)prim->AddAttr(dpico::kBlockHeight, api::MakeValue<int64_t>(block_height));

      std::vector<uint8_t> block_height_attr(sizeof(uint32_t));
      if (memcpy_s(block_height_attr.data(), block_height_attr.size() * sizeof(uint8_t), &block_height,
                   sizeof(uint32_t)) != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return nullptr;
      }
      custom_attrs[dpico::kBlockHeight] = block_height_attr;
    }
    if (pass_through_param.has_block_width()) {
      uint32_t block_width = pass_through_param.block_width();
      (void)prim->AddAttr(dpico::kBlockWidth, api::MakeValue<int64_t>(block_width));

      std::vector<uint8_t> block_width_attr(sizeof(uint32_t));
      if (memcpy_s(block_width_attr.data(), block_width_attr.size() * sizeof(uint8_t), &block_width,
                   sizeof(uint32_t)) != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return nullptr;
      }
      custom_attrs[dpico::kBlockWidth] = block_width_attr;
    }
  }

  prim->set_attr(custom_attrs);
  return prim;
}

CaffeNodeRegistrar g_caffePassThroughParser("PassThrough", new CaffePassThroughParser());
}  // namespace lite
}  // namespace mindspore
