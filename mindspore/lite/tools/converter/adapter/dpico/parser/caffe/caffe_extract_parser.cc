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

#include "parser/caffe/caffe_extract_parser.h"
#include <memory>
#include <vector>
#include <map>
#include <string>
#include "common/check_base.h"
#include "common/op_attr.h"
#include "ops/custom.h"
#include "ops/op_name.h"
#include "third_party/securec/include/securec.h"

namespace mindspore {
namespace lite {
namespace {
STATUS SetExtractAxis(const caffe::ExtractParameter &extract_param, ops::Custom *prim,
                      std::map<std::string, std::vector<uint8_t>> *custom_attrs) {
  MS_ASSERT(prim != nullptr);
  MS_ASSERT(custom_attrs != nullptr);
  int32_t axis = 1;
  if (extract_param.has_axis()) {
    axis = extract_param.axis();
  } else if (extract_param.has_slice_dim()) {
    axis = static_cast<int32_t>(extract_param.slice_dim());
  }
  (void)prim->AddAttr(ops::kAxis, api::MakeValue<int64_t>(axis));

  std::vector<uint8_t> axis_attr(sizeof(int32_t));
  if (memcpy_s(axis_attr.data(), axis_attr.size() * sizeof(uint8_t), &axis, sizeof(int32_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return RET_ERROR;
  }
  (*custom_attrs)[ops::kAxis] = axis_attr;
  return RET_OK;
}

STATUS SetExtractSlicePointBegin(const caffe::ExtractParameter &extract_param, ops::Custom *prim,
                                 std::map<std::string, std::vector<uint8_t>> *custom_attrs) {
  MS_ASSERT(prim != nullptr);
  MS_ASSERT(custom_attrs != nullptr);
  uint32_t slice_point_begin = 0;
  if (extract_param.has_slice_point_begin()) {
    slice_point_begin = extract_param.slice_point_begin();
  }
  (void)prim->AddAttr(dpico::kSlicePointBegin, api::MakeValue<int64_t>(slice_point_begin));

  std::vector<uint8_t> slice_point_begin_attr(sizeof(uint32_t));
  if (memcpy_s(slice_point_begin_attr.data(), slice_point_begin_attr.size() * sizeof(uint8_t), &slice_point_begin,
               sizeof(uint32_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return RET_ERROR;
  }
  (*custom_attrs)[dpico::kSlicePointBegin] = slice_point_begin_attr;
  return RET_OK;
}

STATUS SetExtractSlicePointEnd(const caffe::ExtractParameter &extract_param, ops::Custom *prim,
                               std::map<std::string, std::vector<uint8_t>> *custom_attrs) {
  MS_ASSERT(prim != nullptr);
  MS_ASSERT(custom_attrs != nullptr);
  uint32_t slice_point_end = 1;
  if (extract_param.has_slice_point_end()) {
    slice_point_end = extract_param.slice_point_end();
  }
  (void)prim->AddAttr(dpico::kSlicePointEnd, api::MakeValue<int64_t>(slice_point_end));

  std::vector<uint8_t> slice_point_end_attr(sizeof(uint32_t));
  if (memcpy_s(slice_point_end_attr.data(), slice_point_end_attr.size() * sizeof(uint8_t), &slice_point_end,
               sizeof(uint32_t)) != EOK) {
    MS_LOG(ERROR) << "memcpy_s failed.";
    return RET_ERROR;
  }
  (*custom_attrs)[dpico::kSlicePointEnd] = slice_point_end_attr;
  return RET_OK;
}
}  // namespace
BaseOperatorPtr CaffeExtractParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("Extract");

  std::map<std::string, std::vector<uint8_t>> custom_attrs{};
  const caffe::ExtractParameter &extract_param = proto.extract_param();
  if (SetExtractAxis(extract_param, prim.get(), &custom_attrs) != RET_OK) {
    MS_LOG(ERROR) << "set extract axis failed.";
    return nullptr;
  }
  if (SetExtractSlicePointBegin(extract_param, prim.get(), &custom_attrs) != RET_OK) {
    MS_LOG(ERROR) << "set extract slice point begin failed.";
    return nullptr;
  }
  if (SetExtractSlicePointEnd(extract_param, prim.get(), &custom_attrs) != RET_OK) {
    MS_LOG(ERROR) << "set extract slice point end failed.";
    return nullptr;
  }

  prim->set_attr(custom_attrs);
  return prim;
}

CaffeNodeRegistrar g_caffeExtractParser("Extract", new CaffeExtractParser());
}  // namespace lite
}  // namespace mindspore
