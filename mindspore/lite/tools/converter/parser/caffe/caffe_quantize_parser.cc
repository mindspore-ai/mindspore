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

#ifdef ENABLE_ACL_QUANT_PARAM
#include "tools/converter/parser/caffe/caffe_quantize_parser.h"
#include <memory>
#include "ops/quant_dtype_cast.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr CaffeQuantizeParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::QuantDTypeCast>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  if (proto.type() == "Quant") {
    prim->set_src_t(kNumberTypeFloat32);
    prim->set_dst_t(kNumberTypeInt8);
  } else if (proto.type() == "DeQuant") {
    prim->set_src_t(kNumberTypeInt32);
    prim->set_dst_t(kNumberTypeFloat32);
  } else {
    MS_LOG(ERROR) << "Unsupported nodeType: " << proto.type();
    return nullptr;
  }

  return prim->GetPrim();
}

CaffeNodeRegistrar g_caffeQuantizeParser("Quant", new CaffeQuantizeParser());
CaffeNodeRegistrar g_caffeDeQuantizeParser("DeQuant", new CaffeQuantizeParser());
}  // namespace lite
}  // namespace mindspore
#endif  // ENABLE_ACL_QUANT_PARAM
