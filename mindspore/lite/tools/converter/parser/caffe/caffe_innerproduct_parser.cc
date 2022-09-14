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

#include "tools/converter/parser/caffe/caffe_innerproduct_parser.h"
#include <memory>
#include "ops/fusion/full_connection.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr CaffeInnerProductParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::FullConnection>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::NO_ACTIVATION);

  const caffe::InnerProductParameter &innerProductParam = proto.inner_product_param();
  if (!innerProductParam.has_num_output()) {
    MS_LOG(ERROR) << "InnerProduct Parse num_output for " << proto.name().c_str() << " failed.";
    return nullptr;
  }
  int64_t num_output = static_cast<int64_t>(innerProductParam.num_output());
  auto value_ptr = MakeValue(num_output);
  MS_CHECK_TRUE_RET(value_ptr != nullptr, nullptr);
  (void)prim_c->AddAttr(ops::kNumOutput, value_ptr);

  if (innerProductParam.axis() == 1) {
    prim->set_axis(1);
    prim->set_use_axis(true);
  } else {
    MS_LOG(ERROR) << "InnerProduct Parse axis only support default 1, but actually " << innerProductParam.axis();
    return nullptr;
  }
  if (innerProductParam.bias_term()) {
    prim->set_has_bias(true);
  }

  return prim->GetPrim();
}

CaffeNodeRegistrar g_caffeInnerProductParser("InnerProduct", new CaffeInnerProductParser());
}  // namespace lite
}  // namespace mindspore
