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

#include "parser/caffe/caffe_matmul_parser.h"
#include <memory>
#include "common/op_attr.h"
#include "ops/fusion/mat_mul_fusion.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeMatmulParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::MatMulFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  if (proto.has_matmul_param()) {
    const caffe::MatMulParameter &matmul_param = proto.matmul_param();
    if (matmul_param.has_dim_1()) {
      (void)prim->AddAttr(dpico::kDim1, api::MakeValue<int64_t>(matmul_param.dim_1()));
    }
    if (matmul_param.has_dim_2()) {
      (void)prim->AddAttr(dpico::kDim2, api::MakeValue<int64_t>(matmul_param.dim_2()));
    }
    if (matmul_param.has_dim_3()) {
      (void)prim->AddAttr(dpico::kDim3, api::MakeValue<int64_t>(matmul_param.dim_3()));
    }
  }
  prim->set_transpose_a(false);
  prim->set_transpose_b(true);
  return prim;
}

CaffeNodeRegistrar g_caffeMatmulParser("MatMul", new CaffeMatmulParser());
}  // namespace lite
}  // namespace mindspore
