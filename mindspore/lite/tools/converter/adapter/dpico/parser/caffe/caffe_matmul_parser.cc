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
#include "ops/mat_mul.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeMatmulParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::MatMul>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  if (proto.has_matmul_param()) {
    const caffe::MatMulParameter &matmul_param = proto.matmul_param();
    if (matmul_param.has_dim_1()) {
      prim->AddAttr(dpico::kDim1, MakeValue<uint32_t>(matmul_param.dim_1()));
    }
    if (matmul_param.has_dim_2()) {
      prim->AddAttr(dpico::kDim2, MakeValue<uint32_t>(matmul_param.dim_2()));
    }
    if (matmul_param.has_dim_3()) {
      prim->AddAttr(dpico::kDim3, MakeValue<uint32_t>(matmul_param.dim_3()));
    }
  }
  prim->set_transpose_a(false);
  prim->set_transpose_b(true);
  return prim.release();
}

CaffeNodeRegistrar g_caffeMatmulParser("MatMul", new CaffeMatmulParser());
}  // namespace lite
}  // namespace mindspore
