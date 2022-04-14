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

#include "src/expression/ops/transpose.h"
#include <memory>
#include "nnacl/transpose.h"
#include "inner/model_generated.h"
#include "src/expression/import.h"

namespace mindspore {
namespace lite {
TransposeM::TransposeM(const std::vector<int> &vector) {
  auto param = calloc(1, sizeof(TransposeParameter));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate transpose parameter";
    return;
  }
  SetOpParam(param);
  expr()->SetSize(C2NUM);
  set_primitive(schema::PrimitiveType_Transpose);
  std::vector<int> dims = {static_cast<int>(vector.size())};
  set_name(UniqueName("Transpose"));
  CreateConstTensor(C1NUM, dims, kNumberTypeInt32, KHWC, "axis", vector.data());
}

std::vector<int> TransposeM::Invert(const std::vector<int> &vector) {
  std::vector<int> res;
  for (size_t i = 0; i < vector.size(); i++) {
    int idx = static_cast<int>(i);
    auto val = std::find_if(vector.begin(), vector.end(), [idx](int x) { return (x == idx) ? true : false; });
    if (val == vector.end()) {
      MS_LOG(ERROR) << "Wrong index for " << idx;
      return {};
    }
    res.push_back(std::distance(vector.begin(), val));
  }
  return res;
}

std::vector<EXPR *> TransposeM::Grad(EXPR *yt) {
  auto tensor = input(1)->node();
  auto data = tensor->data();
  auto vec = data->Extract<int>();
  auto invert = Invert(vec);
  auto tran = new (std::nothrow) TransposeM(invert);
  if (tran == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate transpose grad";
    return {};
  }
  tran->set_name(kGradName + "/" + name() + "/" + tran->name());
  PushOp(tran);
  auto grad = (*tran)({yt});
  return grad;
}

int TransposeM::UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) {
  auto prim = new (std::nothrow) schema::TransposeT;
  if (prim == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate primitive";
    return RET_ERROR;
  }
  cnode->primitive->value.value = prim;
  return RET_OK;
}

static ImportReg reg(schema::PrimitiveType_Transpose, ReturnNode<TransposeM>);

namespace NN {
Node *Transpose(const std::vector<int> &permute) {
  auto node = new (std::nothrow) TransposeM(permute);
  return node;
}
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
