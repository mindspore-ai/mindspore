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

#define USE_DEPRECATED_API
#include "tools/converter/adapter/acl/mapper/lstm_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "ops/lstm.h"
#include "nnacl/op_base.h"
#include "ops/op_name.h"
#include "src/common/log_util.h"
namespace mindspore {
namespace {
constexpr size_t kNumOnnxInputSize = 7;
}  // namespace
namespace lite {
STATUS LSTMMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  if (value_node == nullptr || src_prim == nullptr) {
    MS_LOG(ERROR) << "value node or src prim is nullptr.";
    return lite::RET_ERROR;
  }
  auto dst_prim = std::make_shared<acl::CommonLSTM>();
  if (dst_prim == nullptr) {
    MS_LOG(ERROR) << "make dst prim failed.";
    return lite::RET_ERROR;
  }
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  auto fmk_attr = src_prim->GetAttr(ops::kFmkType);
  if (fmk_attr == nullptr) {
    MS_LOG(ERROR) << "attr val is nullptr.";
    return lite::RET_ERROR;
  }
  int fmk_type = GetValue<int64_t>(fmk_attr);
  if (fmk_type == converter::kFmkTypeOnnx) {
    if (cnode->inputs().size() < kNumOnnxInputSize) {
      MS_LOG(ERROR) << "onnx lstm op input size is: " << cnode->inputs().size()
                    << ", but export size is: " << kNumOnnxInputSize;
      return lite::RET_ERROR;
    }
    auto bidirectional_attr = src_prim->GetAttr("bidirectional");
    bool bidirectional = GetValue<bool>(bidirectional_attr);
    if (!bidirectional) {
      MS_LOG(ERROR) << "not support bidirectional is false.";
      return RET_ERROR;
    }
    std::vector<AnfNodePtr> new_inputs;
    dst_prim->SetAttrs({{"direction", MakeValue("bidirectional")}});
    new_inputs.insert(new_inputs.end(), cnode->inputs().begin(), cnode->inputs().begin() + kNumOnnxInputSize);
    cnode->set_inputs(new_inputs);
  } else {
    MS_LOG(ERROR) << "not support in lstm mapper.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
REGISTER_PRIMITIVE_MAPPER(kNameLSTM, LSTMMapper)
}  // namespace lite
}  // namespace mindspore
