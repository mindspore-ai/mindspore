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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/base/broadcast_to.h"
#include "ops/broadcast_to.h"
using mindspore::ops::kNameBroadcastTo;
using mindspore::schema::PrimitiveType_BroadcastTo;
namespace mindspore {
namespace lite {
OpParameter *PopulateBroadcastToOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<BroadcastToParameter *>(PopulateOpParameter<BroadcastToParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "Make OpParameter ptr failed";
    return nullptr;
  }

  auto op = dynamic_cast<ops::BroadcastTo *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to BroadcastTo failed";
    free(param);
    return nullptr;
  }

  auto dst_shape = op->get_shape();
  if (dst_shape.empty()) {
    MS_LOG(INFO) << "broadcast_to has not shape const tensor.";
  } else {
    param->shape_size_ = dst_shape.size();
    if (param->shape_size_ > MAX_SHAPE_SIZE) {
      MS_LOG(ERROR) << "Invalid shape size: " << param->shape_size_;
      free(param);
      return nullptr;
    }
    for (size_t i = 0; i < param->shape_size_; ++i) {
      param->shape_[i] = dst_shape[i];
    }
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameBroadcastTo, PrimitiveType_BroadcastTo, PopulateBroadcastToOpParameter)
}  // namespace lite
}  // namespace mindspore
