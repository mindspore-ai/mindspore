/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/softmax_cross_entropy.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> SoftmaxCrossEntropy::GetAxis() const { return this->primitive_->value.AsSoftmaxCrossEntropy()->axis; }

void SoftmaxCrossEntropy::SetAxis(const std::vector<int> &axis) {
  this->primitive_->value.AsSoftmaxCrossEntropy()->axis = axis;
}

#else

std::vector<int> SoftmaxCrossEntropy::GetAxis() const {
  auto fb_vector = this->primitive_->value_as_SoftmaxCrossEntropy()->axis();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
int SoftmaxCrossEntropy::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_SoftmaxCrossEntropy();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_SoftmaxCrossEntropy return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> axis;
  if (attr->axis() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->axis()->size()); i++) {
      axis.push_back(attr->axis()->data()[i]);
    }
  }
  auto val_offset = schema::CreateSoftmaxCrossEntropyDirect(*fbb, &axis);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_SoftmaxCrossEntropy, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif
}  // namespace lite
}  // namespace mindspore
