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
#include "src/ops/apply_momentum.h"
namespace mindspore {
namespace lite {

#ifdef PRIMITIVE_WRITEABLE

#else
int ApplyMomentum::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_ApplyMomentum();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_ApplyMomentum return nullptr";
    return RET_ERROR;
  }
  auto val_offset = schema::CreateApplyMomentum(*fbb);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_ActivationGrad, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
#endif

int ApplyMomentum::InferShape(std::vector<Tensor *> inputs, std::vector<Tensor *> outputs) {
  if (5 != inputs.size()) {
    MS_LOG(ERROR) << "ApplyMomentum should have at 5 input tensors";
    return RET_ERROR;
  }
  // if (outputs.empty()) {
  //  MS_LOG(ERROR) << "ApplyMomentumCPUKernel error input output size!";
  //  return RET_ERROR;
  // }

  if (inputs[0]->ElementsNum() != inputs[1]->ElementsNum() || inputs[0]->ElementsNum() != inputs[3]->ElementsNum() ||
      inputs[2]->ElementsNum() != 1 || inputs[4]->ElementsNum() != 1) {
    MS_LOG(ERROR) << "error input data size!";
    return RET_ERROR;
  }
  if (!outputs.empty()) {
    auto *out = outputs.front();
    MS_ASSERT(out != nullptr);
    out->set_data_type(inputs[0]->data_type());
    out->SetFormat(inputs[0]->GetFormat());
  }

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
