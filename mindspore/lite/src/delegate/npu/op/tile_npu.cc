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

#include "src/delegate/npu/op/tile_npu.h"
#include <memory>
#include "src/delegate/npu/npu_converter_utils.h"

namespace mindspore {
int TileNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                         const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors.size() != 2) {
    return RET_ERROR;
  }
  auto multiple_tensor = in_tensors[1];
  if (multiple_tensor.ElementNum() > NPU_SHAPE_SIZE || multiple_tensor.Data() == nullptr) {
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int TileNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                    const std::vector<mindspore::MSTensor> &out_tensors) {
  tile_ = new (std::nothrow) hiai::op::Tile(name_);
  if (tile_ == nullptr) {
    MS_LOG(ERROR) << "New tile npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int TileNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors,
                            const std::vector<ge::Operator *> &npu_inputs) {
  tile_->set_input_x(*npu_inputs[0]);

  std::vector<int> multiples;
  if (in_tensors[1].Data() == nullptr) {
    return RET_ERROR;
  }
  auto multiple_data = reinterpret_cast<const int *>(in_tensors[1].Data().get());
  for (int i = 0; i < in_tensors[1].ElementNum(); ++i) {
    multiples.push_back(multiple_data[i]);
  }
  ge::TensorDesc multiple_tensor_desc(ge::Shape({static_cast<int64_t>(multiples.size())}), ge::FORMAT_NCHW,
                                      ge::DT_INT32);
  ge::TensorPtr multiple_tensor = std::make_shared<hiai::Tensor>(multiple_tensor_desc);
  multiple_tensor->SetData(reinterpret_cast<uint8_t *>(multiples.data()), multiples.size() * sizeof(int));
  multiple_ = new hiai::op::Const(name_ + "multiples");
  if (multiple_ == nullptr) {
    MS_LOG(ERROR) << "New multiple const for tile npu operator failed.";
    return RET_ERROR;
  }
  multiple_->set_attr_value(multiple_tensor);
  tile_->set_input_multiples(*multiple_);
  return RET_OK;
}

ge::Operator *TileNPUOp::GetNPUOp() { return this->tile_; }

TileNPUOp::~TileNPUOp() {
  if (tile_ != nullptr) {
    delete tile_;
    tile_ = nullptr;
  }
  if (multiple_ != nullptr) {
    delete multiple_;
    multiple_ = nullptr;
  }
}
}  // namespace mindspore
