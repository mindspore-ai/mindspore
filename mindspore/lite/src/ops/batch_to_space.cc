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

#include "src/ops/batch_to_space.h"
#include "src/common/common.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"

#ifndef PRIMITIVE_WRITEABLE
#include "src/ops/ops_register.h"
#endif

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> BatchToSpace::GetBlockShape() const { return this->primitive_->value.AsBatchToSpace()->blockShape; }
std::vector<int> BatchToSpace::GetCrops() const { return this->primitive_->value.AsBatchToSpace()->crops; }

void BatchToSpace::SetBlockShape(const std::vector<int> &block_shape) {
  this->primitive_->value.AsBatchToSpace()->blockShape = block_shape;
}
void BatchToSpace::SetCrops(const std::vector<int> &crops) { this->primitive_->value.AsBatchToSpace()->crops = crops; }

#else
int BatchToSpace::UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) {
  MS_ASSERT(nullptr != primitive);
  MS_ASSERT(nullptr != fbb);
  auto attr = primitive->value_as_BatchToSpace();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "value_as_BatchToSpace return nullptr";
    return RET_ERROR;
  }
  std::vector<int32_t> blockShape;
  if (attr->blockShape() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->blockShape()->size()); i++) {
      blockShape.push_back(attr->blockShape()->data()[i]);
    }
  }
  std::vector<int32_t> crops;
  if (attr->crops() != nullptr) {
    for (int i = 0; i < static_cast<int>(attr->crops()->size()); i++) {
      crops.push_back(attr->crops()->data()[i]);
    }
  }
  auto val_offset = schema::CreateBatchToSpaceDirect(*fbb, &blockShape, &crops);
  auto prim_offset = schema::CreatePrimitive(*fbb, schema::PrimitiveType_BatchToSpace, val_offset.o);
  fbb->Finish(prim_offset);
  return RET_OK;
}
std::vector<int> BatchToSpace::GetBlockShape() const {
  auto fb_vector = this->primitive_->value_as_BatchToSpace()->blockShape();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}
std::vector<int> BatchToSpace::GetCrops() const {
  auto fb_vector = this->primitive_->value_as_BatchToSpace()->crops();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

PrimitiveC *BatchToSpaceCreator(const schema::Primitive *primitive) {
  return PrimitiveC::NewPrimitiveC<BatchToSpace>(primitive);
}
Registry BatchToSpaceRegistry(schema::PrimitiveType_BatchToSpace, BatchToSpaceCreator);
#endif

namespace {
constexpr int kBatchToSpaceOutputNum = 1;
constexpr int kBatchToSpaceInputNum = 1;
constexpr int kBlockShapeSize = 2;
constexpr int kCropsSize = 4;
}  // namespace

int BatchToSpace::InferShape(std::vector<lite::Tensor *> inputs, std::vector<lite::Tensor *> outputs) {
  MS_ASSERT(this->primitive_ != nullptr);
  if (outputs.size() != kBatchToSpaceOutputNum || inputs.size() != kBatchToSpaceInputNum) {
    MS_LOG(ERROR) << "Invalid output/input size! output size: " << outputs.size() << ",input size: " << inputs.size();
    return RET_PARAM_INVALID;
  }

  auto input = inputs.at(0);
  if (input->format() != schema::Format::Format_NHWC) {
    MS_LOG(ERROR) << "batch_to_space only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  outputs[0]->set_format(input->format());
  outputs[0]->set_data_type(input->data_type());
  if (!infer_flag()) {
    return RET_INFER_INVALID;
  }
  auto input_shape = input->shape();
  if (input_shape.size() != kQuadrupleNum) {
    MS_LOG(ERROR) << "input shape dimension size should == " << kQuadrupleNum;
    return RET_PARAM_INVALID;
  }

  auto block_shape = GetBlockShape();
  if (block_shape.size() != kBlockShapeSize) {
    MS_LOG(ERROR) << "Block shape size should be " << kBlockShapeSize;
    return RET_PARAM_INVALID;
  }
  auto crops = GetCrops();
  if (crops.size() != kCropsSize) {
    MS_LOG(ERROR) << "Crops size should be " << kCropsSize;
    return RET_PARAM_INVALID;
  }
  int mul_block_shape = 1;

  for (size_t i = 0; i < kBlockShapeSize; ++i) {
    if (block_shape[i] <= 0) {
      MS_LOG(ERROR) << "Input block_shape should > 0!";
      return RET_PARAM_INVALID;
    }
    if (input_shape[NHWC_N] % block_shape[i]) {
      MS_LOG(ERROR) << "Dimension n " << input_shape[NHWC_N] << " can not divide block_shape[" << i << "] "
                    << block_shape[i];
      return 1;
    }
    mul_block_shape *= block_shape[i];
  }

  if (input_shape[NHWC_N] < mul_block_shape) {
    MS_LOG(ERROR) << "Dimension n " << input_shape[NHWC_N] << " < product of block shape!";
    return RET_PARAM_INVALID;
  }
  for (size_t i = 0; i < kCropsSize; ++i) {
    if (crops[i] < 0) {
      MS_LOG(ERROR) << "Input crops should >= 0";
      return RET_PARAM_INVALID;
    }
  }
  std::vector<int32_t> output_shape(input_shape.size());
  output_shape[NHWC_N] = input_shape[NHWC_N] / mul_block_shape;
  output_shape[NHWC_H] = input_shape[NHWC_H] * block_shape[0] - crops[0] - crops[1];
  output_shape[NHWC_W] = input_shape[NHWC_W] * block_shape[1] - crops[2] - crops[3];
  output_shape[NHWC_C] = input_shape[NHWC_C];

  outputs[0]->set_shape(output_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
