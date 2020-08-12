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

#include "src/ops/ops.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore::lite {
int Reshape::CalNewShape(const tensor::Tensor *in_tensor, std::vector<int> *out_shape) const {
  size_t in_shape_size = 1;
  for (size_t i = 0; i < in_tensor->shape().size(); i++) {
    in_shape_size *= in_tensor->shape()[i];
  }

  int64_t inferIndex = -1;
  size_t out_shapeSize = 1;
  for (size_t i = 0; i < out_shape->size(); i++) {
    if (out_shape->at(i) == -1) {
      if (inferIndex == -1) {
        inferIndex = i;
      } else {
        MS_LOG(ERROR) << "output shape should has no more than one dim which need infer";
        return RET_INFER_ERR;
      }
    } else if (out_shape->at(i) < 0) {
      MS_LOG(ERROR) << "output shape dim should be non-negative";
      return RET_INFER_ERR;
    } else if (out_shape->at(i) == 0) {
      out_shape->at(i) = in_tensor->shape().at(i);
      out_shapeSize *= out_shape->at(i);
    } else {
      out_shapeSize *= out_shape->at(i);
    }
  }

  if (inferIndex == -1 && out_shapeSize != in_shape_size) {
    MS_LOG(ERROR) << "output shapeSize: " << out_shapeSize << " should be equal to input shapeSize: " << in_shape_size;
    return RET_INFER_ERR;
  }
  if (inferIndex != -1) {
    out_shape->at(inferIndex) = in_shape_size / out_shapeSize;
  }
  return RET_OK;
}

template <typename T>
void CalShape(const T *data, const std::vector<tensor::Tensor *> &inputs, std::vector<int> *out_shape, int shape_size) {
  int input_count = inputs[0]->ElementsNum();

  int index = 0;
  int size = 1;
  for (size_t i = 0; i < shape_size; i++) {
    if (data[i] == -1) {
      index = i;
    } else {
      size *= data[i];
    }
    out_shape->push_back(data[i]);
  }
  if (data[index] == -1) {
    (*out_shape)[index] = input_count / size;
  }
}

int Reshape::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  auto reshape_prim = this->primitive->value_as_Reshape();
  MS_ASSERT(reshape_prim != nullptr);

  std::vector<int> out_shape;
  if (inputs_.size() == kDoubleNum) {
    auto shape_tensor = inputs_.at(1);
    if (shape_tensor->Data() == nullptr) {
      MS_LOG(INFO) << "Do infer shape in runtime.";
      return RET_INFER_INVALID;
    }
    size_t shape_size = shape_tensor->ElementsNum();
    switch (shape_tensor->data_type()) {
      case kNumberTypeInt8: {
        auto data = reinterpret_cast<int8_t *>(shape_tensor->Data());
        CalShape<int8_t>(data, inputs_, &out_shape, shape_size);
      } break;
      case kNumberTypeInt32: {
        auto data = reinterpret_cast<int32_t *>(shape_tensor->Data());
        CalShape<int32_t>(data, inputs_, &out_shape, shape_size);
      } break;
      case kNumberTypeInt64: {
        auto data = reinterpret_cast<int64_t *>(shape_tensor->Data());
        CalShape<int64_t>(data, inputs_, &out_shape, shape_size);
      } break;
      case kNumberTypeFloat: {
        auto data = reinterpret_cast<float *>(shape_tensor->Data());
        CalShape<float>(data, inputs_, &out_shape, shape_size);
      } break;
      case kNumberTypeUInt32: {
        auto data = reinterpret_cast<uint32_t *>(shape_tensor->Data());
        CalShape<uint32_t>(data, inputs_, &out_shape, shape_size);
      } break;
      default: {
        MS_LOG(ERROR) << "Reshape weight tensor has unsupported dataType: " << shape_tensor->data_type();
        return RET_INFER_ERR;
      }
    }
  } else if (inputs_.size() == kSingleNum) {
    std::copy(reshape_prim->shape()->begin(), reshape_prim->shape()->end(), std::back_inserter(out_shape));
  } else {
    MS_LOG(ERROR) << "inputs tensor size invalid.";
    return RET_INFER_ERR;
  }

  auto ret = CalNewShape(inputs_.front(), &out_shape);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CalNewShape error";
    return ret;
  }

  output->set_shape(out_shape);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());

  return RET_OK;
}
}  // namespace mindspore::lite
