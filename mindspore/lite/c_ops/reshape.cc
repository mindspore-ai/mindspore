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

#include "c_ops/reshape.h"
#include <algorithm>

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
int Reshape::GetFormat() const { return this->primitive->value.AsReshape()->format; }
std::vector<long> Reshape::GetShape() const { return this->primitive->value.AsReshape()->shape; }

void Reshape::SetFormat(int format) { this->primitive->value.AsReshape()->format = format; }
void Reshape::SetShape(const std::vector<long> &shape) { this->primitive->value.AsReshape()->shape = shape; }

#else

int Reshape::GetFormat() const { return this->primitive->value_as_Reshape()->format(); }
std::vector<long> Reshape::GetShape() const {
  auto fb_vector = this->primitive->value_as_Reshape()->shape();
  return std::vector<long>(fb_vector->begin(), fb_vector->end());
}

void Reshape::SetFormat(int format) {}
void Reshape::SetShape(const std::vector<long> &shape) {}
#endif

int Reshape::CalNewShape(const lite::tensor::Tensor *in_tensor, std::vector<int> *out_shape) const {
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
        return 1;
      }
    } else if (out_shape->at(i) < 0) {
      MS_LOG(ERROR) << "output shape dim should be non-negative";
      return 1;
    } else if (out_shape->at(i) == 0) {
      out_shape->at(i) = in_tensor->shape().at(i);
      out_shapeSize *= out_shape->at(i);
    } else {
      out_shapeSize *= out_shape->at(i);
    }
  }

  if (inferIndex == -1 && out_shapeSize != in_shape_size) {
    MS_LOG(ERROR) << "output shapeSize: " << out_shapeSize << " should be equal to input shapeSize: " << in_shape_size;
    return 1;
  }
  if (inferIndex != -1) {
    out_shape->at(inferIndex) = in_shape_size / out_shapeSize;
  }
  return 0;
}

template <typename T>
void CalShape(const T *data, const std::vector<lite::tensor::Tensor *> &inputs, std::vector<int> *out_shape,
              int shape_size) {
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

int Reshape::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);

  std::vector<int> out_shape;
  if (inputs_.size() == kDoubleNum) {
    auto shape_tensor = inputs_.at(1);
    if (shape_tensor->Data() == nullptr) {
      MS_LOG(INFO) << "Do infer shape in runtime.";
      return 1;
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
        return 1;
      }
    }
  } else if (inputs_.size() == kSingleNum) {
    std::copy(GetShape().begin(), GetShape().end(), std::back_inserter(out_shape));
  } else {
    MS_LOG(ERROR) << "inputs tensor size invalid.";
    return 1;
  }

  auto ret = CalNewShape(inputs_.front(), &out_shape);
  if (ret != 0) {
    MS_LOG(ERROR) << "CalNewShape error";
    return ret;
  }

  output->set_shape(out_shape);
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());

  return 0;
}
}  // namespace mindspore
