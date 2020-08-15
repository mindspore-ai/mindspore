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

#include "c_ops/squeeze.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Squeeze::GetAxis() const { return this->primitive->value.AsSqueeze()->axis; }

void Squeeze::SetAxis(const std::vector<int> &axis) { this->primitive->value.AsSqueeze()->axis = axis; }

#else

std::vector<int> Squeeze::GetAxis() const {
  auto fb_vector = this->primitive->value_as_Squeeze()->axis();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void Squeeze::SetAxis(const std::vector<int> &axis) {}
#endif
namespace {
constexpr int kSqueezeInputNum = 1;
constexpr int kSqueezeOutputNum = 1;
}  // namespace
int Squeeze::InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
  if (kSqueezeInputNum != inputs_.size()) {
    MS_LOG(ERROR) << "Add should has " << kSqueezeInputNum << " inputs";
    return -1;
  }
  if (kSqueezeOutputNum != outputs_.size()) {
    MS_LOG(ERROR) << "Add should has " << kSqueezeOutputNum << " outputs";
    return -1;
  }
  auto *in_tensor = inputs_.front();
  auto in_shape = in_tensor->shape();
  std::vector<int> out_shape;

  // todo: getAxis
  auto axis = GetAxis();
  std::vector<int> axes_;
  for (auto iter = axis.begin(); iter != axis.end(); iter++) {
    axes_.push_back(*iter);
  }

  if (axes_.size() == 0) {
    for (int i = 0; i < in_shape.size(); i++) {
      if (in_shape[i] != 1) {
        out_shape.push_back(in_shape[i]);
      }
    }
  } else {
    int axisIdx = 0;
    for (int i = 0; i < in_shape.size(); i++) {
      if (axisIdx < axes_.size() && axes_[axisIdx] == i) {
        MS_ASSERT(in_shape[i] == 1);
        axisIdx++;
        continue;
      } else {
        out_shape.push_back(in_shape[i]);
      }
    }
  }

  outputs_.front()->set_shape(out_shape);
  outputs_.front()->set_data_type(in_tensor->data_type());
  outputs_.front()->SetFormat(in_tensor->GetFormat());

  return 0;
}
}  // namespace mindspore
