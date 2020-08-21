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

#include "src/ops/unsqueeze.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"
#include "src/ir/tensor.h"

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
std::vector<int> Unsqueeze::GetAxis() const { return this->primitive_->value.AsUnsqueeze()->axis; }

void Unsqueeze::SetAxis(const std::vector<int> &axis) { this->primitive_->value.AsUnsqueeze()->axis = axis; }

#else
bool predicate(int n) { return n != 1; }
std::vector<int> Unsqueeze::GetAxis() const {
  auto fb_vector = this->primitive_->value_as_Unsqueeze()->axis();
  return std::vector<int>(fb_vector->begin(), fb_vector->end());
}

void Unsqueeze::SetAxis(const std::vector<int> &axis) {}
#endif

int Unsqueeze::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive_ != nullptr);
  auto input = inputs_.front();
  MS_ASSERT(input != nullptr);
  auto output = outputs_.front();
  MS_ASSERT(output != nullptr);
  if (inputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "input size is invalid";
  }
  if (outputs_.size() != kSingleNum) {
    MS_LOG(ERROR) << "output size is invalid";
  }
  output->set_data_type(input->data_type());
  output->SetFormat(input->GetFormat());
  if (!GetInferFlag()) {
    return RET_OK;
  }

  auto dims = GetAxis().data();
  auto in_shape = input->shape();
  auto in_rank = in_shape.size();
  auto dim_rank = GetAxis().size();
  std::vector<int> out_shape;
  if (dim_rank == 0) {
    for (auto d : in_shape) {
      if (d != 1) {
        out_shape.push_back(d);
      }
    }
  } else {
    auto sz = in_rank + dim_rank;
    size_t in_itr = 0;
    size_t ax_itr = 0;
    for (size_t i = 0; i < sz; i++) {
      if (ax_itr < dim_rank && dims[ax_itr] == static_cast<int>(i)) {
        out_shape.emplace_back(1);
        ax_itr++;
      } else if (ax_itr < dim_rank && dims[ax_itr] + sz == i) {
        out_shape.emplace_back(1);
        ax_itr++;
      } else {
        if (in_shape[in_itr] > 1) {
          out_shape.emplace_back(in_shape[in_itr]);
        }
        in_itr++;
      }
    }
  }
  output->set_shape(out_shape);
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
