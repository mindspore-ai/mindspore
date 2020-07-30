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
int Unsqueeze::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  MS_ASSERT(this->primitive != nullptr);
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
  auto unsqueeze_prim = this->primitive->value_as_Unsqueeze();
  auto dims = unsqueeze_prim->axis()->data();
  auto in_shape = input->shape();
  auto in_rank = in_shape.size();
  auto dim_rank = unsqueeze_prim->axis()->size();
  std::vector<int> out_shape;

  if (dim_rank == 0) {
    for (auto d : in_shape) {
      if (d != 1) {
        out_shape.push_back(d);
      }
    }
  } else {
    auto sz = in_rank + dim_rank;
    int in_itr = 0;
    int ax_itr = 0;
    for (int i = 0; i < sz; i++) {
      if (ax_itr < dim_rank && dims[ax_itr] == i) {
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

  output->SetFormat(input->GetFormat());
  output->set_shape(out_shape);
  output->set_data_type(input->data_type());
  return RET_OK;
}
}  // namespace mindspore::lite
