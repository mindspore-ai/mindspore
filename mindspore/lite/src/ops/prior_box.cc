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
namespace {
constexpr int kPriorBoxPoints = 4;
constexpr int kPriorBoxN = 1;
constexpr int kPriorBoxW = 1;
constexpr int kPriorBoxC = 2;
}  // namespace

int PriorBox::InferShape(std::vector<tensor::Tensor *> inputs_, std::vector<tensor::Tensor *> outputs_) {
  auto param = GetAttrbute();
  MS_ASSERT(param != nullptr);
  std::vector<float> different_aspect_ratios{1.0f};
  auto aspect_ratios = param->aspect_ratios();
  MS_ASSERT(aspect_ratios != nullptr);
  for (auto i = 0; i < aspect_ratios->size(); i++) {
    float ratio = (*aspect_ratios)[i];
    bool exist = std::any_of(different_aspect_ratios.begin(), different_aspect_ratios.end(), [&](float v) {
      return abs(ratio - v) < 1e-6;
    });
    if (!exist) {
      different_aspect_ratios.emplace_back(ratio);
      if (param->flip()) {
        different_aspect_ratios.emplace_back(1.0f / ratio);
      }
    }
  }
  int32_t num_priors_box = param->min_sizes()->size() * different_aspect_ratios.size() + param->max_sizes()->size();
  auto input = inputs_.at(0);
  MS_ASSERT(input != nullptr);
  int32_t h = input->Height() * input->Width() * num_priors_box * kPriorBoxPoints;

  std::vector<int> output_shape{kPriorBoxN, h, kPriorBoxW, kPriorBoxC};
  auto output = outputs_.at(0);
  MS_ASSERT(output != nullptr);

  output->set_shape(output_shape);
  output->set_data_type(kNumberTypeFloat32);
  output->SetFormat(input->GetFormat());
  return RET_OK;
}
}  // namespace mindspore::lite
