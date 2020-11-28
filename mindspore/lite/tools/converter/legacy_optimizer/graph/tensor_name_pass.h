/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef LITE_NAME_TENSOR_PASS_H
#define LITE_NAME_TENSOR_PASS_H

#include <memory>
#include "tools/converter/optimizer.h"
#include "tools/common/graph_util.h"

namespace mindspore {
namespace lite {
class TensorNamePass : public GraphPass {
 public:
  TensorNamePass() {}

  ~TensorNamePass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // LITE_NAME_TENSOR_PASS_H
