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

#include <vector>
#include <set>
#include <cmath>
#include "ir/dtype/type_id.h"
#include "src/ops/primitive_c.h"
#ifdef PRIMITIVE_WRITEABLE
#include "schema/inner/model_generated.h"
#else
#include "schema/model_generated.h"
#endif

#ifndef LITE_MINDSPORE_LITE_C_OPS_REVERSE_SEQUENCE_H_
#define LITE_MINDSPORE_LITE_C_OPS_REVERSE_SEQUENCE_H_

namespace mindspore {
namespace lite {
class ReverseSequence : public PrimitiveC {
 public:
  explicit ReverseSequence(OriginPrimitive *primitive) : PrimitiveC(primitive) {}

  int InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) override;
  int GetSeqAxis() const;
  int GetBatchAxis() const;
  std::vector<int> GetSeqLengths() const;
  void SetSeqAxis(int seq_axis);
  void SetBatchAxis(int batch_axis);
  void SetSeqLengths(const std::vector<int> &seq_lengths);
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_REVERSE_SEQUENCE_H_
