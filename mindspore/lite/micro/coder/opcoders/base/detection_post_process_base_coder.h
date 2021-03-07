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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_BASE_DETECTION_POST_PROCESS_BASE_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_BASE_DETECTION_POST_PROCESS_BASE_CODER_H_

#include <string>
#include <vector>
#include <utility>
#include <memory>
#include "coder/opcoders/op_coder.h"
#include "nnacl/detection_post_process_parameter.h"
#include "coder/opcoders/serializers/serializer.h"

namespace mindspore::lite::micro {

class DetectionPostProcessBaseCoder : public OperatorCoder {
 public:
  DetectionPostProcessBaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~DetectionPostProcessBaseCoder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 protected:
  int AllocateBuffer();
  virtual int GetInputData(CoderContext *const context, Serializer *const coder) = 0;
  virtual int MallocInputsBuffer() = 0;

  int num_boxes_{0};
  int num_classes_with_bg_{0};
  float *input_boxes_{nullptr};
  float *input_scores_{nullptr};
  DetectionPostProcessParameter *params_{nullptr};
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_BASE_DETECTION_POST_PROCESS_BASE_CODER_H_
