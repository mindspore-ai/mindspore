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

#ifndef DPICO_PARSER_DETECTION_OUTPUT_PARAM_HOLDER_H_
#define DPICO_PARSER_DETECTION_OUTPUT_PARAM_HOLDER_H_

#include <utility>
#include <memory>
#include "ir/anf.h"
#include "op/detection_output_operator.h"

namespace mindspore {
namespace lite {
class DetectionOutputParamHolder : public Value {
 public:
  explicit DetectionOutputParamHolder(mapper::DetectionOutputParam param) : detection_output_param_(std::move(param)) {}
  ~DetectionOutputParamHolder() override = default;

  MS_DECLARE_PARENT(DetectionOutputParamHolder, Value);

  bool operator==(const Value &rhs) const override {  // unused
    return rhs.isa<DetectionOutputParamHolder>();
  }
  const mapper::DetectionOutputParam &GetDetectionOutputParam() const { return detection_output_param_; }

 private:
  mapper::DetectionOutputParam detection_output_param_;
};
using DetectionOutputParamHolderPtr = std::shared_ptr<DetectionOutputParamHolder>;
}  // namespace lite
}  // namespace mindspore
#endif  // DPICO_PARSER_DETECTION_OUTPUT_PARAM_HOLDER_H_
