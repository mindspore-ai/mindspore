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

#ifndef MINDSPORE_CCSRC_PARALLEL_OPS_INFO_ELEMENTARY_FUNCTION_INFO_H_
#define MINDSPORE_CCSRC_PARALLEL_OPS_INFO_ELEMENTARY_FUNCTION_INFO_H_

#include <string>
#include <unordered_map>
#include <vector>
#include "ir/value.h"
#include "parallel/auto_parallel/operator_costmodel.h"
#include "parallel/ops_info/activation_info.h"
#include "parallel/strategy.h"

namespace mindspore {
namespace parallel {
class PowInfo : public ActivationOther {
 public:
  PowInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape, const PrimitiveAttrs& attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs) {}
  ~PowInfo() override = default;

 protected:
  Status InferMirrorOps() override;
};

class ExpInfo : public ActivationOther {
 public:
  ExpInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape, const PrimitiveAttrs& attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs) {}
  ~ExpInfo() override = default;
};

class LogInfo : public ActivationOther {
 public:
  LogInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape, const PrimitiveAttrs& attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs) {}
  ~LogInfo() override = default;
};

class CosInfo : public ActivationOther {
 public:
  CosInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape, const PrimitiveAttrs& attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs) {}
  ~CosInfo() override = default;
};

class ACosInfo : public ActivationOther {
 public:
  ACosInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
           const PrimitiveAttrs& attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs) {}
  ~ACosInfo() override = default;
};

class LogicalNotInfo : public ActivationOther {
 public:
  LogicalNotInfo(const std::string& name, const Shapes& inputs_shape, const Shapes& outputs_shape,
                 const PrimitiveAttrs& attrs)
      : ActivationOther(name, inputs_shape, outputs_shape, attrs) {}
  ~LogicalNotInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PARALLEL_OPS_INFO_ELEMENTARY_FUNCTION_INFO_H_
