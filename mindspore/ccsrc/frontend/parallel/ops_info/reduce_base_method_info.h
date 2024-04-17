/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_BASE_METHOD_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_BASE_METHOD_INFO_H_

#include <vector>
#include <memory>
#include <string>

#include "frontend/parallel/ops_info/reduce_method_info.h"

namespace mindspore {
namespace parallel {
class ReduceBaseMethod : public ReduceMethod {
 public:
  ReduceBaseMethod(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                   const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : ReduceMethod(name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ReduceBaseMethod() override = default;

 protected:
  Status InferMirrorOps() override;
  std::vector<int64_t> reduce_dim() override;
  Status GetAttrs() override;
};

class ReduceMaxInfo : public ReduceBaseMethod {
 public:
  ReduceMaxInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceMaxCost>()) {
    reduce_method_ = REDUCE_OP_MAX;
  }

  ~ReduceMaxInfo() override = default;
};

class ReduceMeanInfo : public ReduceBaseMethod {
 public:
  ReduceMeanInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceMeanCost>()) {}

  ~ReduceMeanInfo() override = default;

 protected:
  Status InferForwardCommunication() override;
};

class ReduceSumInfo : public ReduceBaseMethod {
 public:
  ReduceSumInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceSumCost>()) {
    reduce_method_ = REDUCE_OP_SUM;
  }

  ~ReduceSumInfo() override = default;
};

class ReduceAnyInfo : public ReduceBaseMethod {
 public:
  ReduceAnyInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceSumCost>()) {
    reduce_method_ = REDUCE_OP_ANY;
  }
  ~ReduceAnyInfo() override = default;

 protected:
  Status InferForwardCommunication() override;
  ForwardOp CreateForwardOp(const std::vector<Group> &forward_group) const;
};

class ReduceMinInfo : public ReduceBaseMethod {
 public:
  ReduceMinInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceMinCost>()) {
    reduce_method_ = REDUCE_OP_MIN;
  }

  ~ReduceMinInfo() override = default;
};

class ReduceProdInfo : public ReduceBaseMethod {
 public:
  ReduceProdInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ReduceBaseMethod(name, inputs_shape, outputs_shape, attrs, std::make_shared<ReduceProdCost>()) {
    reduce_method_ = REDUCE_OP_PROD;
  }

  ~ReduceProdInfo() override = default;
};

class ReduceAllInfo : public ReduceAnyInfo {
 public:
  ReduceAllInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : ReduceAnyInfo(name, inputs_shape, outputs_shape, attrs) {
    reduce_method_ = REDUCE_OP_ALL;
  }

  ~ReduceAllInfo() override = default;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_REDUCE_BASE_METHOD_INFO_H_
