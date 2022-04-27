/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_OPS_POOLING_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_OPS_POOLING_H_

#include <vector>
#include <string>
#include <memory>
#include "src/expression/node.h"
#include "inner/model_generated.h"
#include "src/expression/cfg.h"
#include "nnacl/pooling_parameter.h"

namespace mindspore {
namespace lite {
class PoolingM : public Node {
 public:
  PoolingM() = default;
  explicit PoolingM(const PoolingConfig &cfg);
  template <typename T>
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode);
  template <typename T>
  int UnPopulateGrad(const std::unique_ptr<schema::CNodeT> &cnode);
  std::vector<EXPR *> construct(const std::vector<EXPR *> &inputs);

 private:
  void UpdateRoundMode(const PoolingParameter *param, enum schema::RoundMode *round_mode);
  int GetMode(std::string mode);
};

class MaxPoolM : public PoolingM {
 public:
  MaxPoolM() = default;
  explicit MaxPoolM(const PoolingConfig &cfg);
  std::vector<EXPR *> Grad(EXPR *expr) override;
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
};

class AvgPoolM : public PoolingM {
 public:
  AvgPoolM() = default;
  explicit AvgPoolM(const PoolingConfig &cfg);
  std::vector<EXPR *> Grad(EXPR *expr) override;
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
};

class MaxPoolGradM : public PoolingM {
 public:
  explicit MaxPoolGradM(MaxPoolM *node);
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
};

class AvgPoolGradM : public PoolingM {
 public:
  explicit AvgPoolGradM(AvgPoolM *node);
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXPRESSION_OPS_POOLING_H_
