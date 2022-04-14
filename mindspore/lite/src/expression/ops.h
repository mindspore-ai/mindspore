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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_OPS_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_OPS_H_

#include <vector>
#include <string>
#include <set>
#include "include/api/net.h"
#include "src/expression/cfg.h"
#include "src/expression/net.h"
#include "inner/model_generated.h"

namespace mindspore {
namespace lite {
class InputM : public Node {
 public:
  explicit InputM(const schema::Tensor *tensor);
  explicit InputM(const std::vector<int> &dims, TypeId data_type = kNumberTypeFloat32, int fmt = NHWC);
  Param *data() override { return &data_; }

 private:
  void SetUp(const std::vector<int> &dims, TypeId data_type, int fmt);
  Param data_;
};
namespace NN {
Node *Conv2D(const ConvConfig &cfg);
Node *Relu();
Node *Dense(const DenseConfig &cfg);
Node *Flatten();
Node *Input(const std::vector<int> &dims, TypeId data_type = kNumberTypeFloat32, int fmt = NHWC);
Node *Add();
Node *Sub();
Node *Div();
Node *Mul();
Node *Neg();
Node *SoftmaxCrossEntropy();
Net *Sequential();
Node *Adam(std::set<Node *> &&learn, const AdamConfig &cfg);

Node *Softmax(int axis = -1);
Node *BatchNorm2D(int outp, float momentum = 0.1, float epsilon = 1e-5f);
Node *Sigmoid();
Node *DropOut(float ration = 0.5);
Node *ReLU6();
Node *Reshape(const std::vector<int> &shape);
Node *ReduceMean(bool keep_dims, const std::vector<int> &dims);
Node *ReduceSum(bool keep_dims, const std::vector<int> &dims);
Node *Tile(const std::vector<int> &multiples);
Node *MaxPool2D(const PoolingConfig &cfg);
Node *AvgPool2D(const PoolingConfig &cfg);
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXPRESSION_OPS_H_
