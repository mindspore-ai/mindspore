/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <cmath>
#include <memory>
#include "common/common_test.h"
#include "internal/include/model.h"
#include "internal/include/lite_session.h"
#include "internal/include/context.h"
#include "internal/include/errorcode.h"
#include "internal/include/ms_tensor.h"
#include "nnacl/conv_parameter.h"

namespace mindspore {
class InferTest : public mindspore::CommonTest {
 public:
  InferTest() {}
};

TEST_F(InferTest, TestSession) {
//  Model model;
//  Node *node = (Node *)malloc(sizeof(Node));
//  node->name_ = "conv2d";
//  uint32_t index = model.all_tensors_.size();
//  node->input_indices_ = {index};
//  MSTensor *in = CreateTensor(kNumberTypeFloat32, {3, 3, 24, 24});
//  model.all_tensors_.emplace_back(in);
//
//  index = model.all_tensors_.size();
//  node->output_indices_ = {index};
//  MSTensor *out = CreateTensor(kNumberTypeFloat32, {3, 3, 24, 24});
//  model.all_tensors_.emplace_back(out);
//
//  ConvParameter *param = (ConvParameter *)malloc(sizeof(ConvParameter));
//  param->kernel_w_ = 3;
//  // todo: fill other param fields
//  node->primitive_ = (PrimitiveC *)param;
//  model.nodes_.push_back(node);
//
//  LiteSession session;
//  session.CompileGraph(&model);
//  TensorPtrVector invec = session.GetInputs();
//  ASSERT_EQ(invec.size(), 1);
//  // todo: fill inputs data
//  session.RunGraph();
//  TensorPtrVector outvec = session.GetOutputs();
//  ASSERT_EQ(outvec.size(), 1);
}

}  // namespace mindspore
