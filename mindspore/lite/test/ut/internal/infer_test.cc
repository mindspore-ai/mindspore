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
#include "nnacl/op_base.h"

namespace mindspore {
class InferTest : public mindspore::CommonTest {
 public:
  InferTest() {}
};

TEST_F(InferTest, TestSession) {
  Model model;
  Node *node = reinterpret_cast<Node *>(malloc(sizeof(Node)));

  node->name_ = "Neg";
  node->node_type_ = NodeType::NodeType_CNode;
  PrimitiveC *prim = reinterpret_cast<PrimitiveC *>(malloc(sizeof(PrimitiveC)));
  prim->type_ = KernelType::Neg;
  node->input_indices_.push_back(0);
  node->output_indices_.push_back(1);

  MSTensor *in = CreateTensor(kNumberTypeFloat32, {1, 1, 1, 10});
  model.all_tensors_.push_back(in);
  model.input_indices_.push_back(0);

  MSTensor *out = CreateTensor(kNumberTypeFloat32, {1, 1, 1, 10});
  model.all_tensors_.emplace_back(out);
  node->output_indices_.push_back(1);

  LiteSession session;
  session.CompileGraph(&model);
  TensorPtrVector invec = session.GetInputs();
  ASSERT_EQ(invec.size(), 1);
  constexpr int kOutSize = 10;
  float expect_out[kOutSize];
  for (int i = 0; i < kOutSize; ++i) {
    *(reinterpret_cast<float *>(in->data_) + i) = i + 1;
    expect_out[i] = -(i + 1);
  }
  session.RunGraph();
  TensorPtrVector outvec = session.GetOutputs();
  ASSERT_EQ(outvec.size(), 1);
  for (int i = 0; i < kOutSize; ++i) {
    std::cout << *(reinterpret_cast<float *>(outvec.at(0)->data_)+ i) << " ";
  }
  std::cout << "\n";
  CompareOutputData(reinterpret_cast<float *>(outvec.at(0)->data_), expect_out, kOutSize, 0.000001);
}

}  // namespace mindspore
