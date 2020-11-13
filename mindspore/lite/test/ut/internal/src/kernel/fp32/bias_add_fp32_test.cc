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
#include "test/common/common_test.h"
#include "src/common/file_utils.h"
#include "schema/ops_generated.h"
#include "mindspore/lite/nnacl/fp32/arithmetic.h"
#include "internal/src/allocator.h"
#include "internal/include/model.h"
#include "internal/include/ms_tensor.h"
#include "internal/include/lite_utils.h"
#include "internal/src/kernel/fp32/bias_add.h"
#include "gtest/gtest.h"

class TestInternalBiasAddFp32 : public mindspore::CommonTest {
 public:
  TestInternalBiasAddFp32() {}
};

TEST_F(TestInternalBiasAddFp32, BiasAddTest) {
  auto bias_add_param = new ArithmeticParameter();
  bias_add_param->activation_type_ = mindspore::schema::ActivationType_NO_ACTIVATION;
  bias_add_param->op_parameter_.type_ = KernelType_BiasAdd;
  Node *node = new Node();
  node->name_ = "BiasAdd";
  node->node_type_ = NodeType::NodeType_CNode;
  node->primitive_ = reinterpret_cast<PrimitiveC *>(bias_add_param);
  mindspore::lite::Allocator allocator;
  std::vector<float> data0 = {12.216284, 3.3466918, 15.327419,  5.234958,  0.804376,   9.952188,
                              14.727955, -8.080715, 13.71383,   8.055829,  6.5845337,  -9.25232,
                              -4.24519,  11.550042, 9.262012,   1.2780352, 6.7263746,  -3.9301445,
                              3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  std::vector<float> data1 = {0.16771512, 0.7336843, 0.6768286, 0.4453379};
  std::vector<float> correct_out = {12.3839989, 4.0803761,  16.0042477, 5.6802959,  0.9720911,  10.6858721,
                                    15.4047832, -7.6353774, 13.8815451, 8.7895136,  7.2613621,  -8.8069820,
                                    -4.0774751, 12.2837267, 9.9388399,  1.7233731,  6.8940897,  -3.1964602,
                                    4.4413204,  -8.1567402, -3.1880918, 14.3527193, -1.9926107, 3.6461883};
  TensorPtrVector in_tensors;
  ShapeVector shape0(4);
  shape0[0] = 1;
  shape0[1] = 2;
  shape0[2] = 3;
  shape0[3] = 4;
  MSTensor in0;
  in0.data_ = data0.data();
  in0.shape_ = shape0;
  in0.data_type_ = TypeId::kNumberTypeFloat32;
  in_tensors.push_back(&in0);

  ShapeVector shape1{4};
  MSTensor in1;
  in1.data_ = data1.data();
  in1.shape_ = shape1;
  in1.data_type_ = TypeId::kNumberTypeFloat32;
  in_tensors.push_back(&in1);

  TensorPtrVector out_tensors;
  MSTensor out0;
  out_tensors.push_back(&out0);

  DoBiasAddInferShape(in_tensors, out_tensors, reinterpret_cast<OpParameter *>(bias_add_param));

  ShapeVector out_shape0(4);
  out_shape0[0] = 1;
  out_shape0[1] = 2;
  out_shape0[2] = 3;
  out_shape0[3] = 4;
  ASSERT_EQ(out_tensors.front()->shape_, out_shape0);

  out_tensors[0]->data_ = new float[correct_out.size()];
  DoBiasAdd(in_tensors, out_tensors, node, &allocator);

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(out_tensors.front()->data_), correct_out.data(),
                                 correct_out.size(), 0.00001));

  delete out_tensors[0]->data_;
  delete node;
  delete bias_add_param;
}
