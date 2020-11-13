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
#include "internal/src/kernel/fp32/arithmetic.h"
#include "gtest/gtest.h"

class TestInternalArithmeticFp32 : public mindspore::CommonTest {
 public:
  TestInternalArithmeticFp32() {}
};

TEST_F(TestInternalArithmeticFp32, MulTest) {
  auto mul_param = new ArithmeticParameter();
  mul_param->activation_type_ = mindspore::schema::ActivationType_NO_ACTIVATION;
  mul_param->op_parameter_.type_ = KernelType_Mul;
  mul_param->ndim_ = 4;
  Node *node = new Node();
  node->name_ = "Mul";
  node->node_type_ = NodeType::NodeType_CNode;
  node->primitive_ = reinterpret_cast<PrimitiveC *>(mul_param);
  mindspore::lite::Allocator allocator;
  /* 1x2x3x4 NHWC */
  std::vector<float> data0 = {12.216284, 3.3466918, 15.327419,  5.234958,  0.804376,   9.952188,
                              14.727955, -8.080715, 13.71383,   8.055829,  6.5845337,  -9.25232,
                              -4.24519,  11.550042, 9.262012,   1.2780352, 6.7263746,  -3.9301445,
                              3.764492,  -8.602078, -3.3558068, 13.619035, -2.6694393, 3.2008505};
  std::vector<float> data1 = {0.16771512, 0.7336843, 0.6768286, 0.4453379};
  std::vector<float> correct_out = {2.0488555,   2.4554152,  10.374036,   2.3313253, 0.13490601, 7.3017635,
                                    9.968302,    -3.5986485, 2.3000166,   5.910435,  4.4566007,  -4.120409,
                                    -0.71198255, 8.474085,   6.2687945,   0.5691575, 1.1281147,  -2.8834853,
                                    2.547916,    -3.8308315, -0.56281954, 9.992072,  -1.8067529, 1.42546};

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

  ShapeVector shape1(4);
  shape1[0] = 1;
  shape1[1] = 1;
  shape1[2] = 1;
  shape1[3] = 4;
  MSTensor in1;
  in1.data_ = data1.data();
  in1.shape_ = shape1;
  in1.data_type_ = TypeId::kNumberTypeFloat32;
  in_tensors.push_back(&in1);

  TensorPtrVector out_tensors;
  MSTensor out0;
  out0.shape_.resize(4);
  out_tensors.push_back(&out0);

  DoArithmeticInferShape(in_tensors, out_tensors, reinterpret_cast<OpParameter *>(mul_param));

  ShapeVector out_shape0(4);
  out_shape0[0] = 1;
  out_shape0[1] = 2;
  out_shape0[2] = 3;
  out_shape0[3] = 4;
  ASSERT_EQ(out_tensors.front()->shape_, out_shape0);

  out_tensors[0]->data_ = new float[correct_out.size()];
  DoArithmetic(in_tensors, out_tensors, node, &allocator);

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(out_tensors.front()->data_), correct_out.data(),
                                 correct_out.size(), 0.00001));

  delete[] out_tensors[0]->data_;
  delete node;
  delete mul_param;
}
