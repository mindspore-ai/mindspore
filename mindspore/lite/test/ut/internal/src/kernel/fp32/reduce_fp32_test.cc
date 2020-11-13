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
#include "mindspore/lite/nnacl/reduce_parameter.h"
#include "schema/ops_generated.h"
#include "internal/src/allocator.h"
#include "internal/include/model.h"
#include "internal/include/ms_tensor.h"
#include "internal/include/lite_utils.h"
#include "internal/src/kernel/fp32/reduce.h"
#include "gtest/gtest.h"

class TestInternalReduceFp32 : public mindspore::CommonTest {
 public:
  TestInternalReduceFp32() {}
};

TEST_F(TestInternalReduceFp32, ReduceSumOneAxisTest) {
  Node *node = reinterpret_cast<Node *>(new Node());
  node->name_ = "ReduceSum";
  node->node_type_ = NodeType::NodeType_CNode;

  auto params = new ReduceParameter();
  params->mode_ = mindspore::schema::ReduceMode_ReduceSum;
  params->num_axes_ = 1;
  params->axes_[0] = 1;
  params->keep_dims_ = false;
  node->primitive_ = reinterpret_cast<PrimitiveC *>(params);
  mindspore::lite::Allocator allocator;
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {72.0,  76.0,  80.0,  84.0,  88.0,  92.0,  96.0,  100.0, 104.0, 108.0, 112.0, 116.0,
                       264.0, 268.0, 272.0, 276.0, 280.0, 284.0, 288.0, 292.0, 296.0, 300.0, 304.0, 308.0};

  TensorPtrVector in_tensors;
  ShapeVector shape0(4);
  shape0[0] = 2;
  shape0[1] = 4;
  shape0[2] = 4;
  shape0[3] = 3;
  MSTensor in0;
  in0.data_ = in;
  in0.shape_ = shape0;
  in0.data_type_ = TypeId::kNumberTypeFloat32;
  in_tensors.push_back(&in0);

  TensorPtrVector out_tensors;
  MSTensor out0;
  out0.shape_.resize(3);
  out_tensors.push_back(&out0);

  DoReduceInferShape(in_tensors, out_tensors, reinterpret_cast<OpParameter *>(params));

  ShapeVector out_shape0(3);
  out_shape0[0] = 2;
  out_shape0[1] = 4;
  out_shape0[2] = 3;
  ASSERT_EQ(out_tensors.front()->shape_, out_shape0);
  out_tensors[0]->data_ = new float[24];

  DoReduce(in_tensors, out_tensors, node, &allocator);

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(out_tensors.front()->data_), correct, 24, 0.00001));
  delete out_tensors[0]->data_;
  delete node;
  delete params;
}

TEST_F(TestInternalReduceFp32, ReduceSumAllAxisTest) {
  Node *node = reinterpret_cast<Node *>(new Node());
  node->name_ = "ReduceSum";
  node->node_type_ = NodeType::NodeType_CNode;

  auto params = new ReduceParameter();
  params->mode_ = mindspore::schema::ReduceMode_ReduceSum;
  params->num_axes_ = 0;
  params->keep_dims_ = false;
  node->primitive_ = reinterpret_cast<PrimitiveC *>(params);
  mindspore::lite::Allocator allocator;
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[1] = {4560.0};

  TensorPtrVector in_tensors;
  ShapeVector shape0(4);
  shape0[0] = 2;
  shape0[1] = 4;
  shape0[2] = 4;
  shape0[3] = 3;
  MSTensor in0;
  in0.data_ = in;
  in0.shape_ = shape0;
  in0.data_type_ = TypeId::kNumberTypeFloat32;
  in_tensors.push_back(&in0);

  TensorPtrVector out_tensors;
  MSTensor out0;
  out_tensors.push_back(&out0);

  DoReduceInferShape(in_tensors, out_tensors, reinterpret_cast<OpParameter *>(params));

  ShapeVector out_shape0{};
  ASSERT_EQ(out_tensors.front()->shape_, out_shape0);
  out_tensors[0]->data_ = new float[1];

  DoReduce(in_tensors, out_tensors, node, &allocator);

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(out_tensors.front()->data_), correct, 1, 0.00001));
  delete out_tensors[0]->data_;
  delete node;
  delete params;
}

TEST_F(TestInternalReduceFp32, ReduceMeanOneAxisTest) {
  Node *node = reinterpret_cast<Node *>(new Node());
  node->name_ = "ReduceMean";
  node->node_type_ = NodeType::NodeType_CNode;

  auto params = new ReduceParameter();
  params->mode_ = mindspore::schema::ReduceMode_ReduceMean;
  params->num_axes_ = 1;
  params->axes_[0] = 1;
  params->keep_dims_ = false;
  node->primitive_ = reinterpret_cast<PrimitiveC *>(params);
  mindspore::lite::Allocator allocator;
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[24] = {18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                       66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0};
  TensorPtrVector in_tensors;
  ShapeVector shape0(4);
  shape0[0] = 2;
  shape0[1] = 4;
  shape0[2] = 4;
  shape0[3] = 3;
  MSTensor in0;
  in0.data_ = in;
  in0.shape_ = shape0;
  in0.data_type_ = TypeId::kNumberTypeFloat32;
  in_tensors.push_back(&in0);

  TensorPtrVector out_tensors;
  MSTensor out0;
  out0.shape_.resize(3);
  out_tensors.push_back(&out0);

  DoReduceInferShape(in_tensors, out_tensors, reinterpret_cast<OpParameter *>(params));

  ShapeVector out_shape0(3);
  out_shape0[0] = 2;
  out_shape0[1] = 4;
  out_shape0[2] = 3;
  ASSERT_EQ(out_tensors.front()->shape_, out_shape0);
  out_tensors[0]->data_ = new float[24];

  DoReduce(in_tensors, out_tensors, node, &allocator);

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(out_tensors.front()->data_), correct, 24, 0.00001));
  delete out_tensors[0]->data_;
  delete node;
  delete params;
}

TEST_F(TestInternalReduceFp32, ReduceMeanAllAxisTest) {
  Node *node = reinterpret_cast<Node *>(new Node());
  node->name_ = "ReduceMean";
  node->node_type_ = NodeType::NodeType_CNode;

  auto params = new ReduceParameter();
  params->mode_ = mindspore::schema::ReduceMode_ReduceMean;
  params->num_axes_ = 0;
  params->keep_dims_ = true;
  node->primitive_ = reinterpret_cast<PrimitiveC *>(params);
  mindspore::lite::Allocator allocator;
  float in[96] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                  16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
                  32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
                  48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0,
                  64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0,
                  80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0};
  float correct[1] = {47.5};

  TensorPtrVector in_tensors;
  ShapeVector shape0(4);
  shape0[0] = 2;
  shape0[1] = 4;
  shape0[2] = 4;
  shape0[3] = 3;
  MSTensor in0;
  in0.data_ = in;
  in0.shape_ = shape0;
  in0.data_type_ = TypeId::kNumberTypeFloat32;
  in_tensors.push_back(&in0);

  TensorPtrVector out_tensors;
  MSTensor out0;
  out0.shape_.resize(4);
  out_tensors.push_back(&out0);

  DoReduceInferShape(in_tensors, out_tensors, reinterpret_cast<OpParameter *>(params));

  ShapeVector out_shape0(4);
  out_shape0[0] = 1;
  out_shape0[1] = 1;
  out_shape0[2] = 1;
  out_shape0[3] = 1;
  ASSERT_EQ(out_tensors.front()->shape_, out_shape0);
  out_tensors[0]->data_ = new float[1];

  DoReduce(in_tensors, out_tensors, node, &allocator);

  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(out_tensors.front()->data_), correct, 1, 0.00001));
  delete out_tensors[0]->data_;
  delete node;
  delete params;
}
