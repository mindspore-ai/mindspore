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
#include "common/common_test.h"
#include "src/tensorlist.h"
#include "mindspore/lite/src/runtime/infer_manager.h"

namespace mindspore::lite {

class InferManagerTest : public mindspore::CommonTest {
 public:
  InferManagerTest() {}
};

// PrimitiveType_TensorListGetItem
TEST_F(InferManagerTest, InferManagerTest0) {
  Tensor *tensor0 = new (std::nothrow) Tensor;
  std::vector<int> tensor0_shape = {4, 6, 5};
  tensor0->set_shape(tensor0_shape);
  tensor0->set_data_type(kNumberTypeInt32);
  Tensor *tensor1 = new (std::nothrow) Tensor;
  std::vector<int> tensor1_shape = {1, 2};
  tensor1->set_shape(tensor1_shape);
  std::vector<int> tensor1_data = {-1, 5};
  tensor1->set_data(tensor1_data.data());
  tensor1->set_data_type(kNumberTypeInt32);
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(tensor0);
  inputs.push_back(tensor1);

  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  parameter->type_ = mindspore::schema::PrimitiveType_TensorListFromTensor;

  std::vector<lite::Tensor *> outputs;
  TensorList *tensorList = new (std::nothrow) TensorList;
  tensorList->set_data_type(kObjectTypeTensorType);
  Tensor *output = reinterpret_cast<Tensor *>(tensorList);
  outputs.push_back(output);

  int ret = KernelInferShape(inputs, &outputs, parameter);

  TensorList *out = reinterpret_cast<TensorList *>(outputs[0]);

  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(out->shape()[0], 4);
  ASSERT_EQ(out->data_type(), kObjectTypeTensorType);
  ASSERT_EQ(out->element_shape().size(), 2);
  ASSERT_EQ(out->element_shape()[0], -1);
  ASSERT_EQ(out->element_shape()[1], 5);
  ASSERT_EQ(out->tensors_data_type(), kNumberTypeInt32);
  // ASSERT_EQ(outputs[0]->format_, Format_NHWC);
  for (int i = 0; i < out->shape()[0]; i++) {
    ASSERT_EQ(out->tensors()[i]->shape().size(), 2);
    ASSERT_EQ(out->tensors()[i]->shape()[0], 6);
    ASSERT_EQ(out->tensors()[i]->shape()[1], 5);
  }
  delete parameter;
  for (size_t i = 0; i < 2; i++) {
    if (inputs[i]->data_type() == kObjectTypeTensorType) {
      delete reinterpret_cast<TensorList *>(inputs[i]);
    } else {
      delete inputs[i];
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i]->data_type() == kObjectTypeTensorType) {
      delete reinterpret_cast<TensorList *>(outputs[i]);
    } else {
      delete outputs[i];
    }
  }
}

// PrimitiveType_TensorListGetItem
TEST_F(InferManagerTest, InferManagerTest1) {
  TensorList *tensorList = new TensorList;
  tensorList->set_data_type(kObjectTypeTensorType);
  Tensor *tensor0_0 = new Tensor;
  std::vector<int> tensor0_0_shape = {1, 2};
  tensor0_0->set_shape(tensor0_0_shape);
  Tensor *tensor0_1 = new Tensor;
  std::vector<int> tensor0_1_shape = {3, 4, 5};
  tensor0_1->set_shape(tensor0_1_shape);
  Tensor *tensor0_2 = new Tensor;
  std::vector<int> tensor0_2_shape = {6, 7, 8, 9};
  tensor0_2->set_shape(tensor0_2_shape);
  std::vector<Tensor *> tensor0;
  tensor0.push_back(tensor0_0);
  tensor0.push_back(tensor0_1);
  tensor0.push_back(tensor0_2);
  tensorList->set_tensors(tensor0);
  std::vector<int> tensorlist_shape = {3};
  tensorList->set_shape(tensorlist_shape);

  Tensor *tensor1 = new Tensor;
  std::vector<int> tensor1_shape = {1};
  std::vector<int> tensor1_data = {2};
  tensor1->set_shape(tensor1_shape);
  tensor1->set_data(tensor1_data.data());
  Tensor *tensor2 = new Tensor;

  std::vector<lite::Tensor *> inputs;
  inputs.push_back(reinterpret_cast<Tensor *>(tensorList));
  inputs.push_back(tensor1);
  inputs.push_back(tensor2);

  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  parameter->type_ = mindspore::schema::PrimitiveType_TensorListGetItem;

  std::vector<lite::Tensor *> outputs;
  Tensor *output = new Tensor;
  outputs.push_back(output);
  int res = KernelInferShape(inputs, &outputs, parameter);
  ASSERT_EQ(res, RET_OK);
  ASSERT_EQ(outputs[0]->shape().size(), 4);
  ASSERT_EQ(outputs[0]->shape().at(0), 6);
  ASSERT_EQ(outputs[0]->shape().at(1), 7);
  ASSERT_EQ(outputs[0]->shape().at(2), 8);
  ASSERT_EQ(outputs[0]->shape().at(3), 9);
  delete parameter;
  for (size_t i = 0; i < 3; i++) {
    if (inputs[i]->data_type() == kObjectTypeTensorType) {
      delete reinterpret_cast<TensorList *>(inputs[i]);
    } else {
      delete inputs[i];
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

TEST_F(InferManagerTest, InferManagerTest2) {
  Tensor *tensor0 = new (std::nothrow) Tensor;
  std::vector<int> tensor0_shape = {3};
  tensor0->set_shape(tensor0_shape);
  tensor0->set_data_type(kNumberTypeInt32);
  std::vector<int> tensor0_data = {2, 3, 4};
  tensor0->set_data(tensor0_data.data());
  Tensor *tensor1 = new (std::nothrow) Tensor;
  std::vector<int> tensor1_shape = {1};
  tensor1->set_shape(tensor1_shape);
  std::vector<int> tensor1_data = {5};
  tensor1->set_data(tensor1_data.data());
  tensor1->set_data_type(kNumberTypeInt32);
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(tensor0);
  inputs.push_back(tensor1);

  OpParameter *parameter = new OpParameter;
  parameter->infer_flag_ = true;
  parameter->type_ = mindspore::schema::PrimitiveType_TensorListReserve;

  std::vector<lite::Tensor *> outputs;
  TensorList *tensorList = new (std::nothrow) TensorList;
  tensorList->set_data_type(kObjectTypeTensorType);
  Tensor *output = reinterpret_cast<Tensor *>(tensorList);
  outputs.push_back(output);

  int ret = KernelInferShape(inputs, &outputs, parameter);

  TensorList *out = reinterpret_cast<TensorList *>(outputs[0]);

  ASSERT_EQ(ret, RET_OK);
  ASSERT_EQ(out->shape()[0], 5);
  ASSERT_EQ(out->data_type(), kObjectTypeTensorType);
  ASSERT_EQ(out->element_shape().size(), 3);
  ASSERT_EQ(out->element_shape()[0], 2);
  ASSERT_EQ(out->element_shape()[1], 3);
  ASSERT_EQ(out->element_shape()[2], 4);
  ASSERT_EQ(out->tensors_data_type(), kTypeUnknown);

  delete parameter;
  for (size_t i = 0; i < 2; i++) {
    if (inputs[i]->data_type() == kObjectTypeTensorType) {
      delete reinterpret_cast<TensorList *>(inputs[i]);
    } else {
      delete inputs[i];
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    if (outputs[i]->data_type() == kObjectTypeTensorType) {
      delete reinterpret_cast<TensorList *>(outputs[i]);
    } else {
      delete outputs[i];
    }
  }
}

}  // namespace mindspore::lite
