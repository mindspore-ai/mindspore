/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <memory>
#include <string>
#include "minddata/dataset/core/client.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "securec.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/core/data_type.h"

using namespace mindspore::dataset;

namespace py = pybind11;

class MindDataTestStringTensorDE : public UT::Common {
 public:
  MindDataTestStringTensorDE() = default;

  void SetUp() override { GlobalInit(); }
};

/// Feature: Tensor
/// Description: Test creating a Tensor from a scalar string and 1D string array
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestStringTensorDE, Basics) {
  std::shared_ptr<Tensor> t;
  Tensor::CreateScalar<std::string>("Hi", &t);
  ASSERT_TRUE(t->shape() == TensorShape({}));
  std::string_view s = "";
  t->GetItemAt(&s, {});
  ASSERT_TRUE(s == "Hi");

  std::shared_ptr<Tensor> t2;
  Tensor::CreateFromVector(std::vector<std::string>{"Hi", "Bye"}, &t2);
  ASSERT_TRUE(t2->shape() == TensorShape({2}));
  t2->GetItemAt(&s, {0});
  ASSERT_TRUE(s == "Hi");
  t2->GetItemAt(&s, {1});
  ASSERT_TRUE(s == "Bye");

  std::vector<std::string> strings{"abc", "defg", "hi", "klmno", "123", "789"};
  std::shared_ptr<Tensor> t3;
  Tensor::CreateFromVector(strings, TensorShape({2, 3}), &t3);

  ASSERT_TRUE(t3->shape() == TensorShape({2, 3}));
  uint32_t index = 0;
  for (uint32_t i = 0; i < 2; i++) {
    for (uint32_t j = 0; j < 3; j++) {
      std::string_view s = "";
      t3->GetItemAt(&s, {i, j});
      ASSERT_TRUE(s == strings[index++]);
    }
  }
}

/// Feature: Tensor
/// Description: Test memory of Tensor created from a 1D string array
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestStringTensorDE, Basics2) {
  std::shared_ptr<Tensor> t;
  Tensor::CreateFromVector(std::vector<std::string>{"abc", "defg", "hi", "klmno", "123", "789"}, TensorShape({2, 3}),
                           &t);

  ASSERT_TRUE(t->SizeInBytes() == 6 * 5 + 20 + 4);
  std::vector<uint32_t> offsets = {0, 4, 9, 12, 18, 22, 26};
  uint32_t ctr = 0;
  for (auto i : offsets) {
    ASSERT_TRUE(*(reinterpret_cast<const uint32_t *>(t->GetBuffer() + ctr)) == i + 28);
    ctr += 4;
  }
  const char *buf = reinterpret_cast<const char *>(t->GetBuffer()) + 6 * 4 + 4;
  std::vector<uint32_t> starts = {0, 4, 9, 12, 18, 22};

  uint32_t index = 0;
  for (uint32_t i = 0; i < 2; i++) {
    for (uint32_t j = 0; j < 3; j++) {
      std::string_view s = "";
      t->GetItemAt(&s, {i, j});
      ASSERT_TRUE(s.data() == buf + starts[index++]);
    }
  }
}

/// Feature: Tensor
/// Description: Test creating Tensor from array of strings that contain empty strings
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestStringTensorDE, Empty) {
  std::vector<std::string> strings{"abc", "defg", "", "", "123", ""};
  std::shared_ptr<Tensor> t;
  Tensor::CreateFromVector(strings, TensorShape({2, 3}), &t);
  //  abc_defg___123__
  //  0123456789012345
  ASSERT_TRUE(t->SizeInBytes() == 6 * 5 + 10 + 4);
  std::vector<uint32_t> offsets = {0, 4, 9, 10, 11, 15, 16};
  uint32_t ctr = 0;
  for (auto i : offsets) {
    ASSERT_TRUE(*(reinterpret_cast<const uint32_t *>(t->GetBuffer() + ctr)) == i + 28);
    ctr += 4;
  }
  const char *buf = reinterpret_cast<const char *>(t->GetBuffer()) + 6 * 4 + 4;
  std::vector<uint32_t> starts = {0, 4, 9, 10, 11, 15};

  uint32_t index = 0;
  for (uint32_t i = 0; i < 2; i++) {
    for (uint32_t j = 0; j < 3; j++) {
      std::string_view s = "";
      t->GetItemAt(&s, {i, j});
      ASSERT_TRUE(s.data() == buf + starts[index]);
      ASSERT_TRUE(s == strings[index++]);
    }
  }
}

/// Feature: Tensor
/// Description: Test creating Tensor from empty data
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestStringTensorDE, EmptyData) {
  std::shared_ptr<Tensor> t;
  Tensor::CreateScalar<std::string>("", &t);
  // empty string has 1 element 
  ASSERT_TRUE(t->HasData());

  std::shared_ptr<Tensor> t1;
  Tensor::CreateEmpty(TensorShape({0}), DataType(DataType::DE_STRING), &t1);
  ASSERT_TRUE(!t1->HasData());
}

/// Feature: Tensor
/// Description: Test SetItemAt usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestStringTensorDE, SetItem) {
  std::vector<std::string> strings{"abc", "defg", "hi", "klmno", "123", "789"};
  std::shared_ptr<Tensor> t3;
  Tensor::CreateFromVector(strings, TensorShape({2, 3}), &t3);

  ASSERT_TRUE(t3->shape() == TensorShape({2, 3}));

  t3->SetItemAt({0, 1}, std::string{"xyzz"});
  strings[1] = "xyzz";

  t3->SetItemAt({0, 2}, std::string{"07"});
  strings[2] = "07";

  t3->SetItemAt({1, 2}, std::string{"987"});
  strings[5] = "987";

  uint32_t index = 0;
  for (uint32_t i = 0; i < 2; i++) {
    for (uint32_t j = 0; j < 3; j++) {
      std::string_view s = "";
      t3->GetItemAt(&s, {i, j});
      ASSERT_TRUE(s == strings[index++]);
    }
  }
}

/// Feature: Tensor
/// Description: Test iterating over a Tensor
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestStringTensorDE, Iterator) {
  std::vector<std::string> strings{"abc", "defg", "hi", "klmno", "123", "789"};
  std::shared_ptr<Tensor> t;
  Tensor::CreateFromVector(strings, TensorShape({2, 3}), &t);
  uint32_t index = 0;
  auto itr = t->begin<std::string_view>();
  for (; itr != t->end<std::string_view>(); itr++) {
    ASSERT_TRUE(*itr == strings[index++]);
  }

  index = 0;
  itr = t->begin<std::string_view>();
  for (; itr != t->end<std::string_view>(); itr += 2) {
    ASSERT_TRUE(*itr == strings[index]);
    index += 2;
  }
}
