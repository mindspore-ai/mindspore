/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "dataset/core/client.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "securec.h"
#include "dataset/core/tensor.h"
#include "dataset/core/cv_tensor.h"
#include "dataset/core/data_type.h"

using namespace mindspore::dataset;

namespace py = pybind11;

class MindDataTestTensorDE : public UT::Common {
 public:
  MindDataTestTensorDE() {}

  void SetUp() { GlobalInit(); }
};

TEST_F(MindDataTestTensorDE, Basics) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 3}), DataType(DataType::DE_UINT64));
  ASSERT_TRUE((t->AllocateBuffer(t->SizeInBytes())).IsOk());
  ASSERT_EQ(t->shape(), TensorShape({2, 3}));
  ASSERT_EQ(t->type(), DataType::DE_UINT64);
  ASSERT_EQ(t->SizeInBytes(), 2 * 3 * 8);
  ASSERT_EQ(t->Rank(), 2);
  t->SetItemAt<uint64_t>({0, 0}, 1);
  t->SetItemAt<uint64_t>({0, 1}, 2);
  t->SetItemAt<uint64_t>({0, 2}, 3);
  t->SetItemAt<uint64_t>({1, 0}, 4);
  t->SetItemAt<uint64_t>({1, 1}, 5);
  t->SetItemAt<uint64_t>({1, 2}, 6);
  Status rc = t->SetItemAt<uint64_t>({2, 3}, 7);
  ASSERT_TRUE(rc.IsError());
  uint64_t o;
  t->GetItemAt<uint64_t>(&o, {0, 0});
  ASSERT_EQ(o, 1);
  t->GetItemAt<uint64_t>(&o, {0, 1});
  ASSERT_EQ(o, 2);
  t->GetItemAt<uint64_t>(&o, {0, 2});
  ASSERT_EQ(o, 3);
  t->GetItemAt<uint64_t>(&o, {1, 0});
  ASSERT_EQ(o, 4);
  t->GetItemAt<uint64_t>(&o, {1, 1});
  ASSERT_EQ(o, 5);
  t->GetItemAt<uint64_t>(&o, {1, 2});
  ASSERT_EQ(o, 6);
  rc = t->GetItemAt<uint64_t>(&o, {2, 3});
  ASSERT_TRUE(rc.IsError());
  ASSERT_EQ(t->ToString(), "Tensor (shape: <2,3>, Type: uint64)\n[[1,2,3],[4,5,6]]");
  std::vector<uint64_t> x = {1, 2, 3, 4, 5, 6};
  std::shared_ptr<Tensor> t2;
  Tensor::CreateTensor(&t2, x, TensorShape({2, 3}));

  ASSERT_EQ(*t == *t2, true);
  ASSERT_EQ(*t != *t2, false);
}

TEST_F(MindDataTestTensorDE, Fill) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 2}), DataType(DataType::DE_FLOAT32));
  t->Fill<float>(2.5);
  std::vector<float> x = {2.5, 2.5, 2.5, 2.5};
  std::shared_ptr<Tensor> t2;
  Tensor::CreateTensor(&t2, x, TensorShape({2, 2}));
  ASSERT_EQ(*t == *t2, true);
}

TEST_F(MindDataTestTensorDE, Reshape) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 2}), DataType(DataType::DE_UINT8));
  t->Fill<uint8_t>(254);
  t->Reshape(TensorShape({4}));
  std::vector<uint8_t> x = {254, 254, 254, 254};
  std::shared_ptr<Tensor> t2;
  Tensor::CreateTensor(&t2, x);

  ASSERT_EQ(*t == *t2, true);
  Status rc = t->Reshape(TensorShape({5}));
  ASSERT_TRUE(rc.IsError());
  t2->ExpandDim(0);
  ASSERT_EQ(t2->shape(), TensorShape({1, 4}));
  t2->ExpandDim(2);
  ASSERT_EQ(t2->shape(), TensorShape({1, 4, 1}));
  rc = t2->ExpandDim(4);
  ASSERT_TRUE(rc.IsError());
}

TEST_F(MindDataTestTensorDE, CopyTensor) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({}), DataType(DataType::DE_INT16));
  t->SetItemAt<int16_t>({}, -66);
  ASSERT_EQ(t->shape(), TensorShape({}));
  ASSERT_EQ(t->type(), DataType::DE_INT16);
  int16_t o;
  t->GetItemAt<int16_t>(&o, {});
  ASSERT_EQ(o, -66);
  const unsigned char *addr = t->GetBuffer();
  auto t2 = std::make_shared<Tensor>(std::move(*t));
  ASSERT_EQ(t2->shape(), TensorShape({}));
  ASSERT_EQ(t2->type(), DataType::DE_INT16);
  t2->GetItemAt<int16_t>(&o, {});
  ASSERT_EQ(o, -66);
  const unsigned char *new_addr = t2->GetBuffer();
  ASSERT_EQ(addr, new_addr);
  ASSERT_EQ(t->shape(), TensorShape::CreateUnknownRankShape());
  ASSERT_EQ(t->type(), DataType::DE_UNKNOWN);
  ASSERT_EQ(t->GetBuffer(), nullptr);
  Status rc = t->GetItemAt<int16_t>(&o, {});
  ASSERT_TRUE(rc.IsError());
}

TEST_F(MindDataTestTensorDE, InsertTensor) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 3}), DataType(DataType::DE_FLOAT64));
  std::vector<double> x = {1.1, 2.1, 3.1};
  std::shared_ptr<Tensor> t2;
  Tensor::CreateTensor(&t2, x);

  std::vector<double> y = {1.2, 2.2, 3.2};
  std::shared_ptr<Tensor> t3;
  Tensor::CreateTensor(&t3, y);

  ASSERT_TRUE(t->InsertTensor({0}, t2).OK());
  ASSERT_TRUE(t->InsertTensor({1}, t3).OK());
  std::vector<double> z = {1.1, 2.1, 3.1, 1.2, 2.2, 3.2};

  std::shared_ptr<Tensor> t4;
  Tensor::CreateTensor(&t4, z, TensorShape({2, 3}));
  ASSERT_EQ(*t == *t4, true);

  std::shared_ptr<Tensor> t5;
  Tensor::CreateTensor<double>(&t5, 0);

  ASSERT_TRUE(t->InsertTensor({1, 2}, t5).OK());
  z[5] = 0;
  std::shared_ptr<Tensor> t6;
  Tensor::CreateTensor(&t6, z, TensorShape({2, 3}));

  ASSERT_EQ(*t == *t6, true);
  ASSERT_EQ(t->InsertTensor({2}, t5).get_code(), StatusCode::kUnexpectedError);
  ASSERT_EQ(t->InsertTensor({1}, t5).get_code(), StatusCode::kUnexpectedError);
  ASSERT_EQ(t->InsertTensor({1, 2}, t6).get_code(), StatusCode::kUnexpectedError);
  t6->Fill<double>(-1);
  ASSERT_TRUE(t->InsertTensor({}, t6).OK());
  ASSERT_EQ(*t == *t6, true);
}

// Test the bug of Tensor::ToString will exec failed for Tensor which store bool values
TEST_F(MindDataTestTensorDE, BoolTensor) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2}), DataType(DataType::DE_BOOL));
  t->SetItemAt<bool>({0}, true);
  t->SetItemAt<bool>({1}, true);
  std::string out = t->ToString();
  ASSERT_TRUE(out.find("Template type and Tensor type are not compatible") == std::string::npos);
}

TEST_F(MindDataTestTensorDE, GetItemAt) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 2}), DataType(DataType::DE_UINT8));
  t->Fill<uint8_t>(254);
  uint64_t o1;
  t->GetItemAt<uint64_t>(&o1, {0, 0});
  ASSERT_EQ(o1, 254);
  uint32_t o2;
  t->GetItemAt<uint32_t>(&o2, {0, 1});
  ASSERT_EQ(o2, 254);
  uint16_t o3;
  t->GetItemAt<uint16_t>(&o3, {1, 0});
  ASSERT_EQ(o3, 254);
  uint8_t o4;
  t->GetItemAt<uint8_t>(&o4, {1, 1});
  ASSERT_EQ(o4, 254);
  std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(TensorShape({2, 2}), DataType(DataType::DE_INT8));
  t2->Fill<int8_t>(-10);
  int64_t o5;
  t2->GetItemAt<int64_t>(&o5, {0, 0});
  ASSERT_EQ(o5, -10);
  int32_t o6;
  t2->GetItemAt<int32_t>(&o6, {0, 1});
  ASSERT_EQ(o6, -10);
  int16_t o7;
  t2->GetItemAt<int16_t>(&o7, {1, 0});
  ASSERT_EQ(o7, -10);
  int8_t o8;
  t2->GetItemAt<int8_t>(&o8, {1, 1});
  ASSERT_EQ(o8, -10);
  std::shared_ptr<Tensor> t3 = std::make_shared<Tensor>(TensorShape({2, 2}), DataType(DataType::DE_FLOAT32));
  t3->Fill<float>(1.1);
  double o9;
  t3->GetItemAt<double>(&o9, {0, 0});
  ASSERT_FLOAT_EQ(o9, 1.1);
  float o10;
  t3->GetItemAt<float>(&o10, {0, 1});
  ASSERT_FLOAT_EQ(o10, 1.1);
}

TEST_F(MindDataTestTensorDE, OperatorAssign) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 2}), DataType(DataType::DE_UINT8));
  t->Fill<uint8_t>(1);
  std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(TensorShape({2, 2}), DataType(DataType::DE_UINT8));
  *t2 = std::move(*t);
  uint8_t o;
  t2->GetItemAt(&o, {0, 0});
  ASSERT_EQ(o, 1);
  t2->GetItemAt(&o, {0, 1});
  ASSERT_EQ(o, 1);
  t2->GetItemAt(&o, {1, 0});
  ASSERT_EQ(o, 1);
  t2->GetItemAt(&o, {1, 1});
  ASSERT_EQ(o, 1);
}

TEST_F(MindDataTestTensorDE, Strides) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({4, 2, 2}), DataType(DataType::DE_UINT8));
  std::vector<dsize_t> x1 = t->Strides();
  std::vector<dsize_t> x2 = {4, 2, 1};
  ASSERT_EQ(x1, x2);
  t = std::make_shared<Tensor>(TensorShape({4, 2, 2}), DataType(DataType::DE_UINT32));
  x1 = t->Strides();
  x2 = {16, 8, 4};
  ASSERT_EQ(x1, x2);
}

void checkCvMat(TensorShape shape, DataType type) {
  std::shared_ptr<CVTensor> t = std::make_shared<CVTensor>(shape, type);
  cv::Mat m = t->mat();
  ASSERT_EQ(m.data, t->GetBuffer());
  ASSERT_EQ(static_cast<uchar>(m.type()) & static_cast<uchar>(CV_MAT_DEPTH_MASK), type.AsCVType());
  if (shape.Rank() < 4) {
    if (shape.Rank() > 1) {
      for (dsize_t i = 0; i < 2; i++) ASSERT_EQ(m.size[static_cast<int>(i)], shape[i]);
    } else if (shape.Rank() == 0) {
      ASSERT_EQ(m.size[0], 1);
      ASSERT_EQ(m.size[1], 1);
    } else {
      ASSERT_EQ(m.size[0], shape[0]);
    }
    if (shape.Rank() == 3) {
      ASSERT_EQ(m.channels(), shape[2]);
    }
    ASSERT_EQ(m.dims, 2);
    ASSERT_EQ(m.size.dims(), 2);
    if (shape.Rank() > 0) {
      ASSERT_EQ(m.rows, shape[0]);
    }
    if (shape.Rank() > 1) {
      ASSERT_EQ(m.cols, shape[1]);
    }
  } else {
    for (dsize_t i = 0; i < shape.Rank(); i++) ASSERT_EQ(m.size[static_cast<int>(i)], shape[i]);
    ASSERT_EQ(m.dims, shape.Rank());
    ASSERT_EQ(m.size.dims(), shape.Rank());
    ASSERT_EQ(m.rows, -1);
    ASSERT_EQ(m.cols, -1);
  }
}

TEST_F(MindDataTestTensorDE, CVTensorBasics) {
  checkCvMat(TensorShape({4, 5}), DataType(DataType::DE_UINT8));
  checkCvMat(TensorShape({4, 5, 3}), DataType(DataType::DE_UINT8));
  checkCvMat(TensorShape({4, 5, 10}), DataType(DataType::DE_UINT8));
  checkCvMat(TensorShape({4, 5, 3, 2}), DataType(DataType::DE_UINT8));
  checkCvMat(TensorShape({4}), DataType(DataType::DE_UINT8));
  checkCvMat(TensorShape({}), DataType(DataType::DE_INT16));
  checkCvMat(TensorShape({4, 5}), DataType(DataType::DE_INT16));
  checkCvMat(TensorShape({4, 5, 3}), DataType(DataType::DE_INT16));
  checkCvMat(TensorShape({4, 5, 10}), DataType(DataType::DE_INT16));
  checkCvMat(TensorShape({4, 5, 3, 2}), DataType(DataType::DE_INT16));
  checkCvMat(TensorShape({4}), DataType(DataType::DE_INT16));
  checkCvMat(TensorShape({}), DataType(DataType::DE_INT16));
}

TEST_F(MindDataTestTensorDE, CVTensorFromMat) {
  cv::Mat m(2, 2, CV_8U);
  m.at<uint8_t>(0, 0) = 10;
  m.at<uint8_t>(0, 1) = 20;
  m.at<uint8_t>(1, 0) = 30;
  m.at<uint8_t>(1, 1) = 40;
  std::shared_ptr<CVTensor> cvt = std::make_shared<CVTensor>(m);
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({2, 2}), DataType(DataType::DE_UINT8));
  t->SetItemAt<uint8_t>({0, 0}, 10);
  t->SetItemAt<uint8_t>({0, 1}, 20);
  t->SetItemAt<uint8_t>({1, 0}, 30);
  t->SetItemAt<uint8_t>({1, 1}, 40);
  ASSERT_TRUE(*t == *cvt);
  int size[] = {4};
  cv::Mat m2(1, size, CV_8U);
  m2.at<uint8_t>(0) = 10;
  m2.at<uint8_t>(1) = 20;
  m2.at<uint8_t>(2) = 30;
  m2.at<uint8_t>(3) = 40;
  std::shared_ptr<CVTensor> cvt2 = std::make_shared<CVTensor>(m2);
  std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(TensorShape({4}), DataType(DataType::DE_UINT8));
  t2->SetItemAt<uint8_t>({0}, 10);
  t2->SetItemAt<uint8_t>({1}, 20);
  t2->SetItemAt<uint8_t>({2}, 30);
  t2->SetItemAt<uint8_t>({3}, 40);
  t2->ExpandDim(1);
  ASSERT_TRUE(*t2 == *cvt2);
}

TEST_F(MindDataTestTensorDE, CVTensorAs) {
  std::shared_ptr<Tensor> t = std::make_shared<Tensor>(TensorShape({3, 2}), DataType(DataType::DE_FLOAT64));
  t->Fill<double>(2.2);
  const unsigned char *addr = t->GetBuffer();
  std::shared_ptr<Tensor> t2 = std::make_shared<Tensor>(TensorShape({3, 2}), DataType(DataType::DE_FLOAT64));
  t2->Fill<double>(4.4);
  std::shared_ptr<CVTensor> ctv = CVTensor::AsCVTensor(t);
  ASSERT_EQ(t->GetBuffer(), nullptr);
  ASSERT_EQ(ctv->GetBuffer(), addr);
  cv::Mat m = ctv->mat();
  m = 2 * m;
  ASSERT_EQ(ctv->GetBuffer(), addr);
  ASSERT_TRUE(*t2 == *ctv);
  MS_LOG(DEBUG) << *t2 << std::endl << *ctv;
}

TEST_F(MindDataTestTensorDE, CVTensorMatSlice) {
  cv::Mat m(2, 3, CV_32S);
  m.at<int32_t>(0, 0) = 10;
  m.at<int32_t>(0, 1) = 20;
  m.at<int32_t>(0, 2) = 30;
  m.at<int32_t>(1, 0) = 40;
  m.at<int32_t>(1, 1) = 50;
  m.at<int32_t>(1, 2) = 60;
  std::shared_ptr<CVTensor> cvt = std::make_shared<CVTensor>(m);
  cv::Mat mat;
  cvt->Mat({1}, &mat);
  cv::Mat m2(3, 1, CV_32S);
  m2.at<int32_t>(0) = 40;
  m2.at<int32_t>(1) = 50;
  m2.at<int32_t>(2) = 60;
  std::shared_ptr<CVTensor> cvt2 = std::make_shared<CVTensor>(mat);
  std::shared_ptr<CVTensor> cvt3 = std::make_shared<CVTensor>(m2);

  ASSERT_TRUE(*cvt2 == *cvt3);
  cvt->Mat({0}, &mat);
  m2.at<int32_t>(0) = 10;
  m2.at<int32_t>(1) = 20;
  m2.at<int32_t>(2) = 30;
  cvt2 = std::make_shared<CVTensor>(mat);
  cvt3 = std::make_shared<CVTensor>(m2);
  ASSERT_TRUE(*cvt2 == *cvt3);
}

TEST_F(MindDataTestTensorDE, TensorIterator) {
  std::vector<uint32_t> values = {1, 2, 3, 4, 5, 6};
  std::vector<uint32_t> values2 = {2, 3, 4, 5, 6, 7};

  std::shared_ptr<Tensor> t;
  Tensor::CreateTensor(&t, values);

  auto i = t->begin<uint32_t>();
  auto j = values.begin();
  uint32_t ctr = 0;
  for (; i != t->end<uint32_t>(); i++, j++) {
    ASSERT_TRUE(*i == *j);
    ctr++;
  }
  ASSERT_TRUE(ctr == 6);
  t->Reshape(TensorShape{2, 3});
  i = t->begin<uint32_t>();
  j = values.begin();
  ctr = 0;
  for (; i != t->end<uint32_t>(); i++, j++) {
    ASSERT_TRUE(*i == *j);
    ctr++;
  }
  ASSERT_TRUE(ctr == 6);
  for (auto it = t->begin<uint32_t>(); it != t->end<uint32_t>(); it++) {
    *it = *it + 1;
  }
  i = t->begin<uint32_t>();
  j = values2.begin();
  ctr = 0;
  for (; i != t->end<uint32_t>(); i++, j++) {
    ASSERT_TRUE(*i == *j);
    ctr++;
  }
  ASSERT_TRUE(ctr == 6);
}

TEST_F(MindDataTestTensorDE, TensorSlice) {
  std::shared_ptr<Tensor> t;
  Tensor::CreateTensor(&t, std::vector<dsize_t>{0, 1, 2, 3, 4});
  std::shared_ptr<Tensor> t2;
  auto x = std::vector<dsize_t>{0, 3, 4};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateTensor(&expected, x);
  t->Slice(&t2, x);
  ASSERT_EQ(*t2, *expected);
  t->Slice(&t2, std::vector<dsize_t>{0, 1, 2, 3, 4});
  ASSERT_EQ(*t2, *t);
}

TEST_F(MindDataTestTensorDE, TensorConcatenate) {
  std::vector<uint32_t> values1 = {1, 2, 3, 0, 0, 0};
  std::vector<uint32_t> values2 = {4, 5, 6};
  std::vector<uint32_t> expected = {1, 2, 3, 4, 5, 6};

  std::shared_ptr<Tensor> t1;
  Tensor::CreateTensor(&t1, values1);

  std::shared_ptr<Tensor> t2;
  Tensor::CreateTensor(&t2, values2);

  std::shared_ptr<Tensor> out;
  Tensor::CreateTensor(&out, expected);
  Status s = t1->Concatenate({3}, t2);
  EXPECT_TRUE(s.IsOk());

  auto i = out->begin<uint32_t>();
  auto j = t1->begin<uint32_t>();
  for (; i != out->end<uint32_t>(); i++, j++) {
    ASSERT_TRUE(*i == *j);
  }

  // should fail if the concatenated vector is too large
  s = t1->Concatenate({5}, t2);
  EXPECT_FALSE(s.IsOk());
}
