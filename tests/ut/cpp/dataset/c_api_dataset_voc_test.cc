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
#include "common/common.h"
#include "minddata/dataset/include/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestVOCClassIndex) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVOCClassIndex.";

  // Create a VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::map<std::string, int32_t> class_index;
  class_index["car"] = 0;
  class_index["cat"] = 1;
  class_index["train"] = 9;

  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", class_index, false, SequentialSampler(0, 6));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check if VOCOp read correct labels
  // When we provide class_index, label of ["car","cat","train"] become [0,1,9]
  // std::shared_ptr<Tensor> expect_label;
  // Tensor::CreateFromMemory(TensorShape({1, 1}), DataType(DataType::DE_UINT32), nullptr, &expect_label);

  // uint32_t expect[] = {9, 9, 9, 1, 1, 0};
  uint64_t i = 0;
  while (row.size() != 0) {
    // auto image = row["image"];
    // auto label = row["label"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    // MS_LOG(INFO) << "Tensor label shape: " << label->shape();
    // expect_label->SetItemAt({0, 0}, expect[i]);
    // EXPECT_EQ(*label, *expect_label);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestVOCGetClassIndex) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVOCGetClassIndex.";
  // Create a VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::map<std::string, int32_t> class_index;
  class_index["car"] = 0;
  class_index["cat"] = 1;
  class_index["train"] = 9;

  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", class_index, false, SequentialSampler(0, 6));
  EXPECT_NE(ds, nullptr);

  std::vector<std::pair<std::string, std::vector<int32_t>>> class_index1 = ds->GetClassIndexing();
  EXPECT_EQ(class_index1.size(), 3);
  EXPECT_EQ(class_index1[0].first, "car");
  EXPECT_EQ(class_index1[0].second[0], 0);
  EXPECT_EQ(class_index1[1].first, "cat");
  EXPECT_EQ(class_index1[1].second[0], 1);
  EXPECT_EQ(class_index1[2].first, "train");
  EXPECT_EQ(class_index1[2].second[0], 9);
}

TEST_F(MindDataTestPipeline, TestVOCGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVOCGetters.";

  // Create a VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::map<std::string, int32_t> class_index;
  class_index["car"] = 0;
  class_index["cat"] = 1;
  class_index["train"] = 9;

  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", class_index, false, SequentialSampler(0, 6));
  EXPECT_NE(ds, nullptr);

  ds = ds->Batch(2);
  ds = ds->Repeat(2);

  EXPECT_EQ(ds->GetDatasetSize(), 6);
  std::vector<std::string> column_names = {"image", "bbox", "label", "difficult", "truncate"};
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

TEST_F(MindDataTestPipeline, TestVOCDetection) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVOCDetection.";

  // Create a VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, false, SequentialSampler(0, 4));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check if VOCOp read correct images/labels
  // std::string expect_file[] = {"15", "32", "33", "39"};
  // uint32_t expect_num[] = {5, 5, 4, 3};
  uint64_t i = 0;
  while (row.size() != 0) {
    // auto image = row["image"];
    // auto label = row["label"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    // MS_LOG(INFO) << "Tensor label shape: " << label->shape();

    // std::shared_ptr<Tensor> expect_image;
    // Tensor::CreateFromFile(folder_path + "/JPEGImages/" + expect_file[i] + ".jpg", &expect_image);
    // EXPECT_EQ(*image, *expect_image);

    // std::shared_ptr<Tensor> expect_label;
    // Tensor::CreateFromMemory(TensorShape({1, 1}), DataType(DataType::DE_UINT32), nullptr, &expect_label);
    // expect_label->SetItemAt({0, 0}, expect_num[i]);
    // EXPECT_EQ(*label, *expect_label);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestVOCInvalidTaskOrModeError1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVOCInvalidTaskOrModeError1.";

  // Create a VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds1 = VOC(folder_path, "Classification", "train", {}, false, SequentialSampler(0, 3));
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid Manifest input, invalid task
  EXPECT_EQ(iter1, nullptr);

  std::shared_ptr<Dataset> ds2 = VOC(folder_path, "Segmentation", "validation", {}, false, RandomSampler(false, 4));
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid VOC input, invalid mode
  EXPECT_EQ(iter2, nullptr);
}

TEST_F(MindDataTestPipeline, TestVOCSegmentation) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVOCSegmentation.";

  // Create a VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Segmentation", "train", {}, false, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check if VOCOp read correct images/targets
  // using Tensor = mindspore::dataset::Tensor;
  // std::string expect_file[] = {"32", "33", "39", "32", "33", "39"};
  uint64_t i = 0;
  while (row.size() != 0) {
    // auto image = row["image"];
    // auto target = row["target"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    // MS_LOG(INFO) << "Tensor target shape: " << target->shape();

    // std::shared_ptr<Tensor> expect_image;
    // Tensor::CreateFromFile(folder_path + "/JPEGImages/" + expect_file[i] + ".jpg", &expect_image);
    // EXPECT_EQ(*image, *expect_image);

    // std::shared_ptr<Tensor> expect_target;
    // Tensor::CreateFromFile(folder_path + "/SegmentationClass/" + expect_file[i] + ".png", &expect_target);
    // EXPECT_EQ(*target, *expect_target);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestVOCSegmentationError2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVOCSegmentationError2.";

  // Create a VOC Dataset
  std::map<std::string, int32_t> class_index;
  class_index["car"] = 0;
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Segmentation", "train", class_index, false, RandomSampler(false, 6));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid VOC input, segmentation task with class_index
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestVOCWithNullSamplerError3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVOCWithNullSamplerError3.";

  // Create a VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Segmentation", "train", {}, false, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid VOC input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
