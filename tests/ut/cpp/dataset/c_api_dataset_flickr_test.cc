/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: FlickrDataset
/// Description: Test basic usage of FlickrDataset
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestFlickrBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlickrBasic.";

  std::string dataset_path = datasets_root_path_ + "/testFlickrData/flickr30k/flickr30k-images";
  std::string file_path = datasets_root_path_ + "/testFlickrData/flickr30k/test1.token";

  // Create a Flickr30k Dataset
  std::shared_ptr<Dataset> ds = Flickr(dataset_path, file_path);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: FlickrDataset
/// Description: Test usage of FlickrDataset with pipeline mode
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestFlickrBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlickrBasicWithPipeline.";

  std::string dataset_path = datasets_root_path_ + "/testFlickrData/flickr30k/flickr30k-images";
  std::string file_path = datasets_root_path_ + "/testFlickrData/flickr30k/test1.token";

  // Create two Flickr30k Dataset
  std::shared_ptr<Dataset> ds1 = Flickr(dataset_path, file_path);
  std::shared_ptr<Dataset> ds2 = Flickr(dataset_path, file_path);
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 2;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 3;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds
  std::vector<std::string> column_project = {"image"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: FlickrDataset
/// Description: Test iterator of FlickrDataset with only the "image" column
/// Expectation: Get correct data
TEST_F(MindDataTestPipeline, TestFlickrIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlickrIteratorOneColumn.";
  std::string dataset_path = datasets_root_path_ + "/testFlickrData/flickr30k/flickr30k-images";
  std::string file_path = datasets_root_path_ + "/testFlickrData/flickr30k/test1.token";

  // Create a Flickr30k Dataset
  std::shared_ptr<Dataset> ds = Flickr(dataset_path, file_path);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "image" column and drop others
  std::vector<std::string> columns = {"image"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "image shape:" << v.Shape();
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: FlickrDataset
/// Description: Test iterator of FlickrDataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFlickrIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlickrIteratorWrongColumn.";
  std::string dataset_path = datasets_root_path_ + "/testFlickrData/flickr30k/flickr30k-images";
  std::string file_path = datasets_root_path_ + "/testFlickrData/flickr30k/test1.token";

  // Create a Flickr30k Dataset
  std::shared_ptr<Dataset> ds = Flickr(dataset_path, file_path);
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestFlickrGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlickrGetters.";

  std::string dataset_path = datasets_root_path_ + "/testFlickrData/flickr30k/flickr30k-images";
  std::string file_path1 = datasets_root_path_ + "/testFlickrData/flickr30k/test1.token";
  std::string file_path2 = datasets_root_path_ + "/testFlickrData/flickr30k/test2.token";

  // Create a Flickr30k Dataset
  std::shared_ptr<Dataset> ds1 = Flickr(dataset_path, file_path1);
  std::shared_ptr<Dataset> ds2 = Flickr(dataset_path, file_path2);
  std::vector<std::string> column_names = {"image", "annotation"};

  EXPECT_NE(ds1, nullptr);
  EXPECT_EQ(ds1->GetDatasetSize(), 2);
  EXPECT_EQ(ds1->GetColumnNames(), column_names);

  EXPECT_NE(ds2, nullptr);
  EXPECT_EQ(ds2->GetDatasetSize(), 3);
  EXPECT_EQ(ds2->GetColumnNames(), column_names);
}

/// Feature: FlickrDataset
/// Description: Test usage of FlickrAnnotations
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestFlickrAnnotations) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlickrGetters.";

  std::string dataset_path = datasets_root_path_ + "/testFlickrData/flickr30k/flickr30k-images";
  std::string file_path = datasets_root_path_ + "/testFlickrData/flickr30k/test3.token";
  std::shared_ptr<Dataset> ds = Flickr(dataset_path, file_path);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::shared_ptr<Tensor> a_expect_item;
  std::vector<std::string> annotation_arr;
  annotation_arr.emplace_back("This is a banana.");
  annotation_arr.emplace_back("This is a yellow banana.");
  annotation_arr.emplace_back("This is a banana on the table.");
  annotation_arr.emplace_back("The banana is yellow.");
  annotation_arr.emplace_back("The banana is very big.");

  ASSERT_OK(Tensor::CreateFromVector(annotation_arr, &a_expect_item));
  mindspore::MSTensor expect_item = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(a_expect_item));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto annotation = row["annotation"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor annotation shape: " << annotation.Shape();

    EXPECT_MSTENSOR_EQ(annotation, expect_item);

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: FlickrDataset
/// Description: Test usage of FlickrDataset with RandomSampler
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestFlickrDecode) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlickrDecode.";

  std::string dataset_path = datasets_root_path_ + "/testFlickrData/flickr30k/flickr30k-images";
  std::string file_path = datasets_root_path_ + "/testFlickrData/flickr30k/test1.token";
  // Create a Flickr30k Dataset
  std::shared_ptr<Dataset> ds = Flickr(dataset_path, file_path, true, std::make_shared<RandomSampler>());
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto shape = image.Shape();
    MS_LOG(INFO) << "Tensor image shape size: " << shape.size();
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_GT(shape.size(), 1);  // Verify decode=true took effect
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: FlickrDataset
/// Description: Test usage of FlickrDataset with SequentialSampler
/// Expectation: Get correct piece of data
TEST_F(MindDataTestPipeline, TestFlickrNumSamplers) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlickrNumSamplers.";

  std::string dataset_path = datasets_root_path_ + "/testFlickrData/flickr30k/flickr30k-images";
  std::string file_path = datasets_root_path_ + "/testFlickrData/flickr30k/test1.token";
  // Create a Flickr30k Dataset
  std::shared_ptr<Dataset> ds = Flickr(dataset_path, file_path, true, std::make_shared<SequentialSampler>(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto annotation = row["annotation"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();

    auto a_it = annotation.Shape().begin();
    for (; a_it != annotation.Shape().end(); ++a_it) {
      std::cout << "annotation shape " << *a_it << std::endl;
    }
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: FlickrDataset
/// Description: Test FlickrDataset with invalid inputs
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFlickrError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlickrError.";

  std::string dataset_path = datasets_root_path_ + "/testFlickrData/flickr30k/flickr30k-images";
  std::string file_path = datasets_root_path_ + "/testFlickrData/flickr30k/test1.token";
  // Create a Flickr30k Dataset with non-existing dataset dir
  std::shared_ptr<Dataset> ds0 = Flickr("NotExistFile", file_path);
  EXPECT_NE(ds0, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid Flickr30k input
  EXPECT_EQ(iter0, nullptr);

  // Create a Flickr30k Dataset with non-existing annotation file
  std::shared_ptr<Dataset> ds1 = Flickr(dataset_path, "NotExistFile");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid Flickr30k input
  EXPECT_EQ(iter1, nullptr);

  // Create a Flickr30k Dataset with invalid string of dataset dir
  std::shared_ptr<Dataset> ds2 = Flickr(":*?\"<>|`&;'", file_path);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid Flickr30k input
  EXPECT_EQ(iter2, nullptr);

  // Create a Flickr30k Dataset with invalid string of annotation file
  std::shared_ptr<Dataset> ds3 = Flickr(dataset_path, ":*?\"<>|`&;'");
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid Flickr30k input
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: FlickrDataset
/// Description: Test FlickrDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFlickrWithNullSamplerError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlickrWithNullSamplerError.";

  std::string dataset_path = datasets_root_path_ + "/testFlickrData/flickr30k/flickr30k-images";
  std::string file_path = datasets_root_path_ + "/testFlickrData/flickr30k/test1.token";
  // Create a Flickr30k Dataset
  std::shared_ptr<Dataset> ds = Flickr(dataset_path, file_path, false, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Flickr30k input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
