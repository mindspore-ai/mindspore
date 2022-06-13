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
using mindspore::dataset::DataType;
using mindspore::dataset::dsize_t;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: Test WIDERFace dataset.
/// Description: Read data for default usage.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestWIDERFace) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWIDERFace.";
  // Create a WIDERFace Dataset.
  std::string folder_path = datasets_root_path_ + "/testWIDERFace/";
  std::shared_ptr<Dataset> ds = WIDERFace(folder_path);
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("bbox"), row.end());
  EXPECT_NE(row.find("blur"), row.end());
  EXPECT_NE(row.find("expression"), row.end());
  EXPECT_NE(row.find("illumination"), row.end());
  EXPECT_NE(row.find("occlusion"), row.end());
  EXPECT_NE(row.find("pose"), row.end());
  EXPECT_NE(row.find("invalid"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto bbox = row["bbox"];
    auto blur = row["blur"];
    auto expression = row["expression"];
    auto illumination = row["illumination"];
    auto occlusion = row["occlusion"];
    auto pose = row["pose"];
    auto invalid = row["invalid"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor bbox shape: " << bbox.Shape();
    MS_LOG(INFO) << "Tensor blur shape: " << blur.Shape();
    MS_LOG(INFO) << "Tensor expression shape: " << expression.Shape();
    MS_LOG(INFO) << "Tensor illumination shape: " << illumination.Shape();
    MS_LOG(INFO) << "Tensor occlusion shape: " << occlusion.Shape();
    MS_LOG(INFO) << "Tensor pose shape: " << pose.Shape();
    MS_LOG(INFO) << "Tensor invalid shape: " << invalid.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);
  iter->Stop();
}

/// Feature: Test WIDERFace dataset.
/// Description: Test usage "test".
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestWIDERFaceTest) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWIDERFaceTest.";
  // Create a WIDERFace Dataset.
  std::string folder_path = datasets_root_path_ + "/testWIDERFace/";
  std::shared_ptr<Dataset> ds = WIDERFace(folder_path, "test");
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);
  iter->Stop();
}

/// Feature: Test WIDERFace dataset.
/// Description: Test pipeline.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestWIDERFaceDefaultWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWIDERFaceDefaultWithPipeline.";
  // Create two WIDERFace Dataset.
  std::string folder_path = datasets_root_path_ + "/testWIDERFace/";

  std::shared_ptr<Dataset> ds1 = WIDERFace(folder_path);
  std::shared_ptr<Dataset> ds2 = WIDERFace(folder_path);
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds.
  int32_t repeat_num = 1;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 2;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds.
  std::vector<std::string> column_project = {"image",        "bbox",      "blur", "expression",
                                             "illumination", "occlusion", "pose", "invalid"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds.
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto bbox = row["bbox"];
    auto blur = row["blur"];
    auto expression = row["expression"];
    auto illumination = row["illumination"];
    auto occlusion = row["occlusion"];
    auto pose = row["pose"];
    auto invalid = row["invalid"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor bbox shape: " << bbox.Shape();
    MS_LOG(INFO) << "Tensor blur shape: " << blur.Shape();
    MS_LOG(INFO) << "Tensor expression shape: " << expression.Shape();
    MS_LOG(INFO) << "Tensor illumination shape: " << illumination.Shape();
    MS_LOG(INFO) << "Tensor occlusion shape: " << occlusion.Shape();
    MS_LOG(INFO) << "Tensor pose shape: " << pose.Shape();
    MS_LOG(INFO) << "Tensor invalid shape: " << invalid.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 12);
  iter->Stop();
}

/// Feature: Test WIDERFace dataset.
/// Description: Test WIDERFace getters.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestWIDERFaceGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWIDERFaceGetters.";
  // Create a WIDERFace Dataset.
  std::string folder_path = datasets_root_path_ + "/testWIDERFace/";

  std::shared_ptr<Dataset> ds = WIDERFace(folder_path);
  EXPECT_NE(ds, nullptr);

  std::vector<std::string> column_names = {"image",        "bbox",      "blur", "expression",
                                           "illumination", "occlusion", "pose", "invalid"};
  EXPECT_EQ(ds->GetDatasetSize(), 4);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: Test WIDERFace dataset.
/// Description: Test WIDERFace usage error.
/// Expectation: Throw error messages when certain errors occur.
TEST_F(MindDataTestPipeline, TestWIDERFaceWithUsageError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWIDERFaceWithNullSamplerFail.";
  // Create a WIDERFace Dataset.
  std::string folder_path = datasets_root_path_ + "/testWIDERFace/";

  std::shared_ptr<Dataset> ds = WIDERFace(folder_path, "off");
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid WIDERFace input, sampler cannot be nullptr.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test WIDERFace dataset.
/// Description: Test WIDERFace with SequentialSampler.
/// Expectation: The data is processed successfully.
TEST_F(MindDataTestPipeline, TestWIDERFaceSequentialSampler) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWIDERFaceSequentialSampler.";

  std::string folder_path = datasets_root_path_ + "/testWIDERFace/";
  // Create a WIDERFace Dataset.
  std::shared_ptr<Dataset> ds = WIDERFace(folder_path, "test", false, std::make_shared<SequentialSampler>(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: Test WIDERFace dataset.
/// Description: Test WIDERFace with invalid nullptr sampler.
/// Expectation: Throw error messages when certain errors occur.
TEST_F(MindDataTestPipeline, TestWIDERFaceWithNullSamplerError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWIDERFaceWithNullSamplerError.";

  // Create a WIDERFace Dataset.
  std::string folder_path = datasets_root_path_ + "/testWIDERFace/";
  std::shared_ptr<Dataset> ds = WIDERFace(folder_path, "all", false, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid WIDERFace input, sampler cannot be nullptr.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Test WIDERFace dataset.
/// Description: Test WIDERFace error.
/// Expectation: Throw error messages when certain errors occur.
TEST_F(MindDataTestPipeline, TestWIDERFaceError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestWIDERFaceError.";

  std::string folder_path = datasets_root_path_ + "/testWIDERFace/";
  // Create a WIDERFace Dataset with non-existing file.
  std::shared_ptr<Dataset> ds0 = WIDERFace("NotExistFile", "train");
  EXPECT_NE(ds0, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid WIDERFace input.
  EXPECT_EQ(iter0, nullptr);

  // Create a WIDERFace Dataset with invalid usage.
  std::shared_ptr<Dataset> ds1 = WIDERFace(folder_path, "invalid_usage");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid WIDERFace input.
  EXPECT_EQ(iter1, nullptr);

  // Create a WIDERFace Dataset with invalid string.
  std::shared_ptr<Dataset> ds2 = WIDERFace(":*?\"<>|`&;'", "train");
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid WIDERFace input.
  EXPECT_EQ(iter2, nullptr);
}
