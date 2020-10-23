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

// IR non-leaf nodes
#include "minddata/dataset/engine/ir/datasetops/batch_node.h"
#include "minddata/dataset/engine/ir/datasetops/bucket_batch_by_length_node.h"
#include "minddata/dataset/engine/ir/datasetops/concat_node.h"
#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/engine/ir/datasetops/project_node.h"
#include "minddata/dataset/engine/ir/datasetops/rename_node.h"
#include "minddata/dataset/engine/ir/datasetops/shuffle_node.h"
#include "minddata/dataset/engine/ir/datasetops/skip_node.h"
#include "minddata/dataset/engine/ir/datasetops/zip_node.h"


// IR leaf nodes
#include "minddata/dataset/engine/ir/datasetops/source/cifar100_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar10_node.h"

using namespace mindspore::dataset::api;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestCifar10Dataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10Dataset.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("label"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCifar10GetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10GetDatasetSize.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 10000);
}

TEST_F(MindDataTestPipeline, TestCifar100Dataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar100Dataset.";

  // Create a Cifar100 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  iter->GetNextRow(&row);

  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("coarse_label"), row.end());
  EXPECT_NE(row.find("fine_label"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestCifar100GetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar100GetDatasetSize.";

  // Create a Cifar100 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 10);
}

TEST_F(MindDataTestPipeline, TestCifar100DatasetFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar100DatasetFail1.";

  // Create a Cifar100 Dataset
  std::shared_ptr<Dataset> ds = Cifar100("", "all", RandomSampler(false, 10));
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestPipeline, TestCifar10DatasetFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10DatasetFail1.";

  // Create a Cifar10 Dataset
  std::shared_ptr<Dataset> ds = Cifar10("", "all", RandomSampler(false, 10));
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestPipeline, TestCifar10DatasetWithInvalidUsage) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10DatasetWithNullSampler.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "validation");
  // Expect failure: validation is not a valid usage
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestPipeline, TestCifar10DatasetWithNullSampler) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar10DatasetWithNullSampler.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", nullptr);
  // Expect failure: sampler can not be nullptr
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestPipeline, TestCifar100DatasetWithNullSampler) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar100DatasetWithNullSampler.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", nullptr);
  // Expect failure: sampler can not be nullptr
  EXPECT_EQ(ds, nullptr);
}

TEST_F(MindDataTestPipeline, TestCifar100DatasetWithWrongSampler) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCifar100DatasetWithWrongSampler.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar100Data/";
  std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", RandomSampler(false, -10));
  // Expect failure: sampler is not construnced correctly
  EXPECT_EQ(ds, nullptr);
}
