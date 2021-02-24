/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/tree_adapter.h"
#include "common/common.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/transforms.h"

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

#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/mnist_node.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestTreeAdapter : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestTreeAdapter, TestSimpleTreeAdapter) {
  MS_LOG(INFO) << "Doing MindDataTestTreeAdapter-TestSimpleTreeAdapter.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<SequentialSampler>(0, 4));
  EXPECT_NE(ds, nullptr);

  ds = ds->Batch(2);
  EXPECT_NE(ds, nullptr);

  mindspore::dataset::TreeAdapter tree_adapter;

  Status rc = tree_adapter.Compile(ds->IRNode(), 1);

  EXPECT_TRUE(rc.IsOk());

  const std::unordered_map<std::string, int32_t> map = {{"label", 1}, {"image", 0}};
  EXPECT_EQ(tree_adapter.GetColumnNameMap(), map);

  std::vector<size_t> row_sizes = {2, 2, 0};

  TensorRow row;
  for (size_t sz : row_sizes) {
    rc = tree_adapter.GetNext(&row);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(row.size(), sz);
  }

  rc = tree_adapter.GetNext(&row);
  EXPECT_TRUE(rc.IsError());
  const std::string err_msg = rc.ToString();
  EXPECT_TRUE(err_msg.find("EOF buffer encountered.") != err_msg.npos);
}

TEST_F(MindDataTestTreeAdapter, TestTreeAdapterWithRepeat) {
  MS_LOG(INFO) << "Doing MindDataTestTreeAdapter-TestTreeAdapterWithRepeat.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  ds = ds->Batch(2, false);
  EXPECT_NE(ds, nullptr);

  mindspore::dataset::TreeAdapter tree_adapter;

  Status rc = tree_adapter.Compile(ds->IRNode(), 2);
  EXPECT_TRUE(rc.IsOk());

  const std::unordered_map<std::string, int32_t> map = tree_adapter.GetColumnNameMap();
  EXPECT_EQ(tree_adapter.GetColumnNameMap(), map);

  std::vector<size_t> row_sizes = {2, 2, 0, 2, 2, 0};

  TensorRow row;
  for (size_t sz : row_sizes) {
    rc = tree_adapter.GetNext(&row);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(row.size(), sz);
  }
  rc = tree_adapter.GetNext(&row);
  const std::string err_msg = rc.ToString();
  EXPECT_TRUE(err_msg.find("EOF buffer encountered.") != err_msg.npos);
}

TEST_F(MindDataTestTreeAdapter, TestProjectMapTreeAdapter) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestProjectMap.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot = std::make_shared<transforms::OneHot>(10);
  EXPECT_NE(one_hot, nullptr);

  // Create a Map operation, this will automatically add a project after map
  ds = ds->Map({one_hot}, {"label"}, {"label"}, {"label"});
  EXPECT_NE(ds, nullptr);

  mindspore::dataset::TreeAdapter tree_adapter;

  Status rc = tree_adapter.Compile(ds->IRNode(), 2);

  EXPECT_TRUE(rc.IsOk());

  const std::unordered_map<std::string, int32_t> map = {{"label", 0}};
  EXPECT_EQ(tree_adapter.GetColumnNameMap(), map);

  std::vector<size_t> row_sizes = {1, 1, 0, 1, 1, 0};
  TensorRow row;

  for (size_t sz : row_sizes) {
    rc = tree_adapter.GetNext(&row);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(row.size(), sz);
  }
  rc = tree_adapter.GetNext(&row);
  const std::string err_msg = rc.ToString();
  EXPECT_TRUE(err_msg.find("EOF buffer encountered.") != err_msg.npos);
}
