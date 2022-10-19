/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/transforms.h"

// IR non-leaf nodes
#include "minddata/dataset/engine/ir/datasetops/bucket_batch_by_length_node.h"

#include "minddata/dataset/engine/tree_modifier.h"
#include "minddata/dataset/engine/serdes.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestTreeAdapter : public UT::DatasetOpTesting {};

// Feature: TreeAdapter
// Description: Test TreeAdapter in simple case without IR optimization
// Expectation: Runs successfully
TEST_F(MindDataTestTreeAdapter, TestSimpleTreeAdapter) {
  MS_LOG(INFO) << "Doing MindDataTestTreeAdapter-TestSimpleTreeAdapter.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<SequentialSampler>(0, 4));
  EXPECT_NE(ds, nullptr);

  ds = ds->Batch(2);
  EXPECT_NE(ds, nullptr);

  auto tree_adapter = std::make_shared<TreeAdapter>();

  // Disable IR optimization pass
  tree_adapter->SetOptimize(false);

  Status rc = tree_adapter->Compile(ds->IRNode(), 1);

  EXPECT_TRUE(rc.IsOk());

  const std::unordered_map<std::string, int32_t> map = {{"label", 1}, {"image", 0}};
  EXPECT_EQ(tree_adapter->GetColumnNameMap(), map);

  std::vector<size_t> row_sizes = {2, 2, 0};

  TensorRow row;
  for (size_t sz : row_sizes) {
    rc = tree_adapter->GetNext(&row);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(row.size(), sz);
  }

  rc = tree_adapter->GetNext(&row);
  EXPECT_TRUE(rc.IsError());
  const std::string err_msg = rc.ToString();
  EXPECT_TRUE(err_msg.find("EOF buffer encountered.") != err_msg.npos);
}

// Feature: TreeAdapter
// Description: Test TreeAdapter with repeated row_sizes
// Expectation: Runs successfully
TEST_F(MindDataTestTreeAdapter, TestTreeAdapterWithRepeat) {
  MS_LOG(INFO) << "Doing MindDataTestTreeAdapter-TestTreeAdapterWithRepeat.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  ds = ds->Batch(2, false);
  EXPECT_NE(ds, nullptr);

  auto tree_adapter = std::make_shared<TreeAdapter>();

  Status rc = tree_adapter->Compile(ds->IRNode(), 2);
  EXPECT_TRUE(rc.IsOk());

  const std::unordered_map<std::string, int32_t> map = tree_adapter->GetColumnNameMap();
  EXPECT_EQ(tree_adapter->GetColumnNameMap(), map);

  std::vector<size_t> row_sizes = {2, 2, 0, 2, 2, 0};

  TensorRow row;
  for (size_t sz : row_sizes) {
    rc = tree_adapter->GetNext(&row);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(row.size(), sz);
  }
  rc = tree_adapter->GetNext(&row);
  const std::string err_msg = rc.ToString();
  EXPECT_TRUE(err_msg.find("EOF buffer encountered.") != err_msg.npos);
}

// Feature: TreeAdapter
// Description: Test TreeAdapter on dataset that has been projected
// Expectation: Runs successfully
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
  ds = ds->Map({one_hot}, {"label"}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<ProjectDataset> project_ds = ds->Project({"label"});

  auto tree_adapter = std::make_shared<TreeAdapter>();

  Status rc = tree_adapter->Compile(project_ds->IRNode(), 2);

  EXPECT_TRUE(rc.IsOk());

  const std::unordered_map<std::string, int32_t> map = {{"label", 0}};
  EXPECT_EQ(tree_adapter->GetColumnNameMap(), map);

  std::vector<size_t> row_sizes = {1, 1, 0, 1, 1, 0};
  TensorRow row;

  for (size_t sz : row_sizes) {
    rc = tree_adapter->GetNext(&row);
    EXPECT_TRUE(rc.IsOk());
    EXPECT_EQ(row.size(), sz);
  }
  rc = tree_adapter->GetNext(&row);
  const std::string err_msg = rc.ToString();
  EXPECT_TRUE(err_msg.find("EOF buffer encountered.") != err_msg.npos);
}

// Feature: Test for Serializing and Deserializing an optimized IR Tree after the tree has been modified with
// TreeModifier or in other words through Autotune indirectly.
// Description: Create a simple tree, modify the workers and queue size, serialize the optimized IR Tree, obtain a new
// tree with deserialize and then compare the output of serializing the new optimized IR tree with the first tree.
// Expectation: No failures.
TEST_F(MindDataTestTreeAdapter, TestOptimizedTreeSerializeDeserializeForAutoTune) {
  MS_LOG(INFO) << "Doing MindDataTestTreeAdapter-TestOptimizedTreeSerializeDeserializeForAutoTune.";

  // Create a CSVDataset, with single CSV file
  std::string train_file = datasets_root_path_ + "/testCSV/1.csv";
  std::vector<std::string> column_names = {"col1", "col2", "col3", "col4"};
  std::shared_ptr<Dataset> ds = CSV({train_file}, ',', {}, column_names, 0, ShuffleMode::kFalse);
  ASSERT_NE(ds, nullptr);
  ds = ds->Project({"col1"});
  ASSERT_NE(ds, nullptr);
  ds = ds->Repeat(2);
  ASSERT_NE(ds, nullptr);
  auto to_number = std::make_shared<text::ToNumber>(mindspore::DataType::kNumberTypeInt32);
  ASSERT_NE(to_number, nullptr);
  ds = ds->Map({to_number}, {"col1"}, {"col1"});
  ds->SetNumWorkers(1);
  ds = ds->Batch(1);
  ds->SetNumWorkers(1);

  // Create a tree adapter and compile the IR Tree
  auto tree_adapter1 = std::make_shared<TreeAdapter>();
  ASSERT_OK(tree_adapter1->Compile(ds->IRNode(), 1));

  // Change num_parallel_workers and connector_queue_size for some ops
  auto tree_modifier = std::make_unique<TreeModifier>(tree_adapter1.get());
  tree_modifier->AddChangeRequest(1, std::make_shared<ChangeNumWorkersRequest>(10));
  tree_modifier->AddChangeRequest(1, std::make_shared<ResizeConnectorRequest>(20));
  tree_modifier->AddChangeRequest(0, std::make_shared<ResizeConnectorRequest>(100));
  tree_modifier->AddChangeRequest(0, std::make_shared<ChangeNumWorkersRequest>(10));

  std::vector<int32_t> expected_result = {1, 5, 9, 1, 5, 9};
  TensorRow row;

  uint64_t i = 0;
  ASSERT_OK(tree_adapter1->GetNext(&row));
  while (!row.empty()) {
    auto tensor = row[0];
    int32_t num;
    ASSERT_OK(tensor->GetItemAt(&num, {0}));
    EXPECT_EQ(num, expected_result[i]);
    ASSERT_OK(tree_adapter1->GetNext(&row));
    i++;
  }
  // Expect 6 samples
  EXPECT_EQ(i, 6);

  // Serialize the optimized IR Tree
  nlohmann::json out_json;
  ASSERT_OK(Serdes::SaveToJSON(tree_adapter1->RootIRNode(), "", &out_json));

  // Check that updated values of num_parallel_workers and connector_queue_size are not reflected in the json
  EXPECT_EQ(out_json["op_type"], "Batch");
  EXPECT_NE(out_json["num_parallel_workers"], 10);
  EXPECT_NE(out_json["connector_queue_size"], 100);

  EXPECT_EQ(out_json["children"][0]["op_type"], "Map");
  EXPECT_NE(out_json["children"][0]["num_parallel_workers"], 10);
  EXPECT_NE(out_json["children"][0]["connector_queue_size"], 20);

  // Create an op_id to dataset op mapping
  std::map<int32_t, std::shared_ptr<DatasetOp>> op_mapping;
  auto tree = tree_adapter1->GetExecutionTree();
  ASSERT_NE(tree, nullptr);

  for (auto itr = tree->begin(); itr != tree->end(); ++itr) {
    op_mapping[itr->id()] = itr.get();
  }

  // Update the serialized JSON object of the optimized IR tree
  ASSERT_OK(Serdes::UpdateOptimizedIRTreeJSON(&out_json, op_mapping));

  // Check that updated values of num_parallel_workers and connector_queue_size are reflected in the json now
  EXPECT_EQ(out_json["op_type"], "Batch");
  EXPECT_EQ(out_json["num_parallel_workers"], 10);
  EXPECT_EQ(out_json["connector_queue_size"], 100);

  EXPECT_EQ(out_json["children"][0]["op_type"], "Map");
  EXPECT_EQ(out_json["children"][0]["num_parallel_workers"], 10);
  EXPECT_EQ(out_json["children"][0]["connector_queue_size"], 20);

  // Deserialize the above updated serialized optimized IR Tree
  std::shared_ptr<DatasetNode> deserialized_node;
  ASSERT_OK(Serdes::ConstructPipeline(out_json, &deserialized_node));

  // Create a new tree adapter and compile the IR Tree obtained from deserialization above
  auto tree_adapter2 = std::make_shared<TreeAdapter>();
  ASSERT_OK(tree_adapter2->Compile(deserialized_node, 1));

  // Serialize the new optimized IR Tree
  nlohmann::json out_json1;
  ASSERT_OK(Serdes::SaveToJSON(tree_adapter2->RootIRNode(), "", &out_json1));

  // Ensure that both the serialized outputs are equal
  EXPECT_TRUE(out_json == out_json1);

  i = 0;
  ASSERT_OK(tree_adapter2->GetNext(&row));
  while (!row.empty()) {
    auto tensor = row[0];
    int32_t num;
    ASSERT_OK(tensor->GetItemAt(&num, {0}));
    EXPECT_EQ(num, expected_result[i]);
    ASSERT_OK(tree_adapter2->GetNext(&row));
    i++;
  }

  // Expect 6 samples
  EXPECT_EQ(i, 6);
}

// Feature: Basic test for TreeModifier
// Description: Create simple tree and modify the tree by adding workers, change queue size and then removing workers
// Expectation: No failures.
TEST_F(MindDataTestTreeAdapter, TestSimpleTreeModifier) {
  MS_LOG(INFO) << "Doing MindDataTestTreeAdapter-TestSimpleTreeModifier.";

  // Create a CSVDataset, with single CSV file
  std::string train_file = datasets_root_path_ + "/testCSV/1.csv";
  std::vector<std::string> column_names = {"col1", "col2", "col3", "col4"};
  std::shared_ptr<Dataset> ds = CSV({train_file}, ',', {}, column_names, 0, ShuffleMode::kFalse);
  ASSERT_NE(ds, nullptr);
  ds = ds->Project({"col1"});
  ASSERT_NE(ds, nullptr);
  ds = ds->Repeat(2);
  ASSERT_NE(ds, nullptr);
  auto to_number = std::make_shared<text::ToNumber>(mindspore::DataType::kNumberTypeInt32);
  ASSERT_NE(to_number, nullptr);
  ds = ds->Map({to_number}, {"col1"}, {"col1"});
  ds->SetNumWorkers(1);
  ds = ds->Batch(1);
  ds->SetNumWorkers(1);

  auto tree_adapter = std::make_shared<TreeAdapter>();
  // Disable IR optimization pass
  tree_adapter->SetOptimize(false);
  ASSERT_OK(tree_adapter->Compile(ds->IRNode(), 1));

  auto tree_modifier = std::make_unique<TreeModifier>(tree_adapter.get());
  tree_modifier->AddChangeRequest(1, std::make_shared<ChangeNumWorkersRequest>(2));
  tree_modifier->AddChangeRequest(1, std::make_shared<ChangeNumWorkersRequest>());
  tree_modifier->AddChangeRequest(1, std::make_shared<ChangeNumWorkersRequest>(10));

  tree_modifier->AddChangeRequest(1, std::make_shared<ResizeConnectorRequest>(20));
  tree_modifier->AddChangeRequest(0, std::make_shared<ResizeConnectorRequest>(100));

  tree_modifier->AddChangeRequest(0, std::make_shared<ChangeNumWorkersRequest>(2));
  tree_modifier->AddChangeRequest(0, std::make_shared<ChangeNumWorkersRequest>());
  tree_modifier->AddChangeRequest(0, std::make_shared<ChangeNumWorkersRequest>(10));

  std::vector<int32_t> expected_result = {1, 5, 9, 1, 5, 9};
  TensorRow row;

  uint64_t i = 0;
  ASSERT_OK(tree_adapter->GetNext(&row));

  while (!row.empty()) {
    auto tensor = row[0];
    int32_t num;
    ASSERT_OK(tensor->GetItemAt(&num, {0}));
    EXPECT_EQ(num, expected_result[i]);
    ASSERT_OK(tree_adapter->GetNext(&row));
    i++;
  }

  // Expect 6 samples
  EXPECT_EQ(i, 6);
}

// Feature: Test for TreeModifier on MindDataset
// Description: Create a simple tree with a Mindrecord op first add then add and remove workers afterward. Collect
// file_name of images when executing the first tree and then compare the outputs of the other runs against it.
// Expectation: No failures.
TEST_F(MindDataTestTreeAdapter, TestTreeModifierMindRecord) {
  MS_LOG(INFO) << "Doing MindDataTestTreeAdapter-TestTreeModifierMindRecord.";

  // Create a MindData Dataset
  // Pass one mindrecord shard file to parse dataset info, and search for other mindrecord files with same dataset info,
  // thus all records in imagenet.mindrecord0 ~ imagenet.mindrecord3 will be read (we only collect "file_name" column).
  std::string file_path = datasets_root_path_ + "/../mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0";
  std::shared_ptr<Dataset> ds = MindData(file_path, {"file_name"}, std::make_shared<SequentialSampler>(0, 20));
  EXPECT_NE(ds, nullptr);
  ds->SetNumWorkers(1);

  TensorRow row;
  std::vector<std::string> file_names;

  auto tree_adapter = std::make_shared<TreeAdapter>();
  // Disable IR optimization pass
  tree_adapter->SetOptimize(false);

  ASSERT_OK(tree_adapter->Compile(ds->IRNode(), 1));
  // Iterate the dataset and collect the file_names in the dataset
  ASSERT_OK(tree_adapter->GetNext(&row));
  uint64_t i = 0;
  while (!row.empty()) {
    auto tensor = row[0];
    std::string_view sv;
    ASSERT_OK(tensor->GetItemAt(&sv, {}));
    std::string image_name(sv);
    file_names.push_back(image_name);

    ASSERT_OK(tree_adapter->GetNext(&row));
    i++;
  }
  // Expect 20 samples
  EXPECT_EQ(i, 20);

  auto tree_adapter2 = std::make_shared<TreeAdapter>();
  // Disable IR optimization pass
  tree_adapter2->SetOptimize(false);
  ASSERT_OK(tree_adapter2->Compile(ds->IRNode(), 1));
  auto tree_modifier1 = std::make_unique<TreeModifier>(tree_adapter2.get());
  // Change number of workers for MindDataset from 1 to 5
  tree_modifier1->AddChangeRequest(0, std::make_shared<ChangeNumWorkersRequest>(5));

  i = 0;
  ASSERT_OK(tree_adapter2->GetNext(&row));
  while (!row.empty()) {
    auto tensor = row[0];
    std::string_view sv;
    ASSERT_OK(tensor->GetItemAt(&sv, {}));
    std::string image_name(sv);
    EXPECT_EQ(image_name, file_names[i]);

    ASSERT_OK(tree_adapter2->GetNext(&row));
    i++;
  }
  // Expect 20 samples
  EXPECT_EQ(i, 20);

  auto tree_adapter3 = std::make_shared<TreeAdapter>();
  // Disable IR optimization pass
  tree_adapter3->SetOptimize(false);
  ASSERT_OK(tree_adapter3->Compile(ds->IRNode(), 1));
  auto tree_modifier2 = std::make_unique<TreeModifier>(tree_adapter3.get());
  // Change number of workers for MindDataset from 5 to 2
  tree_modifier2->AddChangeRequest(0, std::make_shared<ChangeNumWorkersRequest>(2));

  i = 0;
  ASSERT_OK(tree_adapter3->GetNext(&row));
  while (!row.empty()) {
    auto tensor = row[0];
    std::string_view sv;
    ASSERT_OK(tensor->GetItemAt(&sv, {}));
    std::string image_name(sv);
    EXPECT_EQ(image_name, file_names[i]);

    ASSERT_OK(tree_adapter3->GetNext(&row));
    i++;
  }
  // Expect 20 samples
  EXPECT_EQ(i, 20);
}
