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
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "gtest/gtest.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/core/client.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include <memory>
#include <vector>
#include <iostream>

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestClientConfig : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestClientConfig, TestClientConfig1) {
  std::shared_ptr<ConfigManager> my_conf = GlobalContext::config_manager();

  ASSERT_EQ(my_conf->num_parallel_workers(), kCfgParallelWorkers);
  ASSERT_EQ(my_conf->worker_connector_size(), kCfgWorkerConnectorSize);
  ASSERT_EQ(my_conf->op_connector_size(), kCfgOpConnectorSize);
  ASSERT_EQ(my_conf->seed(), kCfgDefaultSeed);

  my_conf->set_num_parallel_workers(2);
  my_conf->set_worker_connector_size(3);
  my_conf->set_op_connector_size(4);
  my_conf->set_seed(5);
  my_conf->set_enable_shared_mem(false);

  ASSERT_EQ(my_conf->num_parallel_workers(), 2);
  ASSERT_EQ(my_conf->worker_connector_size(), 3);
  ASSERT_EQ(my_conf->op_connector_size(), 4);
  ASSERT_EQ(my_conf->seed(), 5);
  ASSERT_EQ(my_conf->enable_shared_mem(), false);

  std::string file = datasets_root_path_ + "/declient.cfg";
  ASSERT_TRUE(my_conf->LoadFile(file));

  ASSERT_EQ(my_conf->num_parallel_workers(), kCfgParallelWorkers);
  ASSERT_EQ(my_conf->worker_connector_size(), kCfgWorkerConnectorSize);
  ASSERT_EQ(my_conf->op_connector_size(), kCfgOpConnectorSize);
  ASSERT_EQ(my_conf->seed(), kCfgDefaultSeed);
}

TEST_F(MindDataTestClientConfig, TestClientConfig2) {
  std::shared_ptr<ConfigManager> my_conf = GlobalContext::config_manager();

  my_conf->set_num_parallel_workers(8);

  Status rc;

  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();

  // Test info:
  // Dataset from testDataset1 has 10 rows, 2 columns.
  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testDataset1/testDataset1.data";
  // get defaults for tf_reader
  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  auto op_connector_size = config_manager->op_connector_size();
  std::vector<std::string> columns_to_load = {};
  std::vector<std::string> files = {dataset_path};
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  std::shared_ptr<TFReaderOp> my_tfreader_op = std::make_shared<TFReaderOp>(
    1, 2, 0, files, std::move(schema), op_connector_size, columns_to_load, false, 1, 0, false);
  rc = my_tfreader_op->Init();
  ASSERT_OK(rc);
  ASSERT_EQ(my_tfreader_op->NumWorkers(), 1);
  my_tree->AssociateNode(my_tfreader_op);

  // Set children/root layout.
  my_tree->AssignRoot(my_tfreader_op);

  my_tree->Prepare();
  my_tree->Launch();

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }
  ASSERT_EQ(row_count, 10);  // Should be 10 rows fetched
  ASSERT_EQ(my_tfreader_op->NumWorkers(), 1);
}
