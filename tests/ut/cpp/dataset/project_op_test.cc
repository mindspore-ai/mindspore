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
#include <iostream>
#include <memory>
#include <vector>

#include "common/common.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestProjectOp : public UT::DatasetOpTesting {};

TEST_F(MindDataTestProjectOp, TestProjectProject) {
  // Start with an empty execution tree
  auto my_tree = std::make_shared<ExecutionTree>();
  Status rc;
  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testTFTestAllTypes/test.data";

  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  auto op_connector_size = config_manager->op_connector_size();
  auto num_workers = 1;  // one file, one worker
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  std::vector<std::string> columns_to_load = {};
  std::vector<std::string> files = {dataset_path};
  schema->LoadSchemaFile(datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json", {});
  std::shared_ptr<TFReaderOp> my_tfreader_op = std::make_shared<TFReaderOp>(
    num_workers, 16, 0, files, std::move(schema), op_connector_size, columns_to_load, false, 1, 0, false);
  rc = my_tfreader_op->Init();
  ASSERT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_tfreader_op);
  ASSERT_TRUE(rc.IsOk());

  // ProjectOp
  std::vector<std::string> columns_to_project = {"col_sint16", "col_float", "col_2d"};
  std::shared_ptr<ProjectOp> my_project_op = std::make_shared<ProjectOp>(columns_to_project);
  rc = my_tree->AssociateNode(my_project_op);
  ASSERT_TRUE(rc.IsOk());

  // Set children/root layout.
  rc = my_project_op->AddChild(my_tfreader_op);
  ASSERT_TRUE(rc.IsOk());
  rc = my_tree->AssignRoot(my_project_op);
  ASSERT_TRUE(rc.IsOk());

  MS_LOG(INFO) << "Launching tree and begin iteration.";
  rc = my_tree->Prepare();

  ASSERT_TRUE(rc.IsOk());

  rc = my_tree->Launch();
  ASSERT_TRUE(rc.IsOk());

  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(my_tree);
  TensorRow tensor_list;
  rc = di.FetchNextTensorRow(&tensor_list);
  ASSERT_TRUE(rc.IsOk());

  int row_count = 0;
  while (!tensor_list.empty()) {
    MS_LOG(INFO) << "Row display for row #: " << row_count << ".";

    ASSERT_EQ(tensor_list.size(), columns_to_project.size());

    // Display the tensor by calling the printer on it
    for (int i = 0; i < tensor_list.size(); i++) {
      std::ostringstream ss;
      ss << "(" << tensor_list[i] << "): " << *tensor_list[i] << std::endl;
      MS_LOG(INFO) << "Tensor print: " << ss.str() << ".";
    }

    rc = di.FetchNextTensorRow(&tensor_list);
    ASSERT_TRUE(rc.IsOk());
    row_count++;
  }

  ASSERT_EQ(row_count, 12);
}
