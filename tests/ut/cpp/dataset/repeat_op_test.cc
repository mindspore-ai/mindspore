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
#include "dataset/util/circular_pool.h"
#include "dataset/core/client.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestrepeat_op : public UT::DatasetOpTesting {

};

TEST_F(MindDataTestrepeat_op, Testrepeat_opFuntions) {
  MS_LOG(INFO) << "Doing MindDataTestrepeat_op.";
  auto my_tree = std::make_shared<ExecutionTree>();

  std::shared_ptr<DatasetOp> parent_op = std::make_shared<RepeatOp>(32);
  std::string dataset_path;
  dataset_path = datasets_root_path_ + "/testTFTestAllTypes/test.data";
// TFReaderOp
  std::shared_ptr<TFReaderOp> my_tfreader_op;
  TFReaderOp::Builder builder;
  builder.SetDatasetFilesList({dataset_path})
    .SetRowsPerBuffer(16)
    .SetWorkerConnectorSize(16)
    .SetNumWorkers(16);
  Status rc= builder.Build(&my_tfreader_op);
  ASSERT_TRUE(rc.IsOk());
  rc = my_tree->AssociateNode(my_tfreader_op);
  ASSERT_TRUE(rc.IsOk());
  my_tree->AssociateNode(parent_op);
  ASSERT_NE(parent_op, nullptr);
  ASSERT_NE(my_tfreader_op, nullptr);
  parent_op->AddChild(std::move(my_tfreader_op));
  MS_LOG(INFO) << parent_op;
  my_tree->Prepare();

  RepeatOp RepeatOpOp();

  std::shared_ptr<RepeatOp> repeat_op;
  rc = RepeatOp::Builder(3).Build(&repeat_op);
  ASSERT_NE(repeat_op, nullptr);
}
