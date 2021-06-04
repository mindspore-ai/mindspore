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
#include "minddata/dataset/util/circular_pool.h"
#include "minddata/dataset/core/client.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
namespace de = mindspore::dataset;

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestfilter_op : public UT::DatasetOpTesting {};

std::shared_ptr<de::FilterOp> Filter() {
  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  int32_t op_connector_size = config_manager->op_connector_size();
  int32_t num_workers = config_manager->num_parallel_workers();
  std::shared_ptr<TensorOp> predicate_func;
  std::vector<std::string> in_col_names = {};
  std::shared_ptr<de::FilterOp> op =
    std::make_shared<FilterOp>(in_col_names, num_workers, op_connector_size, predicate_func);
  return op;
}

TEST_F(MindDataTestfilter_op, Testfilter_opFuntions) {
  MS_LOG(INFO) << "Doing MindDataTest  filter_op.";
  auto my_tree = std::make_shared<ExecutionTree>();

  std::shared_ptr<DatasetOp> parent_op = Filter();

  std::shared_ptr<DatasetOp> leaf_op = Filter();
  my_tree->AssociateNode(parent_op);
  my_tree->AssociateNode(leaf_op);
  ASSERT_NE(parent_op, nullptr);
  ASSERT_NE(leaf_op, nullptr);
}
