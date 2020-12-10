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

#include <memory>
#include <string>
#include "minddata/dataset/core/client.h"
#include "common/common.h"
#include "gtest/gtest.h"

#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/opt/post/auto_worker_pass.h"
#include "minddata/dataset/engine/opt/pre/getter_pass.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::MsLogLevel::INFO;

class MindDataTestOptimizationPass : public UT::DatasetOpTesting {};


TEST_F(MindDataTestOptimizationPass, MindDataTestAutoWorkerPass) {
  MS_LOG(INFO) << "Doing MindDataTestOptimizationPass-MindDataTestAutoWorkerPass.";

  std::shared_ptr<SchemaObj> schema = std::make_shared<SchemaObj>();
  ASSERT_TRUE(schema->add_column("label", "uint32", {}));
  std::shared_ptr<Dataset> map_leaf = ImageFolder("dir")->SetNumWorkers(0);
  std::shared_ptr<Dataset> nonmap_leaf = RandomData(44, schema)->SetNumWorkers(0);
  std::shared_ptr<Dataset> batch = Zip({map_leaf, nonmap_leaf})->Batch(1)->SetNumWorkers(0);
  std::shared_ptr<Dataset> map = batch->Map({})->SetNumWorkers(0);
  //  {ImageFolder, RandomData} -> zip -> batch
  EXPECT_EQ(map_leaf->IRNode()->num_workers(), 0);
  EXPECT_EQ(nonmap_leaf->IRNode()->num_workers(), 0);
  EXPECT_EQ(batch->IRNode()->num_workers(), 0);
  EXPECT_EQ(map->IRNode()->num_workers(), 0);

  std::unique_ptr<IRPass> pass = std::make_unique<AutoWorkerPass>();
  bool m = false;
  ASSERT_OK(pass->Run(map->IRNode(), &m));

  // checking that after this pass, num_workers are set correctly (aka a positive number)
  // It is hard to test a exact value because num_threads are different for different machine
  // however, this will for sure succeed bc regardless of the total threads on cpu, this would always be >= 1
  EXPECT_NE(map_leaf->IRNode()->num_workers(), 0);
  EXPECT_NE(nonmap_leaf->IRNode()->num_workers(), 0);
  EXPECT_NE(batch->IRNode()->num_workers(), 0);
  EXPECT_NE(map->IRNode()->num_workers(), 0);
  MS_LOG(DEBUG) << map_leaf->IRNode()->Name() << ": num_worker=" << map_leaf->IRNode()->num_workers();
  MS_LOG(DEBUG) << nonmap_leaf->IRNode()->Name() << ": num_worker=" << nonmap_leaf->IRNode()->num_workers();
  MS_LOG(DEBUG) << batch->IRNode()->Name() << ": num_worker=" << batch->IRNode()->num_workers();
  MS_LOG(DEBUG) << map->IRNode()->Name() << ": num_worker=" << map->IRNode()->num_workers();
}
