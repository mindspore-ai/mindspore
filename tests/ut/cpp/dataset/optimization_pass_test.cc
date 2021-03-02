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

#include <memory>
#include <string>

#include "common/common.h"
#include "gtest/gtest.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/engine/opt/optional/tensor_op_fusion_pass.h"
#include "minddata/dataset/engine/opt/post/auto_worker_pass.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"
#include "minddata/dataset/include/vision_lite.h"
#include "minddata/dataset/kernels/ir/data/transforms_ir.h"
#include "minddata/dataset/kernels/ir/vision/vision_ir.h"

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
  std::shared_ptr<Dataset> map = batch->Map({std::shared_ptr<TensorTransform>()})->SetNumWorkers(0);
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

TEST_F(MindDataTestOptimizationPass, MindDataTestTensorFusionPass) {
  MS_LOG(INFO) << "Doing MindDataTestOptimizationPass-MindDataTestTensorFusionPass.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  auto decode_op = vision::Decode();
  auto random_resized_crop_op = vision::RandomResizedCrop({100});
  std::shared_ptr<Dataset> root = ImageFolder(folder_path, false)->Map({decode_op, random_resized_crop_op}, {"image"});

  TensorOpFusionPass fusion_pass;
  bool modified = false;
  std::shared_ptr<MapNode> map_node = std::dynamic_pointer_cast<MapNode>(root->IRNode());
  // no deepcopy is performed because this doesn't go through tree_adapter
  fusion_pass.Run(root->IRNode(), &modified);
  EXPECT_EQ(modified, true);
  ASSERT_NE(map_node, nullptr);
  auto fused_ops = map_node->operations();
  ASSERT_EQ(fused_ops.size(), 1);
  ASSERT_EQ(fused_ops[0]->Name(), vision::kRandomCropDecodeResizeOperation);
}

TEST_F(MindDataTestOptimizationPass, MindDataTestTensorFusionPassPreBuiltTensorOperation) {
  MS_LOG(INFO) << "Doing MindDataTestOptimizationPass-MindDataTestTensorFusionPassPreBuiltTensorOperation.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  // make prebuilt tensor operation
  auto decode = std::make_shared<transforms::PreBuiltOperation>(vision::DecodeOperation(true).Build());
  auto resize = std::make_shared<transforms::PreBuiltOperation>(
    vision::RandomResizedCropOperation({100}, {0.5}, {0.1}, InterpolationMode::kNearestNeighbour, 5).Build());
  std::vector<std::shared_ptr<TensorOperation>> op_list = {decode, resize};
  std::vector<std::string> op_name = {"image"};
  std::shared_ptr<DatasetNode> root = ImageFolder(folder_path, false)->IRNode();
  std::shared_ptr<MapNode> map_node = std::make_shared<MapNode>(root, op_list, op_name);

  TensorOpFusionPass fusion_pass;
  bool modified = false;
  // no deepcopy is performed because this doesn't go through tree_adapter
  fusion_pass.Run(map_node, &modified);
  EXPECT_EQ(modified, true);
  ASSERT_NE(map_node, nullptr);
  auto fused_ops = map_node->operations();
  ASSERT_EQ(fused_ops.size(), 1);
  ASSERT_EQ(fused_ops[0]->Name(), kRandomCropDecodeResizeOp);
}
