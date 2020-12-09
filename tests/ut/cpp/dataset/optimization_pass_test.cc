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

class MindDataTestOptimizationPass : public UT::DatasetOpTesting {
 public:
  MindDataTestOptimizationPass() = default;
  void SetUp() override { GlobalInit(); }

  // this recursive function helps build a ExecutionTree from a IR node, it is copied from TreeAdapter
  Status DFSBuild(std::shared_ptr<DatasetNode> ir, std::shared_ptr<DatasetOp> *op, ExecutionTree *tree) {
    std::vector<std::shared_ptr<DatasetOp>> ops;
    RETURN_IF_NOT_OK(ir->Build(&ops));
    CHECK_FAIL_RETURN_UNEXPECTED(!ops.empty() && tree != nullptr && op != nullptr, "Fail To Build Tree.");
    (*op) = ops.front();
    RETURN_IF_NOT_OK(tree->AssociateNode(*op));
    for (size_t i = 1; i < ops.size(); i++) {
      RETURN_IF_NOT_OK(tree->AssociateNode(ops[i]));
      RETURN_IF_NOT_OK(ops[i - 1]->AddChild(ops[i]));
    }
    for (std::shared_ptr<DatasetNode> child_ir : ir->Children()) {
      std::shared_ptr<DatasetOp> child_op;
      RETURN_IF_NOT_OK(DFSBuild(child_ir, &child_op, tree));
      RETURN_IF_NOT_OK(ops.back()->AddChild(child_op));  // append children to the last of ops
    }
    return Status::OK();
  }

  // this function will build an execution_tree from a root ir node. nullptr will be returned if error occurs
  std::unique_ptr<ExecutionTree> BuildTree(std::shared_ptr<DatasetNode> ir) {
    std::unique_ptr<ExecutionTree> tree = std::make_unique<ExecutionTree>();
    std::shared_ptr<DatasetOp> root;
    if (DFSBuild(ir, &root, tree.get()).IsError()) return nullptr;
    if (tree->AssignRoot(root).IsError()) return nullptr;
    return tree;
  }
};

TEST_F(MindDataTestOptimizationPass, MindDataTestOutputShapeAndTypePass) {
  MS_LOG(INFO) << "Doing MindDataTestOptimizationPass-MindDataTestOutputShapeAndTypePass.";
  // config leaf_op, use random_data to avoid I/O
  std::shared_ptr<SchemaObj> schema = std::make_shared<SchemaObj>();
  ASSERT_TRUE(schema->add_column("label", "uint32", {}));
  std::shared_ptr<Dataset> ds = RandomData(44, schema)->Repeat(2)->Project({"label"})->Shuffle(10)->Batch(2);

  std::unique_ptr<ExecutionTree> exe_tree = BuildTree(ds->IRNode());

  ASSERT_NE(exe_tree, nullptr);

  // test the optimization pass
  // OptPass is supposed to remove concat, filter repeat, shuffle skip, take and set the callback of map to empty
  std::function<OptPass(OptPass)> pass = [](OptPass pre) {
    // return a new pass, this will override all the existing pre-pass es
    pre.clear();
    pre.push_back(std::make_unique<GetterPass>(GetterPass::kOutputShapeAndType));
    return pre;
  };

  exe_tree->SetPrePassOverride(pass);
  ASSERT_OK(exe_tree->PreAction());
  std::stringstream ss;

  // print the tree in std::string as a way to verify that nodes are indeed removed
  exe_tree->Print(ss);
  std::string ss_str = ss.str();

  // ss_str would look like this
  //  +- ( 0) <BatchOp>: [workers: 4] [batch size: 2]
  //    +- ( 2) <ProjectOp>: [workers: 0 (inlined)]
  //        +- ( 4) <RandomDataOp>: [workers: 4] [total rows: 44]
  //

  // verify that no ops are removed, but Batch and ProjectOp are not
  EXPECT_NE(ss_str.find("ShuffleOp"), ss_str.npos);
  EXPECT_NE(ss_str.find("RepeatOp"), ss_str.npos);
  EXPECT_NE(ss_str.find("ProjectOp"), ss_str.npos);
  EXPECT_NE(ss_str.find("BatchOp"), ss_str.npos);
}

TEST_F(MindDataTestOptimizationPass, MindDataTestDatasetSizePass) {
  MS_LOG(INFO) << "Doing MindDataTestOptimizationPass-MindDataTestDatasetSizePass.";
  // config leaf_op, use random_data to avoid I/O
  std::shared_ptr<SchemaObj> schema = std::make_shared<SchemaObj>();
  ASSERT_TRUE(schema->add_column("label", "uint32", {}));
  std::shared_ptr<Dataset> ds = RandomData(44, schema)->Repeat(2)->Project({"label"})->Shuffle(10)->Batch(2);

  std::unique_ptr<ExecutionTree> exe_tree = BuildTree(ds->IRNode());

  ASSERT_NE(exe_tree, nullptr);

  // test the optimization pass
  // OptPass is supposed to remove concat, filter repeat, shuffle skip, take and set the callback of map to empty
  std::function<OptPass(OptPass)> pass = [](OptPass pre) {
    // return a new pass, this will override all the existing pre-pass es
    pre.clear();  // remove all existing pre pass
    pre.push_back(std::make_unique<GetterPass>(GetterPass::kDatasetSize));
    return pre;
  };

  exe_tree->SetPrePassOverride(pass);
  ASSERT_OK(exe_tree->PreAction());
  std::stringstream ss;
  // print the tree in std::string as a way to verify that nodes are indeed removed
  exe_tree->Print(ss);
  std::string ss_str = ss.str();

  // verify that no ops are removed, but Batch and ProjectOp are not
  EXPECT_NE(ss_str.find("ShuffleOp"), ss_str.npos);
  EXPECT_NE(ss_str.find("RepeatOp"), ss_str.npos);
  EXPECT_NE(ss_str.find("ProjectOp"), ss_str.npos);
  EXPECT_NE(ss_str.find("BatchOp"), ss_str.npos);
}

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
