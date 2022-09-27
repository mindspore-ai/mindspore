/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/opt/pre/skip_pushdown_pass.h"
#include "minddata/dataset/include/dataset/samplers.h"
#include "minddata/dataset/include/dataset/vision.h"

using namespace mindspore::dataset;

class MindDataSkipPushdownTestOptimizationPass : public UT::DatasetOpTesting {
 protected:
  MindDataSkipPushdownTestOptimizationPass() {}

  /// \brief Compile and compare two datasets
  /// \param[in] root_original Original dataset to be added the skip step
  /// \param[in] root_target Target dataset for compare
  /// \param[in] step Skip step
  /// \return Status of the function
  Status prepare_trees(std::shared_ptr<Dataset> root_original, std::shared_ptr<Dataset> root_target, int64_t step = 0) {
    auto ir_tree = std::make_shared<TreeAdapter>(TreeAdapter::UsageFlag::kDeReset);
    // Compile adds a new RootNode to the top of the tree
    RETURN_IF_NOT_OK(ir_tree->Compile(root_original->IRNode(), 1, step));

    auto ir_tree_target = std::make_shared<TreeAdapter>();
    // Compile adds a new RootNode to the top of the tree
    RETURN_IF_NOT_OK(ir_tree_target->Compile(root_target->IRNode(), 1,
                                             0));  // Step is 0 for target node tree

    if (step != 0) {
      RETURN_IF_NOT_OK(compare_pass(ir_tree_target->RootIRNode(), ir_tree->RootIRNode()));
    }
    RETURN_IF_NOT_OK(compare_pass_row(ir_tree_target, ir_tree));
    return Status::OK();
  }

  /// \brief Compare two dataset node trees
  /// \param[in] expect Expected node tree for compare
  /// \param[in] root Root node tree for compare
  /// \return Status of the function
  Status compare_pass(std::shared_ptr<DatasetNode> expect, std::shared_ptr<DatasetNode> root) {
    if (expect->Children().size() == root->Children().size() && expect->Children().size() == 0) {
      return Status::OK();
    }
    if (expect->Children().size() == root->Children().size() && expect->Children().size() != 0) {
      for (int i = 0; i < expect->Children().size(); i++) {
        std::string expect_name = expect->Children()[i]->Name();
        std::string root_name = root->Children()[i]->Name();
        CHECK_FAIL_RETURN_UNEXPECTED(expect_name == root_name,
                                     "Expect child is " + expect_name + ", but got " + root_name);
        RETURN_IF_NOT_OK(compare_pass(expect->Children()[i], root->Children()[i]));
      }
    } else {
      return Status(StatusCode::kMDUnexpectedError, "Skip Optimization is not working as expected, expect to have " +
                                                      std::to_string(expect->Children().size()) +
                                                      " operation, but got " + std::to_string(root->Children().size()));
    }
    return Status::OK();
  }

  /// \brief Compare each row of two dataset node trees
  /// \param[in] expect Expected tree for compare
  /// \param[in] root Root tree for compare
  /// \return Status of the function
  Status compare_pass_row(std::shared_ptr<TreeAdapter> expect, std::shared_ptr<TreeAdapter> root) {
    TensorRow row_expect;
    TensorRow row_root;
    RETURN_IF_NOT_OK(expect->GetNext(&row_expect));
    RETURN_IF_NOT_OK(root->GetNext(&row_root));
    while (row_expect.size() != 0 && row_root.size() != 0) {
      std::vector<std::shared_ptr<Tensor>> e = row_expect.getRow();
      std::vector<std::shared_ptr<Tensor>> r = row_root.getRow();
      for (int i = 0; i < e.size(); i++) {
        nlohmann::json out_json;
        RETURN_IF_NOT_OK(e[i]->to_json(&out_json));
        std::stringstream json_ss;
        json_ss << out_json;

        nlohmann::json out_json1;
        RETURN_IF_NOT_OK(r[i]->to_json(&out_json1));
        std::stringstream json_ss1;
        json_ss1 << out_json1;
        EXPECT_EQ(json_ss.str(), json_ss1.str());
      }
      RETURN_IF_NOT_OK(expect->GetNext(&row_expect));
      RETURN_IF_NOT_OK(root->GetNext(&row_root));
    }
    EXPECT_EQ(row_expect.size(), row_root.size());
    return Status::OK();
  }
};

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Sampler in MappableSourceNode
/// Expectation: Skip node is pushed down and removed after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownMappableSourceNode) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownMappableSourceNode.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  auto root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>());
  auto root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(2));
  EXPECT_OK(prepare_trees(root, root_target, 2));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Batch Operation
/// Expectation: Skip node is pushed down and removed after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownBatch) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownBatch.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Batch(5)->Skip(5);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(25))->Batch(5);
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Batch(5);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(25))->Batch(5);
  EXPECT_OK(prepare_trees(root, root_target, 5));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Rename Operation
/// Expectation: Skip node is pushed down and removed after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownRename) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownRename.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root =
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Rename({"label"}, {"fake_label"})->Skip(5);
  root_target =
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(5))->Rename({"label"}, {"fake_label"});
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Rename({"label"}, {"fake_label"});
  root_target =
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(5))->Rename({"label"}, {"fake_label"});
  EXPECT_OK(prepare_trees(root, root_target, 5));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Project Operation
/// Expectation: Skip node is pushed down and removed for Project after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownProject) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownProject.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"label", "image"})->Skip(10);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(10))->Project({"label", "image"});
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"label", "image"});
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(10))->Project({"label", "image"});
  EXPECT_OK(prepare_trees(root, root_target, 10));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())
           ->Skip(1)
           ->Project({"label", "image"})
           ->Skip(10);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(11))->Project({"label", "image"});
  EXPECT_OK(prepare_trees(root, root_target, 0));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Concat Operation
/// Expectation: Skip node cannot be pushed down for Concat after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownConcat) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownConcat.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  std::vector<std::shared_ptr<Dataset>> datasets = {
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())};
  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Concat(datasets)->Skip(10);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Concat(datasets)->Skip(10);
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Concat(datasets);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Concat(datasets)->Skip(10);
  EXPECT_OK(prepare_trees(root, root_target, 10));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Zip Operation
/// Expectation: Skip node cannot be pushed down for Zip after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownZip) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownZip.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  std::vector<std::shared_ptr<Dataset>> datasets = {
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"label"})};
  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"image"})->Zip(datasets);
  root_target =
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"image"})->Zip(datasets)->Skip(10);
  EXPECT_OK(prepare_trees(root, root_target, 10));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Repeat Operation
/// Expectation: Skip operation cannot be pushed down for Repeat after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownRepeat) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownRepeat.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Repeat(5);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Repeat(5)->Skip(11);
  EXPECT_OK(prepare_trees(root, root_target, 11));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Take Operation
/// Expectation: Skip operation cannot be pushed down for Take after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownTake) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownTake.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Take(20);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Take(20)->Skip(10);
  EXPECT_OK(prepare_trees(root, root_target, 10));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Skip Operation
/// Expectation: Skip node cannot be pushed down after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownSkip) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownSkip.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(2)->Skip(3);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(2)->Skip(3)->Skip(5);
  EXPECT_OK(prepare_trees(root, root_target, 5));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with NonMappableSourceNode(CSV)
/// Expectation: Skip node is pushed down after optimization pass, but cannot be removed
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownNonMappableSourceNode) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownNonMappableSourceNode.";
  std::string folder_path = datasets_root_path_ + "/testCSV/append.csv";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  std::vector<std::string> column_names = {"col1", "col2", "col3", "col4"};
  root = CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)->Rename({"col1"}, {"fake_label"});
  root_target =
    CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)->Skip(1)->Rename({"col1"}, {"fake_label"});
  EXPECT_OK(prepare_trees(root, root_target, 1));

  root = CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)->Rename({"col1"}, {"fake_label"})->Skip(1);
  root_target =
    CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)->Skip(1)->Rename({"col1"}, {"fake_label"});
  EXPECT_OK(prepare_trees(root, root_target, 0));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Batch and Rename Operations
/// Expectation: Skip node is pushed down after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownCombineOperations1) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownCombineOperations1.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())
           ->Batch(5)
           ->Skip(2)
           ->Rename({"label"}, {"fake_label"});
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())
                  ->Batch(5)
                  ->Skip(2)
                  ->Skip(2)
                  ->Rename({"label"}, {"fake_label"});
  EXPECT_OK(prepare_trees(root, root_target, 2));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Batch and Skip Operations and Sampler
/// Expectation: Skip node is pushed down after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownCombineOperations2) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownCombineOperations2.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(2)->Batch(5);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(2)->Skip(10)->Batch(5);
  EXPECT_OK(prepare_trees(root, root_target, 2));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Take and Project Operations
/// Expectation: Skip node is pushed down for Project but not for Take
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownCombineOperations3) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownCombineOperations3.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Take(20)->Project({"label", "image"});
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())
                  ->Take(20)
                  ->Skip(2)
                  ->Project({"label", "image"});
  EXPECT_OK(prepare_trees(root, root_target, 2));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with no Skip/ Skip(0) Operation
/// Expectation: Skip(0) shows the same result as no Skip operation
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownSkip0) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownSkip0.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"label", "image"})->Take(5);
  root_target =
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"label", "image"})->Take(5);
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"label", "image"})->Skip(0);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"label", "image"});
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())
           ->Skip(0)
           ->Project({"label", "image"})
           ->Skip(0);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"label", "image"});
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(0)->Project({"label", "image"});
  root_target =
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Project({"label", "image"});
  EXPECT_OK(prepare_trees(root, root_target, 1));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())
           ->Skip(2)
           ->Skip(1)
           ->Project({"label", "image"});
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())
                  ->Skip(2)
                  ->Skip(1)
                  ->Skip(1)
                  ->Project({"label", "image"});
  EXPECT_OK(prepare_trees(root, root_target, 1));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Repeat Operation
/// Expectation: Skip operation cannot be pushed down for Repeat operation
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownRepeat2) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownRepeat2.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Repeat(3)->Skip(1);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Repeat(3)->Skip(1);
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Repeat(3);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Repeat(3)->Skip(50);
  EXPECT_OK(prepare_trees(root, root_target, 50));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Repeat(3);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Repeat(3)->Skip(50);
  EXPECT_OK(prepare_trees(root, root_target, 50));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Take Operation
/// Expectation: Skip operation cannot be pushed down for Take operation
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownTake2) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownTake2.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Take(3);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Take(3);
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Take(3);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Take(3)->Skip(1);
  EXPECT_OK(prepare_trees(root, root_target, 1));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Take(5);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Take(5)->Skip(1);
  EXPECT_OK(prepare_trees(root, root_target, 1));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Concat/Zip Operation
/// Expectation: Skip node cannot be removed for Concat/Zip after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownUnsupported) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownUnsupported.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  std::vector<std::shared_ptr<Dataset>> datasets = {
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"label"})};
  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"label"})->Concat(datasets);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())
                  ->Project({"label"})
                  ->Concat(datasets)
                  ->Skip(2);
  EXPECT_OK(prepare_trees(root, root_target, 2));

  root =
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"image"})->Zip(datasets)->Skip(1);
  root_target =
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"image"})->Zip(datasets)->Skip(1);
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"image"})->Zip(datasets);
  root_target =
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Project({"image"})->Zip(datasets)->Skip(1);
  EXPECT_OK(prepare_trees(root, root_target, 1));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Filter Operation
/// Expectation: Skip node cannot be pushed down for Filter after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownUnsupported_Filter) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownUnsupported_Filter.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Filter(Predicate3, {"label"});
  root_target =
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Filter(Predicate3, {"label"});
  EXPECT_OK(prepare_trees(root, root_target, 0));
  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Filter(Predicate3, {"label"})->Skip(1);
  root_target =
    ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Filter(Predicate3, {"label"})->Skip(1);
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Filter(Predicate3, {"label"});
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())
                  ->Skip(1)
                  ->Filter(Predicate3, {"label"})
                  ->Skip(1);
  EXPECT_OK(prepare_trees(root, root_target, 1));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with SubsetSampler as child
/// Expectation: Skip node is removed for Rename/Project/Map after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownSubsetSampler) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownSubsetSampler.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;
  std::vector<int64_t> indices = {0, 1, 2, 3, 4, 5};
  root = ImageFolder(folder_path, false, std::make_shared<SubsetRandomSampler>(indices, 3))->Skip(1);
  auto sampler = std::make_shared<SequentialSampler>(1);
  sampler->AddChild(std::make_shared<SubsetRandomSampler>(indices, 3));
  root_target = ImageFolder(folder_path, false, sampler);
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SubsetRandomSampler>(indices, 4))
           ->Rename({"label"}, {"fake_label"});
  sampler = std::make_shared<SequentialSampler>(1);
  sampler->AddChild(std::make_shared<SubsetRandomSampler>(indices, 4));
  root_target = ImageFolder(folder_path, false, sampler)->Rename({"label"}, {"fake_label"});
  EXPECT_OK(prepare_trees(root, root_target, 1));

  root =
    ImageFolder(folder_path, false, std::make_shared<SubsetRandomSampler>(indices, 10))->Project({"label", "image"});
  sampler = std::make_shared<SequentialSampler>(1);
  sampler->AddChild(std::make_shared<SubsetRandomSampler>(indices, 10));
  root_target = ImageFolder(folder_path, false, sampler)->Project({"label", "image"});
  EXPECT_OK(prepare_trees(root, root_target, 1));

  std::vector<std::shared_ptr<TensorTransform>> transforms;
  std::vector<int32_t> size = {80, 80};
  std::vector<uint32_t> ignore = {20, 20, 20, 20};
  std::shared_ptr<TensorTransform> operation1 = std::make_shared<vision::AutoContrast>(0.5, ignore);
  std::shared_ptr<TensorTransform> operation2 = std::make_shared<vision::CenterCrop>(size);
  transforms.push_back(operation1);
  transforms.push_back(operation2);
  root = ImageFolder(folder_path, true, std::make_shared<SubsetRandomSampler>(indices, 3))->Map(transforms);
  sampler = std::make_shared<SequentialSampler>(1);
  sampler->AddChild(std::make_shared<SubsetRandomSampler>(indices, 3));
  root_target = ImageFolder(folder_path, true, sampler)->Map(transforms);
  EXPECT_OK(prepare_trees(root, root_target, 1));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with Batch Operation
/// Expectation: Skip node is pushed down for Batch after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownBatch2) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownBatch2.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Batch(5);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(1)->Batch(5);
  EXPECT_OK(prepare_trees(root, root_target, 0));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(3)->Batch(5);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Skip(3)->Skip(15)->Batch(5);
  EXPECT_OK(prepare_trees(root, root_target, 3));

  root = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>())->Batch(5);
  root_target = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(15))->Batch(5);
  EXPECT_OK(prepare_trees(root, root_target, 3));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with SkipPushdownNonMappableSourceNode(CSV)
/// Expectation: Skip node is pushed down for Rename/Batch/Project after optimization pass, but cannot be removed for
/// NonMappableSourceNode(CSV)
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownNonMappableSourceNode2) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownNonMappableSourceNode2.";
  std::string folder_path = datasets_root_path_ + "/testCSV/append.csv";
  std::vector<std::string> column_names = {"col1", "col2", "col3", "col4"};

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  root = CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)->Rename({"col1"}, {"fake_label"});
  root_target =
    CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)->Skip(1)->Rename({"col1"}, {"fake_label"});
  EXPECT_OK(prepare_trees(root, root_target, 1));

  root = CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)->Skip(1)->Rename({"col1"}, {"fake_label"});
  root_target = CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)
                  ->Skip(1)
                  ->Skip(1)
                  ->Rename({"col1"}, {"fake_label"});
  EXPECT_OK(prepare_trees(root, root_target, 1));

  root = CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)
           ->Repeat(5)
           ->Skip(1)
           ->Batch(2)
           ->Project({"col1", "col2"})
           ->Rename({"col1"}, {"fake_label"});
  root_target = CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)
                  ->Repeat(5)
                  ->Skip(1)
                  ->Skip(2)
                  ->Batch(2)
                  ->Project({"col1", "col2"})
                  ->Rename({"col1"}, {"fake_label"});
  EXPECT_OK(prepare_trees(root, root_target, 1));
}

/// Feature: MindData Skip Pushdown Optimization Pass Test
/// Description: Test MindData Skip Pushdown Optimization Pass with combined Operations
/// Expectation: Skip node is removed/reduced for Rename/Batch after optimization pass
TEST_F(MindDataSkipPushdownTestOptimizationPass, SkipPushdownCombineOperations4) {
  MS_LOG(INFO) << "Doing MindDataSkipPushdownTestOptimizationPass-SkipPushdownCombineOperations4.";
  std::string folder_path = datasets_root_path_ + "/testPK/data/";

  std::shared_ptr<Dataset> root;
  std::shared_ptr<Dataset> root_target;

  std::vector<int64_t> indices = {0, 1, 2, 3, 4, 5};
  root = ImageFolder(folder_path, false, std::make_shared<SubsetRandomSampler>(indices, 6))
           ->Repeat(4)
           ->Skip(4)
           ->Take(10)
           ->Skip(2)
           ->Take(2)
           ->Rename({"label"}, {"fake_label"});
  auto sampler = std::make_shared<SequentialSampler>(1);
  sampler->AddChild(std::make_shared<SubsetRandomSampler>(indices, 6));
  root_target = ImageFolder(folder_path, false, sampler)
                  ->Repeat(4)
                  ->Skip(4)
                  ->Take(10)
                  ->Skip(2)
                  ->Take(2)
                  ->Skip(1)
                  ->Rename({"label"}, {"fake_label"});
  EXPECT_OK(prepare_trees(root, root_target, 1));

  root = ImageFolder(folder_path, false, std::make_shared<SubsetRandomSampler>(indices, 5))
           ->Repeat(8)
           ->Skip(4)
           ->Skip(2)
           ->Take(10)
           ->Rename({"label"}, {"fake_label"})
           ->Batch(4);
  sampler = std::make_shared<SequentialSampler>(1);
  sampler->AddChild(std::make_shared<SubsetRandomSampler>(indices, 5));
  root_target = ImageFolder(folder_path, false, std::make_shared<SubsetRandomSampler>(indices, 5))
                  ->Repeat(8)
                  ->Skip(4)
                  ->Skip(2)
                  ->Take(10)
                  ->Skip(8)
                  ->Rename({"label"}, {"fake_label"})
                  ->Batch(4);
  EXPECT_OK(prepare_trees(root, root_target, 2));

  folder_path = datasets_root_path_ + "/testCSV/append.csv";
  std::vector<std::string> column_names = {"col1", "col2", "col3", "col4"};

  root = CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)
           ->Skip(1)
           ->Repeat(4)
           ->Batch(3)
           ->Skip(1)
           ->Rename({"col1"}, {"fake_label"});
  root_target = CSV({folder_path}, ',', {}, column_names, 0, ShuffleMode::kFalse)
                  ->Skip(1)
                  ->Repeat(4)
                  ->Skip(3)
                  ->Batch(3)
                  ->Rename({"col1"}, {"fake_label"});
  EXPECT_OK(prepare_trees(root, root_target, 0));
}
