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

#include <memory>
#include <list>

#include "common/common.h"
#include "minddata/dataset/callback/ds_callback.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/tree_adapter.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/kernels/ir/data/transforms_ir.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

namespace mindspore {
namespace dataset {
namespace test {

class TestCallback : public DSCallback {
 public:
  TestCallback(int32_t step_size)
      : DSCallback(step_size),
        begin_(true),
        epoch_begin_(true),
        step_begin_(true),
        end_(false),
        epoch_end_(true),
        step_end_(true) {
    all_names_.reserve(32);
    all_step_nums_.reserve(32);
    all_ep_nums_.reserve(32);
  }

  Status DSBegin(const CallbackParam &cb_param) override {
    std::lock_guard<std::mutex> guard(lock_);
    all_names_.push_back("BGN");
    all_step_nums_.push_back(cb_param.cur_step_num_);
    all_ep_nums_.push_back(cb_param.cur_epoch_num_);
    return Status::OK();
  }
  Status DSEpochBegin(const CallbackParam &cb_param) override {
    std::lock_guard<std::mutex> guard(lock_);
    all_names_.push_back("EPBGN");
    all_step_nums_.push_back(cb_param.cur_step_num_);
    all_ep_nums_.push_back(cb_param.cur_epoch_num_);
    return Status::OK();
  }
  Status DSNStepBegin(const CallbackParam &cb_param) override {
    std::lock_guard<std::mutex> guard(lock_);
    all_names_.push_back("SPBGN");
    all_step_nums_.push_back(cb_param.cur_step_num_);
    all_ep_nums_.push_back(cb_param.cur_epoch_num_);
    return Status::OK();
  }
  Status DSEnd(const CallbackParam &cb_param) override {
    std::lock_guard<std::mutex> guard(lock_);
    all_names_.push_back("END");
    all_step_nums_.push_back(cb_param.cur_step_num_);
    all_ep_nums_.push_back(cb_param.cur_epoch_num_);
    return Status::OK();
  }
  Status DSEpochEnd(const CallbackParam &cb_param) override {
    std::lock_guard<std::mutex> guard(lock_);
    all_names_.push_back("EPEND");
    all_step_nums_.push_back(cb_param.cur_step_num_);
    all_ep_nums_.push_back(cb_param.cur_epoch_num_);
    return Status::OK();
  }
  Status DSNStepEnd(const CallbackParam &cb_param) override {
    std::lock_guard<std::mutex> guard(lock_);
    all_names_.push_back("SPEND");
    all_step_nums_.push_back(cb_param.cur_step_num_);
    all_ep_nums_.push_back(cb_param.cur_epoch_num_);
    return Status::OK();
  }

  bool IsBeginNeeded() override { return begin_; }
  bool IsEpochBeginNeeded() override { return epoch_begin_; }
  bool IsNStepBeginNeeded() override { return step_begin_; }
  bool IsEndNeeded() override { return end_; }
  bool IsEpochEndNeeded() override { return epoch_end_; }
  bool IsNStepEndNeeded() override { return step_end_; }

  std::vector<std::string> all_names(size_t len) {
    std::vector<std::string> res(all_names_.begin(), all_names_.begin() + len);
    std::sort(res.begin(), res.end());
    return res;
  }

  std::vector<int64_t> all_step_nums(size_t len) {
    std::vector<int64_t> res(all_step_nums_.begin(), all_step_nums_.begin() + len);
    std::sort(res.begin(), res.end());
    return res;
  }

  std::vector<int64_t> all_ep_nums(size_t len) {
    std::vector<int64_t> res(all_ep_nums_.begin(), all_ep_nums_.begin() + len);
    std::sort(res.begin(), res.end());
    return res;
  }

  // flag for turning callback on and off
  bool begin_, epoch_begin_, step_begin_, end_, epoch_end_, step_end_;
  // name of the callback function in sequence, BGN, EPBGN, SPB, END, EPEND, SPEND
  std::vector<std::string> all_names_;
  std::vector<int64_t> all_step_nums_, all_ep_nums_;
  std::mutex lock_;
};

}  // namespace test
}  // namespace dataset
}  // namespace mindspore

class MindDataTestCallback : public UT::DatasetOpTesting {
 public:
  void SetUp() override {
    DatasetOpTesting::SetUp();
    GlobalInit();
  }

  /// Feature: Callback
  /// Description: Test basic callbacks with mappable dataset (RandomDataset)
  /// Expectation: Number and order of callbacks generated are correct
  void TestBasicCallback(std::shared_ptr<ExecutionTree> tree, std::shared_ptr<DatasetOp> callback_node,
                         int32_t step_size) {
    // config callback
    Status rc;
    std::shared_ptr<test::TestCallback> tst_cb = std::make_shared<test::TestCallback>(step_size);
    std::shared_ptr<DSCallback> cb1 = tst_cb;
    std::vector<std::shared_ptr<DSCallback>> cbs = {};
    cbs.push_back(cb1);
    callback_node->AddCallbacks(std::move(cbs));

    ASSERT_OK(tree->Prepare());
    ASSERT_OK(tree->Launch());
    // Start the loop of reading tensors from our pipeline
    DatasetIterator di(tree);
    TensorMap tensor_map;
    rc = di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());
    while (!tensor_map.empty()) {
      rc = di.GetNextAsMap(&tensor_map);
      EXPECT_TRUE(rc.IsOk());
    }

    std::vector<std::string> callback_names = {"BGN", "EPBGN", "SPBGN", "SPEND", "SPBGN", "SPEND", "EPEND"};
    std::sort(callback_names.begin(), callback_names.end());
    std::vector<int64_t> all_steps = {0, 0, 1, 1, 65, 65, 88};
    std::vector<int64_t> all_epochs = {0, 1, 1, 1, 1, 1, 1};
    // doing resize to make sure no unexpected epoch_end or extra epoch_begin is called
    size_t len = 7;
    EXPECT_EQ(tst_cb->all_names(len), callback_names);
    EXPECT_EQ(tst_cb->all_step_nums(len), all_steps);
    EXPECT_EQ(tst_cb->all_ep_nums(len), all_epochs);
  }
  std::vector<std::shared_ptr<DatasetOp>> GenerateNodes() {
    // config leaf_op, use random_data to avoid I/O
    std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
    TensorShape shape({});  // empty shape is a 1-value scalar Tensor
    ColDescriptor col("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &shape);
    EXPECT_OK(schema->AddColumn(col));

    std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
    int32_t op_connector_size = config_manager->op_connector_size();
    int32_t num_workers = config_manager->num_parallel_workers();
    int32_t num_rows = 44;
    std::shared_ptr<RandomDataOp> leaf =
      std::make_shared<RandomDataOp>(num_workers, op_connector_size, num_rows, std::move(schema));
    // config mapOp
    std::vector<std::string> input_columns = {"label"};
    std::vector<std::string> output_columns = {};
    std::vector<std::shared_ptr<TensorOperation>> op_list;
    std::shared_ptr<TensorOperation> my_op =
      std::make_shared<transforms::TypeCastOperation>(DataType(DataType::DE_FLOAT64));
    op_list.push_back(my_op);
    std::shared_ptr<MapOp> map_op =
      std::make_shared<MapOp>(input_columns, output_columns, std::move(op_list), num_workers, op_connector_size);

    PadInfo pad_map;
    std::shared_ptr<BatchOp> batch_op =
      std::make_shared<BatchOp>(1, false, false, op_connector_size, num_workers, std::vector<std::string>{}, pad_map);

    // config RepeatOp
    int32_t num_repeats = 2;
    std::shared_ptr<RepeatOp> repeat_op = std::make_shared<RepeatOp>(num_repeats);
    // start build then launch tree
    leaf->SetTotalRepeats(num_repeats);
    leaf->SetNumRepeatsPerEpoch(num_repeats);
    map_op->SetTotalRepeats(num_repeats);
    map_op->SetNumRepeatsPerEpoch(num_repeats);
    batch_op->SetTotalRepeats(num_repeats);
    batch_op->SetNumRepeatsPerEpoch(num_repeats);

    return {leaf, map_op, batch_op, repeat_op};
  }
};

/// Feature: Callback
/// Description: Test callbacks with mappable dataset (RandomDataset)
/// Expectation: Number and order of callbacks generated are correct
TEST_F(MindDataTestCallback, TestBasicCallback) {
  MS_LOG(INFO) << "Doing: MindDataTestCallback-TestBasicCallback";
  // Test Mapop
  auto nodes = GenerateNodes();
  auto tree = Build(nodes);
  TestBasicCallback(tree, nodes[1], 64);
  // Test LeafOp
  nodes = GenerateNodes();
  tree = Build(nodes);
  TestBasicCallback(tree, nodes[0], 64);
  // Test BatchOp
  nodes = GenerateNodes();
  tree = Build(nodes);
  TestBasicCallback(tree, nodes[2], 64);
}

/// Feature: Callback
/// Description: Test callbacks with multiple epochs
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestCallback, TestMultiEpochCallback) {
  MS_LOG(INFO) << "Doing: MindDataTestCallback-TestMultiEpochCallback";
  // config callback
  Status rc;
  std::shared_ptr<test::TestCallback> tst_cb = std::make_shared<test::TestCallback>(4);
  std::shared_ptr<DSCallback> cb1 = tst_cb;
  // config leaf_op, use random_data to avoid I/O
  std::shared_ptr<ConfigManager> config_manager = GlobalContext::config_manager();
  int32_t op_connector_size = config_manager->op_connector_size();
  int32_t num_workers = config_manager->num_parallel_workers();
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape shape({});  // empty shape is a 1-value scalar Tensor
  ColDescriptor col("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &shape);
  ASSERT_OK(schema->AddColumn(col));
  std::shared_ptr<RandomDataOp> leaf = std::make_shared<RandomDataOp>(4, op_connector_size, 4, std::move(schema));
  // config mapOp
  std::vector<std::string> input_columns = {"label"};
  std::vector<std::string> output_columns = {};
  std::vector<std::shared_ptr<TensorOperation>> op_list;
  std::shared_ptr<TensorOperation> my_op =
    std::make_shared<transforms::TypeCastOperation>(DataType(DataType::DE_FLOAT64));
  op_list.push_back(my_op);
  std::shared_ptr<MapOp> map_op =
    std::make_shared<MapOp>(input_columns, output_columns, std::move(op_list), num_workers, op_connector_size);
  std::vector<std::shared_ptr<DSCallback>> cbs = {};
  cbs.push_back(cb1);

  map_op->AddCallbacks(std::move(cbs));
  EXPECT_TRUE(rc.IsOk());
  int32_t num_repeats = 2;
  // config RepeatOp
  std::shared_ptr<RepeatOp> repeat_op = std::make_shared<RepeatOp>(num_repeats);
  // config EpochCtrlOp
  std::shared_ptr<EpochCtrlOp> epoch_ctrl_op = std::make_shared<EpochCtrlOp>(num_repeats);
  // start build then launch tree
  leaf->SetTotalRepeats(4);
  leaf->SetNumRepeatsPerEpoch(2);
  map_op->SetTotalRepeats(4);
  map_op->SetNumRepeatsPerEpoch(2);
  std::shared_ptr<ExecutionTree> tree = Build({leaf, map_op, repeat_op, epoch_ctrl_op});
  rc = tree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = tree->Launch();
  EXPECT_TRUE(rc.IsOk());
  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(tree);
  TensorMap tensor_map;
  size_t num_epochs = 2;
  for (int ep_num = 0; ep_num < num_epochs; ++ep_num) {
    ASSERT_OK(di.GetNextAsMap(&tensor_map));
    EXPECT_TRUE(rc.IsOk());

    while (tensor_map.size() != 0) {
      rc = di.GetNextAsMap(&tensor_map);
      EXPECT_TRUE(rc.IsOk());
    }
  }

  std::vector<std::string> callback_names = {"BGN",   "EPBGN", "SPBGN", "SPEND", "SPBGN", "SPEND", "EPEND",
                                             "EPBGN", "SPBGN", "SPEND", "SPBGN", "SPEND", "EPEND"};
  std::sort(callback_names.begin(), callback_names.end());

  std::vector<int64_t> all_steps = {0, 0, 1, 1, 5, 5, 8, 8, 9, 9, 13, 13, 16};
  std::vector<int64_t> all_epochs = {0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};

  size_t len = 13;
  EXPECT_EQ(tst_cb->all_names(len), callback_names);
  EXPECT_EQ(tst_cb->all_ep_nums(len), all_epochs);
  EXPECT_EQ(tst_cb->all_step_nums(len), all_steps);
}

/// Feature: Callback
/// Description: Test selected callbacks and turning off the epochs
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestCallback, TestSelectedCallback) {
  MS_LOG(INFO) << "Doing: MindDataTestCallback-TestSelectedCallback";
  // config callback
  Status rc;
  std::shared_ptr<test::TestCallback> tst_cb = std::make_shared<test::TestCallback>(4);
  // turn off the epochs
  tst_cb->epoch_begin_ = false;
  tst_cb->epoch_end_ = false;
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("label", mindspore::DataType::kNumberTypeUInt32, {}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  ASSERT_NE(ds, nullptr);
  ds->SetNumWorkers(1);
  // config mapOp
  ds = ds->Map({std::make_shared<transforms::TypeCast>(mindspore::DataType::kNumberTypeUInt64)}, {"label"}, {},
               nullptr, {tst_cb});
  ds->SetNumWorkers(1);
  ASSERT_NE(ds, nullptr);
  ds = ds->Repeat(2);
  ASSERT_NE(ds, nullptr);
  int32_t num_epochs = 2;
  auto itr = ds->CreateIterator(num_epochs);
  for (int ep_num = 0; ep_num < num_epochs; ++ep_num) {
    std::unordered_map<std::string, mindspore::MSTensor> row;
    ASSERT_OK(itr->GetNextRow(&row));
    while (!row.empty()) {
      ASSERT_OK(itr->GetNextRow(&row));
    }
  }

  std::vector<std::string> callback_names = {"BGN",   "SPBGN", "SPEND", "SPBGN", "SPEND",
                                             "SPBGN", "SPEND", "SPBGN", "SPEND"};
  std::sort(callback_names.begin(), callback_names.end());

  std::vector<int64_t> all_steps = {0, 1, 1, 5, 5, 9, 9, 13, 13};
  std::vector<int64_t> all_epochs = {0, 1, 1, 1, 1, 2, 2, 2, 2};

  size_t len = 9;
  EXPECT_EQ(tst_cb->all_names(len), callback_names);
  EXPECT_EQ(tst_cb->all_ep_nums(len), all_epochs);
  EXPECT_EQ(tst_cb->all_step_nums(len), all_steps);
}

/// Feature: Callback
/// Description: Test Cpp API callbacks and disabling IR optimization pass and use tree_adapter to set num_epochs=1
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestCallback, TestCAPICallback) {
  MS_LOG(INFO) << "Doing: MindDataTestCallback-TestCAPICallback";
  // config callback
  std::shared_ptr<test::TestCallback> tst_cb = std::make_shared<test::TestCallback>(64);
  std::shared_ptr<DSCallback> cb1 = tst_cb;
  // Create a RandomDataset.  Use random_data to avoid I/O
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("label", mindspore::DataType::kNumberTypeUInt32, {}));
  std::shared_ptr<Dataset> ds = RandomData(44, schema);
  ASSERT_NE(ds, nullptr);
  ds = ds->Map({std::make_shared<transforms::TypeCast>(mindspore::DataType::kNumberTypeUInt64)}, {"label"}, {},
               nullptr, {cb1});
  ASSERT_NE(ds, nullptr);
  ds = ds->Repeat(2);
  ASSERT_NE(ds, nullptr);

  auto tree_adapter = std::make_shared<TreeAdapter>();

  // Disable IR optimization pass
  tree_adapter->SetOptimize(false);

  // using tree_adapter to set num_epoch = 1
  ASSERT_OK(tree_adapter->Compile(ds->IRNode(), 1));

  TensorRow row;
  ASSERT_OK(tree_adapter->GetNext(&row));
  while (!row.empty()) {
    ASSERT_OK(tree_adapter->GetNext(&row));
  }
  std::vector<std::string> callback_names = {"BGN", "EPBGN", "SPBGN", "SPEND", "SPBGN", "SPEND", "EPEND"};
  std::sort(callback_names.begin(), callback_names.end());
  std::vector<int64_t> all_steps = {0, 0, 1, 1, 65, 65, 88};
  std::vector<int64_t> all_epochs = {0, 1, 1, 1, 1, 1, 1};
  // doing resize to make sure no unexpected epoch_end or extra epoch_begin is called
  size_t len = 7;
  EXPECT_EQ(tst_cb->all_names(len), callback_names);
  EXPECT_EQ(tst_cb->all_step_nums(len), all_steps);
  EXPECT_EQ(tst_cb->all_ep_nums(len), all_epochs);
}
