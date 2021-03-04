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
#include <list>

#include "common/common.h"
#include "minddata/dataset/callback/ds_callback.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/tree_adapter.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/kernels/data/no_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::MsLogLevel::INFO;

namespace mindspore {
namespace dataset {
namespace test {

std::shared_ptr<ExecutionTree> BuildTree(std::vector<std::shared_ptr<DatasetOp>> ops) {
  std::shared_ptr<ExecutionTree> tree = std::make_shared<ExecutionTree>();
  Status rc;
  for (int i = 0; i < ops.size(); i++) {
    rc = tree->AssociateNode(ops[i]);
    EXPECT_TRUE(rc.IsOk());
    if (i > 0) {
      rc = ops[i]->AddChild(ops[i - 1]);
      EXPECT_TRUE(rc.IsOk());
    }
    if (i == ops.size() - 1) {
      rc = tree->AssignRoot(ops[i]);
      EXPECT_TRUE(rc.IsOk());
    }
  }
  return tree;
}

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
    all_names_.push_back("BGN");
    all_step_nums_.push_back(cb_param.cur_step_num_);
    all_ep_nums_.push_back(cb_param.cur_epoch_num_);
    return Status::OK();
  }
  Status DSEpochBegin(const CallbackParam &cb_param) override {
    all_names_.push_back("EPBGN");
    all_step_nums_.push_back(cb_param.cur_step_num_);
    all_ep_nums_.push_back(cb_param.cur_epoch_num_);
    return Status::OK();
  }
  Status DSNStepBegin(const CallbackParam &cb_param) override {
    all_names_.push_back("SPBGN");
    all_step_nums_.push_back(cb_param.cur_step_num_);
    all_ep_nums_.push_back(cb_param.cur_epoch_num_);
    return Status::OK();
  }
  Status DSEnd(const CallbackParam &cb_param) override {
    all_names_.push_back("END");
    all_step_nums_.push_back(cb_param.cur_step_num_);
    all_ep_nums_.push_back(cb_param.cur_epoch_num_);
    return Status::OK();
  }
  Status DSEpochEnd(const CallbackParam &cb_param) override {
    all_names_.push_back("EPEND");
    all_step_nums_.push_back(cb_param.cur_step_num_);
    all_ep_nums_.push_back(cb_param.cur_epoch_num_);
    return Status::OK();
  }
  Status DSNStepEnd(const CallbackParam &cb_param) override {
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
    return std::vector<std::string>(all_names_.begin(), all_names_.begin() + len);
  }

  std::vector<int64_t> all_step_nums(size_t len) {
    return std::vector<int64_t>(all_step_nums_.begin(), all_step_nums_.begin() + len);
  }

  std::vector<int64_t> all_ep_nums(size_t len) {
    return std::vector<int64_t>(all_ep_nums_.begin(), all_ep_nums_.begin() + len);
  }

  // flag for turning callback on and off
  bool begin_, epoch_begin_, step_begin_, end_, epoch_end_, step_end_;
  // name of the callback function in sequence, BGN, EPBGN, SPB, END, EPEND, SPEND
  std::vector<std::string> all_names_;
  std::vector<int64_t> all_step_nums_, all_ep_nums_;
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
};

TEST_F(MindDataTestCallback, TestBasicCallback) {
  MS_LOG(INFO) << "Doing: MindDataTestCallback-TestBasicCallback";
  // config callback
  Status rc;
  std::shared_ptr<test::TestCallback> tst_cb = std::make_shared<test::TestCallback>(64);
  std::shared_ptr<DSCallback> cb1 = tst_cb;
  // config leaf_op, use random_data to avoid I/O
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape shape({});  // empty shape is a 1-value scalar Tensor
  ColDescriptor col("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &shape);
  ASSERT_OK(schema->AddColumn(col));
  std::shared_ptr<RandomDataOp> leaf;
  rc = RandomDataOp::Builder().SetRowsPerBuffer(1).SetDataSchema(std::move(schema)).SetTotalRows(44).Build(&leaf);
  EXPECT_TRUE(rc.IsOk());
  // config mapOp
  std::shared_ptr<MapOp> map_op;
  auto map_b = MapOp::Builder();
  rc = map_b.SetInColNames({"label"}).SetTensorFuncs({std::make_shared<NoOp>()}).AddCallbacks({cb1}).Build(&map_op);
  EXPECT_TRUE(rc.IsOk());
  // config RepeatOp
  std::shared_ptr<RepeatOp> repeat_op;
  rc = RepeatOp::Builder(2).Build(&repeat_op);
  // start build then launch tree
  leaf->set_total_repeats(2);
  leaf->set_num_repeats_per_epoch(2);
  map_op->set_total_repeats(2);
  map_op->set_num_repeats_per_epoch(2);
  std::shared_ptr<ExecutionTree> tree = test::BuildTree({leaf, map_op, repeat_op});
  rc = tree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = tree->Launch();
  EXPECT_TRUE(rc.IsOk());
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
  std::vector<int64_t> all_steps = {0, 0, 1, 1, 65, 65, 88};
  std::vector<int64_t> all_epochs = {0, 1, 1, 1, 1, 1, 1};
  // doing resize to make sure no unexpected epoch_end or extra epoch_begin is called
  size_t len = 7;
  EXPECT_EQ(tst_cb->all_names(len), callback_names);
  EXPECT_EQ(tst_cb->all_step_nums(len), all_steps);
  EXPECT_EQ(tst_cb->all_ep_nums(len), all_epochs);
}

TEST_F(MindDataTestCallback, TestMultiEpochCallback) {
  MS_LOG(INFO) << "Doing: MindDataTestCallback-TestMultiEpochCallback";
  // config callback
  Status rc;
  std::shared_ptr<test::TestCallback> tst_cb = std::make_shared<test::TestCallback>(4);
  std::shared_ptr<DSCallback> cb1 = tst_cb;
  // config leaf_op, use random_data to avoid I/O
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape shape({});  // empty shape is a 1-value scalar Tensor
  ColDescriptor col("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &shape);
  ASSERT_OK(schema->AddColumn(col));
  std::shared_ptr<RandomDataOp> leaf;
  rc = RandomDataOp::Builder().SetRowsPerBuffer(1).SetDataSchema(std::move(schema)).SetTotalRows(4).SetNumWorkers(4).Build(&leaf);
  EXPECT_TRUE(rc.IsOk());
  // config mapOp
  std::shared_ptr<MapOp> map_op;
  auto map_b = MapOp::Builder();
  rc = map_b.SetInColNames({"label"}).SetTensorFuncs({std::make_shared<NoOp>()}).AddCallbacks({cb1}).Build(&map_op);
  EXPECT_TRUE(rc.IsOk());
  // config RepeatOp
  std::shared_ptr<RepeatOp> repeat_op;
  rc = RepeatOp::Builder(2).Build(&repeat_op);
  // config EpochCtrlOp
  std::shared_ptr<EpochCtrlOp> epoch_ctrl_op;
  rc = EpochCtrlOp::Builder(-1).Build(&epoch_ctrl_op);
  // start build then launch tree
  leaf->set_total_repeats(-2);
  leaf->set_num_repeats_per_epoch(2);
  map_op->set_total_repeats(-2);
  map_op->set_num_repeats_per_epoch(2);
  std::shared_ptr<ExecutionTree> tree = test::BuildTree({leaf, map_op, repeat_op, epoch_ctrl_op});
  rc = tree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = tree->Launch();
  EXPECT_TRUE(rc.IsOk());
  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(tree);
  TensorMap tensor_map;
  size_t num_epochs = 2;
  for (int ep_num = 0; ep_num < num_epochs; ++ep_num) {
    di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());

    while (tensor_map.size() != 0) {
      rc = di.GetNextAsMap(&tensor_map);
      EXPECT_TRUE(rc.IsOk());
    }
  }

  std::vector<std::string> callback_names = {"BGN",   "EPBGN", "SPBGN", "SPEND", "SPBGN", "SPEND", "EPEND",
                                             "EPBGN", "SPBGN", "SPEND", "SPBGN", "SPEND", "EPEND"};

  std::vector<int64_t> all_steps = {0, 0, 1, 1, 5, 5, 8, 8, 9, 9, 13, 13, 16};
  std::vector<int64_t> all_epochs = {0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};

  size_t len = 13;
  EXPECT_EQ(tst_cb->all_names(len), callback_names);
  EXPECT_EQ(tst_cb->all_ep_nums(len), all_epochs);
  EXPECT_EQ(tst_cb->all_step_nums(len), all_steps);
}

TEST_F(MindDataTestCallback, TestSelectedCallback) {
  MS_LOG(INFO) << "Doing: MindDataTestCallback-TestSelectedCallback";
  // config callback
  Status rc;
  std::shared_ptr<test::TestCallback> tst_cb = std::make_shared<test::TestCallback>(4);
  std::shared_ptr<DSCallback> cb1 = tst_cb;
  // turn off the epochs
  tst_cb->epoch_begin_ = false;
  tst_cb->epoch_end_ = false;

  // config leaf_op, use random_data to avoid I/O
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape shape({});  // empty shape is a 1-value scalar Tensor
  ColDescriptor col("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &shape);
  ASSERT_OK(schema->AddColumn(col));
  std::shared_ptr<RandomDataOp> leaf;
  rc = RandomDataOp::Builder().SetRowsPerBuffer(1).SetDataSchema(std::move(schema)).SetTotalRows(4).SetNumWorkers(4).Build(&leaf);
  EXPECT_TRUE(rc.IsOk());
  // config mapOp
  std::shared_ptr<MapOp> map_op;
  auto map_b = MapOp::Builder();
  rc = map_b.SetInColNames({"label"}).SetTensorFuncs({std::make_shared<NoOp>()}).AddCallbacks({cb1}).Build(&map_op);
  EXPECT_TRUE(rc.IsOk());
  // config RepeatOp
  std::shared_ptr<RepeatOp> repeat_op;
  rc = RepeatOp::Builder(2).Build(&repeat_op);
  // config EpochCtrlOp
  std::shared_ptr<EpochCtrlOp> epoch_ctrl_op;
  rc = EpochCtrlOp::Builder(-1).Build(&epoch_ctrl_op);
  // start build then launch tree
  leaf->set_total_repeats(-2);
  leaf->set_num_repeats_per_epoch(2);
  map_op->set_total_repeats(-2);
  map_op->set_num_repeats_per_epoch(2);
  std::shared_ptr<ExecutionTree> tree = test::BuildTree({leaf, map_op, repeat_op, epoch_ctrl_op});
  rc = tree->Prepare();
  EXPECT_TRUE(rc.IsOk());
  rc = tree->Launch();
  EXPECT_TRUE(rc.IsOk());
  // Start the loop of reading tensors from our pipeline
  DatasetIterator di(tree);
  TensorMap tensor_map;
  size_t num_epochs = 2;
  for (int ep_num = 0; ep_num < num_epochs; ++ep_num) {
    di.GetNextAsMap(&tensor_map);
    EXPECT_TRUE(rc.IsOk());

    while (tensor_map.size() != 0) {
      rc = di.GetNextAsMap(&tensor_map);
      EXPECT_TRUE(rc.IsOk());
    }
  }

  std::vector<std::string> callback_names = {"BGN",   "SPBGN", "SPEND", "SPBGN", "SPEND",
                                             "SPBGN", "SPEND", "SPBGN", "SPEND"};

  std::vector<int64_t> all_steps = {0, 1, 1, 5, 5, 9, 9, 13, 13};
  std::vector<int64_t> all_epochs = {0, 1, 1, 1, 1, 2, 2, 2, 2};

  size_t len = 9;
  EXPECT_EQ(tst_cb->all_names(len), callback_names);
  EXPECT_EQ(tst_cb->all_ep_nums(len), all_epochs);
  EXPECT_EQ(tst_cb->all_step_nums(len), all_steps);
}

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
  ds = ds->Map({std::make_shared<transforms::TypeCast>("uint64")}, {"label"}, {}, {}, nullptr, {cb1});
  ASSERT_NE(ds, nullptr);
  ds = ds->Repeat(2);
  ASSERT_NE(ds, nullptr);

  TreeAdapter tree_adapter;
  // using tree_adapter to set num_epoch = 1
  ASSERT_OK(tree_adapter.Compile(ds->IRNode(), 1));

  TensorRow row;
  ASSERT_OK(tree_adapter.GetNext(&row));
  while (!row.empty()) {
    ASSERT_OK(tree_adapter.GetNext(&row));
  }
  std::vector<std::string> callback_names = {"BGN", "EPBGN", "SPBGN", "SPEND", "SPBGN", "SPEND", "EPEND"};
  std::vector<int64_t> all_steps = {0, 0, 1, 1, 65, 65, 88};
  std::vector<int64_t> all_epochs = {0, 1, 1, 1, 1, 1, 1};
  // doing resize to make sure no unexpected epoch_end or extra epoch_begin is called
  size_t len = 7;
  EXPECT_EQ(tst_cb->all_names(len), callback_names);
  EXPECT_EQ(tst_cb->all_step_nums(len), all_steps);
  EXPECT_EQ(tst_cb->all_ep_nums(len), all_epochs);
}
