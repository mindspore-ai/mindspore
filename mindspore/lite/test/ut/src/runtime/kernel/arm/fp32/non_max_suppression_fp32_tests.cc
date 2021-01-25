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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32/l2_norm_fp32.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32/non_max_suppression_fp32.h"
#include "src/kernel_registry.h"
#include "src/lite_kernel.h"
using mindspore::schema::Format_NHWC;

namespace mindspore {
class TestNMSFp32 : public mindspore::CommonTest {
 public:
  TestNMSFp32() = default;
  void Init(const std::vector<int> &box_tensor_shape, float *box_data, const std::vector<int> &score_tensor_shape,
            float *score_data, int32_t max_output, float iou_threshold, float score_threshold, int center_box_point);
  void TearDown() override;

 public:
  float err_tol_ = 1e-5;
  lite::Tensor box_tensor_;
  lite::Tensor score_tensor_;
  lite::Tensor max_output_box_per_class_tensor_;
  lite::Tensor iou_threshold_tensor_;
  lite::Tensor score_threshold_tensor_;
  lite::Tensor out_tensor_;
  int32_t max_output_;
  float iou_threshold_;
  float score_threshold_;
  std::vector<lite::Tensor *> inputs_{&box_tensor_, &score_tensor_, &max_output_box_per_class_tensor_,
                                      &iou_threshold_tensor_, &score_threshold_tensor_};
  std::vector<lite::Tensor *> outputs_{&out_tensor_};
  NMSParameter param_;
  kernel::KernelKey desc_ = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_NonMaxSuppression};
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::KernelCreator creator_ = nullptr;
  kernel::LiteKernel *kernel_ = nullptr;
};

void TestNMSFp32::TearDown() {
  box_tensor_.set_data(nullptr);
  score_tensor_.set_data(nullptr);
  max_output_box_per_class_tensor_.set_data(nullptr);
  iou_threshold_tensor_.set_data(nullptr);
  score_threshold_tensor_.set_data(nullptr);
  out_tensor_.FreeData();
}

void TestNMSFp32::Init(const std::vector<int> &box_tensor_shape, float *box_data,
                       const std::vector<int> &score_tensor_shape, float *score_data, int32_t max_output,
                       float iou_threshold, float score_threshold, int center_box_point) {
  box_tensor_.set_data_type(kNumberTypeFloat32);
  box_tensor_.set_format(Format_NHWC);
  box_tensor_.set_shape(box_tensor_shape);
  box_tensor_.set_data(box_data);

  score_tensor_.set_data_type(kNumberTypeFloat32);
  score_tensor_.set_format(Format_NHWC);
  score_tensor_.set_shape(score_tensor_shape);
  score_tensor_.set_data(score_data);

  max_output_ = max_output;
  max_output_box_per_class_tensor_.set_data(&max_output_);
  iou_threshold_ = iou_threshold;
  iou_threshold_tensor_.set_data(&iou_threshold_);
  score_threshold_ = score_threshold;
  score_threshold_tensor_.set_data(&score_threshold_);

  out_tensor_.set_data_type(kNumberTypeInt32);

  param_.center_point_box_ = center_box_point;
  ctx_ = lite::InnerContext();
  ASSERT_EQ(lite::RET_OK, ctx_.Init());
  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc_);
  ASSERT_NE(creator_, nullptr);
  kernel_ = creator_(inputs_, outputs_, reinterpret_cast<OpParameter *>(&param_), &ctx_, desc_);
  ASSERT_NE(kernel_, nullptr);
}

TEST_F(TestNMSFp32, TestCase1) {
  std::vector<int> box_tensor_shape{1, 6, 4};  // batch 1, num 6, box coord 4
  float box_data[24] = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.1f, 1.0f, 1.0f, 0.1f, 0.0f, 1.0f, 1.0f,
                        0.0f, 3.0f, 1.0f, 1.0f, 0.0f, 3.1f, 1.0f, 1.0f, 0.0f, 6.0f, 1.0f, 1.0f};
  std::vector<int> score_tensor_shape{1, 1, 6};  // batch 1, class 1, num 6
  float score_data[6] = {0.9f, 0.8f, 0.7f, 0.95f, 0.6f, 0.5f};
  int64_t max_output = 3;
  float iou_threshold = 0.5f;
  float score_threshold = 0.0f;
  int center_box_point = 1;
  auto output_size = 9;

  Init(box_tensor_shape, box_data, score_tensor_shape, score_data, max_output, iou_threshold, score_threshold,
       center_box_point);
  auto ret = kernel_->PreProcess();
  EXPECT_EQ(0, ret);
  ret = kernel_->Run();
  EXPECT_EQ(0, ret);

  std::vector<int32_t> expect{0, 0, 3, 0, 0, 0, 0, 0, 5};
  ASSERT_EQ(0,
            CompareOutputData(reinterpret_cast<int32_t *>(out_tensor_.data_c()), expect.data(), output_size, err_tol_));
}

}  // namespace mindspore
