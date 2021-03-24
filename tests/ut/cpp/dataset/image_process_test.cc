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

#include "common/common.h"
#include "lite_cv/lite_mat.h"
#include "lite_cv/image_process.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <fstream>

using namespace mindspore::dataset;
class MindDataImageProcess : public UT::Common {
 public:
  MindDataImageProcess() {}

  void SetUp() {}
};

void CompareMat(cv::Mat cv_mat, LiteMat lite_mat) {
  int cv_h = cv_mat.rows;
  int cv_w = cv_mat.cols;
  int cv_c = cv_mat.channels();
  int lite_h = lite_mat.height_;
  int lite_w = lite_mat.width_;
  int lite_c = lite_mat.channel_;
  ASSERT_TRUE(cv_h == lite_h);
  ASSERT_TRUE(cv_w == lite_w);
  ASSERT_TRUE(cv_c == lite_c);
}

void Lite3CImageProcess(LiteMat &lite_mat_bgr, LiteMat &lite_norm_mat_cut) {
  bool ret;
  LiteMat lite_mat_resize;
  ret = ResizeBilinear(lite_mat_bgr, lite_mat_resize, 256, 256);
  ASSERT_TRUE(ret == true);
  LiteMat lite_mat_convert_float;
  ret = ConvertTo(lite_mat_resize, lite_mat_convert_float, 1.0);
  ASSERT_TRUE(ret == true);

  LiteMat lite_mat_crop;
  ret = Crop(lite_mat_convert_float, lite_mat_crop, 16, 16, 224, 224);
  ASSERT_TRUE(ret == true);
  std::vector<float> means = {0.485, 0.456, 0.406};
  std::vector<float> stds = {0.229, 0.224, 0.225};
  SubStractMeanNormalize(lite_mat_crop, lite_norm_mat_cut, means, stds);
  return;
}

cv::Mat cv3CImageProcess(cv::Mat &image) {
  cv::Mat resize_256_image;
  cv::resize(image, resize_256_image, cv::Size(256, 256), CV_INTER_LINEAR);
  cv::Mat float_256_image;
  resize_256_image.convertTo(float_256_image, CV_32FC3);

  cv::Mat roi_224_image;
  cv::Rect roi;
  roi.x = 16;
  roi.y = 16;
  roi.width = 224;
  roi.height = 224;

  float_256_image(roi).copyTo(roi_224_image);

  float meanR = 0.485;
  float meanG = 0.456;
  float meanB = 0.406;
  float varR = 0.229;
  float varG = 0.224;
  float varB = 0.225;
  cv::Scalar mean = cv::Scalar(meanR, meanG, meanB);
  cv::Scalar var = cv::Scalar(varR, varG, varB);

  cv::Mat imgMean(roi_224_image.size(), CV_32FC3, mean);
  cv::Mat imgVar(roi_224_image.size(), CV_32FC3, var);

  cv::Mat imgR1 = roi_224_image - imgMean;
  cv::Mat imgR2 = imgR1 / imgVar;
  return imgR2;
}

void AccuracyComparison(const std::vector<std::vector<double>> &expect, LiteMat &value) {
  for (int i = 0; i < expect.size(); i++) {
    for (int j = 0; j < expect[0].size(); j++) {
      double middle = std::fabs(expect[i][j] - value.ptr<double>(i)[j]);
      ASSERT_TRUE(middle <= 0.005);
    }
  }
}

TEST_F(MindDataImageProcess, testRGB) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat rgba_mat;
  cv::cvtColor(image, rgba_mat, CV_BGR2RGB);

  bool ret = false;
  LiteMat lite_mat_rgb;
  ret = InitFromPixel(rgba_mat.data, LPixelType::RGB, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_rgb);
  ASSERT_TRUE(ret == true);

  cv::Mat dst_image(lite_mat_rgb.height_, lite_mat_rgb.width_, CV_8UC3, lite_mat_rgb.data_ptr_);
}

TEST_F(MindDataImageProcess, testLoadByMemPtr) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat rgba_mat;
  cv::cvtColor(image, rgba_mat, CV_BGR2RGB);

  bool ret = false;
  int width = rgba_mat.cols;
  int height = rgba_mat.rows;
  uchar *p_rgb = (uchar *)malloc(width * height * 3 * sizeof(uchar));
  for (int i = 0; i < height; i++) {
    const uchar *current = rgba_mat.ptr<uchar>(i);
    for (int j = 0; j < width; j++) {
      p_rgb[i * width * 3 + 3 * j + 0] = current[3 * j + 0];
      p_rgb[i * width * 3 + 3 * j + 1] = current[3 * j + 1];
      p_rgb[i * width * 3 + 3 * j + 2] = current[3 * j + 2];
    }
  }

  LiteMat lite_mat_rgb(width, height, 3, (void *)p_rgb, LDataType::UINT8);
  LiteMat lite_mat_resize;
  ret = ResizeBilinear(lite_mat_rgb, lite_mat_resize, 256, 256);
  ASSERT_TRUE(ret == true);
  LiteMat lite_mat_convert_float;
  ret = ConvertTo(lite_mat_resize, lite_mat_convert_float, 1.0);
  ASSERT_TRUE(ret == true);

  LiteMat lite_mat_crop;
  ret = Crop(lite_mat_convert_float, lite_mat_crop, 16, 16, 224, 224);
  ASSERT_TRUE(ret == true);
  std::vector<float> means = {0.485, 0.456, 0.406};
  std::vector<float> stds = {0.229, 0.224, 0.225};
  LiteMat lite_norm_mat_cut;
  ret = SubStractMeanNormalize(lite_mat_crop, lite_norm_mat_cut, means, stds);

  int pad_width = lite_norm_mat_cut.width_ + 20;
  int pad_height = lite_norm_mat_cut.height_ + 20;
  float *p_rgb_pad = (float *)malloc(pad_width * pad_height * 3 * sizeof(float));

  LiteMat makeborder(pad_width, pad_height, 3, (void *)p_rgb_pad, LDataType::FLOAT32);
  ret = Pad(lite_norm_mat_cut, makeborder, 10, 30, 40, 10, PaddBorderType::PADD_BORDER_CONSTANT, 255, 255, 255);
  cv::Mat dst_image(pad_height, pad_width, CV_8UC3, p_rgb_pad);
  free(p_rgb);
  free(p_rgb_pad);
}

TEST_F(MindDataImageProcess, test3C) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat cv_image = cv3CImageProcess(image);

  // convert to RGBA for Android bitmap(rgba)
  cv::Mat rgba_mat;
  cv::cvtColor(image, rgba_mat, CV_BGR2RGBA);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);
  LiteMat lite_norm_mat_cut;
  Lite3CImageProcess(lite_mat_bgr, lite_norm_mat_cut);

  cv::Mat dst_image(lite_norm_mat_cut.height_, lite_norm_mat_cut.width_, CV_32FC3, lite_norm_mat_cut.data_ptr_);
  CompareMat(cv_image, lite_norm_mat_cut);
}

bool ReadYUV(const char *filename, int w, int h, uint8_t **data) {
  FILE *f = fopen(filename, "rb");
  if (f == nullptr) {
    return false;
  }
  fseek(f, 0, SEEK_END);
  int size = ftell(f);
  int expect_size = w * h + 2 * ((w + 1) / 2) * ((h + 1) / 2);
  if (size != expect_size) {
    fclose(f);
    return false;
  }
  fseek(f, 0, SEEK_SET);
  *data = (uint8_t *)malloc(size);
  size_t re = fread(*data, 1, size, f);
  if (re != size) {
    fclose(f);
    return false;
  }
  fclose(f);
  return true;
}

TEST_F(MindDataImageProcess, TestRGBA2GRAY) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, CV_BGR2GRAY);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);
  bool ret = false;
  LiteMat lite_mat_gray;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_gray);
  ASSERT_TRUE(ret == true);

  double distance = 0.f;
  int total_size = gray_image.cols * gray_image.rows * gray_image.channels();
  for (int i = 0; i < total_size; i++) {
    distance += pow((uint8_t)gray_image.data[i] - ((uint8_t *)lite_mat_gray)[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(MindDataImageProcess, testNV21ToBGR) {
  //  ffmpeg -i ./data/dataset/apple.jpg  -s 1024*800 -pix_fmt nv21 ./data/dataset/yuv/test_nv21.yuv
  const char *filename = "data/dataset/yuv/test_nv21.yuv";
  int w = 1024;
  int h = 800;
  uint8_t *yuv_data = nullptr;
  bool ret = ReadYUV(filename, w, h, &yuv_data);
  ASSERT_TRUE(ret == true);

  cv::Mat yuvimg(h * 3 / 2, w, CV_8UC1);
  memcpy(yuvimg.data, yuv_data, w * h * 3 / 2);
  cv::Mat rgbimage;

  cv::cvtColor(yuvimg, rgbimage, cv::COLOR_YUV2BGR_NV21);

  LiteMat lite_mat_bgr;

  ret = InitFromPixel(yuv_data, LPixelType::NV212BGR, LDataType::UINT8, w, h, lite_mat_bgr);
  ASSERT_TRUE(ret == true);
  cv::Mat dst_image(lite_mat_bgr.height_, lite_mat_bgr.width_, CV_8UC3, lite_mat_bgr.data_ptr_);
}

TEST_F(MindDataImageProcess, testNV12ToBGR) {
  //  ffmpeg -i ./data/dataset/apple.jpg  -s 1024*800 -pix_fmt nv12 ./data/dataset/yuv/test_nv12.yuv
  const char *filename = "data/dataset/yuv/test_nv12.yuv";
  int w = 1024;
  int h = 800;
  uint8_t *yuv_data = nullptr;
  bool ret = ReadYUV(filename, w, h, &yuv_data);
  ASSERT_TRUE(ret == true);

  cv::Mat yuvimg(h * 3 / 2, w, CV_8UC1);
  memcpy(yuvimg.data, yuv_data, w * h * 3 / 2);
  cv::Mat rgbimage;

  cv::cvtColor(yuvimg, rgbimage, cv::COLOR_YUV2BGR_NV12);
  LiteMat lite_mat_bgr;
  ret = InitFromPixel(yuv_data, LPixelType::NV122BGR, LDataType::UINT8, w, h, lite_mat_bgr);
  ASSERT_TRUE(ret == true);
  cv::Mat dst_image(lite_mat_bgr.height_, lite_mat_bgr.width_, CV_8UC3, lite_mat_bgr.data_ptr_);
}

TEST_F(MindDataImageProcess, testExtractChannel) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat dst_image;
  cv::extractChannel(src_image, dst_image, 2);
  // convert to RGBA for Android bitmap(rgba)
  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);

  LiteMat lite_B;
  ret = ExtractChannel(lite_mat_bgr, lite_B, 0);
  ASSERT_TRUE(ret == true);

  LiteMat lite_R;
  ret = ExtractChannel(lite_mat_bgr, lite_R, 2);
  ASSERT_TRUE(ret == true);
  cv::Mat dst_imageR(lite_R.height_, lite_R.width_, CV_8UC1, lite_R.data_ptr_);
  // cv::imwrite("./test_lite_r.jpg", dst_imageR);
}

TEST_F(MindDataImageProcess, testSplit) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  std::vector<cv::Mat> dst_images;
  cv::split(src_image, dst_images);
  // convert to RGBA for Android bitmap(rgba)
  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);
  std::vector<LiteMat> lite_all;
  ret = Split(lite_mat_bgr, lite_all);
  ASSERT_TRUE(ret == true);
  ASSERT_TRUE(lite_all.size() == 3);
  LiteMat lite_r = lite_all[2];
  cv::Mat dst_imageR(lite_r.height_, lite_r.width_, CV_8UC1, lite_r.data_ptr_);
}

TEST_F(MindDataImageProcess, testMerge) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  std::vector<cv::Mat> dst_images;
  cv::split(src_image, dst_images);
  // convert to RGBA for Android bitmap(rgba)
  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);
  std::vector<LiteMat> lite_all;
  ret = Split(lite_mat_bgr, lite_all);
  ASSERT_TRUE(ret == true);
  ASSERT_TRUE(lite_all.size() == 3);
  LiteMat lite_r = lite_all[2];
  cv::Mat dst_imageR(lite_r.height_, lite_r.width_, CV_8UC1, lite_r.data_ptr_);

  LiteMat merge_mat;
  EXPECT_TRUE(Merge(lite_all, merge_mat));
  EXPECT_EQ(merge_mat.height_, lite_mat_bgr.height_);
  EXPECT_EQ(merge_mat.width_, lite_mat_bgr.width_);
  EXPECT_EQ(merge_mat.channel_, lite_mat_bgr.channel_);
}

void Lite1CImageProcess(LiteMat &lite_mat_bgr, LiteMat &lite_norm_mat_cut) {
  LiteMat lite_mat_resize;
  int ret = ResizeBilinear(lite_mat_bgr, lite_mat_resize, 256, 256);
  ASSERT_TRUE(ret == true);
  LiteMat lite_mat_convert_float;
  ret = ConvertTo(lite_mat_resize, lite_mat_convert_float);
  ASSERT_TRUE(ret == true);
  LiteMat lite_mat_cut;
  ret = Crop(lite_mat_convert_float, lite_mat_cut, 16, 16, 224, 224);
  ASSERT_TRUE(ret == true);
  std::vector<float> means = {0.485};
  std::vector<float> stds = {0.229};
  ret = SubStractMeanNormalize(lite_mat_cut, lite_norm_mat_cut, means, stds);
  ASSERT_TRUE(ret == true);
  return;
}

cv::Mat cv1CImageProcess(cv::Mat &image) {
  cv::Mat gray_image;
  cv::cvtColor(image, gray_image, CV_BGR2GRAY);

  cv::Mat resize_256_image;
  cv::resize(gray_image, resize_256_image, cv::Size(256, 256), CV_INTER_LINEAR);
  cv::Mat float_256_image;
  resize_256_image.convertTo(float_256_image, CV_32FC3);

  cv::Mat roi_224_image;
  cv::Rect roi;
  roi.x = 16;
  roi.y = 16;
  roi.width = 224;
  roi.height = 224;

  float_256_image(roi).copyTo(roi_224_image);

  float meanR = 0.485;
  float varR = 0.229;
  cv::Scalar mean = cv::Scalar(meanR);
  cv::Scalar var = cv::Scalar(varR);

  cv::Mat imgMean(roi_224_image.size(), CV_32FC1, mean);
  cv::Mat imgVar(roi_224_image.size(), CV_32FC1, var);

  cv::Mat imgR1 = roi_224_image - imgMean;
  cv::Mat imgR2 = imgR1 / imgVar;
  return imgR2;
}

TEST_F(MindDataImageProcess, test1C) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat cv_image = cv1CImageProcess(image);

  // convert to RGBA for Android bitmap(rgba)
  cv::Mat rgba_mat;
  cv::cvtColor(image, rgba_mat, CV_BGR2RGBA);

  LiteMat lite_mat_bgr;
  bool ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);
  LiteMat lite_norm_mat_cut;
  Lite1CImageProcess(lite_mat_bgr, lite_norm_mat_cut);
  cv::Mat dst_image(lite_norm_mat_cut.height_, lite_norm_mat_cut.width_, CV_32FC1, lite_norm_mat_cut.data_ptr_);
  CompareMat(cv_image, lite_norm_mat_cut);
}

TEST_F(MindDataImageProcess, TestPadd) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  int left = 10;
  int right = 20;
  int top = 30;
  int bottom = 40;
  cv::Mat b_image;
  cv::Scalar color = cv::Scalar(255, 255, 255);
  cv::copyMakeBorder(image, b_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
  cv::Mat rgba_mat;
  cv::cvtColor(image, rgba_mat, CV_BGR2RGBA);

  LiteMat lite_mat_bgr;
  bool ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);
  LiteMat makeborder;
  ret = Pad(lite_mat_bgr, makeborder, top, bottom, left, right, PaddBorderType::PADD_BORDER_CONSTANT, 255, 255, 255);
  ASSERT_TRUE(ret == true);
  size_t total_size = makeborder.height_ * makeborder.width_ * makeborder.channel_;
  double distance = 0.0f;
  for (size_t i = 0; i < total_size; i++) {
    distance += pow((uint8_t)b_image.data[i] - ((uint8_t *)makeborder)[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(MindDataImageProcess, TestPadZero) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  int left = 0;
  int right = 0;
  int top = 0;
  int bottom = 0;
  cv::Mat b_image;
  cv::Scalar color = cv::Scalar(255, 255, 255);
  cv::copyMakeBorder(image, b_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
  cv::Mat rgba_mat;
  cv::cvtColor(image, rgba_mat, CV_BGR2RGBA);

  LiteMat lite_mat_bgr;
  bool ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);
  LiteMat makeborder;
  ret = Pad(lite_mat_bgr, makeborder, top, bottom, left, right, PaddBorderType::PADD_BORDER_CONSTANT, 255, 255, 255);
  ASSERT_TRUE(ret == true);
  size_t total_size = makeborder.height_ * makeborder.width_ * makeborder.channel_;
  double distance = 0.0f;
  for (size_t i = 0; i < total_size; i++) {
    distance += pow((uint8_t)b_image.data[i] - ((uint8_t *)makeborder)[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(MindDataImageProcess, TestPadReplicate) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  int left = 20;
  int right = 20;
  int top = 20;
  int bottom = 20;
  cv::Mat b_image;
  cv::copyMakeBorder(image, b_image, top, bottom, left, right, cv::BORDER_REPLICATE);

  cv::Mat rgba_mat;
  cv::cvtColor(image, rgba_mat, CV_BGR2RGBA);
  LiteMat lite_mat_bgr;
  bool ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);

  LiteMat makeborder;
  ret = Pad(lite_mat_bgr, makeborder, top, bottom, left, right, PaddBorderType::PADD_BORDER_REPLICATE);
  ASSERT_TRUE(ret == true);

  size_t total_size = makeborder.height_ * makeborder.width_ * makeborder.channel_;
  double distance = 0.0f;
  for (size_t i = 0; i < total_size; i++) {
    distance += pow((uint8_t)b_image.data[i] - ((uint8_t *)makeborder)[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(MindDataImageProcess, TestPadReflect101) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  int left = 20;
  int right = 20;
  int top = 20;
  int bottom = 20;
  cv::Mat b_image;
  cv::copyMakeBorder(image, b_image, top, bottom, left, right, cv::BORDER_REFLECT_101);

  cv::Mat rgba_mat;
  cv::cvtColor(image, rgba_mat, CV_BGR2RGBA);
  LiteMat lite_mat_bgr;
  bool ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);

  LiteMat makeborder;
  ret = Pad(lite_mat_bgr, makeborder, top, bottom, left, right, PaddBorderType::PADD_BORDER_REFLECT_101);
  ASSERT_TRUE(ret == true);

  size_t total_size = makeborder.height_ * makeborder.width_ * makeborder.channel_;
  double distance = 0.0f;
  for (size_t i = 0; i < total_size; i++) {
    distance += pow((uint8_t)b_image.data[i] - ((uint8_t *)makeborder)[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(MindDataImageProcess, TestGetDefaultBoxes) {
  std::string benchmark = "data/dataset/testLite/default_boxes.bin";
  BoxesConfig config;
  config.img_shape = {300, 300};
  config.num_default = {3, 6, 6, 6, 6, 6};
  config.feature_size = {19, 10, 5, 3, 2, 1};
  config.min_scale = 0.2;
  config.max_scale = 0.95;
  config.aspect_rations = {{2}, {2, 3}, {2, 3}, {2, 3}, {2, 3}, {2, 3}};
  config.steps = {16, 32, 64, 100, 150, 300};
  config.prior_scaling = {0.1, 0.2};

  int rows = 1917;
  int cols = 4;
  std::vector<double> benchmark_boxes(rows * cols);
  std::ifstream in(benchmark, std::ios::in | std::ios::binary);
  in.read(reinterpret_cast<char *>(benchmark_boxes.data()), benchmark_boxes.size() * sizeof(double));
  in.close();

  std::vector<std::vector<float>> default_boxes = GetDefaultBoxes(config);
  EXPECT_EQ(default_boxes.size(), rows);
  EXPECT_EQ(default_boxes[0].size(), cols);

  double distance = 0.0f;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      distance += pow(default_boxes[i][j] - benchmark_boxes[i * cols + j], 2);
    }
  }
  distance = sqrt(distance);
  EXPECT_LT(distance, 1e-5);
}

TEST_F(MindDataImageProcess, TestApplyNms) {
  std::vector<std::vector<float>> all_boxes = {{1, 1, 2, 2}, {3, 3, 4, 4}, {5, 5, 6, 6}, {5, 5, 6, 6}};
  std::vector<float> all_scores = {0.6, 0.5, 0.4, 0.9};
  std::vector<int> keep = ApplyNms(all_boxes, all_scores, 0.5, 10);
  ASSERT_TRUE(keep[0] == 3);
  ASSERT_TRUE(keep[1] == 0);
  ASSERT_TRUE(keep[2] == 1);
}

TEST_F(MindDataImageProcess, TestAffineInput) {
  LiteMat src(3, 3);
  LiteMat dst;
  double M[6] = {1};
  EXPECT_FALSE(Affine(src, dst, M, {}, UINT8_C1(0)));
  EXPECT_FALSE(Affine(src, dst, M, {3}, UINT8_C1(0)));
  EXPECT_FALSE(Affine(src, dst, M, {0, 0}, UINT8_C1(0)));
}

TEST_F(MindDataImageProcess, TestAffine) {
  // The input matrix
  // 0 0 1 0 0
  // 0 0 1 0 0
  // 2 2 3 2 2
  // 0 0 1 0 0
  // 0 0 1 0 0
  size_t rows = 5;
  size_t cols = 5;
  LiteMat src(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (i == 2 && j == 2) {
        static_cast<UINT8_C1 *>(src.data_ptr_)[i * cols + j] = 3;
      } else if (i == 2) {
        static_cast<UINT8_C1 *>(src.data_ptr_)[i * cols + j] = 2;
      } else if (j == 2) {
        static_cast<UINT8_C1 *>(src.data_ptr_)[i * cols + j] = 1;
      } else {
        static_cast<UINT8_C1 *>(src.data_ptr_)[i * cols + j] = 0;
      }
    }
  }

  // Expect output matrix
  // 0 0 2 0 0
  // 0 0 2 0 0
  // 1 1 3 1 1
  // 0 0 2 0 0
  // 0 0 2 0 0
  LiteMat expect(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (i == 2 && j == 2) {
        static_cast<UINT8_C1 *>(expect.data_ptr_)[i * cols + j] = 3;
      } else if (i == 2) {
        static_cast<UINT8_C1 *>(expect.data_ptr_)[i * cols + j] = 1;
      } else if (j == 2) {
        static_cast<UINT8_C1 *>(expect.data_ptr_)[i * cols + j] = 2;
      } else {
        static_cast<UINT8_C1 *>(expect.data_ptr_)[i * cols + j] = 0;
      }
    }
  }

  double angle = 90.0f;
  cv::Point2f center(rows / 2, cols / 2);
  cv::Mat rotate_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
  double M[6];
  for (size_t i = 0; i < 6; i++) {
    M[i] = rotate_matrix.at<double>(i);
  }
  LiteMat dst;
  EXPECT_TRUE(Affine(src, dst, M, {rows, cols}, UINT8_C1(0)));

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      EXPECT_EQ(static_cast<UINT8_C1 *>(expect.data_ptr_)[i * cols + j].c1,
                static_cast<UINT8_C1 *>(dst.data_ptr_)[i * cols + j].c1);
    }
  }
}

TEST_F(MindDataImageProcess, TestSubtractUint8) {
  const size_t cols = 4;
  // Test uint8
  LiteMat src1_uint8(1, cols);
  LiteMat src2_uint8(1, cols);
  LiteMat expect_uint8(1, cols);
  for (size_t i = 0; i < cols; i++) {
    static_cast<UINT8_C1 *>(src1_uint8.data_ptr_)[i] = 3;
    static_cast<UINT8_C1 *>(src2_uint8.data_ptr_)[i] = 2;
    static_cast<UINT8_C1 *>(expect_uint8.data_ptr_)[i] = 1;
  }
  LiteMat dst_uint8;
  EXPECT_TRUE(Subtract(src1_uint8, src2_uint8, &dst_uint8));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<UINT8_C1 *>(expect_uint8.data_ptr_)[i].c1,
              static_cast<UINT8_C1 *>(dst_uint8.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestSubtractInt8) {
  const size_t cols = 4;
  // Test int8
  LiteMat src1_int8(1, cols, LDataType(LDataType::INT8));
  LiteMat src2_int8(1, cols, LDataType(LDataType::INT8));
  LiteMat expect_int8(1, cols, LDataType(LDataType::INT8));
  for (size_t i = 0; i < cols; i++) {
    static_cast<INT8_C1 *>(src1_int8.data_ptr_)[i] = 2;
    static_cast<INT8_C1 *>(src2_int8.data_ptr_)[i] = 3;
    static_cast<INT8_C1 *>(expect_int8.data_ptr_)[i] = -1;
  }
  LiteMat dst_int8;
  EXPECT_TRUE(Subtract(src1_int8, src2_int8, &dst_int8));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<INT8_C1 *>(expect_int8.data_ptr_)[i].c1, static_cast<INT8_C1 *>(dst_int8.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestSubtractUInt16) {
  const size_t cols = 4;
  // Test uint16
  LiteMat src1_uint16(1, cols, LDataType(LDataType::UINT16));
  LiteMat src2_uint16(1, cols, LDataType(LDataType::UINT16));
  LiteMat expect_uint16(1, cols, LDataType(LDataType::UINT16));
  for (size_t i = 0; i < cols; i++) {
    static_cast<UINT16_C1 *>(src1_uint16.data_ptr_)[i] = 2;
    static_cast<UINT16_C1 *>(src2_uint16.data_ptr_)[i] = 3;
    static_cast<UINT16_C1 *>(expect_uint16.data_ptr_)[i] = 0;
  }
  LiteMat dst_uint16;
  EXPECT_TRUE(Subtract(src1_uint16, src2_uint16, &dst_uint16));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<UINT16_C1 *>(expect_uint16.data_ptr_)[i].c1,
              static_cast<UINT16_C1 *>(dst_uint16.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestSubtractInt16) {
  const size_t cols = 4;
  // Test int16
  LiteMat src1_int16(1, cols, LDataType(LDataType::INT16));
  LiteMat src2_int16(1, cols, LDataType(LDataType::INT16));
  LiteMat expect_int16(1, cols, LDataType(LDataType::INT16));
  for (size_t i = 0; i < cols; i++) {
    static_cast<INT16_C1 *>(src1_int16.data_ptr_)[i] = 2;
    static_cast<INT16_C1 *>(src2_int16.data_ptr_)[i] = 3;
    static_cast<INT16_C1 *>(expect_int16.data_ptr_)[i] = -1;
  }
  LiteMat dst_int16;
  EXPECT_TRUE(Subtract(src1_int16, src2_int16, &dst_int16));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<INT16_C1 *>(expect_int16.data_ptr_)[i].c1,
              static_cast<INT16_C1 *>(dst_int16.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestSubtractUInt32) {
  const size_t cols = 4;
  // Test uint16
  LiteMat src1_uint32(1, cols, LDataType(LDataType::UINT32));
  LiteMat src2_uint32(1, cols, LDataType(LDataType::UINT32));
  LiteMat expect_uint32(1, cols, LDataType(LDataType::UINT32));
  for (size_t i = 0; i < cols; i++) {
    static_cast<UINT32_C1 *>(src1_uint32.data_ptr_)[i] = 2;
    static_cast<UINT32_C1 *>(src2_uint32.data_ptr_)[i] = 3;
    static_cast<UINT32_C1 *>(expect_uint32.data_ptr_)[i] = 0;
  }
  LiteMat dst_uint32;
  EXPECT_TRUE(Subtract(src1_uint32, src2_uint32, &dst_uint32));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<UINT32_C1 *>(expect_uint32.data_ptr_)[i].c1,
              static_cast<UINT32_C1 *>(dst_uint32.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestSubtractInt32) {
  const size_t cols = 4;
  // Test int32
  LiteMat src1_int32(1, cols, LDataType(LDataType::INT32));
  LiteMat src2_int32(1, cols, LDataType(LDataType::INT32));
  LiteMat expect_int32(1, cols, LDataType(LDataType::INT32));
  for (size_t i = 0; i < cols; i++) {
    static_cast<INT32_C1 *>(src1_int32.data_ptr_)[i] = 2;
    static_cast<INT32_C1 *>(src2_int32.data_ptr_)[i] = 4;
    static_cast<INT32_C1 *>(expect_int32.data_ptr_)[i] = -2;
  }
  LiteMat dst_int32;
  EXPECT_TRUE(Subtract(src1_int32, src2_int32, &dst_int32));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<INT32_C1 *>(expect_int32.data_ptr_)[i].c1,
              static_cast<INT32_C1 *>(dst_int32.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestSubtractFloat) {
  const size_t cols = 4;
  // Test float
  LiteMat src1_float(1, cols, LDataType(LDataType::FLOAT32));
  LiteMat src2_float(1, cols, LDataType(LDataType::FLOAT32));
  LiteMat expect_float(1, cols, LDataType(LDataType::FLOAT32));
  for (size_t i = 0; i < cols; i++) {
    static_cast<FLOAT32_C1 *>(src1_float.data_ptr_)[i] = 3.4;
    static_cast<FLOAT32_C1 *>(src2_float.data_ptr_)[i] = 5.7;
    static_cast<FLOAT32_C1 *>(expect_float.data_ptr_)[i] = -2.3;
  }
  LiteMat dst_float;
  EXPECT_TRUE(Subtract(src1_float, src2_float, &dst_float));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_FLOAT_EQ(static_cast<FLOAT32_C1 *>(expect_float.data_ptr_)[i].c1,
                    static_cast<FLOAT32_C1 *>(dst_float.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestDivideUint8) {
  const size_t cols = 4;
  // Test uint8
  LiteMat src1_uint8(1, cols);
  LiteMat src2_uint8(1, cols);
  LiteMat expect_uint8(1, cols);
  for (size_t i = 0; i < cols; i++) {
    static_cast<UINT8_C1 *>(src1_uint8.data_ptr_)[i] = 8;
    static_cast<UINT8_C1 *>(src2_uint8.data_ptr_)[i] = 4;
    static_cast<UINT8_C1 *>(expect_uint8.data_ptr_)[i] = 2;
  }
  LiteMat dst_uint8;
  EXPECT_TRUE(Divide(src1_uint8, src2_uint8, &dst_uint8));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<UINT8_C1 *>(expect_uint8.data_ptr_)[i].c1,
              static_cast<UINT8_C1 *>(dst_uint8.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestDivideInt8) {
  const size_t cols = 4;
  // Test int8
  LiteMat src1_int8(1, cols, LDataType(LDataType::INT8));
  LiteMat src2_int8(1, cols, LDataType(LDataType::INT8));
  LiteMat expect_int8(1, cols, LDataType(LDataType::INT8));
  for (size_t i = 0; i < cols; i++) {
    static_cast<INT8_C1 *>(src1_int8.data_ptr_)[i] = 8;
    static_cast<INT8_C1 *>(src2_int8.data_ptr_)[i] = -4;
    static_cast<INT8_C1 *>(expect_int8.data_ptr_)[i] = -2;
  }
  LiteMat dst_int8;
  EXPECT_TRUE(Divide(src1_int8, src2_int8, &dst_int8));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<INT8_C1 *>(expect_int8.data_ptr_)[i].c1, static_cast<INT8_C1 *>(dst_int8.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestDivideUInt16) {
  const size_t cols = 4;
  // Test uint16
  LiteMat src1_uint16(1, cols, LDataType(LDataType::UINT16));
  LiteMat src2_uint16(1, cols, LDataType(LDataType::UINT16));
  LiteMat expect_uint16(1, cols, LDataType(LDataType::UINT16));
  for (size_t i = 0; i < cols; i++) {
    static_cast<UINT16_C1 *>(src1_uint16.data_ptr_)[i] = 40000;
    static_cast<UINT16_C1 *>(src2_uint16.data_ptr_)[i] = 20000;
    static_cast<UINT16_C1 *>(expect_uint16.data_ptr_)[i] = 2;
  }
  LiteMat dst_uint16;
  EXPECT_TRUE(Divide(src1_uint16, src2_uint16, &dst_uint16));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<UINT16_C1 *>(expect_uint16.data_ptr_)[i].c1,
              static_cast<UINT16_C1 *>(dst_uint16.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestDivideInt16) {
  const size_t cols = 4;
  // Test int16
  LiteMat src1_int16(1, cols, LDataType(LDataType::INT16));
  LiteMat src2_int16(1, cols, LDataType(LDataType::INT16));
  LiteMat expect_int16(1, cols, LDataType(LDataType::INT16));
  for (size_t i = 0; i < cols; i++) {
    static_cast<INT16_C1 *>(src1_int16.data_ptr_)[i] = 30000;
    static_cast<INT16_C1 *>(src2_int16.data_ptr_)[i] = -3;
    static_cast<INT16_C1 *>(expect_int16.data_ptr_)[i] = -10000;
  }
  LiteMat dst_int16;
  EXPECT_TRUE(Divide(src1_int16, src2_int16, &dst_int16));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<INT16_C1 *>(expect_int16.data_ptr_)[i].c1,
              static_cast<INT16_C1 *>(dst_int16.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestDivideUInt32) {
  const size_t cols = 4;
  // Test uint16
  LiteMat src1_uint32(1, cols, LDataType(LDataType::UINT32));
  LiteMat src2_uint32(1, cols, LDataType(LDataType::UINT32));
  LiteMat expect_uint32(1, cols, LDataType(LDataType::UINT32));
  for (size_t i = 0; i < cols; i++) {
    static_cast<UINT32_C1 *>(src1_uint32.data_ptr_)[i] = 4000000000;
    static_cast<UINT32_C1 *>(src2_uint32.data_ptr_)[i] = 4;
    static_cast<UINT32_C1 *>(expect_uint32.data_ptr_)[i] = 1000000000;
  }
  LiteMat dst_uint32;
  EXPECT_TRUE(Divide(src1_uint32, src2_uint32, &dst_uint32));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<UINT32_C1 *>(expect_uint32.data_ptr_)[i].c1,
              static_cast<UINT32_C1 *>(dst_uint32.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestDivideInt32) {
  const size_t cols = 4;
  // Test int32
  LiteMat src1_int32(1, cols, LDataType(LDataType::INT32));
  LiteMat src2_int32(1, cols, LDataType(LDataType::INT32));
  LiteMat expect_int32(1, cols, LDataType(LDataType::INT32));
  for (size_t i = 0; i < cols; i++) {
    static_cast<INT32_C1 *>(src1_int32.data_ptr_)[i] = 2000000000;
    static_cast<INT32_C1 *>(src2_int32.data_ptr_)[i] = -2;
    static_cast<INT32_C1 *>(expect_int32.data_ptr_)[i] = -1000000000;
  }
  LiteMat dst_int32;
  EXPECT_TRUE(Divide(src1_int32, src2_int32, &dst_int32));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<INT32_C1 *>(expect_int32.data_ptr_)[i].c1,
              static_cast<INT32_C1 *>(dst_int32.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestDivideFloat) {
  const size_t cols = 4;
  // Test float
  LiteMat src1_float(1, cols, LDataType(LDataType::FLOAT32));
  LiteMat src2_float(1, cols, LDataType(LDataType::FLOAT32));
  LiteMat expect_float(1, cols, LDataType(LDataType::FLOAT32));
  for (size_t i = 0; i < cols; i++) {
    static_cast<FLOAT32_C1 *>(src1_float.data_ptr_)[i] = 12.34f;
    static_cast<FLOAT32_C1 *>(src2_float.data_ptr_)[i] = -2.0f;
    static_cast<FLOAT32_C1 *>(expect_float.data_ptr_)[i] = -6.17f;
  }
  LiteMat dst_float;
  EXPECT_TRUE(Divide(src1_float, src2_float, &dst_float));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_FLOAT_EQ(static_cast<FLOAT32_C1 *>(expect_float.data_ptr_)[i].c1,
                    static_cast<FLOAT32_C1 *>(dst_float.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestMultiplyUint8) {
  const size_t cols = 4;
  // Test uint8
  LiteMat src1_uint8(1, cols);
  LiteMat src2_uint8(1, cols);
  LiteMat expect_uint8(1, cols);
  for (size_t i = 0; i < cols; i++) {
    static_cast<UINT8_C1 *>(src1_uint8.data_ptr_)[i] = 8;
    static_cast<UINT8_C1 *>(src2_uint8.data_ptr_)[i] = 4;
    static_cast<UINT8_C1 *>(expect_uint8.data_ptr_)[i] = 32;
  }
  LiteMat dst_uint8;
  EXPECT_TRUE(Multiply(src1_uint8, src2_uint8, &dst_uint8));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<UINT8_C1 *>(expect_uint8.data_ptr_)[i].c1,
              static_cast<UINT8_C1 *>(dst_uint8.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestMultiplyUInt16) {
  const size_t cols = 4;
  // Test int16
  LiteMat src1_int16(1, cols, LDataType(LDataType::UINT16));
  LiteMat src2_int16(1, cols, LDataType(LDataType::UINT16));
  LiteMat expect_int16(1, cols, LDataType(LDataType::UINT16));
  for (size_t i = 0; i < cols; i++) {
    static_cast<UINT16_C1 *>(src1_int16.data_ptr_)[i] = 60000;
    static_cast<UINT16_C1 *>(src2_int16.data_ptr_)[i] = 2;
    static_cast<UINT16_C1 *>(expect_int16.data_ptr_)[i] = 65535;
  }
  LiteMat dst_int16;
  EXPECT_TRUE(Multiply(src1_int16, src2_int16, &dst_int16));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_EQ(static_cast<UINT16_C1 *>(expect_int16.data_ptr_)[i].c1,
              static_cast<UINT16_C1 *>(dst_int16.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestMultiplyFloat) {
  const size_t cols = 4;
  // Test float
  LiteMat src1_float(1, cols, LDataType(LDataType::FLOAT32));
  LiteMat src2_float(1, cols, LDataType(LDataType::FLOAT32));
  LiteMat expect_float(1, cols, LDataType(LDataType::FLOAT32));
  for (size_t i = 0; i < cols; i++) {
    static_cast<FLOAT32_C1 *>(src1_float.data_ptr_)[i] = 30.0f;
    static_cast<FLOAT32_C1 *>(src2_float.data_ptr_)[i] = -2.0f;
    static_cast<FLOAT32_C1 *>(expect_float.data_ptr_)[i] = -60.0f;
  }
  LiteMat dst_float;
  EXPECT_TRUE(Multiply(src1_float, src2_float, &dst_float));
  for (size_t i = 0; i < cols; i++) {
    EXPECT_FLOAT_EQ(static_cast<FLOAT32_C1 *>(expect_float.data_ptr_)[i].c1,
                    static_cast<FLOAT32_C1 *>(dst_float.data_ptr_)[i].c1);
  }
}

TEST_F(MindDataImageProcess, TestExtractChannel) {
  LiteMat lite_single;
  LiteMat lite_mat = LiteMat(1, 4, 3, LDataType::UINT16);

  EXPECT_FALSE(ExtractChannel(lite_mat, lite_single, 0));
  EXPECT_TRUE(lite_single.IsEmpty());
}
TEST_F(MindDataImageProcess, testROI3C) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat cv_roi = cv::Mat(src_image, cv::Rect(500, 500, 3000, 1500));

  cv::imwrite("./cv_roi.jpg", cv_roi);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret = InitFromPixel(src_image.data, LPixelType::BGR, LDataType::UINT8, src_image.cols, src_image.rows, lite_mat_bgr);
  EXPECT_TRUE(ret);
  LiteMat lite_roi;

  ret = lite_mat_bgr.GetROI(500, 500, 3000, 1500, lite_roi);
  EXPECT_TRUE(ret);

  LiteMat lite_roi_save(3000, 1500, lite_roi.channel_, LDataType::UINT8);

  for (size_t i = 0; i < lite_roi.height_; i++) {
    const unsigned char *ptr = lite_roi.ptr<unsigned char>(i);
    size_t image_size = lite_roi.width_ * lite_roi.channel_ * sizeof(unsigned char);
    unsigned char *dst_ptr = (unsigned char *)lite_roi_save.data_ptr_ + image_size * i;
    (void)memcpy(dst_ptr, ptr, image_size);
  }

  cv::Mat dst_imageR(lite_roi_save.height_, lite_roi_save.width_, CV_8UC3, lite_roi_save.data_ptr_);
  cv::imwrite("./lite_roi.jpg", dst_imageR);
}

TEST_F(MindDataImageProcess, testROI3CFalse) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat cv_roi = cv::Mat(src_image, cv::Rect(500, 500, 3000, 1500));

  cv::imwrite("./cv_roi.jpg", cv_roi);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret = InitFromPixel(src_image.data, LPixelType::BGR, LDataType::UINT8, src_image.cols, src_image.rows, lite_mat_bgr);
  EXPECT_TRUE(ret);
  LiteMat lite_roi;

  ret = lite_mat_bgr.GetROI(500, 500, 1200, -100, lite_roi);
  EXPECT_FALSE(ret);
}

TEST_F(MindDataImageProcess, testROI1C) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, CV_BGR2GRAY);
  cv::Mat cv_roi_gray = cv::Mat(gray_image, cv::Rect(500, 500, 3000, 1500));

  cv::imwrite("./cv_roi_gray.jpg", cv_roi_gray);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);
  bool ret = false;
  LiteMat lite_mat_gray;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_gray);
  EXPECT_TRUE(ret);
  LiteMat lite_roi_gray;

  ret = lite_mat_gray.GetROI(500, 500, 3000, 1500, lite_roi_gray);
  EXPECT_TRUE(ret);

  LiteMat lite_roi_gray_save(3000, 1500, lite_roi_gray.channel_, LDataType::UINT8);

  for (size_t i = 0; i < lite_roi_gray.height_; i++) {
    const unsigned char *ptr = lite_roi_gray.ptr<unsigned char>(i);
    size_t image_size = lite_roi_gray.width_ * lite_roi_gray.channel_ * sizeof(unsigned char);
    unsigned char *dst_ptr = (unsigned char *)lite_roi_gray_save.data_ptr_ + image_size * i;
    (void)memcpy(dst_ptr, ptr, image_size);
  }

  cv::Mat dst_imageR(lite_roi_gray_save.height_, lite_roi_gray_save.width_, CV_8UC1, lite_roi_gray_save.data_ptr_);
  cv::imwrite("./lite_roi.jpg", dst_imageR);
}

// warp
TEST_F(MindDataImageProcess, testWarpAffineBGR) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];
  srcTri[0] = cv::Point2f(0, 0);
  srcTri[1] = cv::Point2f(src_image.cols - 1, 0);
  srcTri[2] = cv::Point2f(0, src_image.rows - 1);

  dstTri[0] = cv::Point2f(src_image.cols * 0.0, src_image.rows * 0.33);
  dstTri[1] = cv::Point2f(src_image.cols * 0.85, src_image.rows * 0.25);
  dstTri[2] = cv::Point2f(src_image.cols * 0.15, src_image.rows * 0.7);

  cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
  ;
  cv::Mat warp_dstImage;
  cv::warpAffine(src_image, warp_dstImage, warp_mat, warp_dstImage.size());
  cv::imwrite("./warpAffine_cv_bgr.png", warp_dstImage);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret = InitFromPixel(src_image.data, LPixelType::BGR, LDataType::UINT8, src_image.cols, src_image.rows, lite_mat_bgr);
  EXPECT_TRUE(ret);
  double *mat_ptr = warp_mat.ptr<double>(0);
  LiteMat lite_M(3, 2, 1, mat_ptr, LDataType::DOUBLE);

  LiteMat lite_warp;
  std::vector<uint8_t> borderValues;
  borderValues.push_back(0);
  borderValues.push_back(0);
  borderValues.push_back(0);
  ret = WarpAffineBilinear(lite_mat_bgr, lite_warp, lite_M, lite_mat_bgr.width_, lite_mat_bgr.height_,
                           PADD_BORDER_CONSTANT, borderValues);
  EXPECT_TRUE(ret);

  cv::Mat dst_imageR(lite_warp.height_, lite_warp.width_, CV_8UC3, lite_warp.data_ptr_);
  cv::imwrite("./warpAffine_lite_bgr.png", dst_imageR);
}

TEST_F(MindDataImageProcess, testWarpAffineBGRScale) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];
  srcTri[0] = cv::Point2f(10, 20);
  srcTri[1] = cv::Point2f(src_image.cols - 1 - 100, 0);
  srcTri[2] = cv::Point2f(0, src_image.rows - 1 - 300);

  dstTri[0] = cv::Point2f(src_image.cols * 0.22, src_image.rows * 0.33);
  dstTri[1] = cv::Point2f(src_image.cols * 0.87, src_image.rows * 0.75);
  dstTri[2] = cv::Point2f(src_image.cols * 0.35, src_image.rows * 0.37);

  cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
  ;
  cv::Mat warp_dstImage;
  cv::warpAffine(src_image, warp_dstImage, warp_mat, warp_dstImage.size());
  cv::imwrite("./warpAffine_cv_bgr_scale.png", warp_dstImage);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret = InitFromPixel(src_image.data, LPixelType::BGR, LDataType::UINT8, src_image.cols, src_image.rows, lite_mat_bgr);
  EXPECT_TRUE(ret);
  double *mat_ptr = warp_mat.ptr<double>(0);
  LiteMat lite_M(3, 2, 1, mat_ptr, LDataType::DOUBLE);

  LiteMat lite_warp;
  std::vector<uint8_t> borderValues;
  borderValues.push_back(0);
  borderValues.push_back(0);
  borderValues.push_back(0);
  ret = WarpAffineBilinear(lite_mat_bgr, lite_warp, lite_M, lite_mat_bgr.width_, lite_mat_bgr.height_,
                           PADD_BORDER_CONSTANT, borderValues);
  EXPECT_TRUE(ret);

  cv::Mat dst_imageR(lite_warp.height_, lite_warp.width_, CV_8UC3, lite_warp.data_ptr_);
  cv::imwrite("./warpAffine_lite_bgr_scale.png", dst_imageR);
}

TEST_F(MindDataImageProcess, testWarpAffineBGRResize) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];
  srcTri[0] = cv::Point2f(10, 20);
  srcTri[1] = cv::Point2f(src_image.cols - 1 - 100, 0);
  srcTri[2] = cv::Point2f(0, src_image.rows - 1 - 300);

  dstTri[0] = cv::Point2f(src_image.cols * 0.22, src_image.rows * 0.33);
  dstTri[1] = cv::Point2f(src_image.cols * 0.87, src_image.rows * 0.75);
  dstTri[2] = cv::Point2f(src_image.cols * 0.35, src_image.rows * 0.37);

  cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
  ;
  cv::Mat warp_dstImage;
  cv::warpAffine(src_image, warp_dstImage, warp_mat, cv::Size(src_image.cols + 200, src_image.rows - 300));
  cv::imwrite("./warpAffine_cv_bgr_resize.png", warp_dstImage);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret = InitFromPixel(src_image.data, LPixelType::BGR, LDataType::UINT8, src_image.cols, src_image.rows, lite_mat_bgr);
  EXPECT_TRUE(ret);
  double *mat_ptr = warp_mat.ptr<double>(0);
  LiteMat lite_M(3, 2, 1, mat_ptr, LDataType::DOUBLE);

  LiteMat lite_warp;
  std::vector<uint8_t> borderValues;
  borderValues.push_back(0);
  borderValues.push_back(0);
  borderValues.push_back(0);
  ret = WarpAffineBilinear(lite_mat_bgr, lite_warp, lite_M, lite_mat_bgr.width_ + 200, lite_mat_bgr.height_ - 300,
                           PADD_BORDER_CONSTANT, borderValues);
  EXPECT_TRUE(ret);

  cv::Mat dst_imageR(lite_warp.height_, lite_warp.width_, CV_8UC3, lite_warp.data_ptr_);
  cv::imwrite("./warpAffine_lite_bgr_resize.png", dst_imageR);
}

TEST_F(MindDataImageProcess, testWarpAffineGray) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, CV_BGR2GRAY);

  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];
  srcTri[0] = cv::Point2f(0, 0);
  srcTri[1] = cv::Point2f(src_image.cols - 1, 0);
  srcTri[2] = cv::Point2f(0, src_image.rows - 1);

  dstTri[0] = cv::Point2f(src_image.cols * 0.0, src_image.rows * 0.33);
  dstTri[1] = cv::Point2f(src_image.cols * 0.85, src_image.rows * 0.25);
  dstTri[2] = cv::Point2f(src_image.cols * 0.15, src_image.rows * 0.7);

  cv::Mat warp_mat = cv::getAffineTransform(srcTri, dstTri);
  ;
  cv::Mat warp_gray_dstImage;
  cv::warpAffine(gray_image, warp_gray_dstImage, warp_mat, cv::Size(src_image.cols + 200, src_image.rows - 300));
  cv::imwrite("./warpAffine_cv_gray.png", warp_gray_dstImage);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);
  bool ret = false;
  LiteMat lite_mat_gray;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_gray);
  EXPECT_TRUE(ret);
  double *mat_ptr = warp_mat.ptr<double>(0);
  LiteMat lite_M(3, 2, 1, mat_ptr, LDataType::DOUBLE);

  LiteMat lite_warp;
  std::vector<uint8_t> borderValues;
  borderValues.push_back(0);
  ret = WarpAffineBilinear(lite_mat_gray, lite_warp, lite_M, lite_mat_gray.width_ + 200, lite_mat_gray.height_ - 300,
                           PADD_BORDER_CONSTANT, borderValues);
  EXPECT_TRUE(ret);

  cv::Mat dst_imageR(lite_warp.height_, lite_warp.width_, CV_8UC1, lite_warp.data_ptr_);
  cv::imwrite("./warpAffine_lite_gray.png", dst_imageR);
}

TEST_F(MindDataImageProcess, testWarpPerspectiveBGRResize) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Point2f srcQuad[4], dstQuad[4];
  srcQuad[0].x = 0;
  srcQuad[0].y = 0;
  srcQuad[1].x = src_image.cols - 1.;
  srcQuad[1].y = 0;
  srcQuad[2].x = 0;
  srcQuad[2].y = src_image.rows - 1;
  srcQuad[3].x = src_image.cols - 1;
  srcQuad[3].y = src_image.rows - 1;

  dstQuad[0].x = src_image.cols * 0.05;
  dstQuad[0].y = src_image.rows * 0.33;
  dstQuad[1].x = src_image.cols * 0.9;
  dstQuad[1].y = src_image.rows * 0.25;
  dstQuad[2].x = src_image.cols * 0.2;
  dstQuad[2].y = src_image.rows * 0.7;
  dstQuad[3].x = src_image.cols * 0.8;
  dstQuad[3].y = src_image.rows * 0.9;

  cv::Mat ptran = cv::getPerspectiveTransform(srcQuad, dstQuad, cv::DECOMP_SVD);
  cv::Mat warp_dstImage;
  cv::warpPerspective(src_image, warp_dstImage, ptran, cv::Size(src_image.cols + 200, src_image.rows - 300));
  cv::imwrite("./warpPerspective_cv_bgr.png", warp_dstImage);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret = InitFromPixel(src_image.data, LPixelType::BGR, LDataType::UINT8, src_image.cols, src_image.rows, lite_mat_bgr);
  EXPECT_TRUE(ret);
  double *mat_ptr = ptran.ptr<double>(0);
  LiteMat lite_M(3, 3, 1, mat_ptr, LDataType::DOUBLE);

  LiteMat lite_warp;
  std::vector<uint8_t> borderValues;
  borderValues.push_back(0);
  borderValues.push_back(0);
  borderValues.push_back(0);
  ret = WarpPerspectiveBilinear(lite_mat_bgr, lite_warp, lite_M, lite_mat_bgr.width_ + 200, lite_mat_bgr.height_ - 300,
                                PADD_BORDER_CONSTANT, borderValues);
  EXPECT_TRUE(ret);

  cv::Mat dst_imageR(lite_warp.height_, lite_warp.width_, CV_8UC3, lite_warp.data_ptr_);
  cv::imwrite("./warpPerspective_lite_bgr.png", dst_imageR);
}

TEST_F(MindDataImageProcess, testWarpPerspectiveGrayResize) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, CV_BGR2GRAY);

  cv::Point2f srcQuad[4], dstQuad[4];
  srcQuad[0].x = 0;
  srcQuad[0].y = 0;
  srcQuad[1].x = src_image.cols - 1.;
  srcQuad[1].y = 0;
  srcQuad[2].x = 0;
  srcQuad[2].y = src_image.rows - 1;
  srcQuad[3].x = src_image.cols - 1;
  srcQuad[3].y = src_image.rows - 1;

  dstQuad[0].x = src_image.cols * 0.05;
  dstQuad[0].y = src_image.rows * 0.33;
  dstQuad[1].x = src_image.cols * 0.9;
  dstQuad[1].y = src_image.rows * 0.25;
  dstQuad[2].x = src_image.cols * 0.2;
  dstQuad[2].y = src_image.rows * 0.7;
  dstQuad[3].x = src_image.cols * 0.8;
  dstQuad[3].y = src_image.rows * 0.9;

  cv::Mat ptran = cv::getPerspectiveTransform(srcQuad, dstQuad, cv::DECOMP_SVD);
  cv::Mat warp_dstImage;
  cv::warpPerspective(gray_image, warp_dstImage, ptran, cv::Size(gray_image.cols + 200, gray_image.rows - 300));
  cv::imwrite("./warpPerspective_cv_gray.png", warp_dstImage);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);
  bool ret = false;
  LiteMat lite_mat_gray;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_gray);
  EXPECT_TRUE(ret);
  double *mat_ptr = ptran.ptr<double>(0);
  LiteMat lite_M(3, 3, 1, mat_ptr, LDataType::DOUBLE);

  LiteMat lite_warp;
  std::vector<uint8_t> borderValues;
  borderValues.push_back(0);
  ret = WarpPerspectiveBilinear(lite_mat_gray, lite_warp, lite_M, lite_mat_gray.width_ + 200,
                                lite_mat_gray.height_ - 300, PADD_BORDER_CONSTANT, borderValues);
  EXPECT_TRUE(ret);

  cv::Mat dst_imageR(lite_warp.height_, lite_warp.width_, CV_8UC1, lite_warp.data_ptr_);
  cv::imwrite("./warpPerspective_lite_gray.png", dst_imageR);
}

TEST_F(MindDataImageProcess, testGetRotationMatrix2D) {
  std::vector<std::vector<double>> expect_matrix = {{0.250000, 0.433013, -0.116025}, {-0.433013, 0.250000, 1.933013}};

  double angle = 60.0;
  double scale = 0.5;

  LiteMat M;
  bool ret = false;
  ret = GetRotationMatrix2D(1.0f, 2.0f, angle, scale, M);
  EXPECT_TRUE(ret);
  AccuracyComparison(expect_matrix, M);
}

TEST_F(MindDataImageProcess, testGetPerspectiveTransform) {
  std::vector<std::vector<double>> expect_matrix = {
    {1.272113, 3.665216, -788.484287}, {-0.394146, 3.228247, -134.009780}, {-0.001460, 0.006414, 1}};

  std::vector<Point> src = {Point(165, 270), Point(835, 270), Point(360, 125), Point(615, 125)};
  std::vector<Point> dst = {Point(165, 270), Point(835, 270), Point(100, 100), Point(500, 30)};

  LiteMat M;
  bool ret = false;
  ret = GetPerspectiveTransform(src, dst, M);
  EXPECT_TRUE(ret);
  AccuracyComparison(expect_matrix, M);
}

TEST_F(MindDataImageProcess, testGetPerspectiveTransformFail) {
  std::vector<Point> src = {Point(165, 270), Point(835, 270), Point(360, 125), Point(615, 125)};
  std::vector<Point> dst = {Point(100, 100), Point(500, 30)};

  LiteMat M;
  bool ret = GetPerspectiveTransform(src, dst, M);
  EXPECT_FALSE(ret);

  std::vector<Point> src1 = {Point(360, 125), Point(615, 125)};
  std::vector<Point> dst2 = {Point(165, 270), Point(835, 270), Point(100, 100), Point(500, 30)};

  LiteMat M1;
  bool ret1 = GetPerspectiveTransform(src, dst, M1);
  EXPECT_FALSE(ret1);
}

TEST_F(MindDataImageProcess, testGetAffineTransform) {
  std::vector<std::vector<double>> expect_matrix = {{0.400000, 0.066667, 16.666667}, {0.000000, 0.333333, 23.333333}};

  std::vector<Point> src = {Point(50, 50), Point(200, 50), Point(50, 200)};
  std::vector<Point> dst = {Point(40, 40), Point(100, 40), Point(50, 90)};

  LiteMat M;
  bool ret = false;
  ret = GetAffineTransform(src, dst, M);
  EXPECT_TRUE(ret);
  AccuracyComparison(expect_matrix, M);
}

TEST_F(MindDataImageProcess, testGetAffineTransformFail) {
  std::vector<Point> src = {Point(50, 50), Point(200, 50)};
  std::vector<Point> dst = {Point(40, 40), Point(100, 40), Point(50, 90)};

  LiteMat M;
  bool ret = GetAffineTransform(src, dst, M);
  EXPECT_FALSE(ret);

  std::vector<Point> src1 = {Point(50, 50), Point(200, 50), Point(50, 200)};
  std::vector<Point> dst1 = {Point(40, 40), Point(100, 40)};

  LiteMat M1;
  bool ret1 = GetAffineTransform(src1, dst1, M1);
  EXPECT_FALSE(ret1);
}

TEST_F(MindDataImageProcess, TestConv2D8U) {
  LiteMat lite_mat_src;
  lite_mat_src.Init(3, 3, 1, LDataType::UINT8);
  uint8_t *src_ptr = lite_mat_src;
  for (int i = 0; i < 9; i++) {
    src_ptr[i] = i % 3;
  }
  LiteMat kernel;
  kernel.Init(3, 3, 1, LDataType::FLOAT32);
  float *kernel_ptr = kernel;
  for (int i = 0; i < 9; i++) {
    kernel_ptr[i] = i % 2;
  }
  LiteMat lite_mat_dst;
  bool ret = Conv2D(lite_mat_src, kernel, lite_mat_dst, LDataType::UINT8);
  ASSERT_TRUE(ret == true);

  std::vector<uint8_t> expected_result = {2, 4, 6, 2, 4, 6, 2, 4, 6};

  size_t total_size = lite_mat_dst.height_ * lite_mat_dst.width_ * lite_mat_dst.channel_;
  float distance = 0.0f;
  for (size_t i = 0; i < total_size; i++) {
    distance += pow(((uint8_t *)lite_mat_dst)[i] - expected_result[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(MindDataImageProcess, TestConv2D32F) {
  LiteMat lite_mat_src;
  lite_mat_src.Init(2, 2, 1, LDataType::FLOAT32);
  float *src_ptr = lite_mat_src;
  for (int i = 0; i < 4; i++) {
    src_ptr[i] = static_cast<float>(i) / 2;
  }
  LiteMat kernel;
  kernel.Init(2, 2, 1, LDataType::FLOAT32);
  float *kernel_ptr = kernel;
  for (int i = 0; i < 4; i++) {
    kernel_ptr[i] = static_cast<float>(i);
  }
  LiteMat lite_mat_dst;
  bool ret = Conv2D(lite_mat_src, kernel, lite_mat_dst, LDataType::FLOAT32);
  ASSERT_TRUE(ret == true);

  std::vector<float> expected_result = {2.f, 3.f, 6.f, 7.f};

  size_t total_size = lite_mat_dst.height_ * lite_mat_dst.width_ * lite_mat_dst.channel_;
  float distance = 0.0f;
  for (size_t i = 0; i < total_size; i++) {
    distance += pow(((float *)lite_mat_dst)[i] - expected_result[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(MindDataImageProcess, TestGaussianBlurSize35) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat dst_image;
  cv::GaussianBlur(src_image, dst_image, cv::Size(3, 5), 3, 3);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);

  LiteMat lite_mat_bgr;
  bool ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);

  LiteMat lite_mat_dst;
  ret = GaussianBlur(lite_mat_bgr, lite_mat_dst, {3, 5}, 3, 3);
  ASSERT_TRUE(ret == true);

  size_t total_size = lite_mat_dst.height_ * lite_mat_dst.width_ * lite_mat_dst.channel_;
  double distance = 0.0f;
  for (size_t i = 0; i < total_size; i++) {
    distance += pow((uint8_t)dst_image.data[i] - ((uint8_t *)lite_mat_dst)[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_LE(distance, 1.0f);
}

TEST_F(MindDataImageProcess, TestGaussianBlurSize13) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat dst_image;
  cv::GaussianBlur(src_image, dst_image, cv::Size(1, 3), 3);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);

  LiteMat lite_mat_bgr;
  bool ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);

  LiteMat lite_mat_dst;
  ret = GaussianBlur(lite_mat_bgr, lite_mat_dst, {1, 3}, 3);
  ASSERT_TRUE(ret == true);

  size_t total_size = lite_mat_dst.height_ * lite_mat_dst.width_ * lite_mat_dst.channel_;
  double distance = 0.0f;
  for (size_t i = 0; i < total_size; i++) {
    distance += pow((uint8_t)dst_image.data[i] - ((uint8_t *)lite_mat_dst)[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_LE(distance, 1.0f);
}

TEST_F(MindDataImageProcess, TestGaussianBlurInvalidParams) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);

  LiteMat lite_mat_bgr;
  bool ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);

  LiteMat lite_mat_dst;

  // even size
  ret = GaussianBlur(lite_mat_bgr, lite_mat_dst, {3, 4}, 3);
  ASSERT_TRUE(ret == false);

  // ksize.size() != 2
  ret = GaussianBlur(lite_mat_bgr, lite_mat_dst, {3, 4, 5}, 3);
  ASSERT_TRUE(ret == false);

  // size less or equal to 0
  ret = GaussianBlur(lite_mat_bgr, lite_mat_dst, {0, 3}, 3);
  ASSERT_TRUE(ret == false);

  // sigmaX less or equal to 0
  ret = GaussianBlur(lite_mat_bgr, lite_mat_dst, {3, 3}, 0);
  ASSERT_TRUE(ret == false);
}

TEST_F(MindDataImageProcess, TestCannySize3) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, CV_BGR2GRAY);
  cv::Mat dst_image;
  cv::Canny(gray_image, dst_image, 100, 200, 3);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);
  bool ret = false;
  LiteMat lite_mat_gray;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_gray);
  ASSERT_TRUE(ret == true);

  LiteMat lite_mat_dst;
  ret = Canny(lite_mat_gray, lite_mat_dst, 100, 200, 3);
  ASSERT_TRUE(ret == true);

  int total_size = lite_mat_dst.height_ * lite_mat_dst.width_ * lite_mat_dst.channel_;
  double distance = 0.0f;
  for (int i = 0; i < total_size; i++) {
    distance += pow((uint8_t)dst_image.data[i] - ((uint8_t *)lite_mat_dst)[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(MindDataImageProcess, TestCannySize5) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, CV_BGR2GRAY);
  cv::Mat dst_image;
  cv::Canny(gray_image, dst_image, 200, 300, 5);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);
  bool ret = false;
  LiteMat lite_mat_gray;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_gray);
  ASSERT_TRUE(ret == true);

  LiteMat lite_mat_dst;
  ret = Canny(lite_mat_gray, lite_mat_dst, 200, 300, 5);
  ASSERT_TRUE(ret == true);

  int total_size = lite_mat_dst.height_ * lite_mat_dst.width_ * lite_mat_dst.channel_;
  double distance = 0.0f;
  for (int i = 0; i < total_size; i++) {
    distance += pow((uint8_t)dst_image.data[i] - ((uint8_t *)lite_mat_dst)[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(MindDataImageProcess, TestCannySize7) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, CV_BGR2GRAY);
  cv::Mat dst_image;
  cv::Canny(gray_image, dst_image, 110, 220, 7);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);
  bool ret = false;
  LiteMat lite_mat_gray;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_gray);
  ASSERT_TRUE(ret == true);

  LiteMat lite_mat_dst;
  ret = Canny(lite_mat_gray, lite_mat_dst, 110, 220, 7);
  ASSERT_TRUE(ret == true);

  int total_size = lite_mat_dst.height_ * lite_mat_dst.width_ * lite_mat_dst.channel_;
  double distance = 0.0f;
  for (int i = 0; i < total_size; i++) {
    distance += pow((uint8_t)dst_image.data[i] - ((uint8_t *)lite_mat_dst)[i], 2);
  }
  distance = sqrt(distance / total_size);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(MindDataImageProcess, TestCannyL2) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, CV_BGR2GRAY);
  cv::Mat dst_image;
  cv::Canny(gray_image, dst_image, 50, 150, 3, true);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);
  bool ret = false;
  LiteMat lite_mat_gray;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_gray);
  ASSERT_TRUE(ret == true);

  LiteMat lite_mat_dst;
  ret = Canny(lite_mat_gray, lite_mat_dst, 50, 150, 3, true);
  ASSERT_TRUE(ret == true);

  int total_size = lite_mat_dst.height_ * lite_mat_dst.width_ * lite_mat_dst.channel_;
  double distance = 0.0f;
  for (int i = 0; i < total_size; i++) {
    distance += pow((uint8_t)dst_image.data[i] - ((uint8_t *)lite_mat_dst)[i], 2);
  }

  distance = sqrt(distance / total_size);
  EXPECT_EQ(distance, 0.0f);
}

TEST_F(MindDataImageProcess, TestCannyInvalidParams) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);

  bool ret = false;
  LiteMat lite_mat_bgr;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2BGR, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_bgr);
  ASSERT_TRUE(ret == true);

  // channel is not 1
  LiteMat lite_mat_dst;
  ret = Canny(lite_mat_bgr, lite_mat_dst, 70, 210, 3);
  ASSERT_TRUE(ret == false);

  LiteMat lite_mat_gray;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_gray);
  ASSERT_TRUE(ret == true);

  // low_thresh less than 0
  ret = Canny(lite_mat_gray, lite_mat_dst, -5, 230, 3);
  ASSERT_TRUE(ret == false);

  // high_thresh less than low_thresh
  ret = Canny(lite_mat_gray, lite_mat_dst, 250, 130, 3);
  ASSERT_TRUE(ret == false);

  // even size
  ret = Canny(lite_mat_gray, lite_mat_dst, 60, 180, 4);
  ASSERT_TRUE(ret == false);

  // size less than 3 or large than 7
  ret = Canny(lite_mat_gray, lite_mat_dst, 10, 190, 9);
  ASSERT_TRUE(ret == false);
}

TEST_F(MindDataImageProcess, TestSobel) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, CV_BGR2GRAY);

  cv::Mat sobel_image_x;
  cv::Mat sobel_image_y;
  cv::Sobel(gray_image, sobel_image_x, CV_32F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
  cv::Sobel(gray_image, sobel_image_y, CV_32F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);

  cv::Mat sobel_cv_x, sobel_cv_y;
  sobel_image_x.convertTo(sobel_cv_x, CV_8UC1);
  sobel_image_y.convertTo(sobel_cv_y, CV_8UC1);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);
  bool ret = false;
  LiteMat lite_mat_gray;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_gray);
  ASSERT_TRUE(ret == true);
  LiteMat lite_mat_x;
  LiteMat lite_mat_y;
  Sobel(lite_mat_gray, lite_mat_x, 1, 0, 3, 1, PaddBorderType::PADD_BORDER_REPLICATE);
  Sobel(lite_mat_gray, lite_mat_y, 0, 1, 3, 1, PaddBorderType::PADD_BORDER_REPLICATE);
  ASSERT_TRUE(ret == true);

  cv::Mat dst_imageX(lite_mat_x.height_, lite_mat_x.width_, CV_32FC1, lite_mat_x.data_ptr_);
  cv::Mat dst_imageY(lite_mat_y.height_, lite_mat_y.width_, CV_32FC1, lite_mat_y.data_ptr_);
  cv::Mat sobel_ms_x, sobel_ms_y;
  dst_imageX.convertTo(sobel_ms_x, CV_8UC1);
  dst_imageY.convertTo(sobel_ms_y, CV_8UC1);

  size_t total_size = lite_mat_x.height_ * lite_mat_x.width_ * lite_mat_x.channel_;
  float distance_x = 0.0f, distance_y = 0.0f;
  for (int i = 0; i < total_size; i++) {
    distance_x += pow((uint8_t)sobel_cv_x.data[i] - (uint8_t)sobel_ms_x.data[i], 2);
    distance_y += pow((uint8_t)sobel_cv_y.data[i] - (uint8_t)sobel_ms_y.data[i], 2);
  }
  distance_x = sqrt(distance_x / total_size);
  distance_y = sqrt(distance_y / total_size);
  EXPECT_EQ(distance_x, 0.0f);
  EXPECT_EQ(distance_y, 0.0f);
}

TEST_F(MindDataImageProcess, TestSobelFlag) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat src_image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat gray_image;
  cv::cvtColor(src_image, gray_image, CV_BGR2GRAY);

  cv::Mat sobel_image_x;
  cv::Sobel(gray_image, sobel_image_x, CV_32F, 3, 1, 5, 1, 0, cv::BORDER_REPLICATE);

  cv::Mat sobel_cv_x;
  sobel_image_x.convertTo(sobel_cv_x, CV_8UC1);

  cv::Mat rgba_mat;
  cv::cvtColor(src_image, rgba_mat, CV_BGR2RGBA);
  bool ret = false;
  LiteMat lite_mat_gray;
  ret =
    InitFromPixel(rgba_mat.data, LPixelType::RGBA2GRAY, LDataType::UINT8, rgba_mat.cols, rgba_mat.rows, lite_mat_gray);
  ASSERT_TRUE(ret == true);
  LiteMat lite_mat_x;
  Sobel(lite_mat_gray, lite_mat_x, 3, 1, 5, 1, PaddBorderType::PADD_BORDER_REPLICATE);
  ASSERT_TRUE(ret == true);

  cv::Mat dst_imageX(lite_mat_x.height_, lite_mat_x.width_, CV_32FC1, lite_mat_x.data_ptr_);
  cv::Mat sobel_ms_x;
  dst_imageX.convertTo(sobel_ms_x, CV_8UC1);

  size_t total_size = lite_mat_x.height_ * lite_mat_x.width_ * lite_mat_x.channel_;
  float distance_x = 0.0f;
  for (int i = 0; i < total_size; i++) {
    distance_x += pow((uint8_t)sobel_cv_x.data[i] - (uint8_t)sobel_ms_x.data[i], 2);
  }
  distance_x = sqrt(distance_x / total_size);
  EXPECT_EQ(distance_x, 0.0f);
}

TEST_F(MindDataImageProcess, testConvertRgbToGray) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat rgb_mat;
  cv::Mat rgb_mat1;

  cv::cvtColor(image, rgb_mat, CV_BGR2GRAY);
  cv::imwrite("./opencv_image.jpg", rgb_mat);

  cv::cvtColor(image, rgb_mat1, CV_BGR2RGB);

  LiteMat lite_mat_rgb;
  lite_mat_rgb.Init(rgb_mat1.cols, rgb_mat1.rows, rgb_mat1.channels(), rgb_mat1.data, LDataType::UINT8);
  LiteMat lite_mat_gray;
  bool ret = ConvertRgbToGray(lite_mat_rgb, LDataType::UINT8, image.cols, image.rows, lite_mat_gray);
  ASSERT_TRUE(ret == true);

  cv::Mat dst_image(lite_mat_gray.height_, lite_mat_gray.width_, CV_8UC1, lite_mat_gray.data_ptr_);
  cv::imwrite("./mindspore_image.jpg", dst_image);
  CompareMat(rgb_mat, lite_mat_gray);
}

TEST_F(MindDataImageProcess, testConvertRgbToGrayFail) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);
  cv::Mat rgb_mat;
  cv::Mat rgb_mat1;

  cv::cvtColor(image, rgb_mat, CV_BGR2GRAY);
  cv::imwrite("./opencv_image.jpg", rgb_mat);

  cv::cvtColor(image, rgb_mat1, CV_BGR2RGB);

  // The width and height of the output image is different from the original image.
  LiteMat lite_mat_rgb;
  lite_mat_rgb.Init(rgb_mat1.cols, rgb_mat1.rows, rgb_mat1.channels(), rgb_mat1.data, LDataType::UINT8);
  LiteMat lite_mat_gray;
  bool ret = ConvertRgbToGray(lite_mat_rgb, LDataType::UINT8, 1000, 1000, lite_mat_gray);
  ASSERT_TRUE(ret == false);

  // The input lite_mat_rgb object is null.
  LiteMat lite_mat_rgb1;
  LiteMat lite_mat_gray1;
  bool ret1 = ConvertRgbToGray(lite_mat_rgb1, LDataType::UINT8, image.cols, image.rows, lite_mat_gray1);
  ASSERT_TRUE(ret1 == false);

  // The channel of output image object is not 1.
  LiteMat lite_mat_rgb2;
  lite_mat_rgb2.Init(rgb_mat1.cols, rgb_mat1.rows, rgb_mat1.channels(), rgb_mat1.data, LDataType::UINT8);
  LiteMat lite_mat_gray2;
  lite_mat_gray2.Init(rgb_mat1.cols, rgb_mat1.rows, rgb_mat1.channels(), rgb_mat1.data, LDataType::UINT8);
  bool ret2 = ConvertRgbToGray(lite_mat_rgb2, LDataType::UINT8, image.cols, image.rows, lite_mat_gray2);
  ASSERT_TRUE(ret2 == false);
}

TEST_F(MindDataImageProcess, testResizePreserveARWithFillerv) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  LiteMat lite_mat_rgb;
  lite_mat_rgb.Init(image.cols, image.rows, image.channels(), image.data, LDataType::UINT8);
  LiteMat lite_mat_resize;
  float ratioShiftWShiftH[3] = {0};
  float invM[2][3] = {{0, 0, 0}, {0, 0, 0}};
  int h = 1000;
  int w = 1000;
  bool ret = ResizePreserveARWithFiller(lite_mat_rgb, lite_mat_resize, h, w, &ratioShiftWShiftH, &invM, 0);
  ASSERT_TRUE(ret == true);
  cv::Mat dst_image(lite_mat_resize.height_, lite_mat_resize.width_, CV_32FC3, lite_mat_resize.data_ptr_);
  cv::imwrite("./mindspore_image.jpg", dst_image);
}

TEST_F(MindDataImageProcess, testResizePreserveARWithFillervFail) {
  std::string filename = "data/dataset/apple.jpg";
  cv::Mat image = cv::imread(filename, cv::ImreadModes::IMREAD_COLOR);

  // The input lite_mat_rgb object is null.
  LiteMat lite_mat_rgb;
  LiteMat lite_mat_resize;
  float ratioShiftWShiftH[3] = {0};
  float invM[2][3] = {{0, 0, 0}, {0, 0, 0}};
  int h = 1000;
  int w = 1000;
  bool ret = ResizePreserveARWithFiller(lite_mat_rgb, lite_mat_resize, h, w, &ratioShiftWShiftH, &invM, 0);
  ASSERT_TRUE(ret == false);

  // The channel of input lite_mat_rgb object is not 3.
  LiteMat lite_mat_rgb1;
  lite_mat_rgb1.Init(image.cols, image.rows, 1, image.data, LDataType::UINT8);
  LiteMat lite_mat_resize1;
  float ratioShiftWShiftH1[3] = {0};
  float invM1[2][3] = {{0, 0, 0}, {0, 0, 0}};
  int h1 = 1000;
  int w1 = 1000;
  bool ret1 = ResizePreserveARWithFiller(lite_mat_rgb1, lite_mat_resize1, h1, w1, &ratioShiftWShiftH1, &invM1, 0);
  ASSERT_TRUE(ret1 == false);

  // The ratioShiftWShiftH2 and invM2 is null.
  LiteMat lite_mat_rgb2;
  lite_mat_rgb2.Init(image.cols, image.rows, image.channels(), image.data, LDataType::UINT8);
  LiteMat lite_mat_resize2;
  int h2 = 1000;
  int w2 = 1000;
  bool ret2 = ResizePreserveARWithFiller(lite_mat_rgb2, lite_mat_resize2, h2, w2, nullptr, nullptr, 0);
  ASSERT_TRUE(ret2 == false);

  // The width and height of the output image is less than or equal to 0.
  LiteMat lite_mat_rgb3;
  lite_mat_rgb3.Init(image.cols, image.rows, image.channels(), image.data, LDataType::UINT8);
  LiteMat lite_mat_resize3;
  float ratioShiftWShiftH3[3] = {0};
  float invM3[2][3] = {{0, 0, 0}, {0, 0, 0}};
  int h3 = -1000;
  int w3 = 1000;
  bool ret3 = ResizePreserveARWithFiller(lite_mat_rgb3, lite_mat_resize3, h3, w3, &ratioShiftWShiftH3, &invM3, 0);
  ASSERT_TRUE(ret3 == false);
}
