/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.hms.scenedetection;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;

import com.alibaba.android.arouter.facade.annotation.Route;
import com.alibaba.android.arouter.launcher.ARouter;
import com.bumptech.glide.Glide;
import com.huawei.hmf.tasks.OnFailureListener;
import com.huawei.hmf.tasks.OnSuccessListener;
import com.huawei.hmf.tasks.Task;
import com.huawei.hms.mlsdk.common.MLException;
import com.huawei.hms.mlsdk.common.MLFrame;
import com.huawei.hms.mlsdk.scd.MLSceneDetection;
import com.huawei.hms.mlsdk.scd.MLSceneDetectionAnalyzer;
import com.huawei.hms.mlsdk.scd.MLSceneDetectionAnalyzerFactory;
import com.huawei.hms.mlsdk.scd.MLSceneDetectionAnalyzerSetting;
import com.mindspore.hms.BitmapUtils;
import com.mindspore.hms.R;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.NumberFormat;
import java.util.HashMap;
import java.util.List;

@Route(path = "/hms/SenceDetectionStillAnalyseActivity")
public class SenceDetectionStillAnalyseActivity extends AppCompatActivity {
    private static final String TAG = SenceDetectionStillAnalyseActivity.class.getSimpleName();

    private static final String[] SENCE_DATA_EN = new String[]{
            "stage", "beach", "bluesky", "sunset", "food", "flower", "greenplant", "snow", "night", "Text",
            "cat", "dog", "fireworks", "overcast", "fallen", "panda", "car", "oldbuildings", "bicycle", "waterfall",
            "playground", "corridor", "cabin", "washroom", "kitchen", "bedroom", "diningroom", "livingroom", "skyscraper", "bridge",
            "waterside", "mountain", "overlook", "construction", "islam", "european", "footballfield", "baseballfield", "tenniscourt", "carinside",
            "badmintoncourt", "pingpangcourt", "swimmingpool", "alpaca", "library", "supermarket", "restaurant", "tiger", "penguin", "elephant",
            "dinosaur", "watersurface", "indoorbasketballco", "bowlingalley", "classroom", "rabbit", "rhinoceros", "camel", "tortoise", "leopard",
            "giraffe", "peacock", "kangaroo", "lion", "motorcycle", "aircraft", "train", "ship", "glasses", "watch",
            "highwheels", "washingmachine", "airconditioner", "camera", "map", "keyboard", "redenvelope", "fucharacter", "xicharacter", "dragondance",
            "liondance", "go", "teddybear", "transformer", "thesmurfs", "littlepony", "butterfly", "ladybug", "dragonfly", "billiardroom",
            "meetingroom", "office", "bar", "mallcourtyard", "deer", "cathedralhall", "bee", "helicopter", "mahjong", "chess",
            "mcDonalds", "ornamentalfish", "widebuildings", "other"};

    private static  final  String[] SCENCE_DATA_ZH = new String[]{
            "舞台", "沙滩", "蓝天", "日出日落", "美食", "花朵", "绿植", "雪景", "夜景", "文字", "猫",
            "狗", "烟花", "阴天", "秋叶", "熊猫", "汽车", "中式建筑", "自行车", "瀑布", "游乐场", "走廊巷道",
            "火车飞机内", "卫生间", "厨房", "卧室", "餐厅", "客厅", "摩天大楼+塔", "桥梁", "海滨/湖滨", "山峰",
            "城市俯瞰", "工地", "伊斯兰式建", "欧式建筑", "足球场", "棒球场", "网球场", "轿车内部", "羽毛球馆", "乒乓球馆",
            "游泳馆", "羊驼", "图书馆", "超市", "饭店", "老虎", "企鹅", "大象", "恐龙", "水面",
            "室内篮球场", "保龄球馆", "教室", "兔子", "犀牛", "骆驼", "乌龟", "豹子", "长颈鹿", "孔雀",
            "袋鼠", "狮子", "电动车/摩", "飞机", "火车", "轮船", "眼镜", "手表", "高跟鞋", "洗衣机",
            "空调", "相机", "地图", "键盘", "红包", "福 字", "囍 字", "舞龙", "舞狮", "围棋",
            "泰迪熊", "京剧脸谱", "蓝精灵", "小马宝莉", "蝴蝶", "瓢虫", "蜻蜓", "桌球房", "会议室", "办公室",
            "酒吧", "商场中庭", "梅花鹿", "教堂正厅", "蜜蜂", "直升机", "麻将", "象棋", "麦当劳", "观赏鱼",
            "广角城楼", "其他"
    };
    private static final int RC_CHOOSE_PHOTO = 1;
    private static final int RC_CHOOSE_CAMERA = 2;

    private boolean isPreViewShow = false;

    private ImageView imgPreview;
    private TextView textOriginImage;
    private Uri imageUri;

    private Bitmap originBitmap;

    private TextView mTextView;
    private MLSceneDetectionAnalyzer analyzer;
    private Integer maxWidthOfImage;
    private Integer maxHeightOfImage;
    private boolean isLandScape;
    public static HashMap<String, String> mHashMap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sence_detection_still_analyse);
        init();
    }

    private void init() {
        Toolbar mToolbar = findViewById(R.id.segmentation_toolbar);
        mToolbar.setNavigationOnClickListener(view -> finish());

        mTextView = findViewById(R.id.text_text);
        imgPreview = findViewById(R.id.img_origin);
        textOriginImage = findViewById(R.id.tv_image);

        mTextView.setMovementMethod(ScrollingMovementMethod.getInstance());

        MLSceneDetectionAnalyzerSetting setting = new MLSceneDetectionAnalyzerSetting.Factory()

                .setConfidence(0.0f)
                .create();
        analyzer = MLSceneDetectionAnalyzerFactory.getInstance().getSceneDetectionAnalyzer(setting);

        mHashMap = new HashMap<>();
      if (SENCE_DATA_EN.length == SCENCE_DATA_ZH.length ){
          for (int i = 0; i < SENCE_DATA_EN.length; i++) {
              mHashMap.put(SENCE_DATA_EN[i],SCENCE_DATA_ZH[i]);
          }
      }
    }

    public void onClickPhoto(View view) {
        openGallay(RC_CHOOSE_PHOTO);
        textOriginImage.setVisibility(View.GONE);
    }

    public void onClickCamera(View view) {
        openCamera();
        textOriginImage.setVisibility(View.GONE);
    }


    public void onClickRealTime(View view) {
        ARouter.getInstance().build("/hms/SenceDetectionLiveAnalyseActivity").navigation();
    }

    private void openGallay(int request) {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, request);
    }

    private void openCamera() {
        Intent intentToTakePhoto = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        String mTempPhotoPath = Environment.getExternalStorageDirectory() + File.separator + "photo.jpeg";
        imageUri = FileProvider.getUriForFile(this, getApplicationContext().getPackageName() + ".fileprovider", new File(mTempPhotoPath));
        intentToTakePhoto.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        startActivityForResult(intentToTakePhoto, RC_CHOOSE_CAMERA);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (RC_CHOOSE_PHOTO == requestCode) {
                if (null != data && null != data.getData()) {
                    this.imageUri = data.getData();
                    showOriginImage();
                } else {
                    finish();
                }
            } else if (RC_CHOOSE_CAMERA == requestCode) {
                showOriginCamera();
            }
        } else {
            textOriginImage.setVisibility(!isPreViewShow ? View.VISIBLE : View.GONE);
        }
    }

    private void showOriginImage() {
        File file = BitmapUtils.getFileFromMediaUri(this, imageUri);
        Bitmap photoBmp = BitmapUtils.getBitmapFormUri(this, Uri.fromFile(file));
        int degree = BitmapUtils.getBitmapDegree(file.getAbsolutePath());
        originBitmap = BitmapUtils.rotateBitmapByDegree(photoBmp, degree).copy(Bitmap.Config.ARGB_8888, true);
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
            isPreViewShow = true;
            showTextRecognition();
        } else {
            isPreViewShow = false;
        }
    }

    private void showOriginCamera() {
        try {
            Pair<Integer, Integer> targetedSize = this.getTargetSize();
            int targetWidth = targetedSize.first;
            int maxHeight = targetedSize.second;
            Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
            originBitmap = BitmapUtils.zoomImage(bitmap, targetWidth, maxHeight);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        // Determine how much to scale down the image.
        Log.e(TAG, "resized image size width:" + originBitmap.getWidth() + ",height: " + originBitmap.getHeight());
        if (originBitmap != null) {
            Glide.with(this).load(originBitmap).into(imgPreview);
            isPreViewShow = true;
            showTextRecognition();
        } else {
            isPreViewShow = false;
        }
    }

    private Pair<Integer, Integer> getTargetSize() {
        Integer targetWidth;
        Integer targetHeight;
        Integer maxWidth = this.getMaxWidthOfImage();
        Integer maxHeight = this.getMaxHeightOfImage();
        targetWidth = this.isLandScape ? maxHeight : maxWidth;
        targetHeight = this.isLandScape ? maxWidth : maxHeight;
        Log.i(TAG, "height:" + targetHeight + ",width:" + targetWidth);
        return new Pair<>(targetWidth, targetHeight);
    }

    private Integer getMaxWidthOfImage() {
        if (this.maxWidthOfImage == null) {
            if (this.isLandScape) {
                this.maxWidthOfImage = ((View) this.imgPreview.getParent()).getHeight();
            } else {
                this.maxWidthOfImage = ((View) this.imgPreview.getParent()).getWidth();
            }
        }
        return this.maxWidthOfImage;
    }

    private Integer getMaxHeightOfImage() {
        if (this.maxHeightOfImage == null) {
            if (this.isLandScape) {
                this.maxHeightOfImage = ((View) this.imgPreview.getParent()).getWidth();
            } else {
                this.maxHeightOfImage = ((View) this.imgPreview.getParent()).getHeight();
            }
        }
        return this.maxHeightOfImage;
    }

    private void showTextRecognition() {
        MLFrame frame = MLFrame.fromBitmap(originBitmap);
        Task<List<MLSceneDetection>> task = analyzer.asyncAnalyseFrame(frame);
        task.addOnSuccessListener(new OnSuccessListener<List<MLSceneDetection>>() {
            public void onSuccess(List<MLSceneDetection> result) {

                if (result != null && !result.isEmpty()) {
                    SenceDetectionStillAnalyseActivity.this.displaySuccess(result);
                } else {
                    SenceDetectionStillAnalyseActivity.this.displayFailure();
                }
            }
        })
                .addOnFailureListener(new OnFailureListener() {
                    public void onFailure(Exception e) {

                        // failure.
                        if (e instanceof MLException) {
                            MLException mlException = (MLException) e;

                            int errorCode = mlException.getErrCode();

                            String errorMessage = mlException.getMessage();
                        } else {

                        }
                    }
                });
    }

    private void displaySuccess(List<MLSceneDetection> sceneInfos) {
        String str = getResources().getString(R.string.image_scene_count) + "：" + sceneInfos.size() + "\n";
        for (int i = 0; i < sceneInfos.size(); i++) {
            MLSceneDetection sceneInfo = sceneInfos.get(i);
            String result = sceneInfo.getResult().toLowerCase();
            if (mHashMap.containsKey(result)) {
                result = mHashMap.get(result);
            }
            NumberFormat fmt = NumberFormat.getPercentInstance();
            fmt.setMaximumFractionDigits(2);
            str += getResources().getString(R.string.image_scene) + "：" + result + "\n" + getResources().getString(R.string.image_score) + "：" + fmt.format(sceneInfo.getConfidence()) + "\n";
        }
        mTextView.setText(str);
    }

    private void displayFailure() {
        Toast.makeText(this.getApplicationContext(), "Fail", Toast.LENGTH_SHORT).show();
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (this.analyzer != null) {
            this.analyzer.stop();
        }

    }
}