/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
package com.mindspore.imageobject.objectdetection.bean;

import android.text.TextUtils;

import com.mindspore.common.utils.Utils;
import com.mindspore.imageobject.R;

import java.util.ArrayList;
import java.util.List;

public class RecognitionObjectBean {
    private String rectID;
    private String imgID;
    private String objectName;
    private float score;
    private float left;
    private float top;
    private float right;
    private float bottom;

    private RecognitionObjectBean(Builder builder) {
        this.rectID = builder.rectID;
        this.imgID = builder.imgID;
        this.objectName = builder.objectName;
        this.score = builder.score;
        this.left = builder.left;
        this.top = builder.top;
        this.right = builder.right;
        this.bottom = builder.bottom;
    }


    public static class Builder {
        private String rectID;
        private String imgID;
        private String objectName;
        private float score;
        private float left;
        private float top;
        private float right;
        private float bottom;

        public RecognitionObjectBean build() {
            return new RecognitionObjectBean(this);
        }

        public Builder setRectID(String rectID) {
            this.rectID = rectID;
            return this;
        }

        public Builder setImgID(String imgID) {
            this.imgID = imgID;
            return this;
        }

        public Builder setObjectName(String objectName) {
            this.objectName = objectName;
            return this;
        }

        public Builder setScore(float score) {
            this.score = score;
            return this;
        }

        public Builder setLeft(float left) {
            this.left = left;
            return this;
        }

        public Builder setTop(float top) {
            this.top = top;
            return this;
        }

        public Builder setRight(float right) {
            this.right = right;
            return this;
        }

        public Builder setBottom(float bottom) {
            this.bottom = bottom;
            return this;
        }
    }


    public String getImgID() {
        return imgID;
    }

    public String getRectID() {
        return rectID;
    }

    public String getObjectName() {
        int i = Integer.parseInt(objectName);
        String[] IMAGECONTENT = Utils.getApp().getResources().getStringArray(R.array.object_category);
        return IMAGECONTENT[i];
    }

    public float getScore() {
        return score;
    }

    public float getLeft() {
        return left;
    }

    public float getTop() {
        return top;
    }

    public float getRight() {
        return right;
    }

    public float getBottom() {
        return bottom;
    }


    public static List<RecognitionObjectBean> getRecognitionList(String result) {
        if (!TextUtils.isEmpty(result)) {
            String[] resultArray = result.split(";");
            List<RecognitionObjectBean> list = new ArrayList<>();
            for (int i = 0; i < resultArray.length; i++) {
                String singleRecognitionResult = resultArray[i];
                String[] singleResult = singleRecognitionResult.split("_");
                RecognitionObjectBean bean = new RecognitionObjectBean.Builder()
                        .setRectID(String.valueOf(i + 1))
                        .setImgID(null != getData(0, singleResult) ? getData(0, singleResult) : "")
                        .setObjectName(null != getData(1, singleResult) ? getData(1, singleResult) : "")
                        .setScore(null != getData(2, singleResult) ? Float.parseFloat(getData(2, singleResult)) : 0)
                        .setLeft(null != getData(3, singleResult) ? Float.parseFloat(getData(3, singleResult)) : 0)
                        .setTop(null != getData(4, singleResult) ? Float.parseFloat(getData(4, singleResult)) : 0)
                        .setRight(null != getData(5, singleResult) ? Float.parseFloat(getData(5, singleResult)) : 0)
                        .setBottom(null != getData(6, singleResult) ? Float.parseFloat(getData(6, singleResult)) : 0)
                        .build();
                list.add(bean);
            }
            return list;
        } else {
            return null;
        }
    }

    /**
     * @param index
     * @param singleResult
     * @return
     */
    private static String getData(int index, String[] singleResult) {
        if (index > singleResult.length) {
            return null;
        } else {
            if (!TextUtils.isEmpty(singleResult[index])) {
                return singleResult[index];
            }
        }
        return null;
    }
}
