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
package com.mindspore.himindspore.ui.college.bean;

public class CollegeItemBean {
    private int itemType;
    private int imageUncheck;
    private int imagechecked;
    private String title;
    private boolean isHasChecked;

    public static final int TYPE_LEFT_IMAGE_RIGHT_TEXT = 1;
    public static final int TYPE_MIX = 2;
    public static final int TYPE_PURE_TEXT = 3;

    public CollegeItemBean(int itemType, int imageUncheck,int imagechecked, String title,boolean isHasChecked) {
        this.itemType = itemType;
        this.imageUncheck = imageUncheck;
        this.imagechecked = imagechecked;
        this.title = title;
        this.isHasChecked = isHasChecked;
    }

    public int getItemType() {
        return itemType;
    }

    public int getImageUncheck() {
        return imageUncheck;
    }

    public int getImagechecked() {
        return imagechecked;
    }

    public String getTitle() {
        return title;
    }

    public boolean isHasChecked() {
        return isHasChecked;
    }

    public CollegeItemBean setHasChecked(boolean hasChecked) {
        isHasChecked = hasChecked;
        return this;
    }
}
