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
package com.mindspore.dance.view.mvp;

import com.mindspore.common.base.mvp.BaseFragmentPresenter;
import com.mindspore.common.net.FileDownLoadObserver;
import com.mindspore.common.net.RetrofitHelper;
import com.mindspore.dance.global.Constants;
import com.mindspore.dance.view.fragment.PrepareFragment;

import java.io.File;

import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;

public class PreparePresenter extends BaseFragmentPresenter<PrepareFragment> implements PrepareContract.Presenter {

    private PrepareContract.View view;

    private RetrofitHelper retrofitHelper;

    public PreparePresenter(PrepareContract.View view) {
        this.view = view;
        retrofitHelper = new RetrofitHelper();
    }

    @Override
    public void downloadDanceVideo(FileDownLoadObserver<File> fileDownLoadObserver) {
        retrofitHelper.downloadDanceVideo()
                .subscribeOn(Schedulers.io())
                .observeOn(Schedulers.io())
                .observeOn(Schedulers.computation())
                .map(responseBody -> fileDownLoadObserver.saveFile(responseBody, Constants.VIDEO_PATH, Constants.VIDEO_NAME))
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(fileDownLoadObserver);
    }
}
