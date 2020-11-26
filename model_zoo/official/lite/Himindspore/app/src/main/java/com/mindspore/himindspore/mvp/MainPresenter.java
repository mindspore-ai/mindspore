package com.mindspore.himindspore.mvp;
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

import android.util.Log;

import com.mindspore.himindspore.SplashActivity;
import com.mindspore.himindspore.base.BasePresenter;
import com.mindspore.himindspore.net.FileDownLoadObserver;
import com.mindspore.himindspore.net.RetrofitHelper;
import com.mindspore.himindspore.net.UpdateInfoBean;

import java.io.File;

import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.annotations.NonNull;
import io.reactivex.functions.Function;
import io.reactivex.schedulers.Schedulers;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainPresenter extends BasePresenter<SplashActivity> implements MainContract.Presenter {

    private static final String TAG = "MainPresenter";
    private RetrofitHelper retrofitHelper;

    public MainPresenter(SplashActivity mainActivity) {
        this.view = mainActivity;
        retrofitHelper = new RetrofitHelper();
    }

    @Override
    public void getUpdateInfo() {
        retrofitHelper.getUpdateInfo().enqueue(new Callback<UpdateInfoBean>() {
            @Override
            public void onResponse(Call<UpdateInfoBean> call, Response<UpdateInfoBean> response) {
                Log.i(TAG, "onResponse" + response.toString());
                view.showUpdateResult(response.body());
            }

            @Override
            public void onFailure(Call<UpdateInfoBean> call, Throwable t) {
                Log.e(TAG, "onFailure>>>" + t.toString());
                view.showFail(call.toString());
            }
        });
    }

    @Override
    public void downloadApk(final String destDir, final String fileName, final FileDownLoadObserver<File> fileDownLoadObserver) {
        retrofitHelper.downlaodApk()
                .subscribeOn(Schedulers.io())
                .observeOn(Schedulers.io())
                .observeOn(Schedulers.computation())
                .map(new Function<ResponseBody, File>() {
                    @Override
                    public File apply(@NonNull ResponseBody responseBody) throws Exception {
                        return fileDownLoadObserver.saveFile(responseBody, destDir, fileName);
                    }
                })
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe(fileDownLoadObserver);
    }

}
