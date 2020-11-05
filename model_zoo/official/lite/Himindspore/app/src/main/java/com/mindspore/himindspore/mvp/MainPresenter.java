package com.mindspore.himindspore.mvp;

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
                Log.e(TAG, "onFailure" + t.toString());
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
