//
// Created by Big_CoCo on 2021/3/22.
//
float mnist_inputs_data[784] = {
    -1.5255959e+00, -7.5023180e-01, -6.5398091e-01, -1.6094848e+00,
    -1.0016718e-01, -6.0918891e-01, -9.7977227e-01, -1.6090963e+00,
    -7.1214461e-01, 3.0372199e-01, -7.7731431e-01, -2.5145525e-01,
    -2.2227049e-01, 1.6871134e+00, 2.2842517e-01, 4.6763551e-01,
    -6.9697243e-01, -1.1607615e+00, 6.9954240e-01, 1.9908163e-01,
    8.6569238e-01, 2.4440390e-01, -6.6291136e-01, 8.0730826e-01,
    1.1016806e+00, -1.7593604e-01, -2.2455578e+00, -1.4464580e+00,
    6.1155282e-02, -6.1774445e-01, -7.9806983e-01, -1.3162321e-01,
    1.8793458e+00, -7.2131783e-02, 1.5777060e-01, -7.7345490e-01,
    1.9905651e-01, 4.5702778e-02, 1.5295692e-01, -4.7567880e-01,
    -1.1101983e-01, 2.9273525e-01, -1.5784515e-01, -2.8787140e-02,
    2.3571110e+00, -1.0373387e+00, 1.5747981e+00, -6.2984723e-01,
    -9.2739171e-01, 5.4514152e-01, 6.6280261e-02, -4.3704012e-01,
    7.6260060e-01, 4.4151092e-01, 1.1651385e+00, 2.0153918e+00,
    1.3741246e-01, 9.3864471e-01, -1.8600109e-01, -6.4463931e-01,
    1.5392458e+00, -8.6958760e-01, -3.3311536e+00, -7.4787223e-01,
    -2.5502462e-02, -1.0233306e+00, -5.9618515e-01, -1.0055307e+00,
    -2.1060631e-01, -7.5475276e-03, 1.6734272e+00, 1.0342831e-02,
    -7.0395666e-01, -1.8526579e-01, -9.9623507e-01, -8.3125526e-01,
    -4.6102202e-01, -5.6008244e-01, 3.9557618e-01, -9.8227710e-01,
    -5.0648659e-01, 9.9775404e-02, -6.5397340e-01, 7.3169369e-01,
    -1.4343859e+00, -5.0081307e-01, 1.7163314e-01, -1.5999313e-01,
    2.5463349e-01, -5.0195730e-01, -1.0412000e+00, 7.3226720e-01,
    -1.0483401e+00, -4.7087720e-01, 2.9113635e-01, 1.9907043e+00,
    6.6144532e-01, 1.1899205e+00, 8.1653392e-01, -9.1352361e-01,
    1.3851457e+00, -8.1384623e-01, -9.2757654e-01, 1.1119633e+00,
    1.3352057e+00, 6.0427362e-01, -1.0344208e-01, -1.5121692e-01,
    -2.1020830e+00, -6.2002194e-01, -1.4782310e+00, -1.1334175e+00,
    8.7379628e-01, -5.6025940e-01, 1.2857845e+00, 8.1682384e-01,
    2.0530410e-01, 3.0510718e-01, 5.3568703e-01, -4.3118501e-01,
    2.5581384e+00, -2.3336388e-01, -1.3472130e-02, 1.8606348e+00,
    -1.9804063e+00, 1.7985829e+00, 1.0181159e-01, 3.4000599e-01,
    7.1236455e-01, -1.7765073e+00, 3.5386458e-01, 1.1996132e+00,
    -3.0299741e-01, -1.7618417e+00, 6.3484460e-01, -8.0435908e-01,
    -1.6111118e+00, -1.8716129e+00, 5.4308361e-01, 6.6067863e-01,
    2.2952116e+00, 6.7490596e-01, 1.7133216e+00, -1.7942734e+00,
    -1.3632672e+00, -9.8321962e-01, 1.5112667e+00, 6.4187074e-01,
    4.7296381e-01, -4.2859009e-01, 5.5137074e-01, -1.5473709e+00,
    5.1811212e-01, 1.0653535e-01, 2.6924077e-01, 1.3247679e+00,
    1.7460191e+00, 1.8549690e+00, -7.0636910e-01, 2.5570862e+00,
    4.1753429e-01, -2.1271861e-01, -8.3995801e-01, -4.2001787e-01,
    -6.2403631e-01, -9.7729611e-01, 8.7484282e-01, 9.8728138e-01,
    3.0957633e-01, 1.5206900e+00, 1.2052339e+00, -1.8155910e+00,
    -4.0346155e-01, -9.5914519e-01, -5.2077039e-03, -7.8863136e-02,
    8.4365427e-01, 1.1657013e+00, 5.2693218e-01, 1.6192533e+00,
    -9.6397626e-01, 1.4152038e-01, -1.6366096e-01, -3.5822257e-01,
    1.7222793e+00, -3.0357561e-01, 2.3887420e-01, 1.3440012e+00,
    1.0322569e-01, 1.1003542e+00, -3.4168020e-01, 9.4733888e-01,
    -5.6851596e-01, 8.3759618e-01, 1.7836607e+00, -1.9542466e-01,
    5.1491612e-01, -1.8474776e+00, -2.9167426e+00, -5.6732988e-01,
    -5.4128021e-01, 8.9517403e-01, -8.8250703e-01, 5.3181124e-01,
    -1.5457772e+00, -1.7329982e-01, 7.2824633e-01, 5.7061020e-02,
    9.0551722e-01, 1.0462948e+00, -5.2059698e-01, 1.3547838e+00,
    2.3519313e-01, 1.9142433e+00, 1.8364111e+00, 1.3245324e+00,
    -9.6900916e-01, 1.2516364e+00, 1.2103242e+00, -5.2792060e-01,
    2.1856615e-01, -5.7430726e-01, 1.4571251e+00, 1.7709557e+00,
    1.6499138e+00, -4.3200457e-01, -2.7102691e-01, -1.4391626e+00,
    1.2470404e+00, 1.2738512e+00, 3.9094925e-01, 3.8721049e-01,
    -7.9828717e-02, 3.4172431e-01, 9.4882733e-01, -1.3839359e+00,
    1.7240863e+00, -2.3647652e+00, -9.2949092e-01, 2.9362530e-01,
    2.1513203e-01, 9.3846369e-01, 1.4657077e+00, -5.5647439e-01,
    -7.4484080e-01, -2.0215721e-01, -2.2966790e-01, 1.3313366e-03,
    3.7527591e-01, -5.8106792e-01, -5.7230884e-01, 1.0097175e+00,
    -1.0564939e-01, -1.1796960e+00, -9.0779595e-02, 5.6311435e-01,
    -1.2560141e+00, 8.9555502e-01, 1.6747737e-01, 7.5142086e-01,
    2.4142299e+00, 1.0205840e+00, -4.4048381e-01, -1.7341677e+00,
    -1.2362250e+00, 1.5785813e+00, -1.1160507e+00, 7.6777023e-01,
    -5.8820677e-01, 2.1188903e+00, -5.4219025e-01, -2.4592547e+00,
    -1.1108288e+00, -1.1187209e+00, 7.5799555e-01, -4.9565765e-01,
    -1.9700006e-01, -3.3396218e-02, 7.1929151e-01, 1.0644146e+00,
    8.3402544e-01, -1.9162164e+00, -3.4202927e-01, -6.6049206e-01,
    3.1508535e-01, 1.1422518e+00, 3.0550566e-01, -5.7888174e-01,
    -2.3828252e-01, -1.3541743e+00, 2.6868939e-01, 1.1455697e-01,
    -1.5562972e+00, -1.0757437e+00, -8.7519461e-01, -4.7281876e-01,
    9.9123681e-01, -5.8622282e-02, 1.1787646e+00, 6.2218499e-01,
    7.8785008e-01, 1.3685523e+00, -8.5068983e-01, 5.1260746e-01,
    1.0476325e+00, -3.1758463e-01, 1.3948506e-01, 2.3402624e+00,
    -6.1160916e-01, 8.1602710e-01, 2.4772300e-01, -3.8672671e-01,
    1.9948451e-01, 7.9926956e-01, -2.6190341e-01, 1.5132962e-01,
    1.1981666e+00, -2.2832582e+00, -1.0129594e+00, -8.8789088e-01,
    6.5221924e-01, -8.7262028e-01, 3.5253752e-02, -3.3653030e-01,
    1.4023319e+00, 4.8412141e-01, -7.0304507e-01, -8.2676607e-01,
    7.7439600e-01, 6.9199395e-01, -1.0184799e+00, -8.0337167e-01,
    -7.0711321e-01, 7.5211829e-01, -1.9208279e-02, 1.1033330e+00,
    -6.0679215e-01, -5.2522349e-01, -5.6618774e-01, 6.6039857e-04,
    7.2245878e-01, 1.5263520e-01, 1.4495978e-01, -2.3442194e+00,
    3.6000299e-01, 4.6668175e-01, 1.2830665e+00, 1.2678007e+00,
    1.9883296e-01, 5.4408771e-01, -3.9781693e-01, -1.9291055e+00,
    2.3236869e-01, 8.6146563e-01, 6.2175733e-01, -1.7811896e+00,
    -7.8206092e-01, -1.4236701e+00, 1.6090765e+00, -3.2787595e-02,
    8.5323340e-01, 5.5063650e-02, -1.7425371e+00, 8.7500376e-01,
    -2.7188172e+00, -2.2192061e-01, 3.4208494e-01, 1.1093477e+00,
    -5.7314759e-01, 9.5778459e-01, 9.8202319e-04, -1.3847686e+00,
    -9.9650228e-01, 8.0734813e-01, 1.1738863e+00, -9.3984646e-01,
    1.3109189e+00, -3.1670693e-01, -1.8610410e-01, -5.7646018e-01,
    6.8665183e-01, 4.2086706e-01, -1.0213808e+00, 9.8856664e-01,
    -5.6187165e-01, -1.5792575e-01, 1.5042593e+00, -1.3950295e+00,
    8.0079097e-01, -6.6194439e-01, 1.2563107e+00, 4.9999446e-01,
    -2.7133808e-01, 1.8469073e+00, -3.1249959e-02, -9.3872704e-02,
    -6.1907429e-01, -6.3632655e-01, -4.2415860e-01, -2.0271668e+00,
    4.0962908e-01, -1.5421267e+00, -1.0128618e+00, -2.9737514e-02,
    -2.8895226e-01, 1.5219319e-01, -2.9803404e-01, -1.3135384e-01,
    -6.2809873e-01, 1.1968799e+00, 6.1099350e-01, -4.5477438e-01,
    -9.6037018e-01, 2.7690458e-01, -6.8010890e-01, -5.4578751e-01,
    -4.5518342e-01, 3.1859580e-01, -3.5494208e-01, 6.8589437e-01,
    -3.7613729e-01, -2.4106996e+00, -1.2778088e+00, -6.2887415e-02,
    -9.4712764e-02, -2.3144305e+00, 5.5653399e-01, 5.0569206e-01,
    -2.0759584e-01, 6.9363183e-01, 4.1949040e-01, 2.2523544e+00,
    9.3852311e-01, 1.4252927e+00, 1.5083258e+00, 1.0539497e-01,
    -1.6049961e+00, -1.0644839e-01, 2.4656655e-01, 6.1250836e-01,
    7.3980182e-01, -1.7860015e-01, 7.8490011e-02, -4.3981805e-01,
    -3.6079338e-01, -1.2617406e+00, 1.9146918e+00, -1.8612741e+00,
    -9.6749123e-03, 2.6038763e-01, 2.8203353e-01, 2.5829947e-01,
    -4.2654869e-01, 9.8075122e-01, 1.8588890e+00, -1.0920147e+00,
    7.6300204e-01, 2.2761525e-01, -1.4569789e+00, 1.7043737e+00,
    -3.2686386e+00, 4.7498712e-01, -2.1142473e+00, -1.5002301e+00,
    1.0692973e+00, 1.4393831e+00, 5.0645941e-01, 8.3597529e-01,
    1.1752968e+00, -3.4211743e-01, -3.8716367e-01, 5.4765379e-01,
    -1.5891987e-01, -7.3604894e-01, -2.3351878e-01, -5.4039150e-01,
    1.5708433e-01, -5.9762299e-01, -8.8390934e-01, 6.0767305e-01,
    -3.8843614e-01, -3.1578582e-02, -5.6058836e-01, -6.5552413e-01,
    7.2615027e-01, 6.7892069e-01, -4.3017429e-01, -3.8485083e-01,
    -1.5082921e+00, -7.1995616e-01, -1.1909670e+00, 1.3271062e+00,
    -2.1984124e+00, 2.8614265e-01, -2.0104712e-01, -2.5348804e+00,
    -1.5848289e+00, 2.1679449e-01, -1.4276333e-01, 1.4274154e+00,
    1.6425379e-01, -3.1606898e-01, 1.2852281e-01, -5.2765143e-01,
    1.0834497e+00, 7.2746372e-01, 5.7725620e-01, 5.3688127e-01,
    -4.3616110e-01, 2.7676934e-01, 2.9459488e-01, -5.6314898e-01,
    5.1899290e-01, 1.3394899e+00, -2.3876244e-01, -6.7961216e-02,
    -1.5035529e-01, 5.2330041e-01, -2.1156418e-01, -1.2541972e+00,
    1.8176029e-02, 1.4141930e+00, -1.7437581e+00, 1.1289321e-01,
    4.5267120e-01, 3.1554270e-01, -6.9010293e-01, -2.8289640e-01,
    3.5618150e-01, -6.5616649e-01, 6.7499673e-01, 1.2909728e+00,
    2.8768075e-01, 1.1313233e+00, -1.9227705e-03, -2.3545134e-01,
    -7.7834469e-01, 1.7674841e-02, 1.1869689e+00, -5.9568787e-01,
    -1.5738513e+00, 9.0094990e-01, 1.0499262e+00, 4.2925611e-01,
    3.4665063e-01, 1.1960464e+00, 5.0744399e-02, -2.4047236e+00,
    6.6365647e-01, -3.9687249e-01, 4.0486488e-01, 3.4154087e-01,
    -5.9558362e-01, 1.1019011e+00, 5.5386519e-01, -9.5087808e-01,
    -5.0393552e-01, 1.7358937e+00, 1.1365190e+00, 7.3528785e-01,
    -6.3713288e-01, -8.8953024e-01, 5.9735751e-01, -6.1928016e-01,
    1.2089928e+00, 8.0966818e-01, -3.7273017e-01, -5.3331411e-01,
    -4.9985203e-01, 3.9947726e-02, -7.8146380e-01, 3.1946027e-01,
    8.2106584e-01, 8.6431539e-01, 4.9166805e-01, 4.4538009e-01,
    -8.8726664e-01, 5.2979738e-01, 2.6839951e-01, 3.5011527e-01,
    -2.7225810e-01, 1.0665658e+00, -8.9532214e-01, 1.4147978e+00,
    -9.1728181e-01, 8.3720893e-01, 1.4950181e+00, -8.3034581e-01,
    -1.9900607e+00, -8.7786657e-01, 2.2035673e-01, -1.9547749e+00,
    8.5329479e-01, -1.4188342e+00, 9.8297036e-01, -5.3868419e-01,
    1.3784917e-01, 9.2474985e-01, 2.9384881e-01, 3.0301414e+00,
    -1.4259109e+00, 3.3642095e-01, -6.0710046e-02, -2.7827954e+00,
    1.3488874e+00, 2.6844734e-01, -1.1277022e+00, -5.9944046e-01,
    -2.7945054e-01, -2.1999671e-01, 1.1315615e+00, -5.5813056e-01,
    -8.4985018e-01, -5.9133893e-01, 9.1871524e-01, -1.7054160e+00,
    -6.2452555e-01, -1.5477768e+00, -4.3917063e-01, -8.2900178e-01,
    -4.2779538e-01, 1.2994735e+00, -1.0199753e+00, -8.5336286e-01,
    -1.8470149e+00, -5.6316632e-01, -2.9311785e-01, -1.5726203e+00,
    -1.0079967e+00, -1.1254747e+00, 2.0839548e+00, 2.8445369e-01,
    -2.0898786e-01, 2.7948596e+00, 9.4693983e-01, 1.1613066e+00,
    2.1592824e-02, 2.1849406e+00, 3.7046966e-01, 8.3229375e-01,
    1.0294781e+00, -4.6743554e-01, 1.2099822e+00, -9.2927051e-01,
    1.5964565e+00, -3.5177864e-02, 1.9276363e-01, 9.4458717e-01,
    4.0307879e-01, 7.8339100e-01, 1.6240975e+00, -1.9683785e+00,
    9.2987645e-01, 1.5981036e+00, 4.2616895e-01, 2.5072601e+00,
    4.4090030e-01, -2.0394561e+00, 1.0628663e+00, 7.7601296e-01,
    8.3457164e-02, 1.7073935e+00, -2.0758156e-01, -2.7201766e-01,
    -6.5246433e-01, 2.3190866e+00, -3.1556660e-01, 1.2293459e+00,
    1.9086858e-02, 1.6939967e+00, -9.7426087e-01, 1.0000985e-01,
    1.6331865e-01, 1.1104544e+00, 6.5858930e-01, -1.8446711e-01,
    -6.9782162e-01, 5.4673910e-01, -1.0919048e+00, -2.0058967e-01,
    -2.1976221e-01, -7.5056171e-01, 9.1047740e-01, 1.4996040e+00,
    -2.7725294e-01, 9.9202655e-02, -1.5756993e+00, 7.4856669e-01,
    -2.4229655e-01, -1.8000333e-01, 9.5837879e-01, 3.7814003e-01,
    1.9289158e-01, 2.4711327e-01, -3.1152922e-01, 4.4534847e-02,
    -7.7046400e-01, 4.5658717e-01, -1.3150460e+00, -5.0721991e-01,
    4.1748023e-01, 9.2643857e-01, 6.3569260e-01, -1.6128796e-01,
    1.0286627e+00, 4.7581047e-02, 4.1486391e-01, -2.7009306e+00,
    -1.5045499e+00, -1.8634710e-01, -9.3207240e-01, 3.0545831e-01,
    -5.1035285e-01, 8.7927073e-01, 1.7738712e+00, -1.3286506e-01,
    1.3458737e+00, -4.6432903e-01, -3.7430039e-01, 9.7058731e-01,
    -1.9518436e+00, -6.4998013e-01, 1.3482264e+00, 3.0995172e-01,
    -1.5216483e+00, 9.7610706e-01, 3.9083481e-01, 2.7913565e-02,
    -4.1744223e-01, 1.7064806e+00, -2.5080970e-01, -3.3612009e-02,
    5.8338016e-01, 1.6178854e+00, -1.3733586e+00, -8.5550433e-01,
    1.5778065e+00, 1.0752751e-01, 1.1045673e+00, 5.9758538e-01,
    7.1269102e-02, -5.0374931e-01, 8.0341589e-01, 1.1834451e+00,
    6.3811505e-01, -5.0269210e-01, -9.9724096e-01, -5.6425828e-01,
    -3.4610125e-01, 2.7074468e-01, -1.3578615e+00, -9.6113062e-01,
    1.1768451e+00, 1.1981529e-01, 6.6130060e-01, 1.7996032e+00,
    -1.4726470e+00, -1.4529139e+00, 2.5632006e-01, -7.5283742e-01,
    1.2143371e+00, 5.3680718e-01, -5.9180927e-01, 1.1358957e+00,
    1.4462845e+00, -1.1436753e+00, 7.8876835e-01, -6.7686230e-01,
    -9.3259799e-01, 7.4118137e-02, 2.1128911e-01, 2.6312185e-02,
    -2.2259822e-02, -1.5083861e+00, -2.7273307e+00, -8.5954350e-01,
    -4.6734902e-01, 1.5499024e+00, 4.5016751e-01, 1.2971551e+00,
    2.9964414e-01, -1.0238653e+00, 1.0269226e+00, -1.9246057e-01};