# CapsNet_classification 使用說明
程式碼如網址：
https://github.com/106601015/CapsNet_classification
資料科學導論期末專題，用CapsNet代替天氣學上人工判別台中定量降水圖的強度

------------------

## 1. 專案下載與環境建置

用法：下載或clone該專案，並且確定python環境具有以下套件：
* os
* sys
* numpy
* tensorflow
* tqdm
* sklearn.metrics
* scipy

作者環境：windows10 x64/anaconda虛擬環境
* numpy=1.19.4
* tensorflow=2.4.0(2.1.0亦可)
* sklearn=0.24.0
* scipy=1.5.2

------------------

## 2. .gitignore中省略資料的重新製作
.gitignore檔案中省略了龐大的資料/圖片上傳，只有quantitative_precipitation_deal資料夾中含有圖片集。
首先進入quantitative_precipitation_deal資料夾，執行quantitative_precipitation_datadeal.py內的make_data()以製作train/test等圖片的.npy檔案形式(另外make_origin_data()以及make_rotated_data()等可針對不同資料製做.npy檔案形式)

檢查data資料夾的quantitative_precipitation子資料夾是否出現相對應的.npy檔案

------------------

## 3. 執行main進行訓練

在終端機開啟具有1.提及套件的虛擬環境，並且執行
```shell=
python main.py train <dataset名稱> <epoch次數>
```
dataset名稱有：
* 'mnist',
* 'fashion-mnist',
* 'myself',
* 'quantitative_precipitation',
* 'quantitative_precipitation_origin'
我們採用quantitative_precipitation

epoch次數則為正整數的字串數字即可

若環境設置ok即可看到capsnet模型開始使用supervisor訓練，並且在訓練完成後會有fd_train_acc以及fd_loss的檔案輸出紀錄訓練的準確率以及損失率，並以logdir記錄模型的訓練參數。

------------------

## 4. 執行main進行測試

在終端機開啟具有1.提及套件的虛擬環境，並且執行
```shell=
python main.py test <dataset名稱> <epoch次數>
```
dataset名稱 以及 epoch次數 規定同3.
即可看到模型在test資料集中的
* accuracy_score
* recall_score
* precision_score
* f1_score
* roc_auc_score
並且記錄以上數據至fd_test_ARPFR，同時記錄label以及predict到fd_test_LandP