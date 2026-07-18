# factorial_ab — A案(事前学習×拡張×データ量) / B案(luxベンチマーク) 実験コード

## 前提

- データ規約は既存パイプラインと同一:
  `<data_root>/<time_id>/dataset/img/<ep>_<view>.npy` (48x64x3 float32 [0,1]) と
  `<data_root>/<time_id>/dataset/vel/data.csv` (episode,center,left,right[,lux])
- 学習は center/left/right 3視点(各列がラベル)、評価は center のみ(既存の比較スクリプトと同じ流儀)
- `scratch` 以外は frozen 特徴+MLPヘッド(特徴は `feature_cache/` に一度だけ抽出されるので、
  2回目以降とヘッド再学習は高速。CPUでも回る)
- 事前学習の重みは初回のみダウンロードが必要

### 使えるエンコーダ(「事前学習の種類」の軸で選ぶ)

| encoder | 系統 | 特徴次元 | 補足 |
|---|---|---|---|
| `scratch` | 事前学習なし | - | `scripts/net.py` と同一CNNを画素から学習 |
| `resnet18` | 教師あり ImageNet / CNN | 512 | 軽量な基準線 |
| `vit_b_16` | 教師あり ImageNet / ViT | 768 | resnet18 と比べると「構造(CNN/ViT)」と「事前学習」を分離できる |
| `dinov2_vits14` | 自己教師 / ViT-S | 384 | 本命。`dinov2_vitb14`(768)もあり |
| `clip_vitb16` | 言語対比 CLIP / ViT-B | 512 | 要 `pip install open_clip_torch`(導入済み) |

推奨比較セット: `scratch,resnet18,dinov2_vits14`(3本柱)。余裕があれば `vit_b_16,clip_vitb16` を
足すと「教師あり/自己教師/言語対比 × CNN/ViT」の表が埋まる。

## A案: 一括実行 (推奨)

```bash
bash run_all_A.sh              # フル: 5エンコーダ×2拡張×4データ量×3シード=120セル
QUICK=1 bash run_all_A.sh      # 短縮: scratch/resnet18/dinov2 × シード1 = 24セル
DEVICE=cuda bash run_all_A.sh  # GPU明示(通常は自動検出)
```

- 中断→再実行で続きから走る。図表(plots/)まで自動生成
- データや条件は環境変数で差し替え可能(スクリプト冒頭を参照)

所要時間の目安(train 1万行、テスト3〜4セットの場合):

| 構成 | GPU (RTX 4060) | CPUのみ |
|---|---|---|
| QUICK=1 (24セル) | 2〜4時間 | 丸1日級 |
| フル (120セル) | 半日〜1日 (scratchの画素学習が大半) | 非現実的(数日) |

- 事前学習系は「特徴抽出(エンコーダごとに数分〜1時間)+ヘッド学習(セルごとに数十秒)」で、
  特徴はデータ量条件・シード(拡張なし時)をまたいでキャッシュ共有される
- テストセットの特徴もキャッシュされるため、評価は2モデル目以降ほぼ一瞬
- 遅いのは scratch の画素学習(100エポック×24セル)。急ぐ場合は `--epochs_scratch 50` 相当を
  run_factorial.py 直呼びで

## A案: 因子計画の実行(個別)

```bash
cd ~/challenge_ws/src/nav_cloning/scripts/factorial_ab

# 屋内 hour-matrix データでの例(train 1本、同時間帯テスト+ギャップテスト)
python3 run_factorial.py \
  --data_root /home/shin/challenge_ws/nav_cloning_data \
  --train_times 20260308_120428 \
  --test_times_same 20260308_130242 \
  --test_times_gap 20260308_100554 20260310_180327 \
  --rows_list 1000,2000,4000,all \
  --encoders scratch,dinov2_vits14 \
  --augs none,taw3op \
  --seeds 1,2,3 \
  --out_dir /home/shin/challenge_ws/nav_cloning_data/_factorial_A_$(date +%Y%m%d)
```

- セル数 = encoders × augs × rows × seeds(上の例で 2×2×4×3 = 48)
- 学習済みモデル・評価済み行は自動スキップ(中断→再実行でレジューム)
- 拡張は `taw3op` = Equalize/Brightness/AutoContrast の TrivialAugment 風を
  K=3 コピー追加(既存のオフライン拡張と同じ意味論、サンプル毎に決定的なので再現可)
- 結果は `results_long.csv` に追記される

1セルだけ手動で回す場合は `train_cell.py` を直接使う(`--help` 参照)。

## A案: 集計・プロット

```bash
python3 plot_factorial.py \
  --results <out_dir>/results_long.csv \
  --out_dir <out_dir>/plots
```

- `01_data_efficiency_{same,gap}.png` … データ効率曲線(誤差棒=シード間std)
- `02_aug_effect.png` … 「事前学習は拡張を吸収するか」の交互作用プロット
- `data_requirement.csv` … 目標MAE(既定: scratch+none の全量時MAE)到達に必要な行数と、
  scratch 比のデータ削減倍率

## B案: lux ベンチマーク

```bash
cat > models.csv <<EOF
model,model_path,train_time
scratch_clean,/home/shin/challenge_ws/nav_cloning_data/_baseline_hour_matrix_20260308_0310/model/....pt,20260308_120428
dinov2_none,<out_dir>/models/dinov2_vits14__none__rall__s1.pt,20260308_120428
EOF

python3 benchmark_lux.py \
  --data_root /home/shin/challenge_ws/nav_cloning_data \
  --models_csv models.csv \
  --test_times 20260308_100554 20260308_111130 20260308_120428 20260308_130242 \
               20260308_140206 20260308_150431 20260308_160405 20260310_170214 20260310_180327 \
  --out_dir /home/shin/challenge_ws/nav_cloning_data/_lux_benchmark_$(date +%Y%m%d)
```

- 既存パイプラインの `.pt`(`net.py` の state_dict)はそのまま読める。
  本ディレクトリで学習したモデルは隣の `.meta.json` から encoder を自動判別
- lux は data.csv の lux 列を使用、無い場合は center 画像平均輝度で代用
  (`lux_source` 列で区別。**混在させた比較は不可**)
- `degradation_fit.csv` の `slope_mae_per_100lux` が「照度ギャップ100lxあたりのMAE劣化」で、
  モデルの照明頑健性を1つの数値で比較できる(論文の主図は `01_mae_vs_lux_gap.png`)

## 注意

- GPUがあれば自動使用(`--device cuda` / `cpu` で強制可)。dinov2 の特徴抽出はCPUだと
  6000行×3視点×(1+K) で数十分かかるが、キャッシュされるので1回だけ
- 屋外データ(20260426/0428系)でも `--data_root` と時刻を変えるだけで同じように動く
