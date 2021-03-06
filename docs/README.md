# docs

## データセット概要

### 建物(site)、階数(floor)について

- metadataの建物の数は全部で204。trainの建物も204でありすべての建物がtrainとして与えられている。
- またmetadataとtrainの階数についても同数であり、すべての階がtrainとして与えられている。
- 階数について
  - 表記ゆれがある。(おそらくF1 == 1F == L1, B1 == LG1 などと推測される)
    - B,G,L1,LMは site_id == "5cd56bc1e2acfd2d33b6404d"でのみ使用。この場合GがF1と考えられるため。B>G>LM>L1の順か？
      - LMはかなり他の階とは形が異なる。
    - BFは site_id == "5d27099303f801723c32364d"のみで、BF, 1F, 2F ... 7Fとなっているため、B1と考えてよいだろう。
    - BMは site_id == "5d2709dd03f801723c32cfb6"のみで、他はB1, L1, ... L4 なのでB1とL1の間かなと予測。
    - P1,P2は屋上を表現。MはF1とF2の間か。
      - https://aiwaok.jp/elevator-button-alphabet
    - LG1, LG2はB1, B2と推測。
    - L1, L2はF1,F2。
  - 順序不明な階数がある(LM,Mなど)
  - 抜けている階数もなる様子。
- testの建物数は24であり、階数は不明だが、trainに含まれている建物・階数のものであると考えられる。

### geometry.json
- 用途不明。地図情報がわかるのだろうか。
- いかに解説あり。
  - https://www.kaggle.com/rafaelcartenet/scaled-floors-geojsons-new-dataset

### 参考
- https://www.kaggle.com/andradaolteanu/indoor-navigation-complete-data-understanding

## 実装の戦略1
- 相対的な移動情報から絶対的な位置・階数を推定しなければならない。
  - 絶対的な位置・階数の推定には、以下の情報が使える。
    1. WiFiのID,RSS
    2. iBeaconのID,RSS
    3. 地図情報(地図上の通路だけがおそらく対象範囲と推測)
  - 階数により見えるID,強度に異なりがあるかどうか調べる必要がある。
    - ID毎に各階でどのように見えるか可視化をする。(★AI)
      - 階をまたがって、同じIDがほぼ見えない状況であれば、階数推定にはかなり有力。
  - 階数が正確にわかれば、絶対位置もかなり正確に分かる可能性がある。
  - ただし、絶対位置がわかるためには階数があれば分かる可能性がある。(卵と鶏の関係)
    1. 2つのモデルの相互学習が必要？
      1. 最初にwifi/iBeacon/その他センサ/正解waypointで階数推定モデルを学習
        - モデル学習できるほど、各建物にデータ数があるか要確認。(懸念A)
          - 建物毎のデータ数を確認する(★AI)
            - データが不足する場合は、もっと問題を抽象化する必要がある。
      2. 次に正解階数とwifi/iBeacon/その他センサで、waypoint推定モデルを学習
        - (懸念A)はこちらも同様にある。

- 機種ごとの個体差はあると思う。後々考慮が必要。

## 実装の戦略2
### 階数の推定

- ある建物について、各階の同一のwifiのbssidやibeaconのidを調査。
  - おおむね、階によってrssiの強いidは異なる傾向にありそう。
  - 上位のidを使えば階数推定はある程度できるのではないか。
  - 順序だけでははずれを引く可能性もある。rssiの値そのものも使う。
  - モデルは建物毎に必要だが致し方ない。

## 入力情報の統計量
- path_fileに含まれるセンサ値は、100～55000サンプル程度のサイズ。
- 1サンプル20msecのため、5秒～20分程度か。あり得る数字だが、幅が大きい。


## 実装の戦略3

- 建物はファイル名から自明である。
- 
- 階数推定に使えそうな情報は以下。
  - 各種センサーデータ
  - WiFiのRSSIDとRSSI
    - 建物で共通のモデルを使うために、ID自体は使わない方が良いと考えられる。
    - その場合、RSSI強度順に上位N個、下位N個を使うなどを考える。
    - サンプリングはセンサーデータより低い。補間するか別系統で入力するか要検討。
  - iBeaconのIDとRSSI
    - 同上
  - 絶対位置も使えそうだが、推論時にはないことが想定されるため使わない。
 
- そのうえで位置推定に使えそうなデータは以下
  - 推論した階数
    - ただしここを階数そのものとすると、建物で共通のモデルを使うのが難しくなる。
    - WiFi/iBeaconのRSSI強度順に上位N個、下位N個の絶対位置とRSSIの分布を使う。
  - 各種センサーデータ
  - タイムスタンプ
    - self-attentionを用い、transformerのencoderのようなモデルを使いたい。

## submission履歴

|name|CV|LB(Public)||explanation|
|:---|:---|:---|:---|
| 2021-04-14-001 | none   | 13.168 | wifi feats baseline |
| 2021-04-14-002 | 16.333 | 19.396 | cv |
| 2021-04-14-003 | 13.839 | 13.283 | lgbm parameter adjustment |
| 2021-04-14-004 |  5.137 | 12.836 | GroupFold -> KFold |
| 2021-04-14-004 |  3.450 |  8.333 | upper 1000 -> all |
| 2021-04-16-001 |  3.455 |  8.264 | wifi features recalc me |

## 参考ベースライン

### wifi features

- https://www.kaggle.com/devinanzelmo/wifi-features
  - wifiのbssid情報から、timestampをキーとして最も近いwaypoint,floorを正解とするデータセットを作成。
  - ユニークなのはtimestampであり、正解waypoint,floorはユニークではない。
  - test側はtimestampに最も近いwifi情報を使う。
  - これを建物毎に作成する。

- https://www.kaggle.com/devinanzelmo/wifi-features-lightgbm-starter/data
  - 上記の特徴を元にしたモデル。

- この手法の課題は時系列の情報を使えてない点。ピンポイントの時間のrssiのみを使っている。

### wifi features2

- 1000以上ではなくすべてのbssidを使う方法。
- https://www.kaggle.com/hiro5299834/wifi-features-with-lightgbm-kfold

- データは以下に公開されている。
  - https://www.kaggle.com/hiro5299834/indoor-navigation-and-location-wifi-features

### wifi features sort highest rssi

- 生成するコード
  - https://www.kaggle.com/kokitanisaka/create-unified-wifi-features-example
- データ
  - https://www.kaggle.com/kokitanisaka/indoorunifiedwifids
- これにより、一つのモデルで全建物を推定するモデルが作成できる。