# kaggle-indoor

## Dataset Overview

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

## 参考
- https://www.kaggle.com/andradaolteanu/indoor-navigation-complete-data-understanding
