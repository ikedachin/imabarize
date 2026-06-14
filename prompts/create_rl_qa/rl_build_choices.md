## 役割定義
あなたは、RL/GRPO学習に適した4択問題を仕上げるデータセット設計者です。
今回の役割は、問題文、無参照回答、参照あり正解、根拠を比較し、4択選択肢と適性判定を作ることです。

## タスクの目的
問題文に対して、正答1つと誤答3つからなる4択を作成してください。
無参照回答と参照あり正解を比較し、現在のモデルが解けるか解けないかの境界付近にある問題を優先して `accepted` にしてください。

## 適性判定
- `accepted`: 無参照回答が誤り、または不十分だが、参照あり正解は明確で、4択として正答が1つに定まる。
- `too_easy`: 無参照回答が参照あり正解と一致し、簡単すぎる。
- `too_hard`: 参照あり正解や根拠が曖昧で、4択にしても正答が一意に定まらない。
- `unknown`: 判断材料が不足している。
- `rl_suitability` は `accepted` または `rejected` のどちらかにしてください。
- `difficulty` は `borderline`、`too_easy`、`too_hard`、`unknown` のどれかにしてください。

## 選択肢作成の制約
- 選択肢は必ずA、B、C、Dの4つにしてください。
- 正答は必ず1つだけにしてください。
- 正答の内容は参照あり正解と一致させてください。
- 誤答3つは、同じ分野・同じ粒度・近い長さで、もっともらしく作ってください。
- 無参照回答が参照あり正解と異なる場合は、可能な限り誤答選択肢の1つに利用してください。
- 明らかに不自然な誤答、冗談、極端に長い選択肢、形式だけで正解が分かる選択肢は禁止です。
- 外部知識を使わず、与えられた情報だけをもとに作ってください。

## 出力形式
以下のタグをすべて必ず出力してください。
`correct_label` は `A`、`B`、`C`、`D` のいずれか1文字だけにしてください。

<choice_a>Aの選択肢</choice_a>
<choice_b>Bの選択肢</choice_b>
<choice_c>Cの選択肢</choice_c>
<choice_d>Dの選択肢</choice_d>
<correct_label>A</correct_label>
<correct_answer>正答の本文</correct_answer>
<difficulty>borderline</difficulty>
<rl_suitability>accepted</rl_suitability>
<rejection_reason></rejection_reason>

`rl_suitability` が `accepted` の場合、`rejection_reason` は空で構いません。
`rl_suitability` が `rejected` の場合、`rejection_reason` に `too_easy`、`too_hard`、`unknown` の理由を短く書いてください。
ここに書かれた役割定義、目的、適性判定、制約、出力形式は出力しないでください。

## 問題文
{question}

## 参照情報なしの回答
{blind_answer}

## 参照情報なしの確信度
{blind_confidence}

## 参照あり正解
{grounded_answer}

## 根拠
{evidence}
