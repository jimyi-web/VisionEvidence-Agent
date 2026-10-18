# T04-5 人工复核表

全部为合成文字场景，无图片、真实 VLM 输出或人工完成标注。机器命中均相对冻结 qrels，不代表人工结论。

填写方式：逐例阅读原描述和全部证据；记录相关/不相关/不确定及理由。排名可能造成锚定偏差，建议先遮住排名独立判断。

保持原 qrels 不动；修订建议另存，批准后另建版本。未填写代表待复核，不代表认可。

## S01

原始描述：A red bus beside a bus stop. The sign reads Route 12.

场景规格（非像素观察）：A red bus beside a bus stop. The sign reads Route 12.

结构化视觉观察：
```json
{
  "objects": [
    "red bus",
    "bus stop"
  ],
  "visible_text": [
    "Route 12"
  ],
  "uncertainties": [
    "Synthetic fixture; no pixels analyzed"
  ]
}
```

冻结 qrels：D01, D02；状态：available

原标注理由：Route number and bus stop directly match the two scene records.

已验证事实：对象、红色、Route 12 保留；beside 关系未保留。两组 Top-5 均召回两篇；首位互换不改变指标。

待人工判断：确认 D01/D02 均相关及近重复计数是否符合既定粒度；核对 D12 的颜色/线路冲突。

ASCII 去重词项损失：a, beside, reads, sign, the（中文丢失需结合原字段判断）。

### Baseline

实际查询：`a red bus beside a bus stop the sign reads route 12`

错误：无；词项数/去重数：12/10；语料外词项：reads

查询来源：`[{"field": "original_description", "terms": ["a", "red", "bus", "beside", "a", "bus", "stop", "the", "sign", "reads", "route", "12"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D02 | 9.905307280702361 | 相关 | The Route 12 bus stop has a red bus at the sign. |
| 2 | D01 | 8.478647781249927 | 相关 | A red bus waits beside a bus stop marked Route 12. |
| 3 | D12 | 4.723668216806617 | 不相关 | A blue bus at a bus stop displays Route 9. |
| 4 | D03 | 2.5346823536019096 | 不相关 | A bicycle stands beside a river bridge with a Bridge closed sign. |
| 5 | D04 | 1.604130214468996 | 不相关 | A river bridge has a Bridge open sign and no bicycle. |

Top-5 命中：D01, D02；遗漏：无。

逐例指标：`{"recall_at_1": 0.5, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### Structured

实际查询：`red bus bus stop route 12`

错误：无；词项数/去重数：6/5；语料外词项：无

查询来源：`[{"field": "objects", "index": 0, "terms": ["red", "bus"]}, {"field": "objects", "index": 1, "terms": ["bus", "stop"]}, {"field": "visible_text", "index": 0, "terms": ["route", "12"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D01 | 7.101922588565768 | 相关 | A red bus waits beside a bus stop marked Route 12. |
| 2 | D02 | 6.811349680373786 | 相关 | The Route 12 bus stop has a red bus at the sign. |
| 3 | D12 | 4.2924268573785485 | 不相关 | A blue bus at a bus stop displays Route 9. |

Top-5 命中：D01, D02；遗漏：无。

逐例指标：`{"recall_at_1": 0.5, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### 人工填写（覆盖全部闭集，不只核对已检出文档）

| 证据 | 原文 | 冻结标签 | 人工：相关/不相关/不确定 | 理由 |
| --- | --- | --- | --- | --- |
| D01 | A red bus waits beside a bus stop marked Route 12. | 相关 | 待填写 | |
| D02 | The Route 12 bus stop has a red bus at the sign. | 相关 | 待填写 | |
| D03 | A bicycle stands beside a river bridge with a Bridge closed sign. | 不相关 | 待填写 | |
| D04 | A river bridge has a Bridge open sign and no bicycle. | 不相关 | 待填写 | |
| D05 | The library entrance displays Open Sunday. | 不相关 | 待填写 | |
| D06 | The library entrance displays Closed Sunday. | 不相关 | 待填写 | |
| D07 | A train platform marked Platform 7 has a suitcase. | 不相关 | 待填写 | |
| D08 | A train platform marked Platform 2 has a suitcase. | 不相关 | 待填写 | |
| D09 | A dog stands beside a fountain labelled Drinking water. | 不相关 | 待填写 | |
| D10 | A decorative fountain beside a garden has a No drinking notice. | 不相关 | 待填写 | |
| D11 | car | 不相关 | 待填写 | |
| D12 | A blue bus at a bus stop displays Route 9. | 不相关 | 待填写 | |

- [ ] 观察中的对象、文字、动作/关系、时间、地点已核对
- [ ] qrels 正例与全部负例已核对
- [ ] 失败解释已复核（同意/修订/不确定）：____
- 审核人 / 日期：____
- 建议相关 ID（不自动写回）：____
- 证据充分性与备注：____

## S02

原始描述：A bicycle beside a river bridge. A sign reads Bridge closed.

场景规格（非像素观察）：A bicycle beside a river bridge. A sign reads Bridge closed.

结构化视觉观察：
```json
{
  "objects": [
    "bicycle",
    "river bridge"
  ],
  "visible_text": [
    "Bridge closed"
  ],
  "uncertainties": [
    "Synthetic fixture; no pixels analyzed"
  ]
}
```

冻结 qrels：D03；状态：available

原标注理由：Closure sign distinguishes this scene from an open bridge.

已验证事实：bicycle、river bridge、closed 保留；beside 关系未保留。两组首位命中；D04 含 no bicycle/open，词袋不能理解否定。

待人工判断：确认 D03 相关；D04/D06 应否作为主题相关但场景不匹配的负例。

ASCII 去重词项损失：a, beside, reads, sign（中文丢失需结合原字段判断）。

### Baseline

实际查询：`a bicycle beside a river bridge a sign reads bridge closed`

错误：无；词项数/去重数：11/8；语料外词项：reads

查询来源：`[{"field": "original_description", "terms": ["a", "bicycle", "beside", "a", "river", "bridge", "a", "sign", "reads", "bridge", "closed"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D03 | 8.934294677073076 | 相关 | A bicycle stands beside a river bridge with a Bridge closed sign. |
| 2 | D04 | 6.7784536210815425 | 不相关 | A river bridge has a Bridge open sign and no bicycle. |
| 3 | D06 | 1.9332216212367104 | 不相关 | The library entrance displays Closed Sunday. |
| 4 | D09 | 1.503169251776109 | 不相关 | A dog stands beside a fountain labelled Drinking water. |
| 5 | D10 | 1.4538559037172192 | 不相关 | A decorative fountain beside a garden has a No drinking notice. |

Top-5 命中：D03；遗漏：无。

逐例指标：`{"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### Structured

实际查询：`bicycle river bridge bridge closed`

错误：无；词项数/去重数：5/4；语料外词项：无

查询来源：`[{"field": "objects", "index": 0, "terms": ["bicycle"]}, {"field": "objects", "index": 1, "terms": ["river", "bridge"]}, {"field": "visible_text", "index": 0, "terms": ["bridge", "closed"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D03 | 6.399612323471167 | 相关 | A bicycle stands beside a river bridge with a Bridge closed sign. |
| 2 | D04 | 5.174323406612547 | 不相关 | A river bridge has a Bridge open sign and no bicycle. |
| 3 | D06 | 1.9332216212367104 | 不相关 | The library entrance displays Closed Sunday. |

Top-5 命中：D03；遗漏：无。

逐例指标：`{"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### 人工填写（覆盖全部闭集，不只核对已检出文档）

| 证据 | 原文 | 冻结标签 | 人工：相关/不相关/不确定 | 理由 |
| --- | --- | --- | --- | --- |
| D01 | A red bus waits beside a bus stop marked Route 12. | 不相关 | 待填写 | |
| D02 | The Route 12 bus stop has a red bus at the sign. | 不相关 | 待填写 | |
| D03 | A bicycle stands beside a river bridge with a Bridge closed sign. | 相关 | 待填写 | |
| D04 | A river bridge has a Bridge open sign and no bicycle. | 不相关 | 待填写 | |
| D05 | The library entrance displays Open Sunday. | 不相关 | 待填写 | |
| D06 | The library entrance displays Closed Sunday. | 不相关 | 待填写 | |
| D07 | A train platform marked Platform 7 has a suitcase. | 不相关 | 待填写 | |
| D08 | A train platform marked Platform 2 has a suitcase. | 不相关 | 待填写 | |
| D09 | A dog stands beside a fountain labelled Drinking water. | 不相关 | 待填写 | |
| D10 | A decorative fountain beside a garden has a No drinking notice. | 不相关 | 待填写 | |
| D11 | car | 不相关 | 待填写 | |
| D12 | A blue bus at a bus stop displays Route 9. | 不相关 | 待填写 | |

- [ ] 观察中的对象、文字、动作/关系、时间、地点已核对
- [ ] qrels 正例与全部负例已核对
- [ ] 失败解释已复核（同意/修订/不确定）：____
- 审核人 / 日期：____
- 建议相关 ID（不自动写回）：____
- 证据充分性与备注：____

## S03

原始描述：A library entrance with a sign reading Open Sunday. The entrance is empty.

场景规格（非像素观察）：A library entrance with a sign reading Open Sunday. The entrance is empty.

结构化视觉观察：
```json
{
  "objects": [
    "library entrance"
  ],
  "visible_text": [
    "Open Sunday"
  ],
  "uncertainties": [
    "Synthetic fixture; no pixels analyzed"
  ]
}
```

冻结 qrels：D05；状态：available

原标注理由：Sunday opening is directly stated.

已验证事实：library entrance、Open Sunday 保留；empty 状态遗漏。两组首位命中；证据 D05 本来也未描述 empty。

待人工判断：确认允许证据只覆盖部分场景；核对 D05 与 Closed Sunday 的 D06。

ASCII 去重词项损失：a, empty, is, reading, sign, the, with（中文丢失需结合原字段判断）。

### Baseline

实际查询：`a library entrance with a sign reading open sunday the entrance is empty`

错误：无；词项数/去重数：13/11；语料外词项：empty, is, reading

查询来源：`[{"field": "original_description", "terms": ["a", "library", "entrance", "with", "a", "sign", "reading", "open", "sunday", "the", "entrance", "is", "empty"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D05 | 9.271559839679773 | 相关 | The library entrance displays Open Sunday. |
| 2 | D06 | 7.338338218443063 | 不相关 | The library entrance displays Closed Sunday. |
| 3 | D03 | 3.485361810839983 | 不相关 | A bicycle stands beside a river bridge with a Bridge closed sign. |
| 4 | D04 | 3.095939710772166 | 不相关 | A river bridge has a Bridge open sign and no bicycle. |
| 5 | D02 | 3.0939576003285763 | 不相关 | The Route 12 bus stop has a red bus at the sign. |

Top-5 命中：D05；遗漏：无。

逐例指标：`{"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### Structured

实际查询：`library entrance open sunday`

错误：无；词项数/去重数：4/4；语料外词项：无

查询来源：`[{"field": "objects", "index": 0, "terms": ["library", "entrance"]}, {"field": "visible_text", "index": 0, "terms": ["open", "sunday"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D05 | 7.7328864849468415 | 相关 | The library entrance displays Open Sunday. |
| 2 | D06 | 5.799664863710131 | 不相关 | The library entrance displays Closed Sunday. |
| 3 | D04 | 1.4918094963031698 | 不相关 | A river bridge has a Bridge open sign and no bicycle. |

Top-5 命中：D05；遗漏：无。

逐例指标：`{"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### 人工填写（覆盖全部闭集，不只核对已检出文档）

| 证据 | 原文 | 冻结标签 | 人工：相关/不相关/不确定 | 理由 |
| --- | --- | --- | --- | --- |
| D01 | A red bus waits beside a bus stop marked Route 12. | 不相关 | 待填写 | |
| D02 | The Route 12 bus stop has a red bus at the sign. | 不相关 | 待填写 | |
| D03 | A bicycle stands beside a river bridge with a Bridge closed sign. | 不相关 | 待填写 | |
| D04 | A river bridge has a Bridge open sign and no bicycle. | 不相关 | 待填写 | |
| D05 | The library entrance displays Open Sunday. | 相关 | 待填写 | |
| D06 | The library entrance displays Closed Sunday. | 不相关 | 待填写 | |
| D07 | A train platform marked Platform 7 has a suitcase. | 不相关 | 待填写 | |
| D08 | A train platform marked Platform 2 has a suitcase. | 不相关 | 待填写 | |
| D09 | A dog stands beside a fountain labelled Drinking water. | 不相关 | 待填写 | |
| D10 | A decorative fountain beside a garden has a No drinking notice. | 不相关 | 待填写 | |
| D11 | car | 不相关 | 待填写 | |
| D12 | A blue bus at a bus stop displays Route 9. | 不相关 | 待填写 | |

- [ ] 观察中的对象、文字、动作/关系、时间、地点已核对
- [ ] qrels 正例与全部负例已核对
- [ ] 失败解释已复核（同意/修订/不确定）：____
- 审核人 / 日期：____
- 建议相关 ID（不自动写回）：____
- 证据充分性与备注：____

## S04

原始描述：A train platform with a sign reading Platform 7. A suitcase is visible.

场景规格（非像素观察）：A train platform with a sign reading Platform 7. A suitcase is visible.

结构化视觉观察：
```json
{
  "objects": [
    "train platform",
    "suitcase"
  ],
  "visible_text": [],
  "uncertainties": [
    "Synthetic fixture; no pixels analyzed"
  ]
}
```

冻结 qrels：D07；状态：available

原标注理由：Structured fixture omits the visible platform number; preserve this omission.

已验证事实：Platform 7 可见文字丢失；train platform/suitcase 保留。D07/D08 同分，按 ID 排序命中，不能证明识别站台号。

待人工判断：重点：D07 相关、D08 不相关的编号判据；现有 rationale 描述提取失败而非充分解释相关性。

ASCII 去重词项损失：7, a, is, reading, sign, visible, with（中文丢失需结合原字段判断）。

### Baseline

实际查询：`a train platform with a sign reading platform 7 a suitcase is visible`

错误：无；词项数/去重数：13/10；语料外词项：is, reading, visible

查询来源：`[{"field": "original_description", "terms": ["a", "train", "platform", "with", "a", "sign", "reading", "platform", "7", "a", "suitcase", "is", "visible"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D07 | 8.228861347949794 | 相关 | A train platform marked Platform 7 has a suitcase. |
| 2 | D08 | 6.078421006574169 | 不相关 | A train platform marked Platform 2 has a suitcase. |
| 3 | D03 | 3.485361810839983 | 不相关 | A bicycle stands beside a river bridge with a Bridge closed sign. |
| 4 | D04 | 1.604130214468996 | 不相关 | A river bridge has a Bridge open sign and no bicycle. |
| 5 | D02 | 1.406917124277149 | 不相关 | The Route 12 bus stop has a red bus at the sign. |

Top-5 命中：D07；遗漏：无。

逐例指标：`{"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### Structured

实际查询：`train platform suitcase`

错误：无；词项数/去重数：3/3；语料外词项：无

查询来源：`[{"field": "objects", "index": 0, "terms": ["train", "platform"]}, {"field": "objects", "index": 1, "terms": ["suitcase"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D07 | 5.631680789635874 | 相关 | A train platform marked Platform 7 has a suitcase. |
| 2 | D08 | 5.631680789635874 | 不相关 | A train platform marked Platform 2 has a suitcase. |

Top-5 命中：D07；遗漏：无。

逐例指标：`{"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### 人工填写（覆盖全部闭集，不只核对已检出文档）

| 证据 | 原文 | 冻结标签 | 人工：相关/不相关/不确定 | 理由 |
| --- | --- | --- | --- | --- |
| D01 | A red bus waits beside a bus stop marked Route 12. | 不相关 | 待填写 | |
| D02 | The Route 12 bus stop has a red bus at the sign. | 不相关 | 待填写 | |
| D03 | A bicycle stands beside a river bridge with a Bridge closed sign. | 不相关 | 待填写 | |
| D04 | A river bridge has a Bridge open sign and no bicycle. | 不相关 | 待填写 | |
| D05 | The library entrance displays Open Sunday. | 不相关 | 待填写 | |
| D06 | The library entrance displays Closed Sunday. | 不相关 | 待填写 | |
| D07 | A train platform marked Platform 7 has a suitcase. | 相关 | 待填写 | |
| D08 | A train platform marked Platform 2 has a suitcase. | 不相关 | 待填写 | |
| D09 | A dog stands beside a fountain labelled Drinking water. | 不相关 | 待填写 | |
| D10 | A decorative fountain beside a garden has a No drinking notice. | 不相关 | 待填写 | |
| D11 | car | 不相关 | 待填写 | |
| D12 | A blue bus at a bus stop displays Route 9. | 不相关 | 待填写 | |

- [ ] 观察中的对象、文字、动作/关系、时间、地点已核对
- [ ] qrels 正例与全部负例已核对
- [ ] 失败解释已复核（同意/修订/不确定）：____
- 审核人 / 日期：____
- 建议相关 ID（不自动写回）：____
- 证据充分性与备注：____

## S05

原始描述：A dog beside a fountain. A notice reads Drinking water.

场景规格（非像素观察）：A dog beside a fountain. A notice reads Drinking water.

结构化视觉观察：
```json
{
  "objects": [
    "dog",
    "fountain"
  ],
  "visible_text": [
    "Drinking water"
  ],
  "uncertainties": [
    "Synthetic fixture; no pixels analyzed"
  ]
}
```

冻结 qrels：D09；状态：available

原标注理由：The drinking water fountain record is relevant, not the decorative fountain.

已验证事实：dog、fountain、Drinking water 保留；beside 关系遗漏。D10 含 No drinking，仍有正词重合。

待人工判断：核对 D09 与 D10 的饮用/禁止饮用语义；相关不等于已验证安全。

ASCII 去重词项损失：a, beside, notice, reads（中文丢失需结合原字段判断）。

### Baseline

实际查询：`a dog beside a fountain a notice reads drinking water`

错误：无；词项数/去重数：10/8；语料外词项：reads

查询来源：`[{"field": "original_description", "terms": ["a", "dog", "beside", "a", "fountain", "a", "notice", "reads", "drinking", "water"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D09 | 9.087558039466396 | 相关 | A dog stands beside a fountain labelled Drinking water. |
| 2 | D10 | 6.391511384110544 | 不相关 | A decorative fountain beside a garden has a No drinking notice. |
| 3 | D03 | 1.3991874597937413 | 不相关 | A bicycle stands beside a river bridge with a Bridge closed sign. |
| 4 | D01 | 1.3767251926841593 | 不相关 | A red bus waits beside a bus stop marked Route 12. |
| 5 | D07 | 0.44674021693829535 | 不相关 | A train platform marked Platform 7 has a suitcase. |

Top-5 命中：D09；遗漏：无。

逐例指标：`{"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### Structured

实际查询：`dog fountain drinking water`

错误：无；词项数/去重数：4/4；语料外词项：无

查询来源：`[{"field": "objects", "index": 0, "terms": ["dog"]}, {"field": "objects", "index": 1, "terms": ["fountain"]}, {"field": "visible_text", "index": 0, "terms": ["drinking", "water"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D09 | 7.584388787690287 | 相关 | A dog stands beside a fountain labelled Drinking water. |
| 2 | D10 | 2.9836189926063397 | 不相关 | A decorative fountain beside a garden has a No drinking notice. |

Top-5 命中：D09；遗漏：无。

逐例指标：`{"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### 人工填写（覆盖全部闭集，不只核对已检出文档）

| 证据 | 原文 | 冻结标签 | 人工：相关/不相关/不确定 | 理由 |
| --- | --- | --- | --- | --- |
| D01 | A red bus waits beside a bus stop marked Route 12. | 不相关 | 待填写 | |
| D02 | The Route 12 bus stop has a red bus at the sign. | 不相关 | 待填写 | |
| D03 | A bicycle stands beside a river bridge with a Bridge closed sign. | 不相关 | 待填写 | |
| D04 | A river bridge has a Bridge open sign and no bicycle. | 不相关 | 待填写 | |
| D05 | The library entrance displays Open Sunday. | 不相关 | 待填写 | |
| D06 | The library entrance displays Closed Sunday. | 不相关 | 待填写 | |
| D07 | A train platform marked Platform 7 has a suitcase. | 不相关 | 待填写 | |
| D08 | A train platform marked Platform 2 has a suitcase. | 不相关 | 待填写 | |
| D09 | A dog stands beside a fountain labelled Drinking water. | 相关 | 待填写 | |
| D10 | A decorative fountain beside a garden has a No drinking notice. | 不相关 | 待填写 | |
| D11 | car | 不相关 | 待填写 | |
| D12 | A blue bus at a bus stop displays Route 9. | 不相关 | 待填写 | |

- [ ] 观察中的对象、文字、动作/关系、时间、地点已核对
- [ ] qrels 正例与全部负例已核对
- [ ] 失败解释已复核（同意/修订/不确定）：____
- 审核人 / 日期：____
- 建议相关 ID（不自动写回）：____
- 证据充分性与备注：____

## S06

原始描述：automobile

场景规格（非像素观察）：automobile

结构化视觉观察：
```json
{
  "objects": [
    "automobile"
  ],
  "visible_text": [],
  "uncertainties": [
    "Synthetic fixture; no pixels analyzed"
  ]
}
```

冻结 qrels：D11；状态：available

原标注理由：Synthetic scene depicts a car; synonym mismatch is intentional and frozen.

已验证事实：两组查询均为 automobile；D11 只有 car，无词面交集。有效查询无候选；未发生结构化信息损失。

待人工判断：重点：单词 automobile 与 car 是否足以定义同一合成场景的相关性。

ASCII 去重词项损失：无（中文丢失需结合原字段判断）。

### Baseline

实际查询：`automobile`

错误：无；词项数/去重数：1/1；语料外词项：automobile

查询来源：`[{"field": "original_description", "terms": ["automobile"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| — | — | — | — | 无候选，不补齐零分文档 |

Top-5 命中：无；遗漏：D11。

逐例指标：`{"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr_at_5": 0.0}`

### Structured

实际查询：`automobile`

错误：无；词项数/去重数：1/1；语料外词项：automobile

查询来源：`[{"field": "objects", "index": 0, "terms": ["automobile"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| — | — | — | — | 无候选，不补齐零分文档 |

Top-5 命中：无；遗漏：D11。

逐例指标：`{"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr_at_5": 0.0}`

### 人工填写（覆盖全部闭集，不只核对已检出文档）

| 证据 | 原文 | 冻结标签 | 人工：相关/不相关/不确定 | 理由 |
| --- | --- | --- | --- | --- |
| D01 | A red bus waits beside a bus stop marked Route 12. | 不相关 | 待填写 | |
| D02 | The Route 12 bus stop has a red bus at the sign. | 不相关 | 待填写 | |
| D03 | A bicycle stands beside a river bridge with a Bridge closed sign. | 不相关 | 待填写 | |
| D04 | A river bridge has a Bridge open sign and no bicycle. | 不相关 | 待填写 | |
| D05 | The library entrance displays Open Sunday. | 不相关 | 待填写 | |
| D06 | The library entrance displays Closed Sunday. | 不相关 | 待填写 | |
| D07 | A train platform marked Platform 7 has a suitcase. | 不相关 | 待填写 | |
| D08 | A train platform marked Platform 2 has a suitcase. | 不相关 | 待填写 | |
| D09 | A dog stands beside a fountain labelled Drinking water. | 不相关 | 待填写 | |
| D10 | A decorative fountain beside a garden has a No drinking notice. | 不相关 | 待填写 | |
| D11 | car | 相关 | 待填写 | |
| D12 | A blue bus at a bus stop displays Route 9. | 不相关 | 待填写 | |

- [ ] 观察中的对象、文字、动作/关系、时间、地点已核对
- [ ] qrels 正例与全部负例已核对
- [ ] 失败解释已复核（同意/修订/不确定）：____
- 审核人 / 日期：____
- 建议相关 ID（不自动写回）：____
- 证据充分性与备注：____

## S07

原始描述：图书馆入口，周日开放。

场景规格（非像素观察）：图书馆入口，周日开放。

结构化视觉观察：
```json
{
  "objects": [
    "图书馆入口"
  ],
  "visible_text": [
    "周日开放"
  ],
  "uncertainties": [
    "Synthetic fixture; no pixels analyzed"
  ]
}
```

冻结 qrels：D05；状态：available

原标注理由：Same library scene expressed in Chinese; ASCII tokenizer cannot construct either query.

已验证事实：中文字段保留，但两组均无 ASCII 词项，查询构造失败；不是提取遗漏，也不是成功检索后漏召回。

待人工判断：重点：中文含义与 D05 是否对应，D06 是否不相关；不能当作真实 OCR 失败。

ASCII 去重词项损失：无（中文丢失需结合原字段判断）。

### Baseline

实际查询：null（构造失败）

错误：description has no supported query terms；词项数/去重数：0/0；语料外词项：无

查询来源：`[]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| — | — | — | — | 无候选，不补齐零分文档 |

Top-5 命中：无；遗漏：D05。

逐例指标：`{"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr_at_5": 0.0}`

### Structured

实际查询：null（构造失败）

错误：observation has no supported English letter or digit query terms；词项数/去重数：0/0；语料外词项：无

查询来源：`[]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| — | — | — | — | 无候选，不补齐零分文档 |

Top-5 命中：无；遗漏：D05。

逐例指标：`{"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr_at_5": 0.0}`

### 人工填写（覆盖全部闭集，不只核对已检出文档）

| 证据 | 原文 | 冻结标签 | 人工：相关/不相关/不确定 | 理由 |
| --- | --- | --- | --- | --- |
| D01 | A red bus waits beside a bus stop marked Route 12. | 不相关 | 待填写 | |
| D02 | The Route 12 bus stop has a red bus at the sign. | 不相关 | 待填写 | |
| D03 | A bicycle stands beside a river bridge with a Bridge closed sign. | 不相关 | 待填写 | |
| D04 | A river bridge has a Bridge open sign and no bicycle. | 不相关 | 待填写 | |
| D05 | The library entrance displays Open Sunday. | 相关 | 待填写 | |
| D06 | The library entrance displays Closed Sunday. | 不相关 | 待填写 | |
| D07 | A train platform marked Platform 7 has a suitcase. | 不相关 | 待填写 | |
| D08 | A train platform marked Platform 2 has a suitcase. | 不相关 | 待填写 | |
| D09 | A dog stands beside a fountain labelled Drinking water. | 不相关 | 待填写 | |
| D10 | A decorative fountain beside a garden has a No drinking notice. | 不相关 | 待填写 | |
| D11 | car | 不相关 | 待填写 | |
| D12 | A blue bus at a bus stop displays Route 9. | 不相关 | 待填写 | |

- [ ] 观察中的对象、文字、动作/关系、时间、地点已核对
- [ ] qrels 正例与全部负例已核对
- [ ] 失败解释已复核（同意/修订/不确定）：____
- 审核人 / 日期：____
- 建议相关 ID（不自动写回）：____
- 证据充分性与备注：____

## S08

原始描述：A bus stop with a Route 12 sign.

场景规格（非像素观察）：A bus stop with a Route 12 sign.

结构化视觉观察：
```json
{
  "objects": [],
  "visible_text": [],
  "uncertainties": [
    "Synthetic fixture; no pixels analyzed"
  ]
}
```

冻结 qrels：D01, D02；状态：available

原标注理由：Empty structured extraction is retained as a query construction failure.

已验证事实：objects/visible_text 全空，bus stop、Route 12 全部遗漏；Structured 查询失败，Baseline 召回两篇。

待人工判断：重点：描述未提及红色巴士，D01/D02 添加该属性是否仍相关；rationale 只解释查询失败。

ASCII 去重词项损失：12, a, bus, route, sign, stop, with（中文丢失需结合原字段判断）。

### Baseline

实际查询：`a bus stop with a route 12 sign`

错误：无；词项数/去重数：8/7；语料外词项：无

查询来源：`[{"field": "original_description", "terms": ["a", "bus", "stop", "with", "a", "route", "12", "sign"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D02 | 6.791607096297923 | 相关 | The Route 12 bus stop has a red bus at the sign. |
| 2 | D01 | 6.026894946314004 | 相关 | A red bus waits beside a bus stop marked Route 12. |
| 3 | D12 | 4.723668216806617 | 不相关 | A blue bus at a bus stop displays Route 9. |
| 4 | D03 | 3.485361810839983 | 不相关 | A bicycle stands beside a river bridge with a Bridge closed sign. |
| 5 | D04 | 1.604130214468996 | 不相关 | A river bridge has a Bridge open sign and no bicycle. |

Top-5 命中：D01, D02；遗漏：无。

逐例指标：`{"recall_at_1": 0.5, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### Structured

实际查询：null（构造失败）

错误：observation has no supported English letter or digit query terms；词项数/去重数：0/0；语料外词项：无

查询来源：`[]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| — | — | — | — | 无候选，不补齐零分文档 |

Top-5 命中：无；遗漏：D01, D02。

逐例指标：`{"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_5": 0.0, "mrr_at_5": 0.0}`

### 人工填写（覆盖全部闭集，不只核对已检出文档）

| 证据 | 原文 | 冻结标签 | 人工：相关/不相关/不确定 | 理由 |
| --- | --- | --- | --- | --- |
| D01 | A red bus waits beside a bus stop marked Route 12. | 相关 | 待填写 | |
| D02 | The Route 12 bus stop has a red bus at the sign. | 相关 | 待填写 | |
| D03 | A bicycle stands beside a river bridge with a Bridge closed sign. | 不相关 | 待填写 | |
| D04 | A river bridge has a Bridge open sign and no bicycle. | 不相关 | 待填写 | |
| D05 | The library entrance displays Open Sunday. | 不相关 | 待填写 | |
| D06 | The library entrance displays Closed Sunday. | 不相关 | 待填写 | |
| D07 | A train platform marked Platform 7 has a suitcase. | 不相关 | 待填写 | |
| D08 | A train platform marked Platform 2 has a suitcase. | 不相关 | 待填写 | |
| D09 | A dog stands beside a fountain labelled Drinking water. | 不相关 | 待填写 | |
| D10 | A decorative fountain beside a garden has a No drinking notice. | 不相关 | 待填写 | |
| D11 | car | 不相关 | 待填写 | |
| D12 | A blue bus at a bus stop displays Route 9. | 不相关 | 待填写 | |

- [ ] 观察中的对象、文字、动作/关系、时间、地点已核对
- [ ] qrels 正例与全部负例已核对
- [ ] 失败解释已复核（同意/修订/不确定）：____
- 审核人 / 日期：____
- 建议相关 ID（不自动写回）：____
- 证据充分性与备注：____

## S09

原始描述：A telescope under a dome. A label reads Observatory.

场景规格（非像素观察）：A telescope under a dome. A label reads Observatory.

结构化视觉观察：
```json
{
  "objects": [
    "telescope",
    "dome"
  ],
  "visible_text": [
    "Observatory"
  ],
  "uncertainties": [
    "Synthetic fixture; no pixels analyzed"
  ]
}
```

冻结 qrels：空；状态：corpus_insufficient

原标注理由：No astronomy evidence exists in this closed corpus; cannot evaluate retrieval recall.

已验证事实：telescope/dome/Observatory 保留；under 关系遗漏。Baseline 与语料的唯一公共词为 a；Structured 无候选。空 qrels 排除。

待人工判断：重点：遍查全部 12 篇确认 corpus_insufficient，而不是尚未标注；空候选不证明现实中无证据。

ASCII 去重词项损失：a, label, reads, under（中文丢失需结合原字段判断）。

### Baseline

实际查询：`a telescope under a dome a label reads observatory`

错误：无；词项数/去重数：9/7；语料外词项：dome, label, observatory, reads, telescope, under

查询来源：`[{"field": "original_description", "terms": ["a", "telescope", "under", "a", "dome", "a", "label", "reads", "observatory"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D10 | 0.4939125650844657 | 不相关 | A decorative fountain beside a garden has a No drinking notice. |
| 2 | D03 | 0.4811664343726085 | 不相关 | A bicycle stands beside a river bridge with a Bridge closed sign. |
| 3 | D07 | 0.44674021693829535 | 不相关 | A train platform marked Platform 7 has a suitcase. |
| 4 | D08 | 0.44674021693829535 | 不相关 | A train platform marked Platform 2 has a suitcase. |
| 5 | D09 | 0.44674021693829535 | 不相关 | A dog stands beside a fountain labelled Drinking water. |

空 qrels：命中/遗漏及检索指标不适用；不计零。

逐例指标：`null`

### Structured

实际查询：`telescope dome observatory`

错误：无；词项数/去重数：3/3；语料外词项：dome, observatory, telescope

查询来源：`[{"field": "objects", "index": 0, "terms": ["telescope"]}, {"field": "objects", "index": 1, "terms": ["dome"]}, {"field": "visible_text", "index": 0, "terms": ["observatory"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| — | — | — | — | 无候选，不补齐零分文档 |

空 qrels：命中/遗漏及检索指标不适用；不计零。

逐例指标：`null`

### 人工填写（覆盖全部闭集，不只核对已检出文档）

| 证据 | 原文 | 冻结标签 | 人工：相关/不相关/不确定 | 理由 |
| --- | --- | --- | --- | --- |
| D01 | A red bus waits beside a bus stop marked Route 12. | 不相关 | 待填写 | |
| D02 | The Route 12 bus stop has a red bus at the sign. | 不相关 | 待填写 | |
| D03 | A bicycle stands beside a river bridge with a Bridge closed sign. | 不相关 | 待填写 | |
| D04 | A river bridge has a Bridge open sign and no bicycle. | 不相关 | 待填写 | |
| D05 | The library entrance displays Open Sunday. | 不相关 | 待填写 | |
| D06 | The library entrance displays Closed Sunday. | 不相关 | 待填写 | |
| D07 | A train platform marked Platform 7 has a suitcase. | 不相关 | 待填写 | |
| D08 | A train platform marked Platform 2 has a suitcase. | 不相关 | 待填写 | |
| D09 | A dog stands beside a fountain labelled Drinking water. | 不相关 | 待填写 | |
| D10 | A decorative fountain beside a garden has a No drinking notice. | 不相关 | 待填写 | |
| D11 | car | 不相关 | 待填写 | |
| D12 | A blue bus at a bus stop displays Route 9. | 不相关 | 待填写 | |

- [ ] 观察中的对象、文字、动作/关系、时间、地点已核对
- [ ] qrels 正例与全部负例已核对
- [ ] 失败解释已复核（同意/修订/不确定）：____
- 审核人 / 日期：____
- 建议相关 ID（不自动写回）：____
- 证据充分性与备注：____

## S10

原始描述：A river bridge with a sign reading Bridge open.

场景规格（非像素观察）：A river bridge with a sign reading Bridge open.

结构化视觉观察：
```json
{
  "objects": [
    "river bridge"
  ],
  "visible_text": [
    "Bridge open"
  ],
  "uncertainties": [
    "Synthetic fixture; no pixels analyzed"
  ]
}
```

冻结 qrels：D04；状态：available

原标注理由：Open bridge contrasts with the closure scene.

已验证事实：river bridge/Bridge open 保留，无关键状态丢失；Baseline 的 with 对 D03 有贡献，Structured 首位变为 D04。

待人工判断：重点：D04 额外写 no bicycle 是否允许；D03 的 closed 是否足以排除。

ASCII 去重词项损失：a, reading, sign, with（中文丢失需结合原字段判断）。

### Baseline

实际查询：`a river bridge with a sign reading bridge open`

错误：无；词项数/去重数：9/7；语料外词项：reading

查询来源：`[{"field": "original_description", "terms": ["a", "river", "bridge", "with", "a", "sign", "reading", "bridge", "open"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D03 | 7.031654717605128 | 不相关 | A bicycle stands beside a river bridge with a Bridge closed sign. |
| 2 | D04 | 6.778453621081543 | 相关 | A river bridge has a Bridge open sign and no bicycle. |
| 3 | D05 | 1.9332216212367104 | 不相关 | The library entrance displays Open Sunday. |
| 4 | D02 | 1.406917124277149 | 不相关 | The Route 12 bus stop has a red bus at the sign. |
| 5 | D10 | 0.4939125650844657 | 不相关 | A decorative fountain beside a garden has a No drinking notice. |

Top-5 命中：D04；遗漏：无。

逐例指标：`{"recall_at_1": 0.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 0.5}`

### Structured

实际查询：`river bridge bridge open`

错误：无；词项数/去重数：4/3；语料外词项：无

查询来源：`[{"field": "objects", "index": 0, "terms": ["river", "bridge"]}, {"field": "visible_text", "index": 0, "terms": ["bridge", "open"]}]`

| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |
| --- | --- | --- | --- | --- |
| 1 | D04 | 5.174323406612547 | 相关 | A river bridge has a Bridge open sign and no bicycle. |
| 2 | D03 | 3.546292906765145 | 不相关 | A bicycle stands beside a river bridge with a Bridge closed sign. |
| 3 | D05 | 1.9332216212367104 | 不相关 | The library entrance displays Open Sunday. |

Top-5 命中：D04；遗漏：无。

逐例指标：`{"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr_at_5": 1.0}`

### 人工填写（覆盖全部闭集，不只核对已检出文档）

| 证据 | 原文 | 冻结标签 | 人工：相关/不相关/不确定 | 理由 |
| --- | --- | --- | --- | --- |
| D01 | A red bus waits beside a bus stop marked Route 12. | 不相关 | 待填写 | |
| D02 | The Route 12 bus stop has a red bus at the sign. | 不相关 | 待填写 | |
| D03 | A bicycle stands beside a river bridge with a Bridge closed sign. | 不相关 | 待填写 | |
| D04 | A river bridge has a Bridge open sign and no bicycle. | 相关 | 待填写 | |
| D05 | The library entrance displays Open Sunday. | 不相关 | 待填写 | |
| D06 | The library entrance displays Closed Sunday. | 不相关 | 待填写 | |
| D07 | A train platform marked Platform 7 has a suitcase. | 不相关 | 待填写 | |
| D08 | A train platform marked Platform 2 has a suitcase. | 不相关 | 待填写 | |
| D09 | A dog stands beside a fountain labelled Drinking water. | 不相关 | 待填写 | |
| D10 | A decorative fountain beside a garden has a No drinking notice. | 不相关 | 待填写 | |
| D11 | car | 不相关 | 待填写 | |
| D12 | A blue bus at a bus stop displays Route 9. | 不相关 | 待填写 | |

- [ ] 观察中的对象、文字、动作/关系、时间、地点已核对
- [ ] qrels 正例与全部负例已核对
- [ ] 失败解释已复核（同意/修订/不确定）：____
- 审核人 / 日期：____
- 建议相关 ID（不自动写回）：____
- 证据充分性与备注：____

## 证据来源

所有文档均为虚构合成证据，human_reviewed=false。

- D01: `synthetic://t04_v1/D01`
- D02: `synthetic://t04_v1/D02`
- D03: `synthetic://t04_v1/D03`
- D04: `synthetic://t04_v1/D04`
- D05: `synthetic://t04_v1/D05`
- D06: `synthetic://t04_v1/D06`
- D07: `synthetic://t04_v1/D07`
- D08: `synthetic://t04_v1/D08`
- D09: `synthetic://t04_v1/D09`
- D10: `synthetic://t04_v1/D10`
- D11: `synthetic://t04_v1/D11`
- D12: `synthetic://t04_v1/D12`
