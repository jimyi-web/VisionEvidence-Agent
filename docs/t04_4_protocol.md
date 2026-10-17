# T04-4 预先固定的评测协议 v1

本协议与数据在首次检索评测前写入。Windows CPU；无网络、GPU、模型加载或新增依赖。

## 设计审查

- Baseline = 完整原始视觉描述的既有 tokenize 输出；不是声明文本。Structured = T04-1 的 objects + visible_text；uncertainties 不进入查询。此次为描述表示比较，不是声明基线或真实视觉理解实验。
- 两组使用同一 BM25Index、同一完整语料、k1=1.5、b=0.75、top_k=5、稳定 ID 同分排序。不重排、不回退、不扩展查询、不调参。分别计算 macro Recall@1/3/5 和 MRR@5，复用 T02 metrics_at_k。
- 样本文件与 qrels 分开。查询函数只接收原始描述或结构化 JSON，不接收 qrels、证据内容、标注解释。先完成两组排名，再加载 qrels 评分。
- 数据是 Codex 编写的合成场景描述、结构化观察和虚构证据；没有真实图片、真实事件、人工完成的标注或 VLM 输出。visual_observation 是场景文字规格，不能当作读取像素的结果。人工可逐项检查，但 human_reviewed=false。
- 结构化字段采用原始描述中连续片段；不得添加原描述中没有的关键词。特意保留结构化遗漏、词汇差异、中文失败及语料缺证，不能删掉不利样本。语料是成组主题文档与干扰文档，不按运行结果增删。
- 仍存在同一作者构造描述、证据和 qrels 的词汇重合及选择偏差，不能据此推断 Structured 在独立真实集的优势。此集是固定 synthetic_test 合约评测集，不用于策略优化；之后修改数据或策略必须升版本并使用独立 dev 数据。

## 分母和失败分析

有非空 qrels 的所有样本进入端到端分母；解析或查询构造失败排名为空，指标记零。额外报告成功构造查询子集指标及其分母，不能替代主指标。
空 qrels 在所有检索指标中排除并列出，遵循 T02；其中 corpus_insufficient 是明确的构造标注，不把任何空 qrels 自动认定为无证据。
标签允许重叠：query_construction_error（解析/无支持词项）、retrieval_no_hit（有效查询但无候选）、relevant_not_retrieved_at_5（有候选但未命中 qrels）、partial_recall_at_5、evidence_insufficient（预先标明语料不能回答）。这些是可观察诊断，不声称自动判定语义查询错误。结构化遗漏由逐例解释补充。

## 固定数据及审计

data/evaluation/t04_v1 中保存 samples.jsonl、corpus.jsonl、qrels.jsonl、dataset.json。qrels 包含相关 ID、判定理由和证据充分性；相关性定义为直接描述同一合成场景的对象/可见文字，非真假判断。未列为相关的其他文档视为不相关，范围仅此闭集。
固定 seed=42；无随机抽样，种子仅作审计记录。独占输出目录，不覆盖历史；保存实际输入字节、SHA-256、配置、源码哈希、Git HEAD/dirty、Python/依赖/平台、逐例查询来源/候选全文/分数/耗时、汇总。无推理耗时，内存未测量须明确为空。
本轮冻结后允许修复程序错误并记录；不允许按测试分数改变数据或查询策略。最终报告不报告 Accuracy 或视觉准确率提升。
