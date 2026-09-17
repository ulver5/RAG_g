# GreenGovRAG 架构文档 —— 双管道完整运行流程

> 本文件描述代码库架构,按两条主线展开:
> **① 文档解析管道(离线 / ETL)** 与 **② 查询管道(在线 / 逐请求)**。
> 项目路径:`green-gov-rag-main/`。面向澳大利亚环境与规划法规的检索增强问答(RAG)平台。
>
> **两套改造轨道(编号勿混)**
> - **P0 / P1**(改进计划轨道,**已在主链路落地**):混合检索内核(BM25+RRF+Reranker+分级)、
>   事前置信度门控、span 引用、多轮记忆、bad_cases 表、**基础版**四维评测 —— 见 §1、§2、§3、附录。
> - **P1–P6**(学习计划轨道,**评测与交互进阶,优化后完整形态**):LLM-as-Judge 评测、A/B 对比工具、
>   bad case 归因分析、地图交互闭环、数据集工程+CI 门控、可解释可视化 —— 见 **§4 优化后完整架构**。
>
> 关键判断:P1–P6 **基本不改在线主链路 [0]–[8]**,而是在其外围建起"度量—对比—归因—守门"的质量飞轮,
> 并把黑盒信号产品化为可解释交互。因此 §1–§3 仍准确描述在线行为,§4 描述围绕它的优化后完整架构。

---

## 0. 总览

**定位**:用 RAG 回答澳大利亚联邦 / 州 / 地方三级环境与规划法规问题,答案带条款级引用与地理空间(LGA)过滤。

**技术栈**
- 后端:FastAPI + SQLModel/Alembic + LangChain(Python 3.12)
- 稠密向量库:Qdrant(生产)/ FAISS(开发)
- 稀疏检索:BM25 —— `rank_bm25`(内存,默认)/ Elasticsearch(生产,P0 新增)
- 重排:CrossEncoder(P0 新增,特性开关)
- 关系库:PostgreSQL(sources / files / chunks / versions / query_history / **bad_cases**)
- 缓存 / 会话:Redis(三级缓存 + 会话记忆)
- LLM:多 provider 工厂(OpenAI / Azure / Bedrock / Anthropic)
- 前端:React 19 + Vite + Mapbox GL(LGA 地图)+ Plotly
- 编排:Airflow(本地)/ GitHub Actions(生产 ETL)
- 部署:Docker Compose / AWS CDK / Azure Bicep

**组件拓扑**

```
                         ┌───────────────────────────┐
   React + Mapbox UI ───▶│      FastAPI  (/api/*)     │
                         └───────────────────────────┘
                             │            │        │
              ┌──────────────┘            │        └──────────────┐
              ▼                           ▼                       ▼
     ┌──────────────┐          ┌────────────────────┐   ┌──────────────────┐
     │  向量库       │          │   PostgreSQL        │   │   Redis           │
     │ Qdrant/FAISS  │          │ sources/files/      │   │ 缓存 + 会话记忆    │
     │ (稠密)        │          │ chunks/versions/    │   └──────────────────┘
     └──────────────┘          │ query_history/      │
     ┌──────────────┐          │ bad_cases           │   ┌──────────────────┐
     │ BM25 (稀疏)   │          └────────────────────┘   │ LLM 多provider    │
     │ rank_bm25/ES  │                                    │ + CrossEncoder    │
     └──────────────┘                                    └──────────────────┘
              ▲
     ┌────────┴─────────┐   离线写入两套索引
     │  ETL 解析管道     │◀── Airflow / GitHub Actions / CLI
     └──────────────────┘
```

---

## 1. 文档解析管道(离线 / ETL)

**性质**:同步、批处理、按文档迭代(监控/发现层为 `async`)。**产物**:稠密向量索引 + BM25 语料 + 关系库中的 chunk/版本记录,三者以 `file_id` / `chunk_index` 关联。

### 1.1 触发方式
- 本地:`greengovrag-cli etl ...` 或 Airflow DAG `greengovrag_full_pipeline`
- 生产:GitHub Actions 定时(`aws-etl-scheduled` 每 2 天;监控 `*-etl-monitoring`)/ Azure Container App Job
- 云事件:S3/Blob 触发文件到达 → Sensor DAG → 主 DAG

### 1.2 完整运行流程(分步)

```
① 采集 Ingest        configs/documents_config.yml → DocumentSourceFactory
   ├─ 插件源:federal / state / local_government / emissions / cer_emissions
   ├─ 下载(requests + 退避重试 + Cloudflare 检测) / 云:StorageClient(S3/Blob)
   ├─ 类型嗅探(magic bytes)、sha256、sidecar .metadata.json
   └─ 稳定 document_id(jurisdiction_category_topic_filename) → 支持增量
        │
        ▼
② 结构解析 Parse      HierarchicalPDFParser(带回退链)
   ├─ 首选 unstructured.io(hi_res,infer_table_structure,含 Title/Table/List)
   ├─ 回退 PyMuPDF(fitz):按字号/正则判定标题 → 维护 section 栈
   └─ 提取:section_hierarchy、clause_reference(s./cl./reg./Schedule)、page_range
        │
        ▼
③ 结构感知切分 Chunk  TextChunker(RecursiveCharacterTextSplitter, 1000/100)
   └─ chunk_with_hierarchy:子切分同时保留层级元数据 + 全局整型 chunk_id
        │
        ▼
④ 元数据打标 Tag      ESGOpenAITagger(LLM,可选)
   └─ ESGMetadata(scope1/2/3、GHG、frameworks: NGER/ISSB/…)+ DocumentMetadata
        │
        ▼
⑤ 版本化 & 去重       DocumentVersion(content_hash / is_current / superseded_at)
   └─ MonitorableSource:HEAD→Last-Modified→ETag→部分/全量哈希 分级变更检测
        │
        ▼
⑥ 入关系库 DB Writer  upsert:document_sources → document_files → document_chunks
        │
        ├────────────────────────────┐
        ▼                             ▼
⑦a 建稠密索引               ⑦b 建/刷新 BM25 语料 (P0)
   scripts/build_embeddings     rag/bm25_index.py:
   → ChunkEmbedder(HF)          - rank_bm25: 从 document_chunks 表懒构建(内存)
   → VectorStore(Qdrant/FAISS)  - elasticsearch: 由 ETL 写入 ES 索引
```

### 1.3 编排(Airflow / CLI 6 段链)
`ingest → parse → chunk → load-chunks-to-db → rag index → test query`
(生产云 DAG 用 Qdrant 索引;本地 DAG 用 FAISS。BM25 内存索引在 API 首次查询时懒构建,或调用 `refresh_bm25_index()` 重建。)

### 1.4 数据模型(PostgreSQL,SQLModel)
```
document_sources (1) ──< document_files (1) ──< document_chunks
        │                        │
        └──< document_versions ──┘        monitoring_logs
query_history        bad_cases (P1 新增)
```
- `document_chunks`:`text` 供 BM25;`page_number/section_hierarchy/clause_reference/citation/deep_link` 供引用溯源;`embedding_index` 对应向量库。

---

## 2. 查询管道(在线 / 逐请求)

**性质**:延迟敏感,逐请求。入口 `POST /api/query` → `QueryService.execute_query`。
下图 ★ 为 **P0/P1 新增/改造** 环节。

### 2.1 端到端流程图

```
POST /api/query {query, region, lgas, jurisdiction, topics, session_id, ...}
        │
        ▼
[0] 归一化      region→全称 · jurisdiction→小写 · topic 映射 · LGA 前缀 fan-out
        │
        ▼
★[1] 查询改写(多轮 P1)    session_memory.format_history → rag/query_rewrite
        │  (指代消解:"它"→"营业执照";无历史则原样)
        ▼
★[2] 混合召回(P0)   rag/agent_tools.RAGAgent.retrieve → HybridRetrievalPipeline
        │
        │   ┌─ 分级路由 query_router:simple=快路 / complex=全管道
        │   ├─ 稠密召回 Qdrant/FAISS.similarity_search(带 metadata_filters,原生过滤)
        │   ├─ 稀疏召回 BM25(rank_bm25/ES)+ 元数据约束层(宽松匹配)
        │   ├─ ★RRF 融合(rag/fusion,k=60,内容哈希跨路去重)
        │   └─ ★Reranker 精排(CrossEncoder,仅复杂查询;不可用则保留融合序)
        │        → 返回带 0-1 `score` 的 top-k 文档
        ▼
★[3] 置信度门控(P0)   rag/confidence.assess_confidence(取 rerank/融合最高分)
        │
        ├── score < 0.60 ──▶ 反问(clarification),跳过 LLM,answer_type=clarification
        ├── 0.60–0.85 ─────▶ 生成 + 免责提示(caveat),answer_type=partial
        └── ≥ 0.85 ────────▶ 直接生成 + 引用,answer_type=answered
        │
        ▼
[4] 缓存 & 生成      CacheService 三级(内存 LRU → Redis → 语义余弦≥0.95)
        │  命中直返;未命中 → RAGAgent.generate(多 provider LLM)→ 回写缓存
        ▼
★[5] Span 级引用(P0)   api/utils/span_citation:句子↔来源 token 重叠 → 行内 [n]
        │
        ▼
[6] 来源富化         CitationFormatter:页码/条款/PDF 深链接 → SourceDocument[]
        │
        ▼
[7] 事后可信(可选 include_trust_score=true)
        │  ├─ 引用核验 CitationVerificationService(版本表 + difflib 引文 + LLM 相关性裁判)
        │  ├─ 法律层级冲突 RegulatoryHierarchyService(联邦>州>地方)
        │  └─ 信任分 TrustScoreService(相关40/时效25/权威15/冲突10/准确10)
        ▼
[8] 落库 & 闭环       query_history 落库(query_id 供反馈)
        │  ★ session_memory.add_turn(多轮记忆,P1)
        │  ★ 低置信/反问 → bad_cases 入库(P1,周复盘)
        ▼
QueryResponse {answer, sources, ★answer_type, ★confidence_score/level, ★clarification,
               trust_score/breakdown, conflicts, coverage_info, query_id, ...}
```

### 2.2 分步说明表

| 步 | 组件/文件 | 作用 | 归属 |
|---|---|---|---|
| 0 | `query_service.execute_query` | 过滤器归一化、LGA 前缀 fan-out | 原有 |
| 1 | `rag/query_rewrite.py` + `session_memory` | 多轮指代消解改写 | **P1** |
| 2 | `rag/retrieval_pipeline.py`(`fusion`/`bm25_index`/`reranker`/`query_router`) | 稠密‖稀疏→RRF→rerank→路由 | **P0** |
| 3 | `rag/confidence.py` | 事前三级置信度门控 | **P0** |
| 4 | `api/services/cache.py` | 三级缓存 + 生成 | 原有 |
| 5 | `api/utils/span_citation.py` | 句子级行内引用 | **P0** |
| 6 | `api/utils/citation_formatter.py` | 深链接/条款/页码 | 原有 |
| 7 | `citation_verification`/`regulatory_hierarchy`/`trust_score` | 事后可信度 | 原有 |
| 8 | `query_history` / `session_memory` / `bad_cases` | 落库 + 记忆 + 闭环 | 原有 + **P1** |

### 2.3 降级路径(单机 CPU 可跑)
- BM25 空/异常 → 回退纯稠密;单路 RRF = 原序
- Reranker 模型缺失 → 跳过,保留融合序
- 简单查询 → 分级路由自动跳过 Reranker
- 整个 HybridRetrievalPipeline 异常 → `RAGAgent.retrieve` 回退旧稠密 `similarity_search`
- 会话记忆 Redis 不可用 → 内存字典兜底

### 2.4 完整详细阶段图(含模块 3/4 展开:领域约束层、事后可信、记忆/bad case)

```
POST /api/query {query, region, lgas, jurisdiction, topics, session_id,
                 max_sources, include_trust_score}
        │
        ▼
[0] 归一化   region→全称 · jurisdiction→小写 · topic 映射 · LGA 前缀 fan-out
        │
        ▼
★[1] 查询改写(P1,多轮)
        ├─ 无 session 历史 ───────────────► retrieval_query = 原 query
        └─ 有历史 ─► session_memory.format_history
                    ─► rewrite_query(LLM,temp0,消解"它/那个"+补管辖/地区)
                    ─► retrieval_query(标准化)   〔原 query 留作 记忆/落库/回显〕
        │
        ▼
★[2] 混合检索(P0)  HybridRetrievalPipeline
        │  分级路由 query_router: simple / complex
        │  ┌── 稠密召回 Qdrant/FAISS(recall_k, 原生 metadata 过滤)
        │  └── 稀疏召回 BM25(recall_k)+ 宽松元数据约束
        │        │
        │        ▼  RRF 融合(k=60, 内容哈希去重)
        │        │
        │  ┌─────┴──── 领域约束层(模块3: hybrid_search)──────┐
        │  │  federal 永远纳入 · state 匹配州 · local 匹配LGA │
        │  │  管辖 3:1 加权(匹配管辖靠前)                     │
        │  └───────────────────┬────────────────────────────┘
        │      simple ─────────┴─► 取 top-k
        │      complex ─► Reranker(cross-encoder, sigmoid→score)─► top-k
        │           → 文档带 0-1 `score`
        ▼
★[3] 置信度门控(P0)  取 top score
        ├── < 0.60 ──► 反问 clarification ──► 跳过 LLM ─────────────┐
        ├── 0.60–0.85 ► 生成 + 免责提示(partial)                   │
        └── ≥ 0.85 ───► 直接生成(answered)                         │
        │ (要生成时)                                               │
        ▼                                                          │
[4] 缓存 & 生成                                                     │
        │ L1 内存LRU(exact)─hit─┐                                  │
        │ L2 Redis(exact)  ─hit─┤ 未命中↓                          │
        │ L3 语义余弦≥0.95 ─hit─┤ 未命中↓                          │
        │                       │ RAGAgent.generate(prompt=context+query→LLM)
        │        命中/新生成 ◄───┘ → cache.set 回写 L1/L3/L2
        ▼                                                          │
★[5] span 引用(P0)  句子↔来源词元重叠 → 行内 [n](无重叠不标)     │
        ▼                                                          │
[6] 来源富化  CitationFormatter: page/clause/section + PDF 深链接    │
        │           (答案里的 [n] ↔ 第 n 条 SourceDocument 对齐)    │
        ▼                                                          │
[7] 事后可信(仅 include_trust_score=True,模块3)                    │
        │  ├─ 引文核验(前3条): 版本superseded/staleness(DocumentVersion)
        │  │                    + 引文 difflib 比对 + LLM 相关性裁判
        │  ├─ 层级冲突: 联邦>州>地方,冲突标 warning +"联邦优先"
        │  ├─ 权威分: jurisdiction/category/主权/时效
        │  └─ 信任分(40相关/25时效/15权威/10冲突/10引文准确)→ trust_score/level/breakdown
        ▼                                                          ▼
[8] 落库与闭环
        │  ├─ query_history 落库 → query_id(供反馈)
        │  ├─ ★session_memory.add_turn(原query, answer)         (P1 多轮)
        │  ├─ ★answer_type≠answered → bad_cases 入库             (P1 复盘)
        │  └─ region 给定 → coverage_info(LGA 覆盖度 + 贡献入口)
        ▼
QueryResponse { answer, sources[], answer_type, confidence_score/level,
                clarification, trust_score/confidence/breakdown,
                conflicts_detected, hierarchy_explanation,
                citation_warnings, coverage_info, query_id }

〔贯穿〕会话记忆(Redis) · 缓存降本 · (可离线)eval 四维矩阵+回归门控 读 query_history/bad_cases
〔降级〕BM25空→纯稠密 | Reranker缺→保留融合序 | pipeline异常→回退旧稠密search | 记忆Redis挂→内存兜底
```

---

## 3. 贯穿能力

### 3.1 特性开关(`green_gov_rag/config.py` / `.env`)
| 开关 | 默认 | 作用 |
|---|---|---|
| `ENABLE_HYBRID_RETRIEVAL` | true | 启用 P0 混合检索管线 |
| `ENABLE_BM25` / `BM25_BACKEND` | true / rank_bm25 | 稀疏路 + 后端(rank_bm25/elasticsearch) |
| `RECALL_K` / `RRF_K` | 20 / 60 | 每路召回量 / RRF 常数 |
| `ENABLE_RERANKER` / `RERANKER_MODEL` | true / ms-marco-MiniLM | 重排 + 模型(可换 bge-reranker-v2-m3) |
| `ENABLE_TIERED_ROUTING` | true | 简单/复杂分级 |
| `ENABLE_CONFIDENCE_GATING` | true | 事前门控 |
| `CONFIDENCE_HIGH/LOW_THRESHOLD` | 0.85 / 0.60 | 三级阈值 |
| `ENABLE_SPAN_CITATIONS` | true | 句子级引用 |
| `ENABLE_MULTI_TURN` | true | 多轮 + 改写 |
| `ENABLE_BAD_CASE_LOGGING` | true | bad case 入库 |

### 3.2 评估闭环(基础版 · P0/P1 轨道)
- `green_gov_rag/eval/`:四维矩阵(检索命中率 / 答案准确率 / 延迟 P99 / 满意度)+ 回归门控(任一维退化超容差即失败)
- `scripts/evaluate_rag.py`:`--dataset` 跑评估、`--save-baseline` 存基线、`--baseline` 做回归门控
- `eval/eval_dataset.json`:起始标注集;`bad_cases` 表驱动周复盘(检索失败/切分不当/知识库缺失)
- ⚠️ 局限:答案准确率仅 `keyword_answer_score`(关键词命中比例,弱代理);数据集小;门控靠本地手动跑。
  **优化后形态见 §4**(LLM-judge 替换弱代理、A/B 对比、bad case 归因、数据集扩充、CI 门控)。

### 3.3 部署形态
- **Docker Compose**:postgres(pgvector)+ redis + qdrant + backend + (airflow, dev) + frontend
- **AWS**:ECS Fargate(Spot)+ RDS + S3 + DynamoDB(缓存)+ EC2 Qdrant(Spot)+ CloudFront + Lambda
- **Azure**:Container Apps + PostgreSQL Flexible + Table Storage + Static Web App + Container App Jobs

---

## 4. 优化后完整架构(学习计划 P1–P6:评测与交互进阶)

> 本章描述 P1–P6 **全部完成**后的完整形态。前提:在线主链路 [0]–[8]**基本不变**,
> 净增部分集中在两处 —— **评测子系统**(P1/P2/P3/P5)与**交互产品层**(P4/P6)。
> (此处 P1–P6 为**学习计划轨道**,与 §1–§3 的改进计划 P0/P1 是不同编号,勿混。)

### 4.1 整体架构图

```
┌─────────────────────────── 交互产品层(P4/P6 强化)──────────────────────────┐
│  React + Mapbox                                                              │
│  ├─ 地图选区(P4)LGA choropleth → selectedLGAs → filters.lgas → 空间约束检索 │
│  ├─ 置信度门控可视化(P6)引用/提示/反问 三档差异化 UI                        │
│  ├─ span 级行内引用高亮(P6)点答案句 → 跳源文档条款/页码                     │
│  └─ 检索来源面板(P6)展开看 RRF / Reranker 分数、命中路径                    │
└───────────────────────────────────┬──────────────────────────────────────┘
                                     │ POST /api/query
┌────────────────────────── 在线查询主链路(§2,基本不变)────────────────────┐
│ [0]归一化→[1]改写→[2]混合召回(BM25‖稠密→RRF→Rerank→路由)→[3]置信度门控     │
│ →[4]缓存&生成→[5]span引用→[6]来源富化→[7]事后可信→[8]落库                    │
│                                    │ 写 query_history / bad_cases / feedback │
└───────────────────────────────────┼──────────────────────────────────────┘
                                     ▼
┌────────────────────── 评测与质量子系统(P1/P2/P3/P5 新增强)──────────────────┐
│  P5 数据集工程     eval_dataset.json(50+,分层标注:难度/类型/管辖)          │
│      └─ 半自动生成 QA(LLM 生成 + 人工校验)                                  │
│            ▼                                                                 │
│  P1 LLM-as-Judge   eval/llm_judge.py:answer_correctness / faithfulness /     │
│      (替换关键词代理) citation_accuracy  ← 强制JSON + 可靠性方差 + 降级不崩    │
│            ▼                                                                 │
│  P2 A/B 评测工具   scripts/ab_compare.py:两配置子进程隔离跑 → diff_matrices   │
│      (证据化选型)    → 终端表 + 零依赖 HTML 报告(内联SVG:指标条/延迟分布)    │
│            ▼                                                                 │
│  P3 Bad Case 分析  失败归因分类(检索未召回/切分/缺失/幻觉/错引/过度拒答)      │
│      (驱动迭代)      ← 用 judge 中间信号打标 → 周报(分布图 + case卡 + 改进动作)│
│            ▼                                                                 │
│  P5 CI 回归门控    GitHub Actions:PR 触发评测 → 任一维退化 fail → 贴 PR       │
└──────────────────────────────────────────────────────────────────────────┘
        ▲ 回灌:bad case 分类 → 定向补数据集 / 调 chunk / 补覆盖度
```

### 4.2 逐阶段:改了什么 + 好处

| 阶段 | 对应文档部位 | 改造前 | 改造后(P1–P6) | 好处 |
|---|---|---|---|---|
| **P1 LLM-Judge** | §3.2 评估闭环 | 答案准确率 = `keyword_answer_score`(关键词命中比例) | 三裁判:正确性 / **忠实度** / **引用正确率**;强制 JSON、方差监控、降级不崩 | 从"含关键词就算对"→ **能抓幻觉、能验引用**;不再依赖人工标关键词 |
| **P2 A/B 工具** | §3.2 只有单次 + 回归门控 | 只能"历史基线 vs 当前"跑一次 | 任意**两套配置**同集对比,子进程隔离,出终端表 + HTML | 把"感觉 Reranker 更好"变成**"命中+16分但 P99+1800ms"**的证据化选型 |
| **P3 Bad Case** | §2.1 [8] bad_cases 仅入库 | 有表、无分析(只记录低置信/反问) | **失败归因分类** + 周报(分布 + case卡 + 改进动作) | 反馈**真正闭环**:定位检索/切分/缺失/幻觉 → 驱动下一轮 |
| **P4 地图交互** | §0 前端仅列组件 | 地图选区能力零散(主链路未描述联动) | **选区 → filters.lgas → 空间约束检索**完整闭环 + 覆盖度提示 | 可交互产品成形:用户用地图**框定管辖**而非手填过滤器 |
| **P5 数据集+CI** | §3.2 起始小集、本地门控 | 数据集个位数、回归门控手动跑 | 数据集扩至 50+ **分层标注** + **CI 自动门控**贴 PR | 质量**可持续、可拦截**:防"单指标优化劣化整体"进主干 |
| **P6 可解释可视化** | §2.1 [2]/[3]/[5] 信号仅后端 | 档位/span/RRF分 用户看不见 | 前端**暴露并可探查**:三档 UI、句级引用跳转、来源分数面板 | 从"给答案"→ **"可解释、可干预"**;信任与可审计落到界面 |

### 4.3 前后对照总表(扩展文末附录)

| 维度 | 改造前 | 改造后(P1–P6 完成) |
|---|---|---|
| 答案质量度量 | 关键词命中比例(弱代理) | LLM-judge:正确性 + 忠实度 + 引用正确率 |
| 选型决策 | 单次评测 + 历史回归 | A/B 两配置对比(质量↔延迟取舍量化) |
| 失败处理 | bad case 仅入库 | 归因分类 + 周报 + 回灌迭代 |
| 评测集 | 起始小集、无分层 | 50+ 分层标注、半自动扩充 |
| 质量守门 | 本地手动回归 | CI 自动门控,PR 拦截退化 |
| 地图交互 | 组件存在、联动未成体系 | 选区驱动空间约束检索闭环 |
| 结果可解释性 | 信号仅在后端 | 前端可视化 + 可探查(档位/span/分数) |
| 幻觉控制 | 事前门控 + 事后信任分 | **+ 评测层第三道**:faithfulness/citation 裁判持续监控 |

### 4.4 关键新增文件(P1–P6 轨道)
- 评测:`eval/llm_judge.py`(P1)· `eval/ab_config.py` + `eval/report.py` + `scripts/ab_compare.py`(P2)
- 数据/配置:`eval/eval_dataset.json`(P5 扩充分层)· `eval/configs/*.yaml`(P2 A/B 配置)
- CI:`.github/workflows/*`(P5 评测门控)
- 前端(P4/P6):`frontend/src/pages/PlaygroundPage.tsx` · `store/mapStore.ts`(地图选区↔检索、置信度/span/分数可视化)

---

## 5. 关键文件索引

**解析管道**
- `green_gov_rag/etl/`:`ingest.py` `pipeline.py` `chunker.py` `metadata_tagger.py` `db_writer.py`
- `green_gov_rag/etl/parsers/`:`layout_parser.py` `unstructured_parser.py` `pdf_parser.py`
- `green_gov_rag/etl/sources/`:`factory.py` `registry.py` `base.py` + 各源插件
- `green_gov_rag/scripts/build_embeddings.py`

**查询管道**
- 检索(P0):`rag/retrieval_pipeline.py` `rag/fusion.py` `rag/bm25_index.py` `rag/reranker.py` `rag/query_router.py`
- 门控(P0):`rag/confidence.py` · `api/utils/span_citation.py`
- 多轮(P1):`rag/query_rewrite.py` · `api/services/session_memory.py`
- 编排:`api/services/query_service.py` · `rag/agent_tools.py` · `rag/rag_chain.py`
- 可信:`api/services/{citation_verification,trust_score,regulatory_hierarchy}.py`

**评估 / 数据 / 配置**
- `green_gov_rag/eval/` · `scripts/evaluate_rag.py` · `eval/eval_dataset.json`
- `models/`:`chunk.py` `document.py` `document_version.py` `query.py` `bad_case.py`(P1)
- `config.py` · `.env.example` · `alembic/versions/a1b2c3d4e5f6_add_bad_cases_table.py`(P1)

---

## 附:P0/P1 前后对照(检索与幻觉控制核心)

| 维度 | 改造前 | 改造后(当前) |
|---|---|---|
| 召回 | 纯稠密单路 | 稠密 + BM25 双路 |
| 融合 | 3:1 交织 | RRF |
| 精排 | 无 | CrossEncoder(分级开启) |
| 路由 | 无 | 简单/复杂分级 |
| 幻觉控制 | 仅事后信任分 | 事前门控 + 事后信任分 |
| 引用 | 末尾堆叠 | 句子级行内 [n] |
| 多轮 | 无 | 会话记忆 + LLM 改写 |
| 质量度量 | 无基线 | 四维矩阵 + 回归门控 + bad case 闭环 |

