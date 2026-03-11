# README V1

## 项目简介

这是一个面向学生 / 研究生购机场景的笔记本推荐 Agent。

项目目标不是做泛化聊天机器人，而是做一个有明确边界的垂直 Agent，支持：

- 需求澄清
- 候选商品检索
- 参数对比
- 规则打分
- 最终推荐

## 为什么选这个题目

相比小说生成器或全流程网店助手，购物推荐 Agent 更适合作为第一个 Agent 项目，因为它：

- 更容易体现 `LangGraph` 工作流设计
- 更容易加入工具调用和结构化决策
- 更容易做 benchmark 和评估闭环
- 更接近真实业务场景

## 当前范围

`V1` 明确只做一个品类：

- 笔记本电脑推荐

不做全品类，也不同时覆盖化妆品、电子产品、电器三大类。

这样做的原因是：

- 不同品类决策维度差异大
- 数据清洗和规格归一化成本高
- 第一版更重要的是把链路跑通、评估做出来

后续扩展路线：

- `V1`：单品类做深
- `V1.5`：抽象统一 schema 和接口
- `V2`：扩展到有限多品类

## 一句话定义

> 做一个面向学生 / 研究生购机场景的笔记本推荐 Agent，支持预算约束、用途澄清、候选检索、参数对比和最终推荐。

## 最小工作流

当前最小链路设计为：

1. `parse_query`
2. `check_missing_info`
3. `retrieve_products`
4. `score_candidates`
5. `generate_answer`

## 输入设计

输入分为两层：

- `common_fields`
- `category_fields`

### common_fields

适用于未来跨品类复用的字段：

- `budget_min`
- `budget_max`
- `usage_scenarios`
- `brand_preference`
- `brand_avoid`

### category_fields

当前笔记本品类专用字段：

- `need_portability`
- `need_dedicated_gpu`
- `memory_min_gb`
- `storage_min_gb`

## 输出设计

当前推荐输出保留这些关键字段：

- `task_summary`
- `extracted_constraints`
- `missing_info`
- `clarifying_questions`
- `candidates`
- `final_recommendation`
- `recommendation_reason`

## 当前项目骨架

```text
notebook_agent/
├── app/
│   ├── main.py
│   ├── state.py
│   ├── schemas.py
│   ├── nodes.py
│   └── workflow.py
├── data/
│   └── products.json
├── tests/
│   └── test_smoke.py
└── requirements.txt
```

当前骨架已经包含：

- `FastAPI` 入口
- `LangGraph` 风格工作流
- mock 商品数据
- 基础推荐逻辑
- smoke test

## 运行方式

安装依赖：

```bash
pip install -r notebook_agent/requirements.txt
```

如果当前目录在 `travel_planner`，运行：

```bash
uvicorn notebook_agent.app.main:app --reload
```

如果当前目录在 `notebook_agent`，运行：

```bash
uvicorn app.main:app --reload
```

不要直接运行：

```bash
python app/main.py
```

因为这会导致相对导入报错。

## 当前阶段最重要的事

当前不追求“品类很多”，而是优先保证：

- 工作流清晰
- 输入输出稳定
- mock 数据可跑通
- 推荐结果可解释
- 后续可以接 benchmark

## 下一步

1. 保证 `/recommend` 接口正常运行
2. 补全输入 schema
3. 补更多笔记本 mock 数据
4. 增加澄清问题逻辑
5. 细化评分规则
6. 再接 benchmark 和 Web Demo
