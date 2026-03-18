# Notebook Recommender Agent

一个最小可运行的笔记本推荐 Agent 骨架，用来承载后续的 `LangGraph` 工作流、商品数据、评分逻辑和 Web Demo。

## 当前骨架包含

- `FastAPI` 接口入口
- `LangGraph` 风格工作流
- 结构化输入输出模型
- mock 商品数据
- 基础推荐链路
- 一个最小 smoke test

## 目录结构

```text
notebook_agent/
├── app/
│   ├── __init__.py
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

## 运行方式

```bash
pip install -r notebook_agent/requirements.txt
uvicorn notebook_agent.app.main:app --reload
```

接口：

- `GET /health`
- `POST /recommend`

## 当前工作流

1. `parse_query`
2. `check_missing_info`
3. `retrieve_products`
4. `score_candidates`
5. `generate_answer`

## 下一步建议

- 把 `query_text` 解析升级为 LLM + schema 抽取
- 增加澄清问题分支
- 增加 benchmark
- 引入真实商品数据
- 将评分拆分为硬约束和软约束
