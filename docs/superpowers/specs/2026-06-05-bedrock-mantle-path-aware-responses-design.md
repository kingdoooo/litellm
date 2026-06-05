# bedrock_mantle 路径感知 Responses 路由 + 补齐 `/v1/responses`

状态: 设计已批准, 待实现
日期: 2026-06-05
关联: PR #29490 (已合并到 `litellm_oss_staging_040626`), Nasa62 的 config-driven 诉求
base 分支: `litellm_oss_staging_040626`

## 背景与缘起

这份需求最初来自一份 Feishu 文档, 把它定义为 PR #29490 的一个小 follow-up:
在现有 allow-list gate 基础上, 额外让被标 `mode: responses` 的模型也路由到
`BedrockMantleResponsesAPIConfig`, 以便将来非 OpenAI 的 Responses 模型可以
不改代码只改 config 启用。

在动手前对 AWS 官方文档和 LiteLLM 代码做了查证, 发现原方案存在一个会导致
净负效果的设计错误, 因此需求被重新定向为本文档描述的"路径感知版"。

### 查证得到的事实 (AWS 官方文档)

bedrock-mantle 端点上, OpenAI 兼容流量分三条上游路径:

| 路径 | base URL 形态 | 当前命中的模型 |
|---|---|---|
| `/openai/v1/responses` | `.../openai/v1` | 仅 OpenAI 闭源前沿 gpt 系 (gpt-5.5, gpt-5.4, 未来 gpt-6) |
| `/v1/responses` | `.../v1` | 仅 gpt-oss-120b, gpt-oss-20b |
| `/v1/chat/completions` | `.../v1` | 其余所有 OpenAI 兼容模型 (nvidia, qwen, mistral, google, zai, deepseek, minimax, moonshot, writer 等) |

GPT-5.5 model card 的原文确认了两条 Responses 路径并存:
"This model is available on the `openai/v1/responses` path on the `bedrock-mantle`
endpoint. This is different from the `v1/responses` path used by other models on
the responses endpoint."

全 mantle 范围内支持 Responses 的只有 4 个模型: gpt-5.5 / gpt-5.4 (走
`/openai/v1/responses`), gpt-oss-120b / gpt-oss-20b (走 `/v1/responses`)。
没有任何非 OpenAI 厂商的模型支持 Responses。

### 原方案的设计错误

`BedrockMantleResponsesAPIConfig.get_complete_url` 把 base 剥成 host 后写死
`/openai/v1/responses`。原方案让"被标 `mode: responses` 的模型"路由到这个
config, 意味着除 gpt-5.x 外, 任何被标记的模型都会被送到 `/openai/v1/responses`
然后 400, 包括真正支持 Responses 的 gpt-oss-120b (它的 Responses 在
`/v1/responses`)。

`mode: responses` 只表示"模型支不支持 Responses", 它不决定走哪条上游路径。
路径区分 (`/openai/v1` vs `/v1`) 与 `mode` 正交, 原方案把两者混为一谈。

LiteLLM 当前完全不支持 `/v1/responses`: `get_complete_url` 没有任何分支能
产出这条路径。因此今天通过 LiteLLM 对 gpt-oss 发 Responses 请求, 只能落到
chat-completions 仿真 (单轮无状态可用, 但拿不到 `previous_response_id` 多轮
状态 / background / async / store 这些 Responses 原生特性)。

Nasa62 想要的 "config-driven, 让未来某 lab 的 Responses 模型免改代码启用",
只有在 `/v1/responses` 这条标准路径上才成立。原方案把覆盖钩子接到了
`/openai/v1/responses` 这条 gpt-5.x 特例路径上, 是接错了地方。

## 目标

1. 让 LiteLLM 支持 mantle 的 `/v1/responses` 上游路径。
2. gate 按模型决定走哪条 Responses 路径 (路径感知):
   - gpt-frontier (`openai.gpt-*` 减 `gpt-oss`) -> `/openai/v1/responses`
   - 其它被标 `mode: responses` 的模型 -> `/v1/responses`
   - 其余 -> `None` (chat 仿真, 行为不变)
3. gpt-oss 的原生 `/v1/responses` 是显式 opt-in: 用户配
   `model_info: {mode: responses}` 才启用, 默认仍走仿真。
4. 把 Nasa62 的 config-driven 诉求接到正确的 `/v1/responses` 标准路径上。

## 非目标

- 不改 price-map 让 gpt-oss 默认 advertise responses (那会改变现有 gpt-oss
  用户的默认行为, 范围更大)。gpt-oss 走原生路径必须显式 opt-in。
- 不动 chat-completions 仿真路径本身。
- 不处理 SigV4 (那是另一份独立文档的 follow-up)。

## 设计

### 与现有 provider 实现的一致性

本设计刻意对齐 LiteLLM 既有的两个主流模式, 不另起炉灶:

- gate 按 model 分流返回不同 config: Azure (O-series vs 普通, 返回两个不同类)
  和 Databricks (`if "gpt" in model` 才返回 config, 否则 None) 已经这么做。
  BEDROCK_MANTLE 三分支是这两者的组合。
- config 不写死 URL、路径由外部决定: Azure responses config 的
  `get_complete_url` 调用 `_get_base_azure_url` 从 `api_base` / `litellm_params`
  动态解析 deployment / api-version。本设计用构造参数 `use_openai_path` 注入,
  是同一思路 (可变来源不同)。

与 Azure 的唯一区别: Azure O-series 用两个独立类 (因为参数处理逻辑不同);
mantle 两条路径唯一差别是 URL 前缀一个字符串, 其余行为全相同, 因此用单类 +
构造 flag 而非拆类, 避免一个几乎空的子类违反 DRY。

### 改动 1: gate (`litellm/utils.py`, `_get_python_responses_api_config` BEDROCK_MANTLE 分支)

路径决策放在 gate, 因为 `get_complete_url` (responses 版) 签名只有
`(api_base, litellm_params)`, 不接收 `model`; 而 gate 的
`get_provider_responses_api_config(provider, model)` 有 model。

```python
elif litellm.LlmProviders.BEDROCK_MANTLE == provider:
    model_lower = model.lower() if model else ""
    if "openai.gpt-" in model_lower and "gpt-oss" not in model_lower:
        return litellm.BedrockMantleResponsesAPIConfig(use_openai_path=True)
    if model:
        try:
            if get_model_info(model, "bedrock_mantle").get("mode") == "responses":
                return litellm.BedrockMantleResponsesAPIConfig(use_openai_path=False)
        except Exception:
            pass
    return None
```

要点:
- allow-list (gpt-frontier) 在前且短路: gpt-5.x 永不依赖 cost-map 加载状态,
  合并前后都对 (保留"合并期 bootstrap"), 常见路径零 `get_model_info` 开销。
- `declared_responses` 用 try/except 兜底: 未映射模型抛异常时降级 None, 不崩。
- gpt-oss 默认 (price-map mode=chat) 不会被 declared_responses 抓到, 默认仍
  返回 None 走仿真; 用户显式 `model_info: {mode: responses}` (经 register_model
  写入全局 model_cost) 后才被抓到, 走 `/v1/responses`。

路由对照:

| 模型 | 命中分支 | 返回 | 上游路径 |
|---|---|---|---|
| `openai.gpt-5.5` / `5.4` / 假想 gpt-6 | gpt-frontier | `Config(use_openai_path=True)` | `/openai/v1/responses` |
| `openai.gpt-oss-120b` + `model_info:{mode:responses}` | declared_responses | `Config(use_openai_path=False)` | `/v1/responses` |
| 未来非 OpenAI responses 模型 + `mode:responses` | declared_responses | `Config(use_openai_path=False)` | `/v1/responses` |
| `openai.gpt-oss-120b` (默认) | 无 | `None` | chat 仿真 (不变) |
| nvidia / qwen / mistral / google / zai ... | 无 | `None` | chat 仿真 (不变) |
| `model=None` (by-id 操作) | 无 | `None` | 不变 |

### 改动 2: config (`litellm/llms/bedrock_mantle/responses/transformation.py`)

新增构造参数 (默认 True, 保持现有 gpt-5.x 行为与现有测试不变):

```python
class BedrockMantleResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def __init__(self, use_openai_path: bool = True):
        super().__init__()
        self.use_openai_path = use_openai_path
```

`get_complete_url` 路径拼接按 flag 选前缀 (其余 base 归一化逻辑不变):

```python
        base = base.rstrip("/")
        for suffix in _BASE_SUFFIXES_TO_STRIP:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        path = "/openai/v1/responses" if self.use_openai_path else "/v1/responses"
        return f"{base}{path}"
```

`_BASE_SUFFIXES_TO_STRIP` 已含 `/openai/v1/responses` / `/v1/responses` /
`/responses` / `/openai/v1` / `/v1`, 最长优先匹配, 所以两条路径下用户在
`api_base` 里塞任何路径都会先剥成 host 再按 flag 拼, 不会 double。

`validate_environment` (Bearer 鉴权)、`supports_native_file_search` /
`supports_native_websocket` (都 False) 两条路径共用, 不变。gpt-oss 的
`/v1/responses` 同样是 Bearer、无 native file_search/websocket, 共用正确。

实现期需核实 (不在设计阶段假设): 父类 `OpenAIResponsesAPIConfig` 的 by-id
操作 (GET/DELETE/cancel `/responses/{id}`) 是否也经过这个 `get_complete_url`。
若是, 两条路径的 by-id URL 也随 flag 正确切换; by-id 调 gate 时 model=None ->
None, 不命中此 config。用测试锁定。

## 测试

文件: `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py`
(已存在, 扩展; 复用已有 `TestBedrockMantleResponsesRegistry`、
`TestBedrockMantleResponsesURL`、`local_cost_map` fixture)。

标准: mutation kill > 90%, 每条断言到能区分 feature 做没做的程度。

### A. gate 路由 + 路径选择 (TestBedrockMantleResponsesRegistry)

1. 回归 (allow-list / bootstrap): `gpt-5.5` / `gpt-5.4` / 假想 `gpt-6` ->
   返回 config 且 `use_openai_path is True`。断言到 flag, 不只 isinstance。
2. 回归 (防 regression 重现): `nvidia.nemotron-nano-9b-v2` / `mistral.*` /
   `google.gemma-*` / `zai.glm-*` (默认 mode=chat) -> `None`。
3. 新功能核心: 局部 `register_model({"bedrock_mantle/somelab.future-model":
   {"litellm_provider": "bedrock_mantle", "mode": "responses"}})` 标记非 OpenAI
   模型 -> 返回 config 且 `use_openai_path is False` (走 /v1/responses)。
   feature 未做时必失败 (老代码写死 openai 路径)。fixture teardown 还原。
4. gpt-oss opt-in: `openai.gpt-oss-120b` 默认 (无 register) -> `None` (锁定
   默认仿真); 再 register `mode: responses` -> 返回 config 且
   `use_openai_path is False` (走 /v1/responses 而非 frontier 路径)。
5. 未映射模型 (非 gpt-frontier 且未 register, get_model_info 抛异常) -> 不崩,
   降级 `None`。
6. `model=None` -> `None` (不变)。

### B. URL 构造 (TestBedrockMantleResponsesURL)

7. `use_openai_path=False` -> `get_complete_url` 产出 `.../v1/responses`。覆盖
   三种 base 输入 (默认 env region / 用户传 `.../v1` / 用户传全 endpoint URL),
   都不 double 都落 `/v1/responses`。
8. `use_openai_path=True` (默认) -> 仍产出 `.../openai/v1/responses` (复用现有
   URL 测试, 确认无回归)。
9. 默认构造 (不传参) 即 `use_openai_path=True`, 锁定默认不改变 gpt-5.x 行为。

### C. 共享行为对两路径都成立

10. `use_openai_path=False` 实例: `validate_environment` 仍 Bearer,
    `supports_native_file_search()` / `supports_native_websocket()` 仍 False。

### 测试隔离

涉及 `register_model` 的用例一律走 fixture, teardown 还原
`litellm.model_cost` 并 `litellm.get_model_info.cache_clear()` (get_model_info
有 lru_cache, 不清会跨测试串)。参照现有 `local_cost_map` fixture 的 teardown。

## 验收 (Proof of Fix: 真实 proxy + curl)

正向验收用 gpt-oss-120b (真实支持 /v1/responses)。EC2 起真 proxy, 真打 AWS
Bedrock Mantle, curl 命令 + 输出 (不用 python 脚本 / 不用 mock):

1. config 配 `bedrock_mantle/openai.gpt-oss-120b` 加 `model_info: {mode: responses}`。
2. 起 proxy: `python litellm/proxy/proxy_cli.py --config <cfg> --detailed_debug
   --reload --use_v2_migration_resolver 2>&1 | tee litellm.log`。
3. curl `/v1/responses` 打该模型, 带 `previous_response_id` 做一次多轮 -> 证明
   原生有状态 Responses 跑通 (仿真给不了), 且 litellm.log outbound 命中
   `bedrock-mantle.<region>.api.aws/v1/responses`。
4. 对照: 同模型不配 model_info -> 走仿真, outbound 命中 `/v1/chat/completions`,
   证明 opt-in 语义。
5. gpt-5.x 回归 (若有 quota): `bedrock_mantle/openai.gpt-5.5` -> outbound 仍命中
   `/openai/v1/responses`, 证明 frontier 路径无回归。

真机这步由用户在 EC2 跑 (需 AWS 凭证 / quota); 实现完成后提供可直接粘的命令清单。

## PR 流程

- 分支: 从 `litellm_oss_staging_040626` 切, prefix `litellm_`, 无 `/`、无
  `claude/` (本分支名 `litellm_bedrock_mantle_v1_responses`)。
- base: `litellm_oss_staging_040626` (#29490 在此分支)。
- PR body 用 `.github/pull_request_template.md` 结构; 说明这是路径感知 + 补齐
  `/v1/responses` 版, 并解释为何偏离原文 (两条路径 / mode 不决定路径 / LiteLLM
  缺 /v1/responses); `mode: responses` 覆盖是显式 opt-in, 用户需自行确保上游
  支持。措辞: 无 emoji、无破折号、少列表、不用"不是 X 而是 Y"句式。
- 原方案被推翻这个发现只在新 PR 描述里解释, 不另外去 #29490 评论区同步。
- commit 前: 跑测试、format、lint。无 Claude attribution。

## 相关文件清单

- `litellm/utils.py` — `_get_python_responses_api_config` BEDROCK_MANTLE 分支 (gate, 主改动)
- `litellm/llms/bedrock_mantle/responses/transformation.py` — config 加 `use_openai_path` + URL 拼接
- `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py` — 测试 (扩展)
