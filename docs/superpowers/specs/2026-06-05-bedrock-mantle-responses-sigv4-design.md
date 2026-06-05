# 设计:为 bedrock_mantle Responses 路由添加 SigV4 / IAM Role 认证

- **关联 issue**:#29665(以及更早的 #29463)
- **前置 PR**:#29490(已合并,新增 `/openai/v1/responses` 路由,仅 Bearer token 认证)
- **base 分支**:`litellm_oss_staging_040626`(见下方「分支基线」一节,**与原始需求文档写的 `litellm_internal_staging` 不同**)
- **工作分支**:`litellm_bedrock_mantle_responses_sigv4`(从 `origin/litellm_oss_staging_040626` 切,HEAD `141a61b8f8`)

---

## 1. 背景与目标

PR #29490 为 Amazon Bedrock Mantle 的 OpenAI Responses 路由(`/openai/v1/responses`,目标模型
`openai.gpt-5.5` / `openai.gpt-5.4`)增加了支持,但只实现了 Bearer token 认证;SigV4 / IAM Role 认证被明确标为
out-of-scope。

社区 issue #29665 指出:很多企业部署(EKS IRSA、EKS Pod Identity、EC2 instance role)的安全策略强制要求用
IAM Role(即 SigV4)认证,禁止长期静态密钥。当前的三层认证链 `api_key` /
`BEDROCK_MANTLE_API_KEY` / `AWS_BEARER_TOKEN_BEDROCK` 全是静态 bearer 凭证,无法满足这类部署。

已实测确认(us-east-2,真实 IAM 凭证):`bedrock-mantle.{region}.api.aws` 的 `/openai/v1/responses`
路径完整支持并校验 SigV4(service name = `bedrock`)。所以上游能力没问题,缺的只是 LiteLLM 侧的 SigV4 签名代码路径。

**目标**:在 `BedrockMantleResponsesAPIConfig`
(`litellm/llms/bedrock_mantle/responses/transformation.py`)上增加 SigV4 / IAM Role 认证作为一个可选项,
与现有 Bearer token 路径并存。当提供了 AWS 凭证(role / access key / profile / web identity 等)且没有
bearer key 时走 SigV4;否则保持现有 Bearer 行为。

**不改动**:routing(PR #29490 已做:只有 `openai.gpt-` 系且非 gpt-oss 走 Responses)、price-map、URL 构造、
请求/响应 transform、streaming 解析、file_search emulation。

---

## 2. 分支基线(决策记录)

原始需求文档写 base = `litellm_internal_staging`。核验发现 **PR #29490 的 Responses 路由代码不在该分支**:

- PR #29490 实际合并进 `litellm_oss_staging_040626`(2026-06-04 合并,merge commit `066f978`)。
- `litellm_internal_staging` 里**没有** `bedrock_mantle/responses/` 目录,也没有 `BedrockMantleResponsesAPIConfig`
  —— 本需求要改的主文件根本不存在。
- 两个 staging 分支已分叉(internal 有 6 个 oss 没有的提交,oss 有 32 个 internal 没有的提交)。

**决策(已与需求方确认):从 `litellm_oss_staging_040626` 切分支,PR base 也设为它。** 这同时满足「从最新 staging 切」
和「前提代码必须存在」两个约束。

---

## 3. 现状代码核验(本设计的事实依据)

以下均已对照 `litellm_oss_staging_040626` 真实代码核验:

| 事实 | 位置 | 结论 |
|---|---|---|
| Responses handler 调用顺序 | `custom_httpx/llm_http_handler.py` `response_api_handler` / `async_response_api_handler` | `validate_environment`(拿不到 body)→ `get_complete_url` → `transform_responses_api_request` → `normalize_responses_api_request_dict` → `data.update(extra_body)` → `post(json=data)` |
| Responses 两条 post 点都是裸 `json=data` | 同上(sync 行 ~2340/2372,async 同结构) | **没有** signed-body 路径,这是核心缺口 |
| chat/embedding 路径**已有**成熟 signed-body 模式 | `llm_http_handler.py:896–956` | `headers, signed_body = provider_config.sign_request(...)` + `if signed_body is not None: post(data=signed_body) else: post(...)`;注释明确写「默认 `BaseConfig.sign_request` 返回 `(headers, None)` 是 no-op」 |
| `BaseResponsesAPIConfig` **没有** `sign_request` | `base_llm/responses/transformation.py`(320 行) | 但已有多个具体默认方法(`supports_native_file_search`、`supports_native_websocket`、`normalize_responses_api_request_dict`),加默认 `sign_request` 是同款增量模式 |
| 现有 15 个 responses config **无一**定义 `sign_request` | 全 `litellm/llms/**/responses/transformation.py` grep = 0 | 加默认 no-op 对它们零影响 |
| 签名内核现成 | `bedrock/base_aws_llm.py:1464 _sign_request`、`:196 get_credentials`、`:584 _get_aws_region_name` | bearer-first 再 SigV4;返回 `(headers, Optional[bytes])`;覆盖 role/AssumeRole/STS/profile/web-identity/access-key + 凭证缓存 |
| Mythos 路由的 SigV4 接入 | `bedrock/chat/mantle/transformation.py` → `AmazonAnthropicClaudeConfig(AmazonInvokeConfig, AnthropicConfig)` → `AmazonInvokeConfig(BaseConfig, BaseAWSLLM)` | `sign_request`(`base_invoke_transformation.py:114`)是一行直通转发到 `_sign_request(service_name="bedrock", ...)` |

---

## 4. 复用映射(与已有 AnthropicClaudeConfig / Mythos 的关系)

Mythos(`AmazonMantleConfig`)之所以「本来就是 SigV4」,靠两件事:多继承 `BaseAWSLLM` 拿到 `_sign_request`,
以及多继承 `BaseConfig`(chat 基类)让 chat handler 会调 `sign_request`。

**Responses config 走的是另一套基类(`BaseResponsesAPIConfig`)和另一个 handler 分支**,所以不能照搬 Mythos 的类继承
(会触发 MRO 冲突,把 Anthropic Messages 格式整套拖进来,与 OpenAI Responses spec 冲突)。正确做法是:
**复用签名内核,复刻接入模式,但不复用类继承。**

| 层次 | Mythos 做法 | 本设计 | 复用程度 |
|---|---|---|---|
| 签名内核 `_sign_request` / `get_credentials` | 多继承 `BaseAWSLLM` 拿到 | 用**组合**持有 `BaseAWSLLM` 实例来调它 | ✅ **复用同一个函数**(SigV4 行为天然与 Mythos 一致:同端点、同 `service_name="bedrock"`、同凭证链) |
| `sign_request` 钩子转发 | `base_invoke_transformation.py:114` 一行直通 | Mantle Responses config 写一个**几乎逐字相同**的 `sign_request`,同参数、同转发 | ✅ **逐字复刻** |
| handler 调用 + signed_body 透传 | chat handler 已有 | 给 responses handler 两条 post 点补 hook | ⚠️ responses handler 没有,需补;但写法**复刻** chat 路径 `if signed_body is not None: post(data=signed_body)` |

三层里两层是复用/逐字复刻已验证代码,真正「新写」的只有 responses handler 那两条 post 点的增量(且照抄 chat 写法)。

---

## 5. 架构总览(组合,不多继承)

`BedrockMantleResponsesAPIConfig` 继续继承 `OpenAIResponsesAPIConfig`(不动)。SigV4 能力通过**组合**注入:
config 内部持有一个轻量 `BaseAWSLLM` 实例,复用其 `_sign_request` / `get_credentials` / `_get_aws_region_name`,
**完全不改 `base_aws_llm.py`**。

数据流(签名发生在 body 定型之后,从根本上解决「`validate_environment` 拿不到 body」的核心难点):

```
validate_environment ──► 只决定 "用哪种 auth",不再硬塞最终头(也不再因无 bearer 直接抛错)
get_complete_url     ──► .../openai/v1/responses (不变)
transform_request    ──► 产出 data
normalize + extra_body ─► data 最终定型           ← 签名必须在这之后
sign_request (新钩子) ──► (signed_headers, signed_body_bytes)
post(data=signed_body) 或 post(json=data)         ← signed_body 非 None 即用前者
```

---

## 6. 实现方案(方案 A:给 Responses handler 增加 signed-body 能力,对齐 chat 路径)

分三层落地:

### 6.1 基类层 — `BaseResponsesAPIConfig` 加默认 `sign_request`
在 `base_llm/responses/transformation.py` 加一个**具体**(非抽象)方法,签名与 chat 基类
(`base_llm/chat/transformation.py:286`)完全一致:

```python
def sign_request(
    self, headers, optional_params, request_data, api_base,
    api_key=None, model=None, stream=None, fake_stream=None,
) -> Tuple[dict, Optional[bytes]]:
    return headers, None  # 默认 no-op,对所有现有 responses provider 零影响
```

### 6.2 handler 层 — sync + async 两条 responses 路径都改(覆盖 streaming)
在 `data` 定型(transform + normalize + extra_body)之后、`post` 之前,插入:

```python
headers, signed_body = responses_api_provider_config.sign_request(
    headers=headers, optional_params=dict(litellm_params), request_data=data,
    api_base=api_base, model=model,
    stream=stream, fake_stream=fake_stream,
)
```

然后把 sync 的两个 post(stream / 非 stream)和 async 的两个 post 统一改为:
`signed_body is not None` → `post(url=api_base, headers=headers, data=signed_body, ...)`;
否则维持 `post(url=api_base, headers=headers, json=data, ...)`。写法复刻 `llm_http_handler.py:943–956`。

> streaming 也覆盖:SigV4 只签**请求 body**(请求 body 在 stream/非 stream 下都是定型的;只有响应才流式),
> 所以两条路径签名逻辑相同,只是 post 参数透传点不同。

### 6.3 Mantle 层 — `BedrockMantleResponsesAPIConfig` 覆写 `sign_request`
逐字对标 `AmazonInvokeConfig.sign_request`(`base_invoke_transformation.py:114`),委托给组合持有的
`BaseAWSLLM` 实例:

```python
def sign_request(self, headers, optional_params, request_data, api_base,
                 api_key=None, model=None, stream=None, fake_stream=None):
    return self._aws_signer._sign_request(
        service_name="bedrock",
        headers=headers, optional_params=optional_params,
        request_data=request_data, api_base=api_base,
        api_key=api_key, model=model, stream=stream, fake_stream=fake_stream,
    )
```

返回的 `signed_body` 是 `_sign_request` 产出的**同一份 body bytes**,handler 直接 `post(data=signed_body)`,
杜绝二次序列化导致 hash 漂移。

---

## 7. 认证选择逻辑(两种 auth 共存)

把优先级**收敛进 `_sign_request` 自带的 bearer-first 逻辑**(它第一步就是「有 bearer 用 bearer,否则 SigV4」),
避免两处重复:

1. `validate_environment` 退化为**只解析 bearer**(`api_key` / `BEDROCK_MANTLE_API_KEY` /
   `AWS_BEARER_TOKEN_BEDROCK`)。有 bearer → 设 Bearer 头并把它作为 `api_key` 透传给后续 `sign_request`;
   **无 bearer 时不再抛错**(改由 `sign_request` 阶段决定)。
2. `sign_request` 阶段:`api_key` 有值 → `_sign_request` 走 Bearer(行为与现状完全一致);否则从
   `litellm_params` / 环境 / 默认凭证链解析 `aws_*` 凭证 → SigV4。
3. 都拿不到 → `_sign_request` 内 `get_credentials` 报错;Mantle config 负责把错误文案更新为**同时提示 bearer 和 IAM
   两条路**。

region 解析复用 `BaseAWSLLM._get_aws_region_name(optional_params)`(`aws_region_name` →
`AWS_REGION_NAME` → `AWS_REGION`),与 Mantle 现有 `get_complete_url` 的 region 来源保持一致。

---

## 8. 影响面 / 风险评估

**对现有 Bedrock 路由与 LLM 调用功能:无影响。** 依据:

- **路由决策在改动点之前**:走哪个 model family、哪个 path 由 PR #29490 的 registry 门控 + `get_complete_url`
  决定,本改动只发生在 `data` 定型后、`post` 前,碰不到路由。
- **Mantle 四条 path 互不交叉**,各自 config 基类独立:`/v1/chat/completions`(`BedrockMantleChatConfig`)、
  `/anthropic/v1/messages`(`AmazonMantleConfig`,本来就是 SigV4)、`/openai/v1/responses`(本次目标)。只动第三条
  config + 共享 handler 的 responses 分支。
- **纯增量、默认 no-op**:基类 `sign_request` 默认返回 `(headers, None)`;现有 15 个 responses config 无一覆写,
  全部命中 no-op → 走原 `post(json=data)`,逐字节与今天一致。受影响的只有覆写了 `sign_request` 的 Mantle Responses
  一个 config。
- **不碰** transform / streaming 解析 / file_search emulation / price-map / URL 构造。

**唯一新增风险**:SigV4 路径自身的 body 字节一致性(签名 hash 的字节必须 = 实际发送字节)。由 §9 测试第 4 条专门锁死,
并通过「`sign_request` 返回同一份 `_sign_request` 产出的 bytes、handler `post(data=signed_body)`」从实现上规避二次序列化。

---

## 9. 测试要求(扩展现有文件,mutation kill rate > 90%)

文件:`tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py`(已有 283 行,
**扩展不新建**)。单元测试用依赖注入 mock 凭证,不 monkeypatch 类属性。每条都应在功能未实现 / 被 mutate 时失败:

1. **bearer 短路**:提供 bearer key → header 是 `Authorization: Bearer <key>`,不触发 SigV4(断言 `get_credentials`
   未被调用)。
2. **access key SigV4**:无 bearer + `aws_access_key_id` + secret(+ 可选 session_token)→ 产生 SigV4 头
   (`Authorization: AWS4-HMAC-SHA256 ...`、`X-Amz-Date`、可选 `X-Amz-Security-Token`),且 service=`bedrock`、
   region 正确。
3. **AssumeRole SigV4**:无 bearer + `aws_role_name`(+ session_name)→ 走 AssumeRole 路径(mock STS)→ SigV4 头。
4. **body 字节一致性(核心回归)**:签名覆盖的 body 与最终发送的 body 字节一致(防止 transform / extra_body /
   normalize 之后 body 变化导致 hash 不匹配)。最能体现核心难点是否真正解决。
5. **region 解析优先级**:`litellm_params.aws_region_name` > 环境 > 默认;URL 仍为 `.../openai/v1/responses`。
6. **双缺报错**:两种 auth 都缺失时报错,且文案同时提示 bearer 和 IAM。
7. **共享 handler 回归(护栏)**:一个 `sign_request` 返回 `None` 的普通 responses provider,handler 仍走
   `json=data`(保护既有 OpenAI / 其他 responses provider 不被破坏)。

---

## 10. 验收(Proof of Fix:真实 proxy + curl,非 pytest 截图)

在 EC2(IAM Role 环境,us-east-2)起 proxy,config 配 `bedrock_mantle/openai.gpt-5.5` 且**不设任何 bearer key**
(仅靠 instance role / IRSA):

- curl proxy 的 `/v1/responses` → 期望 200,debug 日志显示 outbound 打到 `.../openai/v1/responses` 且认证用
  SigV4(无 `Authorization: Bearer`)。
- 对照:仍提供 bearer key 时,同样 200 且用 Bearer 头。

---

## 11. 分支 / PR / 安全约束

- 工作分支 `litellm_bedrock_mantle_responses_sigv4`,从 `origin/litellm_oss_staging_040626` 切;命名 prefix
  `litellm_`、无 `/`、无 `claude/`。
- **PR base = `litellm_oss_staging_040626`**(⚠️ 与原始文档写的 `litellm_internal_staging` 不同,理由见 §2)。
- PR 描述参考 `.github/pull_request_template.md`;关联 issue #29665 / #29463;Type = 🆕 New Feature。
- 全程不在代码 / commit / PR / issue 里出现任何真实 key、access key、secret、role ARN 明文。

---

## 12. 相关文件清单

| 文件 | 角色 |
|---|---|
| `litellm/llms/bedrock_mantle/responses/transformation.py` | 主改动:加 SigV4 `sign_request`(组合持有 `BaseAWSLLM`)+ `validate_environment` 退化 |
| `litellm/llms/base_llm/responses/transformation.py` | 加默认 `sign_request -> (headers, None)` |
| `litellm/llms/custom_httpx/llm_http_handler.py` | responses sync + async 两条路径加 signed-body hook |
| `litellm/llms/bedrock/base_aws_llm.py` | 复用 `_sign_request` / `get_credentials` / `_get_aws_region_name`(只读参考,不改) |
| `litellm/llms/bedrock/chat/mantle/transformation.py` + `bedrock/chat/invoke_transformations/base_invoke_transformation.py` | Mythos / `AmazonInvokeConfig.sign_request` —— SigV4 复用与逐字复刻的范例(只读参考) |
| `tests/test_litellm/llms/bedrock_mantle/test_bedrock_mantle_responses_transformation.py` | 扩展测试(§9 的 7 条) |
