# 多 Agent 是否值得：MATH-500 对照实验报告

日期：2026-11-14 至 2026-11-15。服务：SiliconFlow。模型：`Qwen/Qwen2.5-7B-Instruct`（小模型）、`deepseek-ai/DeepSeek-V4.1-Flash`（强模型）。温度 0，单次输出上限 4096 token（旧流程为 2048）。

## 一句话结论

1. 旧版多 Agent 在 MATH-500 上只有 30%，原因是流程设计，不是模型能力：改掉之后同一模型达到 84%。
2. 同一个模型扮演多个角色，相对"单次分步推理"几乎没有增益（+3 题 / 100，在抽样误差内）。
3. 小模型 + 强模型裁判可以到 96%，但消耗的强模型 token 比直接让强模型做全部题还多，准确率也不更高（强模型单次推理 97%）。
4. MATH-500 对强模型已接近饱和，这个基准无法体现多 Agent 的价值。

## 实验设置

- **主样本**：从 MATH-500 中排除此前用过的 24 题后随机抽取 100 题（种子 20261115），文件 `data/math500-fresh100.jsonl`。设计流程时没有查看过这些题。
- **辅助样本**：`data/math500-validation20.jsonl`，20 题。新旧流程都跑过，但它的失败记录在设计新流程时被查看过，只用于新旧对比，不作为主要依据。
- **评分**：Math-Verify 对照标准答案判断等价；标准答案不进入任何 Agent 的提示词。API 失败的题计入分母。
- **流程**（`src/math_agents/`）：
  - `baseline`（`workflow.py`）：规划者 → 求解者 → finalizer → 验证者 → 评审者，全部 JSON 输出。
  - `optimized`（`optimized.py`）：去掉 finalizer，答案取自代码输出，仍是 JSON 输出加代码验证者。
  - `consensus`（`consensus.py`）：推理者（分步推理）+ 程序员（写并执行 Python/SymPy）；两者答案等价则采纳，否则交给裁判，三者取多数。

## 结果一：旧流程为什么只有 30%

验证 20 题，全部使用 Qwen2.5-7B：

| 流程 / 模式 | 正确 | API 异常 | 平均模型调用 |
|---|---:|---:|---:|
| baseline direct（只回 JSON 答案） | 2/20 (10%) | 1 | 1 |
| baseline single（工具） | 9/20 (45%) | 1 | 2.95 |
| baseline multi | 6/20 (30%) | 0 | 13.0 |
| optimized multi | 6/20 (30%) | 2 | 6.2 |
| consensus direct（分步推理） | 13/20 (65%) | 0 | 1 |
| consensus single（程序） | 11/20 (55%) | 0 | 1.2 |
| consensus multi | 13/20 (65%) | 0 | 2.7 |

从失败记录中找到的原因：

| 原因 | 证据 |
|---|---|
| 所有 Agent 只能回一个 JSON 答案，没有推理空间 | baseline direct 在 GSM8K 20 题全错、MATH-500 10%；改成自由文本分步推理后同一模型单次调用即 65% |
| 规划者心算出错，求解者照抄 | 求解"代码"为 `print('2403_6')`、`print(2 + 3 + 14)` 之类 |
| 验证者写的 PASS/FAIL 检查比求解者更容易错，却有否决权 | optimized 20 题中 13 题用满 3 轮；`geometry/172` 第 2 轮得到正确答案 8，被否决后第 3 轮改成 12 |
| API 超时整题作废 | 每 20 题有 1–3 题因 ReadTimeout 记零分，已算出的答案丢失 |
| 输出处理过严 | 空输出、`58.99999999999818`、未化简根式被判错 |

## 结果二：同模型多角色的增益

主样本 100 题，全部使用 Qwen2.5-7B，consensus multi：

| 运行 | 最终正确 | 推理者单独 | 程序员单独 | 进入裁判 | 平均调用 |
|---|---:|---:|---:|---:|---:|
| 第一次 | 82 | 80 | 56 | 48 | 2.77 |
| 第二次（沙箱输出加入化简后） | 84 | 81 | 63 | 39 | 2.65 |

第一次运行的分解：

| 情况 | 题数 |
|---|---:|
| 推理者对 | 80 |
| 推理者错、程序员对 | 5 |
| 两者都错 | 15（裁判救回 1 题） |

- 即使仲裁完美，上限也只有约 86%。三个角色是同一个模型，会犯同样的错；多角色只能纠正偶然的计算失误。
- 多 Agent 的实际用处是置信度信号：两条路线答案一致的 52 题中 51 题正确（98%），不一致的 48 题中 31 题正确（65%）。
- 两次运行相差 2 题，说明温度 0 下远程模型仍有波动，2–3 题的差异不能当作结论。

## 结果三：加入强模型

主样本同一批 100 题：

| 配置 | 正确 | 强模型输入 / 输出 token | 平均调用 | 平均延迟 |
|---|---:|---:|---:|---:|
| 全 7B，consensus multi | 84 | 0 | 2.65 | 44 s |
| 7B 推理+程序，DeepSeek 只当裁判 | 96 | 5.4 万 / 11.3 万 | 2.77 | 44 s |
| 全 DeepSeek，单次推理 | 97 | 2.0 万 / 8.5 万 | 1 | 11 s |
| 全 DeepSeek，consensus multi | 99 | 4.9 万 / 24.8 万 | 2.1 | 45 s |

按难度（每级题数：L1 7，L2 22，L3 26，L4 23，L5 22）：

| 配置 | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|
| 全 7B multi | 7 | 19 | 24 | 20 | 14 |
| 7B + DeepSeek 裁判 | 7 | 22 | 23 | 23 | 21 |
| 全 DeepSeek 单次推理 | 7 | 22 | 25 | 22 | 21 |
| 全 DeepSeek multi | 7 | 22 | 25 | 23 | 22 |

混合配置（7B + DeepSeek 裁判）的分解：

| 情况 | 题数 | 最终正确 |
|---|---:|---:|
| 两个 7B 角色一致，直接采纳 | 56 | 53 |
| 进入裁判，推理者原本就对 | 25 | 24 |
| 进入裁判，只有程序员对 | 6 | 6 |
| 进入裁判，两者都错 | 13 | 13 |

观察：

- **强裁判确实有效**：进入裁判的 44 题对了 43 题，包括 13 道两个小模型都做错的题。7B 当裁判时这类题几乎救不回来。
- **但不省钱**：裁判要读两份解答再重新推导，平均每次输出约 2600 token，而强模型单次推理平均约 850。44 次裁判调用消耗的强模型 token 超过强模型直接做完 100 题。
- **3 道"一致但错"的题无法挽回**：两个小模型给出相同的错误答案时不会触发裁判，而这 3 题强模型单次推理都能做对。
- **只把分歧题交给强模型从头解**（用已有结果离线推算，未实际运行）：约 94%，强模型输出 token 约为全量的 65%。分歧题恰好是最难、输出最长的题，所以省得有限。
- **全强模型多 Agent 的 99 对 97**只差 2 题，在抽样误差内，输出 token 约 2.9 倍，延迟约 4 倍。

## 局限

- 样本为 100 题，95% 置信区间宽度约 ±7 到 ±8 个百分点（84% 对应约 76%–90%）。未跑全量 500 题。
- 每个配置只跑一次（全 7B multi 跑了两次）。
- 没有记录各模型的价格，成本只用 token 数比较。
- GSM8K 没有用 consensus 流程实测。
- 强模型在该基准上已接近满分，无法区分各种多 Agent 设计的优劣；要回答"多 Agent 何时有用"，需要强模型单次推理只有 50%–70% 的更难基准。

## 复现

```bash
./lab setup                                   # 构建镜像、运行测试
./lab run math500 20                          # 默认 consensus，三种模式
./lab run math500 20 optimized                # 第二版
./lab run math500 20 baseline                 # 原版
```

主样本各配置（本机 Python 环境，需先 `export` 配置或加载 `.env`）：

```bash
# 全 7B multi
math-lab run --data data/math500-fresh100.jsonl --modes multi --workers 8 --output results/NEW_DIR
# 7B + 强裁判
MATH_REFEREE_MODEL=deepseek-ai/DeepSeek-V4.1-Flash math-lab run --data data/math500-fresh100.jsonl --modes multi --workers 6 --output results/NEW_DIR
# 全强模型，单次推理与 multi
MATH_MODEL=deepseek-ai/DeepSeek-V4.1-Flash math-lab run --data data/math500-fresh100.jsonl --modes direct multi --workers 6 --output results/NEW_DIR
```

## 原始结果目录

每个目录含 `manifest.json`（参数、提示词、题号、数据哈希）、`results.jsonl`（逐题答案、各 Agent 候选答案、完整对话与工具输出、token 用量）和汇总文件。

| 目录 | 内容 |
|---|---|
| `results/validation20-baseline-math500` | 原版三种模式，20 题 |
| `results/validation20-optimized-math500` | optimized multi，20 题 |
| `results/consensus-validation20-math500` | consensus 三种模式，20 题 |
| `results/consensus-fresh100-math500` | 全 7B multi，100 题，第一次（82） |
| `results/consensus-fresh100-math500-v2` | 全 7B multi，100 题，第二次（84） |
| `results/hybrid-fresh100-math500` | 7B + DeepSeek 裁判，100 题（96） |
| `results/deepseek-direct-fresh100-math500` | 全 DeepSeek，direct（97）与 multi（99），100 题 |

GSM8K 的旧流程结果在 `results/validation20-*-gsm8k`，详见 `VALIDATION_REPORT.md`。
