# result.md

## 环境搭建
- 使用 `scripts\setup_env.ps1` 配合 `.`\.venv` 运行；依赖已满足。

## MuJoCo 自检（步骤 2）
- `scripts\\mujoco_smoke_test.py`（默认）：OK（50 步），total_reward=6.16
- `scripts\\mujoco_smoke_test.py --render human --steps 200`：OK，total_reward=6.16
- `scripts\\mujoco_smoke_test.py --render human --steps 200 --ignore-done`：OK，total_reward=234.56
- `src\\check_env.py --steps 200 --seed 42`：mean_reward=0.243，episodes=9

## 实验设计（步骤 3，仅 MLP）
- **算法**：DDPG（自定义实现）。
- **模型结构**：MLP critic（256-128 隐层），MLP actor（256-128）。
- **训练设置**：steps ≥ 1e6，batch=64，gamma=0.99，actor_lr=1e-4，critic_lr=1e-4，replay=1e6，目标网络更新 tau=0.005，noise_sigma=0.1。
- **随机扰动**：训练阶段开启随机摩擦。
- **评估**：20 回合，确定性策略，报告均值/标准差与平均回合长度。
- **消融**：关闭随机摩擦（同样 seed/步数），对比奖励曲线与最终均值。
- **可选敏感性**：更小 MLP（128-64）作为轻量基线。

## 步骤 4 检查（可复现/日志）
- `src/train.py` 会将 `config.json` 写入日志与模型目录（超参、seed、run_id）。
- `metrics.csv` 记录回合奖励/长度与损失曲线。
- 当 `--eval-every` > 0 时生成 `eval.json`。
- 每 100k 步保存 checkpoint，结束时保存 `final.pt`。

## CUDA 安装（venv）
- 安装 `torchvision-0.24.1+cu126`（PyTorch CUDA 12.6 源）。
- venv 内 `torch` 已存在（2.9.1）。
- 安装后检查：`torch.cuda.is_available()` 为 `True`（device_count=1，RTX 3050 Laptop GPU）。

## 训练命令更新（MLP）
- 在 `run_commands.txt` 中加入 MLP 训练命令（使用 `.venv`，未执行）。
- 通过 `--plot-after` 集成绘图；训练后保存 `reward_curve.png` 到 run 目录。
- `--visualize-after` 用于训练后 MuJoCo 展示并保存 `render_eval.json`。
- `--record-video` 保存演示视频到 `logs/<run_id>/videos/` 并写入 `video_eval.json`。
- 安装 MoviePy + imageio-ffmpeg 以支持视频保存。
- 重写 `README.md`，覆盖 Word 要求（目标/步骤/成果/报告/答辩/选题）并结构化。
- `run_commands.txt` 训练步数更新为 `--total-steps 1000000`（长跑，耗时与硬件相关）。
- 添加 20 万步短跑演示命令（低噪声、无随机摩擦）与评估/绘图占位。
- 添加 2 万步快速前进演示命令（更高 lr、低噪声、无随机摩擦）。
- 为 `run_commands.txt` 与 `AGENTS.md` 添加行内注释。

## 文档更新（Venv 路径）
- 将文档中的 `VENV_PATH` 替换为直接 `.\\.venv` 路径。

## 烟雾测试（非最终）
- 训练日志：`logs\ddpg_mlp_seed42_20260115_120357`
- 模型权重：`checkpoints\ddpg_mlp_seed42_20260115_120357\final.pt`
- 曲线图：`logs\ddpg_mlp_seed42_20260115_120357\reward_curve.png`
- 评估（2 回合）：mean_reward=4.69，std=0.42，mean_len=114.0

## 最终结果
- 仍在完善（MLP 正式实验待完成）。

## 运行分析（ddpg_mlp_seed42_20260115_203637）
- 配置：1,000,000 步、随机摩擦开启、noise_sigma=0.1、lr=1e-4。
- 评估（无随机摩擦，20 次评估点）：最终 eval @1e6 步 → mean_reward=1003.93，mean_len=439.0，mean_speed=1.232。
- 渲染/视频（随机摩擦，1 回合）：render_len=308，video_len=435；单回合存在前向摔倒的可能。
- 训练尾部（最后 200 回合）：mean ep_len≈339.8，mean ep_reward≈806.2，max ep_len=1000（存在完整回合）。

## 第一次 1e6 实验摔跤原因分析（重点）
- 第一次完整 1,000,000 步训练已能前进但稳定性不足：平均回合长度 ~339（最后 200 回合均值），远低于 1000 上限，“走着走着摔”属正常现象。
- **随机摩擦训练 + 单回合展示** 放大不稳定性：单回合易抽到“更难摩擦”导致前向摔倒。
- **探索噪声较大且不衰减**（noise_sigma=0.1, noise_decay=0）：策略波动大，步态更易前倾过冲。
- **学习率偏保守**（1e-4）：学习更稳但收敛慢，1e6 步仍未完全稳定。
- 数据印证：eval 的 mean_speed≈1.23 说明已能前进，但 mean_len≈439 表明仍会中途失衡；视频/渲染回合 308/435 是“前进但不够稳”的表现。

## 网页核查：摩擦范围与鲁棒性
- Walker2d-v4 左右脚基础摩擦不对称（左脚 1.9、右脚 0.9），v5 才改为相同摩擦；因此 v4 即便不随机摩擦也存在左右脚差异。
- Gymnasium 文档未给出“随机摩擦范围”推荐值；当前范围为自定义设定，范围越大鲁棒性越强但更难收敛。

## 相关算法说明（非本次作业实现）
- TD3 是对 DDPG 的改进：双 Q、延迟更新、目标策略平滑；官方文档认为其能显著提升稳定性与表现。本作业仍以 DDPG 为主，此处仅作背景说明。

## TD3 支持（保留 DDPG，TD3 作为可选）
- 已新增 TD3 实现与训练入口（`--algo td3`），不影响 DDPG 的主流程。

## 长回合支持
- 训练/评估/渲染/视频支持 `--max-episode-steps`，可突破默认 1000 步上限（仍会因跌倒提前结束）。

## 评估视频支持
- `src/eval.py` 支持 `--record-video` / `--video-episodes` / `--video-dir` 保存评估视频。
- `run_commands.txt` 中已加入 TD3 长回合评估+视频示例。

## 评估指标导出
- `src/eval.py` 会在评估后写出 `eval_metrics.json`，并可同时渲染+录制视频。

## PPO 命令与实现
- `run_commands.txt` 已加入 PPO 训练命令，且已实现 PPO。
- 新增 PPO（高斯策略、价值网络、GAE、clip）实现：`src\algos\ppo.py`。
- `src\train.py` 已接入 `--algo ppo` 与 PPO 专属超参（rollout 长度、epochs、minibatch、clip 比例、entropy/value 系数）。
- `src\eval.py` 支持评估 PPO checkpoint（高斯策略确定性输出）。
- `run_commands.txt` 已加入 PPO 一体命令（训练+评估+绘图+视频）。

## PPO 长视频补录（无需重训）
- 命令：`.\.venv\Scripts\python.exe src\eval.py --model "checkpoints\ppo_mlp_seed42_20260116_115827" --record-video --video-episodes 5 --max-episode-steps 1500 --random-friction --video-dir "logs\ppo_mlp_seed42_20260116_115827\videos_long"`
- 评估指标（5 回合，随机摩擦，max_episode_steps=1500）：mean_reward=1107.54，std=518.53，mean_len=414.0，mean_speed=1.619，speed_std=0.260。
- 评估输出：`checkpoints\ppo_mlp_seed42_20260116_115827\eval_metrics.json`
- 视频位置：`checkpoints\ppo_mlp_seed42_20260116_115827\logs\ppo_mlp_seed42_20260116_115827\videos_long\eval-episode-0.mp4`（共 5 段）

## 三算法回合长度对比（最后 200 回合，metrics.csv）
### DDPG：`logs\ddpg_mlp_seed42_20260115_203637`
- ep_len：均值 344.5，中位数 100.5；P10/P25/P75/P90=20.9/50.8/950.2/980.1；min/max=1/1000。
- ep_reward：均值 818.0，中位数 169.2。
- eval@1e6：eval_len_mean=439.0，eval_speed_mean=1.232。

### PPO：`logs\ppo_mlp_seed42_20260116_115827`
- ep_len：均值 367.0，中位数 309.5；P10/P25/P75/P90=153.6/222.8/451.0/698.0；min/max=24/1000。
- ep_reward：均值 1092.8，中位数 953.9。
- eval@1e6：eval_len_mean=546.4，eval_speed_mean=1.766。

### TD3：`logs\td3_mlp_seed42_20260116_001048`
- ep_len：均值/中位数 229.5；P10/P25/P75/P90=149.9/179.8/279.2/309.1；min/max=130/329。
- ep_reward：均值 754.8，中位数 750.8。
- eval@1e6：eval_len_mean=1000.0，eval_speed_mean=3.139。

## 收敛/合格判据与统计（宽松标准，尽量覆盖更多实验）
**合格判据（固定摩擦评估，连续 2 次）：**
- eval_len_mean ≥ 300
- eval_speed_mean > 0
- eval_reward_mean ≥ 500
- 收敛步数 = 满足连续 2 次条件的第一条 eval step
- 波动方差 = eval_reward_std^2

| run_id | eval_random_friction | 收敛步数 | 最终 eval_len_mean | 最终 eval_speed_mean | 最终 eval_reward_mean | 最终 eval_reward_std | 奖励方差 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ddpg_mlp_seed42_20260115_203637 | False | 700000 | 439.0 | 1.232 | 1003.93 | 442.06 | 195417.83 |
| ppo_mlp_seed42_20260116_115827 | False | 200000 | 546.4 | 1.766 | 1499.98 | 604.50 | 365417.61 |
| td3_mlp_seed42_20260116_001048 | False | 750000 | 1000.0 | 3.139 | 4135.44 | 81.18 | 6590.83 |

## 收敛统计脚本
- `scripts\compute_convergence.py`：从 `logs/<run_id>/eval.json` 自动计算收敛步数、最终指标与奖励方差。
- 示例：
  - `.\.venv\Scripts\python.exe scripts\compute_convergence.py --logdir logs --runs ddpg_mlp_seed42_20260115_203637 ppo_mlp_seed42_20260116_115827 td3_mlp_seed42_20260116_001048`

## 鲁棒合格标准（8s 目标）
- 随机摩擦评估：`--eval-random-friction`
- 连续 3 次评估满足 `eval_len_mean ≥ 1000`（≈8 秒）
- 该标准需要重新评估（当前三个 run 的 eval_random_friction 均为 False）

## PPO 调整（基于公开实现/文档）
- 参考 PPO 官方实现与常用超参设置，加入 **target_kl 早停** 与 **value loss clipping**；并为 PPO 设定单独学习率（actor=3e-4，critic=1e-3），GAE λ 默认 0.97。
- 代码更新：
  - `src\train.py`：新增 `--ppo-actor-lr` / `--ppo-critic-lr` / `--ppo-target-kl` / `--ppo-clip-vloss`，并将 PPO 默认 GAE λ 调为 0.97。
  - `src\algos\ppo.py`：支持 value clipping 与 target_kl 早停，更新 `update()` 签名以使用旧值做裁剪。
- `run_commands.txt`：PPO 训练与一体命令已追加上述稳定性调参参数。
- 追加 PPO 鲁棒优先 1e6 命令（更长 rollout、降低 epochs、增大 minibatch、加熵系数、评估也随机摩擦），用于冲 8 秒目标。
- 追加 PPO 鲁棒优先一体命令（训练+评估+绘图+视频），评估启用随机摩擦。

## PPO 鲁棒优先 1e6 结果（ppo_mlp_seed42_20260116_155452）
- 训练设置要点：随机摩擦训练 + **eval_random_friction=true**，ppo_steps=4096，ppo_epochs=5，minibatch=128，entropy_coef=0.01。
- eval@1e6（随机摩擦）：mean_reward=226.79，std=47.12，mean_len=130.0，mean_speed=0.808。
- render_eval：mean_len=86.0，mean_speed=0.930。
- video_eval（5 回合）：mean_len=94.8，mean_speed=0.853。
- 训练尾部（最后 200 回合）：ep_len 均值 129.6，中位数 111，P10/P90=73/204，min/max=28/498；ep_reward 均值 169.95。
- 结论：在随机摩擦评估下仍偏短且不稳定，未达到 8s 目标。

## 文档中文化
- 将之前英文撰写的文档内容统一改为中文（AGENTS、run_commands、status、result、README）。

## PPT/报告完善
- 已在 `report.md` 补写“实验目标”，内容基于 assignment.docx，包含稳定行走控制与 Q 函数逼近模型对比（MLP vs LSTM，受限仅 MLP）。
- 已在 `report.md` 补写“任务与环境”，覆盖 Walker2d-v4、状态/动作空间、奖励构成与 1000 步≈8 秒回合上限。
- 已细化“任务与环境”条目：补充观测/动作空间维度与含义、奖励公式、终止/截断条件与默认步长。
- 已新增“本实验代码对应与操作要点”，覆盖随机摩擦封装、训练/评估流程、可视化与日志输出路径。
- 已移除独立参考链接小节，改为在回答中提供来源说明。
- 已将“本实验代码对应”合并到“任务与环境”，补充动作空间/奖励函数在代码中的具体实现方式。
- 已将观测空间 17 维说明整理为表格并插入报告。

## 评估备注：td3_mlp_seed42_20260116_001048（2 秒摔倒案例）
- 使用 `--episodes 1 --seed 123 --max-episode-steps 2000 --record-video --render` 的评估结果为 `mean_len=362`（见 `checkpoints\td3_mlp_seed42_20260116_001048\eval_metrics.json`），说明是跌倒导致结束，而非步数上限。
- 单回合 + 不同 seed 会带来波动；建议多回合评估或使用训练评估 seed（10042）。

## 长视频提示
- 为了更容易录到 >8 秒，请使用训练评估 seed（10042）。`run_commands.txt` 内已有对应命令。

## 报告说明：为什么二足会前倾（结合本实验现象）
- **奖励函数偏向速度**：正奖励主要来自前进速度，模型会通过前倾重心提升速度，只要不触发健康终止就不会被惩罚。
- **缺少“直立惩罚”**：环境未显式惩罚轻微前倾，仅在角度/高度超限时终止。
- **回合长度分布外**：训练默认上限 1000 步（≈8 秒），评估拉到 2000 步后进入训练未覆盖的时间段，前倾累积更明显。
- **随机摩擦/噪声影响**：摩擦变化与探索噪声会降低稳定性，使前倾更容易演化为摔倒。

## 报告说明：经验回放与目标网络（已实现）
- **经验回放（Experience Replay）**：将 `(obs, action, reward, next_obs, done)` 写入回放池，随机采样更新以打破时序相关性并提高样本利用率。实现：`src\utils\replay_buffer.py`；训练调用：`src\train.py` 的 `buffer.add()` / `buffer.sample()`。
- **目标网络（Target Network）**：维护 Actor/Critic 的目标网络并使用软更新（Polyak 平均）稳定目标值。DDPG 在 `src\algos\ddpg.py` 中维护 `actor_target` / `critic_target`；TD3 在 `src\algos\td3.py` 中维护 `actor_target` / `critic1_target` / `critic2_target`。

## 运行分析（td3_mlp_seed42_20260116_001048）
- 配置：TD3，1,000,000 步，随机摩擦开启，noise_sigma=0.1，policy_noise=0.2，lr=1e-4。
- 评估（无随机摩擦）：最终 eval @1e6 步 → mean_reward=4135.44，mean_len=1000.0，mean_speed=3.139。
- 视频（随机摩擦，1 回合）：video_len=1000，video_speed≈3.02（视频显示稳定前进）。
- 训练尾部（最后 200 回合）：mean ep_len≈229.5，mean ep_reward≈754.8，max ep_len=329 → 训练回合更短，主要受探索噪声+随机摩擦影响。

## 报告要点：为什么 DDPG 不稳、TD3 更稳（基于两次 1e6 结果）
### 实验对照（同样 1e6 步、随机摩擦）
- **DDPG**（ddpg_mlp_seed42_20260115_203637）
  - 评估（固定摩擦）：mean_len=439，mean_speed≈1.23，mean_reward≈1004
  - 视频（随机摩擦，1 回合）：video_len=435，video_speed≈1.14
  - 训练末尾 200 回合：mean ep_len≈339，max ep_len=1000（偶有完整回合，但整体不稳）
- **TD3**（td3_mlp_seed42_20260116_001048）
  - 评估（固定摩擦）：mean_len=1000，mean_speed≈3.14，mean_reward≈4135
  - 视频（随机摩擦，1 回合）：video_len=1000，video_speed≈3.02
  - 训练末尾 200 回合：mean ep_len≈229，max ep_len=329（训练仍带探索噪声）

### 结论与原因解释
- **DDPG 不稳**：平均回合长度显著低于 1000，波动较大，单回合易摔。
- **TD3 更稳**：评估与视频均能跑满 1000 步，速度显著更高，策略质量更好。
- **算法层面原因**：TD3 通过双 Q、延迟更新、目标策略平滑缓解过估计和策略抖动，稳定性更强。
- **实验层面原因**：相同训练预算下，TD3 达到满回合而 DDPG 未达到，体现 TD3 优势。

### 需要强调的细节（写入报告）
- Walker2d-v4 单回合上限是 **1000 步 ≈ 8 秒**。
  - TD3 的 8 秒“跑满”说明达到了回合上限，并非摔倒。
  - DDPG 平均回合长度明显小于 1000，稳定性不足。

## 参数/指标检查（走路时间与速度）
- 走路时间可量化：训练记录 `ep_len`（`logs/<run_id>/metrics.csv`），评估输出 `mean_len`。
- 速度已显式记录：评估与训练评估/渲染/视频中使用 `info["x_velocity"]`；
  - `src/eval.py` 输出 `mean_speed` / `speed_std`；
  - `src/train.py` 写入 `eval_speed_*` / `render_speed_*` / `video_speed_*`。
- 录视频在 `--video-episodes 1` 时只输出单个文件（避免额外空片段）。
- 确定性：`set_seed()` 自动设置 `CUBLAS_WORKSPACE_CONFIG=":4096:8"`，避免 CUDA 确定性警告。

## 网页核查：参数是否“在合理范围内”
- 参考 OpenAI Spinning Up DDPG 默认参数（gamma=0.99、polyak=0.995 ⇒ tau=0.005、start_steps=10k、update_after=1k、act_noise=0.1、replay_size=1e6、batch=100、lr=1e-3）；当前设置与大多数一致，但 lr 更小（1e-4）且 batch=64。
- Walker2d-v4 文档显示奖励包含前进速度（dx/dt）和健康/控制成本，回合上限为 1000 步；因此“走路时间”对应 mean_len，“速度”对应前进速度奖励。

## 2026-01-16 评估汇总（20 回合，seed=123，固定摩擦）
- 评估命令：`.\\.venv\\Scripts\\python.exe src\\eval.py --model checkpoints\\<run_id> --episodes 20 --seed 123`
- ddpg_mlp_seed42_20260115_203637：mean_reward=956.89，std=730.61，mean_len=420.8，mean_speed=1.151，speed_std=0.316
- ppo_mlp_seed42_20260116_115827：mean_reward=1755.69，std=699.61，mean_len=707.5，mean_speed=1.544，speed_std=0.400
- ppo_mlp_seed42_20260116_155452：mean_reward=220.89，std=21.30，mean_len=114.8，mean_speed=0.934，speed_std=0.069
- td3_mlp_seed42_20260116_001048：mean_reward=3685.84，std=862.01，mean_len=912.0，mean_speed=3.019，speed_std=0.170
- 评估输出：各 `checkpoints\<run_id>\eval_metrics.json` 已更新（覆盖为最新 20 回合结果）。
- 备注：Gymnasium 提示 Walker2d-v4 已过期（建议 v5），但本次评估仍按课程要求使用 v4。
