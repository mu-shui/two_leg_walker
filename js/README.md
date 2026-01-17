# Walker2d 强化学习课程设计环境

本项目用于“基于强化学习的二足行走稳定控制”课设实验。**按 AGENTS 要求仅执行 MLP 实验**，LSTM 不运行。

## 1. 课程要求汇总（来自 Word）

### 1.1 实验目标（assignment.docx）
- 在连续控制任务中训练智能体，实现二足稳定行走。
- 训练智能体协调髋/膝等关节运动策略，使系统快速、平稳行走。
- 对比不同 Q 函数逼近模型性能（如 MLP vs LSTM）。

### 1.2 关键步骤（assignment.docx）
- 训练环境：Gym/Gymnasium 的 MuJoCo `Walker2d-v4`（或等效版本）。
- 状态空间：关节角度、角速度、末端执行器位置。
- 动作空间：连续关节扭矩。
- 奖励函数：行走速度正奖励 + 能量消耗/跌倒惩罚；需平衡速度与稳定性。
- 随机扰动：加入地面摩擦随机变化增强鲁棒性。
- 算法实现：PyTorch + DQN 变体（DDPG/PPO 等）；包含经验回放与目标网络。
- 模型结构：MLP（示例 256-128）/ LSTM（序列依赖）。
- 超参数建议：学习率 1e-4、γ=0.99、batch=64、训练步数 ≥ 1e6。
- 对比与消融：MLP vs LSTM；移除目标网络观察稳定性变化。
- 图示：报告中需包含“二足行走系统示意图”。

### 1.3 预期成果（assignment.docx）
- 训练可视化：奖励曲线（每 1000 步平均奖励）。
- 控制效果：行走视频/轨迹图，标注稳定行走持续时间。
- 性能分析表：收敛步数、最终奖励、波动方差等。
- 改进建议：基于实验结果给出优化方向（示例：注意力机制）。

### 1.4 报告提交与评分（课程设计报告要求及评分规则.docx）
- 报告需写明软件/库版本与硬件环境。
- **3 月 15 日前**提交：纸质版（班长统一交）+ 电子版（学习通）。
- 训练过程与成功动作视频需上传学习通。
- 评分要点：格式规范、算法描述清晰且有文献引用、实验方案与流程结构图、每步结果图表与说明、不同算法对比分析、问题/解决方案/心得体会。

### 1.5 答辩评分（课程设计报告要求及评分规则.docx / 课程设计答辩评分.docx）
- PPT 版面美观、内容完整、逻辑性强（20%）。
- 任务完成度高，展示课程设计过程、组员参与度与实验效果（50%）。
- 呈现形式多样（图表、视频等）（10%）。
- 着装得体、讲述清楚、问答准确（20%）。

### 1.6 选题要求（课程设计要求.txt）
- 题目 1~3：个人独立完成。
- 题目 4~8：小组合作完成，并进行 PPT 汇报答辩。

### 1.7 本项目执行范围（AGENTS 约束）
- **只跑 MLP**（DDPG + MLP critic），忽略 LSTM。
- 报告中如需 MLP vs LSTM 对比，请注明本项目限制或由组员补充。

## 2. 实验步骤（按顺序执行）

### 2.1 环境搭建
PowerShell（Windows）：
```powershell
.\scripts\setup_env.ps1 -VenvPath .\.venv
```
Linux/macOS：
```bash
bash scripts/setup_env.sh
source ./.venv/bin/activate
```

### 2.2 MuJoCo 自检（不含训练）
```powershell
.\.venv\Scripts\python.exe scripts\mujoco_smoke_test.py
.\.venv\Scripts\python.exe scripts\mujoco_smoke_test.py --render human --steps 200
.\.venv\Scripts\python.exe scripts\mujoco_smoke_test.py --render human --steps 200 --ignore-done
.\.venv\Scripts\python.exe src\check_env.py --steps 200 --seed 42
```

### 2.3 训练（MLP + 随机摩擦）
> 满足课程要求：训练步数 **≥ 1e6**。
```powershell
.\.venv\Scripts\python.exe src\train.py --critic-arch mlp --seed 42 --total-steps 1000000 --random-friction --visualize-after --record-video
```
说明：
- `--visualize-after`：保存 `reward_curve.png`，并弹出 MuJoCo 展示。
- `--record-video`：保存 mp4 视频（`logs/<run_id>/videos/`）。

### 2.4 评估（数值指标）
```powershell
.\.venv\Scripts\python.exe src\eval.py --model checkpoints\ddpg_mlp_xxx --episodes 20 --seed 123
```

### 2.5 可视化与展示
- 奖励曲线：训练时加 `--visualize-after` 自动生成；或手动：
  ```powershell
  .\.venv\Scripts\python.exe src\plot.py --logdir logs\ddpg_mlp_xxx
  ```
- MuJoCo 展示：评估时加 `--render`：
  ```powershell
  .\.venv\Scripts\python.exe src\eval.py --model checkpoints\ddpg_mlp_xxx --episodes 3 --seed 123 --render
  ```
- 演示视频：训练时加 `--record-video` 自动保存 mp4。

### 2.6 结果整理与报告
整理奖励曲线、评估指标、演示视频、对比表与改进建议，写入报告与答辩 PPT。

## 3. 输出目录约定
- 训练日志：`logs/<run_id>/`
  - `config.json`、`metrics.csv`、`reward_curve.png`
  - `render_eval.json`、`video_eval.json`
  - `videos/`（mp4）
- 模型权重：`checkpoints/<run_id>/final.pt`

## 4. 待办清单（做完打勾）
- [x] 环境搭建（.venv）
- [x] MuJoCo 自检
- [x] MLP 训练（≥1e6 步，随机摩擦）
- [x] 评估（20 episodes）
- [x] 奖励曲线与可视化
- [x] 演示视频保存
- [ ] 报告与答辩材料整理

## 5. PPT/报告完善 TODO（按学术汇报顺序，逐步完善 report.md）
- [x] 实验目标（来自 assignment.docx）：掌握强化学习在连续控制中实现二足稳定行走，并对比不同 Q 函数逼近模型性能（MLP vs LSTM，本项目仅 MLP）。
- [x] 二足系统介绍：加入“图1 二足行走系统示意图”与“图2 结构/关节示意图”；**补充二足系统理论知识介绍（需后续联网查找资料，此处先标注指令）**。
- [x] 任务与环境：Walker2d-v4、状态/动作空间、奖励构成、回合上限（1000 步≈8 秒）；并在此页解释术语（状态/动作/奖励、回合/步、终止/截断、随机摩擦、seed）。
- [x] 环境与版本（PPT具体要写/画）：硬件配置表（CPU/GPU/内存/系统）；软件版本表（Python/PyTorch/Gymnasium/MuJoCo/NumPy）；环境参数简述（env_id、max_episode_steps、dt/frame_skip）。
- [x] 方法与模型（PPT大纲要点）：算法选择（**TD3 为主**，DDPG/PPO 作为对比，连续动作 RL；依据现有评估 TD3 最稳定）；**算法流程**（交互采样→回放池→双 Q 评估→目标策略平滑→延迟更新→软更新目标网络）；网络结构（Actor/Critic：MLP 256-128 + ReLU，输出 tanh 缩放到动作范围）；关键机制（经验回放、目标网络软更新）；限制说明（仅 MLP，LSTM 未做）；配图建议（TD3 训练流程图/时序箭头图）。
- [x] 训练与评估设置（PPT大纲要点）：训练预算（≥1e6 步，随机摩擦开启）；关键超参表（actor_lr/critic_lr=1e-4，gamma=0.99，batch=64，replay=1e6，tau=0.005，noise_sigma=0.1，可备注 eval_every=50k）；评估协议（确定性策略，20 回合，seed=123，固定摩擦为主/可补随机摩擦对比）；汇报指标（mean_reward/std、mean_len、mean_speed）。
- [x] 结果展示（可视化，PPT具体要点）：放 2-3 张奖励曲线（每张标注 run_id + algo + 是否随机摩擦）；在图下写 1 行结论（如“TD3 曲线更平稳/收敛更快”）；可加同一坐标系的多算法对比小图；来源路径标注 `logs/<run_id>/reward_curve.png`。
- [x] 控制效果展示：视频截图/关键帧（稳定行走 vs 摔倒）并标注现象。
- [x] 性能对比表（PPT具体要点/操作）：做 1 张主表（行=算法 DDPG/PPO/TD3；列=固定摩擦/随机摩擦两组指标），每组指标列包含 mean_reward、std、mean_len、mean_speed；数据来源用 `result.md` 的评估汇总或各 `checkpoints/<run_id>/eval_metrics.json`；在表下注明评估协议（20 回合、seed=123、确定性策略）；如版面紧张，可拆成两张表（固定摩擦一张、随机摩擦一张）。
- [x] 收敛与稳定性分析（PPT具体要点/操作）：定义“收敛判据”（例如 eval_len_mean≥300 且 eval_reward_mean≥500，连续 2 次满足）；从 `logs/<run_id>/eval.json` 或 `result.md` 提取“首次满足的 step”作为收敛步数；最终奖励取最后一次评估的 mean_reward；波动方差=std^2（来自 eval 的 std_reward）；用一张小表展示 3 算法的收敛步数/最终奖励/方差，并在旁边写一句解读（如“TD3 收敛更稳、方差更小”）。
- [ ] 消融/对比说明：MLP vs LSTM（注明本项目限制，仅 MLP）；可选“移除目标网络”讨论。大体写一下框架，因为我还没有做对比和消融实验。
- [x] 问题-原因-改进建议（PPT具体要点/操作）：用“问题→证据→原因→改进”四列表或四条 bullet。示例问题：① 随机摩擦下回合长度偏短、单回合易摔；② 训练奖励波动大、收敛慢；③ 前倾速度快但稳定性差。证据可用 mean_len 低、std 高、视频摔倒帧截图。原因：随机摩擦扰动大、探索噪声不衰减、奖励偏速度/缺少姿态约束、训练上限 1000 步导致长时稳定性欠拟合。改进：减小或退火噪声、调大 batch/延长训练、加入姿态惩罚或能耗权重、引入注意力/时序建模（若允许）、增加鲁棒评估（随机摩擦评估）。
- [ ] 结论与展望：总结最优算法与后续改进方向
- [ ] 致谢：课题名称、成员/分工、课程信息。
