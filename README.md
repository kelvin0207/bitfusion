# bitfusion

Simulator for BitFusion

See bitfusion-generate-graphs.ipynb for details on how to use the simulator.

## Motivation

该项目的注释少，很难看懂，因此我们希望梳理一下主要的模拟流程与实现方式。

## BitFusion 模拟器核心流程说明

BitFusion 模拟器通过参数化配置、增量计算和模块化设计，实现了对加速器硬件（如 systolic 阵列）执行神经网络任务的全流程模拟。其核心目标是高效计算不同硬件配置下的性能指标（如执行周期、内存访问量等），为硬件设计优化提供数据支持。


### 核心流程总览

模拟器从配置到输出结果的完整流程分为 **6 个关键阶段**，形成“配置→检查→模拟→计算→收集”的闭环。流程如下：

```
配置解析 → 重复计算检查 → 参数扫描 → 基准测试执行 → 核心周期计算 → 结果持久化
```


### 阶段1：配置解析与模拟器初始化

**目标**：加载硬件参数，初始化模拟器核心对象。

- **配置文件**：通过 `.ini` 配置文件（如 `bf_s_conf.ini`）定义硬件特性，包括：
  - 计算单元：systolic 阵列尺寸（`N` 行 × `M` 列）、支持的精度范围（`pmax` 最大位宽、`pmin` 最小位宽）。
  - 存储系统：SRAM 大小（WBUF/IBUF/OBUF）、内存接口带宽（`mem_if_width`）。
  - 其他：频率、批量大小（`batch_size`）等。

- **初始化代码**：
  ```python
  from bitfusion.src.simulator.simulator import Simulator

  config_file = "bf_s_conf.ini"
  sim = Simulator(config_file, verbose=False)  # 实例化模拟器
  ```

- **内部逻辑**：
  - 解析配置文件，创建 `Accelerator` 硬件模型，绑定所有参数（如 `sim.accelerator.N` 对应 systolic 阵列行数）。
  - 初始化能量、面积计算模块（后续用于性能评估）。


### 阶段2：重复计算检查（增量计算优化）

**目标**：避免重复模拟相同配置，提升效率。

- **核心函数**：`check_pandas_or_run`，通过以下步骤实现：
  1. 提取当前硬件配置（`N`/`M`/`pmax` 等）和任务参数（`batch_size`），生成查询字典 `ld`。
  2. 检查历史结果文件（如 `bitfusion-eyeriss-sim-sweep.csv`）中是否存在匹配记录：
     - 若存在，直接返回历史结果（`results = lookup_pandas_dataframe(...)`）。
     - 若不存在，触发新的模拟流程。

- **代码示例**：
  ```python
  df = pandas.DataFrame(columns=sim_sweep_columns)  # 结果表结构
  results = check_pandas_or_run(sim, df, "results.csv", batch_size=1)
  ```


### 阶段3：参数扫描（遍历硬件配置空间）

**目标**：批量测试不同硬件配置的性能，生成对比数据。

- **核心类**：`SimulatorSweep`，通过嵌套循环遍历所有参数组合：
  ```python
  from bitfusion.src.simulator.sweep import SimulatorSweep

  sweep = SimulatorSweep("results.csv", "bf_s_conf.ini")
  # 扫描不同 systolic 阵列尺寸和批量大小
  df = sweep.sweep(sim, list_n=[16, 32], list_batch=[1, 2])
  ```

- **内部逻辑**：
  1. 接收参数列表（如 `list_n` 为 `N` 的候选值，`list_batch` 为批量大小候选值）。
  2. 对每个参数组合动态更新 `sim.accelerator` 的配置（如 `sim.accelerator.N = 16`）。
  3. 调用基准测试，收集该配置下的性能数据。


### 阶段4：基准测试执行（加载神经网络任务）

**目标**：在当前硬件配置下运行神经网络模型，测试实际任务性能。

- **核心函数**：`benchmarks.get_bench_numbers`，加载预定义的神经网络（如 AlexNet、VGG）并遍历其层：
  ```python
  nn = benchmarks.get_bench_nn("alexnet")  # 加载AlexNet模型
  stats = benchmarks.get_bench_numbers(nn, sim, batch_size=1)  # 计算各层性能
  ```

- **内部逻辑**：
  1. 从 `graph.op_registry` 提取网络层（如卷积层、全连接层）。
  2. 对每个层调用 `sim.get_cycles(op, batch_size)`，计算执行周期。


### 阶段5：核心计算（获取周期与内存统计）

**目标**：计算单个网络层在当前硬件上的执行周期及资源消耗。

- **核心方法**：`sim.get_cycles(op, batch_size)`，模拟硬件行为并输出统计数据：
  1. 解析层参数（如卷积层的输入尺寸、权重大小、精度需求）。
  2. 模拟数据在 systolic 阵列中的计算流程（计算单元调度、数据复用）。
  3. 模拟内存访问（SRAM 读写延迟、DRAM 带宽限制导致的停顿周期）。
  4. 输出总周期（`total_cycles`）、内存停顿周期（`mem_stall_cycles`）、SRAM/DRAM 访问量等指标。

- **返回结果**：`(stats, layer_info)`，其中 `stats` 包含该层的完整性能数据。

## 我们在这里主要分析如何得到模拟结果，核心是通过调用 `get_conv_cycles` 函数实现

Step0 打包参数与配置

Step1  确定分块（Tiling）维度的候选数量

```python
num_O_tiles = int(math.ceil(log2(O))) + 1  # 输出特征图分块数
num_IC_tiles = int(math.ceil(log2(IC))) + 1  # 输入通道分块数
num_OC_tiles = int(math.ceil(log2(math.ceil(float(OC)/self.accelerator.M)))) + 1  # 输出通道分块数
num_B_tiles = int(math.ceil(log2(B))) + 1  # 批次分块数
```

Step2 优化分块（Tiling）与循环顺序（Ordering）
```python
best_instructions, best_tiling, best_order = optimize_for_order(conv_params)
```

这是核心优化步骤，通过 exhaustive search（穷举搜索）寻找最优的：
- best_tiling：各维度的分块大小（如输入通道分块 T_IC、输出通道分块 T_OC 等），需适配硬件 SRAM 容量和 systolic 阵列尺寸。
- best_order：循环执行顺序（如 B→IC→OC→O→K），决定数据复用效率（如优先复用输入特征图还是权重，减少内存访问）。

优化目标是最小化总周期数，因为分块和顺序直接影响数据在 SRAM 中的复用率（复用率高则 DRAM 访问少，周期少）。

Step3 计算性能统计信息

```python
stats = get_stats_fast(conv_params, best_tiling, best_order, verbose=False)
```

根据最优分块和循环顺序，计算卷积层的详细性能指标：
- stats.total_cycles：总执行周期数（核心输出）。
- stats.reads/writes：各内存（SRAM 的 WBUF/IBUF/OBUF、DRAM）的读写次数（反映内存访问效率）。
- stats.mem_stall_cycles：内存访问延迟导致的停顿周期（硬件带宽不足时产生）。

Step4 整理并输出结果

重点在 `optimize.py` 文件下面的两个函数 `optimize_for_order` 与 `get_stats_fast`：

`optimize_for_order` 函数是 BitFusion 模拟器中循环顺序与分块策略的核心优化器，其目标是通过遍历所有可能的循环执行顺序，找到能最小化卷积层执行周期（并兼顾能量消耗）的最优方案。

`optimize_for_order` 的主要执行逻辑是，先定义循环维度（计算维度）并生成所有可能的循环顺序，然后绑定子函数`_optimize_for_order`，准备并行计算。子函数四重循环，将批次B、输出宽度高度OW/OH、输入通道IC和输出通道OC进行分块，然后把分块策略与执行顺序输入`get_stats_fast`得到周期。

`get_stats_fast` 是核心函数，根据给定的卷积参数、分块策略和循环顺序，快速估算卷积层的执行周期、内存访问次数及能量消耗。

### 最核心的模拟部分

在 `optimizer.py` 的 `get_stats_fast` 下面，本质是下面一段很简单的代码，核心是分离 “无法重叠的初始 / 最终访存” 和 “可与计算重叠的中间访存”，再评估两者与计算的匹配度：

Step1 区分 DRAM 访存阶段，计算启动 / 结束延迟
```python
# 根据分块策略与读写策略，计算总周期数和能量消耗
# 总周期数 = 计算周期 + 内存停顿周期

# 初始 DRAM 读取：计算开始前加载初始数据（如首个分块的权重、激活值）的总比特数。
initial_dram_reads = 0 
# 最终 DRAM 写入（final_dram_writes）：计算结束后写回最终结果的总比特数。
final_dram_writes = 0
# namespace 包含weight，activation, output，进行累加
for namespace in max_write_size:
    initial_dram_reads += max_write_size[namespace]
for namespace in max_read_size:
    final_dram_writes += max_read_size[namespace]

# latency 这两部分访存是 “串行启动 / 收尾操作”，
# 无法与计算重叠，直接用硬件内存接口的读写延迟函数计算，且只累加一次。
latency = acc_obj.get_mem_read_cycles('dram', initial_dram_reads) + \
        acc_obj.get_mem_write_cycles('dram', final_dram_writes)
```

Step2 计算中间 DRAM 访存量（可与计算重叠）

``` python
# total_dram_accesses 所有分块的读写总量（stats.reads['dram'] + stats.writes['dram']）
# middle_dram_accesses 减去初始和最终访存，得到计算过程中（分块迭代时）的动态访存总量 
# 这部分可通过双缓冲等技术与计算并行（如计算当前分块时，预加载下一分块数据）。
# 因此需要减去initial和final的
total_dram_accesses = stats.reads['dram'] + stats.writes['dram']
middle_dram_accesses = total_dram_accesses - initial_dram_reads - final_dram_writes
```

Step3 计算纯计算周期

```python
# acc_obj.get_compute_cycles()输入分块的配置，估算出每个块计算的时间，然后乘上分块个数就行
compute_cycles = num_tiles * acc_obj.get_compute_cycles(ic, oc, ow, oh, b, kw, kh, iprec, wprec, im2col)
```

Step4 评估计算与访存的重叠度，计算停顿周期
内存所需周期: 中间 DRAM 访问量 ÷ 内存接口带宽（acc_obj.mem_if_width，向上取整），即传输中间数据需要的总周期。
内存访问的时间其实很简单：规模/带宽

```python
memory_cycles_required = ceil_a_by_b(middle_dram_accesses, acc_obj.mem_if_width)
# 检查内存访问是否可以被计算掩盖
memory_stalls = max(0, memory_cycles_required - compute_cycles) + latency
stats.total_cycles = compute_cycles + memory_stalls
stats.mem_stall_cycles = memory_stalls
```


### 阶段6：结果收集与持久化

**目标**：汇总所有配置和网络层的结果，生成可复用的结构化数据。

- **核心操作**：
  1. 单个层的结果通过 `data_line.append(...)` 存入列表，包含配置参数（`N`/`M`）和性能指标（周期、内存访问量）。
  2. 批量结果追加到 DataFrame 并写入 CSV 文件：
     ```python
     df = df.append(pandas.DataFrame(data_line, columns=sim_sweep_columns))
     df.to_csv("results.csv", index=False)  # 持久化结果
     ```

- **结果用途**：后续可通过数据分析工具（如 Pandas、Matplotlib）生成性能对比图（如不同 `N` 对应的周期数曲线）。


### 总结

BitFusion 模拟器通过模块化设计实现了“硬件配置→任务执行→性能评估”的全流程自动化，核心优势在于：
1. **灵活性**：通过配置文件和参数列表快速调整硬件特性。
2. **效率**：增量计算避免重复模拟，适合大规模参数扫描。
3. **可扩展性**：支持新增神经网络模型或硬件组件（如自定义存储层次）。

该流程可直接复用为其他计算任务（如 SpGEMM 稀疏矩阵乘法）的模拟器框架，只需替换阶段4-5的“神经网络层计算”为目标任务的执行逻辑即可。