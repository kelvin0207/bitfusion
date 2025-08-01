import pandas
import os
import logging

from bitfusion.src.simulator.simulator import Simulator
from bitfusion.src.utils.utils import lookup_pandas_dataframe
import bitfusion.src.benchmarks.benchmarks as benchmarks

class SimulatorSweep(object):
    def __init__(self, csv_filename, config_file='conf.ini', verbose=False):
        self.sim_obj = Simulator(config_file, verbose=False)
        self.csv_filename = csv_filename
        if verbose:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        self.logger = logging.getLogger('{}.{}'.format(__name__, 'Simulator'))
        self.logger.setLevel(log_level)

        self.columns = ['N', 'M', 'Max Precision (bits)', 'Min Precision (bits)',
                'Network', 'Layer',
                'Cycles', 'Memory wait cycles',
                'WBUF Read', 'WBUF Write',
                'OBUF Read', 'OBUF Write',
                'IBUF Read', 'IBUF Write',
                'DRAM Read', 'DRAM Write',
                'Bandwidth (bits/cycle)',
                'WBUF Size (bits)', 'OBUF Size (bits)', 'IBUF Size (bits)',
                'Batch size']

        if os.path.exists(csv_filename):
            self.sweep_df = pandas.read_csv(csv_filename)
        else:
            self.sweep_df = pandas.DataFrame(columns=self.columns)

    def sweep(self, sim_obj, list_n=None, list_m=None, list_pmax=None, list_pmin=None, list_bw=None, list_bench=None, list_wbuf=None, list_ibuf=None, list_obuf=None, list_batch=None):
        """
        Sweep the parameters of the accelerator
        """

        if list_n is None:
            list_n = [sim_obj.accelerator.N]
        if list_m is None:
            list_m = [sim_obj.accelerator.M]

        if list_pmax is None:
            list_pmax = [sim_obj.accelerator.pmax]
        if list_pmin is None:
            list_pmin = [sim_obj.accelerator.pmin]

        if list_bw is None:
            list_bw = [sim_obj.accelerator.mem_if_width]

        if list_bench is None:
            list_bench = benchmarks.benchlist

        if list_wbuf is None:
            list_wbuf = [sim_obj.accelerator.sram['wgt']]
        if list_ibuf is None:
            list_ibuf = [sim_obj.accelerator.sram['act']]
        if list_obuf is None:
            list_obuf = [sim_obj.accelerator.sram['out']]

        if list_batch is None:
            list_batch = [1]

        data_line = []
        # Step1 使用嵌套循环遍历所有参数组合
        for batch_size in list_batch:
            for n in list_n:
                for m in list_m:
                    # self.logger.info('N x M = {} x {}'.format(n, m))
                    for pmax in list_pmax:
                        for pmin in list_pmin:
                            if pmin > pmax:
                                continue
                            for wbuf in list_wbuf:
                                for ibuf in list_ibuf:
                                    for obuf in list_obuf:
                                        for bw in list_bw:
                                            # Step2 动态修改模拟器配置
                                            sim_obj.accelerator.N = n
                                            sim_obj.accelerator.M = m
                                            sim_obj.accelerator.pmax = pmax
                                            sim_obj.accelerator.pmin = pmin
                                            sim_obj.accelerator.mem_if_width = bw
                                            sim_obj.accelerator.sram['wgt'] = wbuf
                                            sim_obj.accelerator.sram['out'] = obuf
                                            sim_obj.accelerator.sram['act'] = ibuf
                                            # Step3 使用lookup_pandas_dataframe检查当前参数组合是否已被模拟过
                                            for b in list_bench:
                                                lookup_dict = {}
                                                lookup_dict['N'] = n
                                                lookup_dict['M'] = m
                                                lookup_dict['Max Precision (bits)'] = pmax
                                                lookup_dict['Min Precision (bits)'] = pmin
                                                lookup_dict['Network'] = b
                                                lookup_dict['Bandwidth (bits/cycle)'] = sim_obj.accelerator.mem_if_width
                                                lookup_dict['WBUF Size (bits)'] = wbuf
                                                lookup_dict['OBUF Size (bits)'] = obuf
                                                lookup_dict['IBUF Size (bits)'] = ibuf
                                                lookup_dict['Batch size'] = batch_size
                                                results = lookup_pandas_dataframe(self.sweep_df, lookup_dict)
                                                # 拿神经网络的配置数据用来 benchmark
                                                nn = benchmarks.get_bench_nn(b, WRPN=True)
                                                if len(results) == 0:
                                                    # 新的模拟，否则跳过
                                                    self.logger.info('Simulating Benchmark: {}'.format(b))
                                                    self.logger.info('N x M = {} x {}'.format(n, m))
                                                    self.logger.info('Max Precision (bits): {}'.format(pmax))
                                                    self.logger.info('Min Precision (bits): {}'.format(pmin))
                                                    self.logger.info('Batch size: {}'.format(batch_size))
                                                    self.logger.info('Bandwidth (bits/cycle): {}'.format(bw))
                                                    # Step4 模拟结果统计
                                                    # 通过get_bench_numbers中的sim_obj.get_cycles()得到周期数据
                                                    stats = benchmarks.get_bench_numbers(nn, sim_obj, batch_size)
                                                    for layer in stats:
                                                        cycles = stats[layer].total_cycles
                                                        reads = stats[layer].reads
                                                        writes = stats[layer].writes
                                                        stalls = stats[layer].mem_stall_cycles
                                                        data_line.append((n,m,pmax,pmin,b,layer,
                                                            cycles,stalls,
                                                            reads['wgt'],writes['wgt'],
                                                            reads['out'],writes['out'],
                                                            reads['act'],writes['act'],
                                                            reads['dram'],writes['dram'],
                                                            sim_obj.accelerator.mem_if_width,
                                                            wbuf, obuf, ibuf, batch_size))
                                            if len(data_line) > 0:
                                                if os.path.exists(self.csv_filename):
                                                    self.sweep_df = pandas.read_csv(self.csv_filename)
                                                else:
                                                    self.sweep_df = pandas.DataFrame(columns=self.columns)
                                                self.sweep_df = self.sweep_df.append(pandas.DataFrame(data_line, columns=self.columns))
                                                self.sweep_df.to_csv(self.csv_filename, index=False)
                                                data_line = []
        return self.sweep_df

def check_pandas_or_run(sim, dataframe, sim_sweep_csv, batch_size=1, config_file='./conf.ini'):
    ld = {}
    ld['N'] = sim.accelerator.N
    ld['M'] = sim.accelerator.M
    ld['Max Precision (bits)'] = sim.accelerator.pmax
    ld['Min Precision (bits)'] = sim.accelerator.pmin
    ld['Bandwidth (bits/cycle)'] = sim.accelerator.mem_if_width
    ld['WBUF Size (bits)'] = sim.accelerator.sram['wgt']
    ld['OBUF Size (bits)'] = sim.accelerator.sram['out']
    ld['IBUF Size (bits)'] = sim.accelerator.sram['act']
    ld['Batch size'] = batch_size

    results = lookup_pandas_dataframe(dataframe, ld)

    if len(results) == 0:
        # 未命中，需要模拟
        # 创建SimulatorSweep对象，它负责批量运行模拟器
        sweep_obj = SimulatorSweep(sim_sweep_csv, config_file)
        # 调用sweep方法执行模拟，传入当前模拟器对象和批量大小列表
        dataframe = sweep_obj.sweep(sim, list_batch=[batch_size])
        # 将新的模拟结果保存到 CSV 文件
        dataframe.to_csv(sim_sweep_csv, index=False)
        return lookup_pandas_dataframe(dataframe, ld)
    else:
        return results

