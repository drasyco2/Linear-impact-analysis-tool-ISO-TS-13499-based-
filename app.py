import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.integrate import cumulative_trapezoid
import math
from numba import jit
from datetime import datetime

# ===================== 滤波函数（内嵌） =====================
def filter_cfc(df, channel, type, append_df=True):
    """CFC filter implementation (4 pole linear phase Butterworth digital filter)."""
    time_axis = df["Time"]
    sampling_rate = time_axis[1] - time_axis[0]
    fCut = type * 2.0775  # Fn frequency based on SAE J211
    if fCut > (0.5 / sampling_rate * 0.775):
        st.error(f"Error: Sampling rate {sampling_rate} is lower than the cutoff frequency")
        return df[channel] if not append_df else df
    if type in [60, 180, 600, 1000]:
        if type == 1000: type_str = "A"
        elif type == 600: type_str = "B"
        elif type == 180: type_str = "C"
        elif type == 60: type_str = "D"
    else:
        st.error(f"Error: Frequency Channel Class {type} is not specified in SAE J211")
        return df[channel] if not append_df else df

    chn = df[channel]
    padding_10ms_incr = int(round(0.01 / sampling_rate))
    pad_start_raw = np.array(chn[:padding_10ms_incr])
    pad_start = (pad_start_raw[::-1] * -1) + 2 * chn[0]
    pad_end_raw = np.array(chn[len(chn) - padding_10ms_incr:])
    pad_end = (pad_end_raw[::-1] * -1) + 2 * (chn[len(chn) - 1])
    channel_padded = np.concatenate([pad_start, chn[1:-1], pad_end])

    input_list = channel_padded
    filtered_list = _butterworth_2pole(input_list, type, sampling_rate)
    filtered_list = np.flip(filtered_list)
    filtered_list = _butterworth_2pole(filtered_list, type, sampling_rate)
    filtered_list = np.flip(filtered_list)

    filtered_list = filtered_list[(padding_10ms_incr - 1):-(len(pad_end_raw) - 1)]

    if append_df:
        new_channel_name = channel[:15] + type_str
        df[new_channel_name] = filtered_list
        return df
    else:
        return filtered_list

@jit(nopython=True, cache=True)
def _butterworth_2pole(input_list, type, sampling_rate):
    pi = math.pi
    cfc = type
    x4 = sampling_rate
    wd = 2 * pi * cfc * 2.0775
    wa = math.sin(wd * x4 / 2) / math.cos(wd * x4 / 2)
    a0 = (wa**2) / (1 + (2**0.5) * wa + wa**2)
    a1 = 2 * a0
    a2 = a0
    b1 = -2 * (wa**2 - 1) / (1 + (2**0.5) * wa + wa**2)
    b2 = (-1 + (2**0.5) * wa - wa**2) / (1 + (2**0.5) * wa + wa**2)
    precision = 8
    filtered_list = np.empty_like(input_list)
    for step, each_value in enumerate(input_list):
        if step == 0 or step == 1:
            filtered_list[step] = round(input_list[step], precision)
        else:
            t = step
            y_t = a0 * input_list[t] + a1 * input_list[t - 1] + a2 * input_list[t - 2] + b1 * filtered_list[t - 1] + b2 * filtered_list[t - 2]
            filtered_list[step] = round(y_t, precision)
    return filtered_list

# ===================== 核心解析抽象逻辑 =====================
def parse_mme_raw_logic(raw_groups):
    all_experiments = {}
    for exp_id, files in raw_groups.items():
        if not files['data_files']:
            continue
        channels_dict = {}
        for f in files['data_files']:
            content = f.getvalue().decode('utf-8', errors='ignore')
            lines = content.splitlines()
            metadata = {}
            data_values = []
            for line in lines:
                line_str = line.strip()
                if not line_str: continue
                if ':' in line_str:
                    parts = line_str.split(':', 1)
                    metadata[parts[0].strip().lower()] = parts[1].strip()
                else:
                    try: data_values.append(float(line_str))
                    except ValueError: pass
            if len(data_values) == 0: continue
            sampling_interval = float(metadata.get('sampling interval', 0.0001))
            time_of_first_sample = float(metadata.get('time of first sample', 0.0))
            data_array = np.array(data_values, dtype=float)
            num_samples = len(data_array)
            time_array = np.linspace(time_of_first_sample, time_of_first_sample + (num_samples * sampling_interval), num_samples, endpoint=False)
            file_suffix = f.name.split('.')[-1]
            raw_channel_meaning = metadata.get('name of the channel', f"Channel_{file_suffix}")
            raw_unit = metadata.get('unit', '未知')
            channels_dict[file_suffix] = {
                'data': data_array, 'time': time_array, 'sampling_interval': sampling_interval,
                'raw_meaning': raw_channel_meaning, 'unit': raw_unit, 'default_name': f"通道 {file_suffix}"
            }
        if files['chn'] and channels_dict:
            chn_lines = files['chn'].getvalue().decode('utf-8', errors='ignore').splitlines()
            sorted_suffixes = sorted(channels_dict.keys())
            chn_names = []
            for i, line in enumerate(chn_lines):
                if i >= 2 and line.strip():
                    if len(line) >= 45: chn_names.append(line[29:45].strip())
                    elif len(line) > 29: chn_names.append(line[29:].strip())
            for idx, suffix in enumerate(sorted_suffixes):
                ch_info = channels_dict[suffix]
                if idx < len(chn_names):
                    ch_info['channel_name'] = f"{chn_names[idx]} ({suffix}) [{ch_info['unit']}]"
                else:
                    ch_info['channel_name'] = f"{ch_info['default_name']} [{ch_info['unit']}]"
        else:
            for suffix, ch_info in channels_dict.items():
                ch_info['channel_name'] = f"{ch_info['raw_meaning'].split('/')[0].strip()} ({suffix}) [{ch_info['unit']}]"
        if channels_dict:
            all_experiments[exp_id] = channels_dict
    return all_experiments

@st.cache_data
def process_multi_experiment_files_tab1(uploaded_files):
    if not uploaded_files: return {}
    raw_groups = {}
    for f in uploaded_files:
        if '.' not in f.name: continue
        base_name, ext = f.name.rsplit('.', 1)
        if ext.lower() == 'chn':
            if base_name not in raw_groups: raw_groups[base_name] = {'chn': None, 'data_files': []}
            raw_groups[base_name]['chn'] = f
        elif ext.lower() == 'mme': continue
        else:
            if base_name not in raw_groups: raw_groups[base_name] = {'chn': None, 'data_files': []}
            raw_groups[base_name]['data_files'].append(f)
    return parse_mme_raw_logic(raw_groups)

@st.cache_data
def process_multi_experiment_files_tab2(uploaded_files):
    if not uploaded_files: return {}
    raw_groups = {}
    for f in uploaded_files:
        if '.' not in f.name: continue
        base_name, ext = f.name.rsplit('.', 1)
        if ext.lower() == 'chn':
            if base_name not in raw_groups: raw_groups[base_name] = {'chn': None, 'data_files': []}
            raw_groups[base_name]['chn'] = f
        elif ext.lower() == 'mme': continue
        else:
            if base_name not in raw_groups: raw_groups[base_name] = {'chn': None, 'data_files': []}
            raw_groups[base_name]['data_files'].append(f)
    return parse_mme_raw_logic(raw_groups)

# ===================== 页面配置 =====================
st.set_page_config(page_title="线性冲击实验基本分析看板", layout="wide")
st.title("🛡️ 线性冲击实验标准分析看板")
st.markdown("以ZF_LF内LIP实验文件夹格式的输入参考对象")

# 建立视窗主框架
main_tab1, main_tab2 = st.tabs(["📊 视窗一：核心吸能对比看板（2x3矩阵）", "📂 视窗二：扩展传感器导入对比"])

# 全局变量占位
html_report_ready = False
full_html_data = ""

# =========================================================================
# 📊 视窗一：核心吸能对比看板
# =========================================================================
with main_tab1:
    st.sidebar.header("📁 视窗一：核心数据导入")
    uploaded_files_tab1 = st.sidebar.file_uploader("导入视窗一的核心分析数据：", accept_multiple_files=True, key="uploader_tab1")
    all_experiments_tab1 = process_multi_experiment_files_tab1(uploaded_files_tab1)

    if not all_experiments_tab1:
        st.info("💡 提示：请在左侧边栏“视窗一”上传核心 MME和通道数据包以激活此看板。")
    else:
        st.markdown("### 🎯 核心分析实验与控制参数设定")
        experiment_names_tab1 = list(all_experiments_tab1.keys())
        selected_exps_tab1 = st.multiselect("1. 选择要对比的实验（可多选）：", options=experiment_names_tab1, default=experiment_names_tab1[:1], key="select_exp_t1")
        
        if not selected_exps_tab1:
            st.warning("请至少选择一个实验。")
        else:
            ref_exp = selected_exps_tab1[0]
            exp_chns = all_experiments_tab1[ref_exp]
            channel_options = {info['channel_name']: suffix for suffix, info in exp_chns.items()}
            ac_defaults = [name for name in channel_options.keys() if "AC" in name.upper()]
            default_ac = ac_defaults[0] if ac_defaults else list(channel_options.keys())[0]

            ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
            with ctrl_col1:
                selected_ac_name = st.selectbox("2. 选择基本加速度通道：", options=list(channel_options.keys()), index=list(channel_options.keys()).index(default_ac), key="ac_select_t1")
                suffix = channel_options[selected_ac_name]
                ch_info_ref = exp_chns[suffix]
            t_raw = ch_info_ref['time']
            t_min, t_max = float(t_raw[0]), float(t_raw[-1])

            with ctrl_col2:
                mass_kg = st.number_input("3. 冲击块质量 (Mass, kg)：", value=6.80, step=0.1, key="mass_t1")
                g_factor = st.number_input("重力加速度系数 (g ➔ m/s²)", value=9.81, step=0.01, key="g_t1")
            with ctrl_col3:
                st.markdown("**4. 各实验初速度 (m/s)**")
                v0_inputs = {}
                v0_cols = st.columns(min(len(selected_exps_tab1), 4))
                for i, exp_name in enumerate(selected_exps_tab1):
                    col = v0_cols[i % 4]
                    with col:
                        v0_inputs[exp_name] = st.number_input(
                            f"{exp_name}",
                            value=3.04,
                            step=0.01,
                            key=f"v0_{exp_name}_tab1"
                        )
                reverse_integration = st.checkbox("碰撞减速脉冲积分 (v0 - ∫a dt)", value=True, key="rev_t1")
            with ctrl_col4:
                filter_option = st.selectbox("5. 加速度滤波 (CFC)：", ["无滤波", "CFC60", "CFC180"], index=0, key="filt_t1")
                tick_step = st.selectbox("6. X轴刻度间隔：", ["自动", "10 ms", "20 ms", "50 ms", "100 ms"], index=0, key="tick_t1")
                tick_step_map = {"自动": None, "10 ms": 0.01, "20 ms": 0.02, "50 ms": 0.05, "100 ms": 0.1}
                dtick = tick_step_map[tick_step]

            st.markdown("#### ⏱️ 曲线裁剪区间控制 (Crop Range)")
            crop_col1, crop_col2 = st.columns(2)
            with crop_col1:
                crop_start = st.number_input("裁剪起点时间 (Start Time, 秒 s)：", value=max(0.0, t_min), min_value=t_min, max_value=t_max, step=0.001, format="%.4f", key="start_t1")
            with crop_col2:
                crop_end = st.number_input("裁剪终点时间 (End Time, 秒 s)：", value=min(0.150, t_max), min_value=t_min, max_value=t_max, step=0.001, format="%.4f", key="end_t1")
            
            if crop_start >= crop_end:
                st.error("❌ 错误：起点时间必须小于终点时间！")
            else:
                color_cycle = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
                exp_color_map = {exp_name: color_cycle[i % len(color_cycle)] for i, exp_name in enumerate(selected_exps_tab1)}

                # ========== 物理计算（独立循环，不包含 UI） ==========
                results = {}
                for exp_name in selected_exps_tab1:
                    exp_data = all_experiments_tab1[exp_name]
                    if suffix not in exp_data: continue
                    ch_info = exp_data[suffix]
                    t_raw_exp = ch_info['time']
                    acc_raw_exp = ch_info['data']
                    dt_exp = ch_info['sampling_interval']

                    start_idx = np.abs(t_raw_exp - crop_start).argmin()
                    end_idx = np.abs(t_raw_exp - crop_end).argmin() + 1
                    time_cropped = t_raw_exp[start_idx:end_idx]
                    acc_cropped = acc_raw_exp[start_idx:end_idx]

                    if filter_option != "无滤波":
                        df_temp = pd.DataFrame({"Time": time_cropped, "acc": acc_cropped})
                        filt_type = 60 if filter_option == "CFC60" else 180
                        filtered = filter_cfc(df_temp, "acc", filt_type, append_df=False)
                        if filtered is not None: acc_cropped = np.array(filtered)

                    acc_m_s2 = acc_cropped * g_factor
                    vel_delta = cumulative_trapezoid(acc_m_s2, dx=dt_exp, initial=0)
                    v0 = v0_inputs[exp_name]
                    vel_cropped = v0 - vel_delta if reverse_integration else v0 + vel_delta
                    disp_cropped = cumulative_trapezoid(vel_cropped, dx=dt_exp, initial=0)
                    force_cropped = acc_m_s2 * mass_kg
                    energy_cropped = cumulative_trapezoid(force_cropped, disp_cropped, initial=0)

                    results[exp_name] = {
                        'time': time_cropped, 'acc': acc_cropped, 'vel': vel_cropped,
                        'disp': disp_cropped, 'force': force_cropped, 'energy': energy_cropped
                    }

                # ========== 图表展示（只执行一次） ==========
                st.markdown("---")
                st.subheader("📊 六面展示曲线对比看板（多实验叠加）")

                def create_figure(x_key, y_key, x_label, y_label):
                    fig = go.Figure()
                    for exp_name, res in results.items():
                        fig.add_trace(go.Scatter(x=res[x_key], y=res[y_key], mode='lines', name=exp_name, line=dict(color=exp_color_map[exp_name], width=2)))
                    if dtick is not None and x_key == 'time': fig.update_xaxes(dtick=dtick)
                    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, margin=dict(l=10, r=10, t=20, b=10), height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    return fig

                row1_col1, row1_col2, row1_col3 = st.columns(3)
                with row1_col1:
                    st.markdown("**1. 加速度曲线 vs 时间**")
                    st.plotly_chart(create_figure('time', 'acc', "时间 Time (s)", "加速度 Acceleration (g)"), use_container_width=True)
                with row1_col2:
                    st.markdown("**2. 速度曲线 vs 时间 (一阶积分)**")
                    st.plotly_chart(create_figure('time', 'vel', "时间 Time (s)", "速度 Velocity (m/s)"), use_container_width=True)
                with row1_col3:
                    st.markdown("**3. 位移曲线 vs 时间 (二阶积分)**")
                    st.plotly_chart(create_figure('time', 'disp', "时间 Time (s)", "位移 Displacement (m)"), use_container_width=True)

                row2_col1, row2_col2, row2_col3 = st.columns(3)
                with row2_col1:
                    st.markdown("**4. 力曲线 vs 时间 ($F = m \\cdot a$)**")
                    st.plotly_chart(create_figure('time', 'force', "时间 Time (s)", "冲击力 Force (N)"), use_container_width=True)
                with row2_col2:
                    st.markdown("**5. 弹性表现曲线 (力 vs 位移)**")
                    st.plotly_chart(create_figure('disp', 'force', "位移 Displacement (m)", "冲击力 Force (N)"), use_container_width=True)
                with row2_col3:
                    st.markdown("**6. 吸能曲线 (能量 vs 位移)**")
                    st.plotly_chart(create_figure('disp', 'energy', "位移 Displacement (m)", "吸能量 Energy (J)"), use_container_width=True)

                # ========== 数据导出（只执行一次） ==========
                st.markdown("---")
                st.subheader("📥 经裁剪及积分运算后的完整试验报告数据预览")
                combined_dfs = []
                for exp_name, res in results.items():
                    df_exp = pd.DataFrame({"实验": exp_name, "时间 Time (s)": res['time'], "加速度 Acc (g)": res['acc'], "速度 Velocity (m/s)": res['vel'], "位移 Displacement (m)": res['disp'], "冲击力 Force (N)": res['force'], "吸能累积 Energy (J)": res['energy']})
                    combined_dfs.append(df_exp)
                if combined_dfs:
                    report_df = pd.concat(combined_dfs, ignore_index=True)
                    st.dataframe(report_df.head(50), use_container_width=True)

# =========================================================================
# 📂 视窗二：扩展额外传感器导入与对比页
# =========================================================================
with main_tab2:
    st.header("📂 扩展测量传感器导入与独立对比页")
    st.markdown("💡 **独立分流说明**：此窗口拥有专属的数据导入通道，您可以在此导入**同一个实验新的传感器文件**。此页面数据不会对视窗一产生任何数据混淆与干扰。")
    
    uploaded_files_tab2 = st.file_uploader("📥 请在此处上传视窗二专属的扩展 MME 数据文件：", accept_multiple_files=True, key="uploader_tab2")
    all_experiments_tab2 = process_multi_experiment_files_tab2(uploaded_files_tab2)
    
    results_tab2 = {}
    extra_x_label_input = "时间 Time (s)"
    extra_y_label_input = "物理量幅值"
    
    if not all_experiments_tab2:
        st.info("📂 请在上方导入通道中传入新的数据文件（.chn、.001等），系统将自动为您拉取独立的扩展对比图表。")
    else:
        st.success(f"🎉 视窗二成功加载 {len(all_experiments_tab2)} 个独立扩展实验数据集！")
        
        ext_col1, ext_col2, ext_col3 = st.columns([1, 2, 1])
        with ext_col1:
            selected_exps_tab2 = st.multiselect("选择要分析的扩展实验组：", options=list(all_experiments_tab2.keys()), default=list(all_experiments_tab2.keys())[:1], key="select_exp_t2")
        
        if selected_exps_tab2:
            all_available_channels = {}
            for e_id in selected_exps_tab2:
                for suffix, ch_info in all_experiments_tab2[e_id].items():
                    label = f"{ch_info['channel_name'].split('[')[0].strip()}"
                    if label not in all_available_channels:
                        all_available_channels[label] = suffix
                        
            with ext_col2:
                selected_extra_channels = st.multiselect("请勾选需要同屏叠加对比的扩展通道曲线：", options=list(all_available_channels.keys()), key="extra_chn_select")
            with ext_col3:
                extra_filter_option = st.selectbox("扩展通道专属滤波：", ["无滤波", "CFC60", "CFC180", "CFC600"], index=0, key="filt_t2")
            
            first_exp_ref = selected_exps_tab2[0]
            first_sfx_ref = list(all_experiments_tab2[first_exp_ref].keys())[0]
            t_ref_ext = all_experiments_tab2[first_exp_ref][first_sfx_ref]['time']
            
            st.markdown("##### ⏱️ 视窗二独立区间控制")
            ext_crop_col1, ext_crop_col2 = st.columns(2)
            with ext_crop_col1:
                ext_crop_start = st.number_input("扩展起点时间 (s)：", value=float(t_ref_ext[0]), step=0.005, key="start_t2")
            with ext_crop_col2:
                ext_crop_end = st.number_input("扩展终点时间 (s)：", value=float(t_ref_ext[-1]), step=0.005, key="end_t2")

            # 🟢 需求1：开放横轴、纵轴的重命名机会给使用者（在视窗二展示）
            st.markdown("##### ✍️ 扩展通道 HTML 报告坐标轴自定义命称")
            axis_name_col1, axis_name_col2 = st.columns(2)
            with axis_name_col1:
                extra_x_label_input = st.text_input("自定义横轴名称（如 Time (s) 或 Displacement (mm)）：", value="时间 Time (s)", key="ext_axis_x")
            with axis_name_col2:
                extra_y_label_input = st.text_input("自定义纵轴名称（如 Pressure (bar) 或 Current (A)）：", value="传感器测试物理量幅值", key="ext_axis_y")

            if selected_extra_channels:
                fig_extra = go.Figure()
                color_cycle_ext = ['#FFA15A', '#19D3F3', '#FF6692', '#636EFA', '#EF553B', '#00CC96', '#AB63FA']
                
                trace_counter = 0
                for exp_id in selected_exps_tab2:
                    results_tab2[exp_id] = {}
                    for ch_label in selected_extra_channels:
                        sfx_ext = all_available_channels[ch_label]
                        if sfx_ext in all_experiments_tab2[exp_id]:
                            ch_info_ext = all_experiments_tab2[exp_id][sfx_ext]
                            t_ext = ch_info_ext['time']
                            d_ext = ch_info_ext['data']
                            
                            s_idx = np.abs(t_ext - ext_crop_start).argmin()
                            e_idx = np.abs(t_ext - ext_crop_end).argmin() + 1
                            t_ext_cropped = t_ext[s_idx:e_idx]
                            d_ext_cropped = d_ext[s_idx:e_idx]
                            
                            if extra_filter_option != "无滤波":
                                df_ext_temp = pd.DataFrame({"Time": t_ext_cropped, "val": d_ext_cropped})
                                f_type = 60 if "60" in extra_filter_option else (180 if "180" in extra_filter_option else 600)
                                filtered_ext = filter_cfc(df_ext_temp, "val", f_type, append_df=False)
                                if filtered_ext is not None: d_ext_cropped = np.array(filtered_ext)
                            
                            trace_name = f"{exp_id} - {ch_label}"
                            results_tab2[exp_id][ch_label] = {
                                'x': t_ext_cropped,
                                'y': d_ext_cropped,
                                'name': trace_name,
                                'color': color_cycle_ext[trace_counter % len(color_cycle_ext)]
                            }
                            
                            fig_extra.add_trace(go.Scatter(
                                x=t_ext_cropped, y=d_ext_cropped, mode='lines', 
                                name=trace_name,
                                line=dict(color=color_cycle_ext[trace_counter % len(color_cycle_ext)], width=2)
                            ))
                            trace_counter += 1
                
                # 界面实时显示更新名称
                fig_extra.update_layout(
                    xaxis_title=extra_x_label_input, 
                    yaxis_title=extra_y_label_input, 
                    height=450, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                )
                st.plotly_chart(fig_extra, use_container_width=True)
            else:
                st.info("💡 请在上方勾选要对比的扩展传感器通道。")

# =========================================================================
# 📜 自动化 HTML 报告构建控制区（采用网格布局，离线可用）
# =========================================================================
if 'all_experiments_tab1' in locals() and all_experiments_tab1:
    st.sidebar.markdown("---")
    st.sidebar.header("📊 自动化报告配置中心")
    
    if st.sidebar.button("🚀 导出完整高端 HTML 报告"):
        # 构建 HTML 内容
        html_parts = []
        html_parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset='utf-8'>
<title>MME 冲击吸能与扩展通道高级试验报告</title>
<style>
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; margin: 0; padding: 40px 20px; }
.report-wrapper { max-width: 1300px; margin: 0 auto; }
.header-title { text-align: center; color: #1e3a8a; font-size: 26px; font-weight: 700; margin-bottom: 5px; letter-spacing: 1px; }
.header-sub { text-align: center; color: #666; font-size: 13px; margin-bottom: 35px; }
.card { background: #ffffff; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.07); padding: 25px; margin-bottom: 30px; border-top: 5px solid #1f77b4; }
.card-title { color: #1f77b4; font-size: 18px; font-weight: 600; margin-top: 0; margin-bottom: 20px; padding-bottom: 8px; border-bottom: 1px solid #e5e7eb; }
ul { padding-left: 20px; margin: 0; }
li { margin-bottom: 6px; font-size: 14px; color: #4b5563; }
li strong { color: #111827; }
/* 网格布局 */
.grid-3col { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
.grid-2col { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
.grid-1col { display: grid; grid-template-columns: 1fr; gap: 20px; }
.plot-card { background: #fafafa; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; transition: box-shadow 0.2s; }
.plot-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.plot-header { font-size: 14px; font-weight: 600; color: #374151; margin-bottom: 8px; padding-left: 8px; border-left: 3px solid #1f77b4; }
/* 响应式 */
@media (max-width: 900px) { .grid-3col { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) { .grid-3col, .grid-2col { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class='report-wrapper'>
<h1 class='header-title'>🛡️ MME 冲击吸能与多通道联合试验分析报告</h1>
<p class='header-sub'>报告自动输出时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """ | 工业级结构闭环验证版</p>
""")

        # 卡片 1：试验背景参数
        html_parts.append("""
<div class='card'>
<div class='card-title'>📋 视窗一：主分析试验配置背景</div>
<ul>
<li><strong>激活比对实验场次:</strong> """ + ', '.join(selected_exps_tab1) + """</li>
<li><strong>基准加速度通道选择:</strong> """ + selected_ac_name + """</li>
<li><strong>质量载荷:</strong> """ + str(mass_kg) + """ kg &nbsp;&nbsp;|&nbsp;&nbsp; <strong>重力加速度常数:</strong> """ + str(g_factor) + """ m/s²</li>
<li><strong>各实验初速度:</strong> """ + "; ".join([f"{exp}: {v0_inputs[exp]:.2f} m/s" for exp in selected_exps_tab1]) + """ &nbsp;&nbsp;|&nbsp;&nbsp; <strong>脉冲反向积分状态:</strong> """ + ('开启' if reverse_integration else '关闭') + """</li>
<li><strong>数据滤波器等级:</strong> """ + filter_option + """ &nbsp;&nbsp;|&nbsp;&nbsp; <strong>时域裁剪区间:</strong> """ + f"{crop_start}s ~ {crop_end}s" + """</li>
</ul>
</div>
""")

        # 卡片 2：核心 6 面看板（2行3列网格）
        html_parts.append("""
<div class='card'>
<div class='card-title'>📊 模块二：核心吸能指标特性对比（2×3 矩阵）</div>
<div class='grid-3col'>
""")

        titles_6 = [
            "1. 加速度曲线 vs 时间",
            "2. 速度曲线 vs 时间",
            "3. 位移曲线 vs 时间",
            "4. 力曲线 vs 时间",
            "5. 弹性表现 (力 vs 位移)",
            "6. 吸能曲线 (能量 vs 位移)"
        ]
        keys_6 = [
            ('time', 'acc', "时间 Time (s)", "加速度 Acceleration (g)"),
            ('time', 'vel', "时间 Time (s)", "速度 Velocity (m/s)"),
            ('time', 'disp', "时间 Time (s)", "位移 Displacement (m)"),
            ('time', 'force', "时间 Time (s)", "冲击力 Force (N)"),
            ('disp', 'force', "位移 Displacement (m)", "冲击力 Force (N)"),
            ('disp', 'energy', "位移 Displacement (m)", "吸能量 Energy (J)")
        ]

        for idx in range(6):
            x_k, y_k, xl, yl = keys_6[idx]
            fig_h = go.Figure()
            for exp_name, res in results.items():
                fig_h.add_trace(go.Scatter(
                    x=res[x_k], y=res[y_k],
                    mode='lines', name=exp_name,
                    line=dict(color=exp_color_map[exp_name], width=2)
                ))
            if dtick is not None and x_k == 'time':
                fig_h.update_xaxes(dtick=dtick)
            # 设置美观的边距和字体
            fig_h.update_layout(
                xaxis_title=xl,
                yaxis_title=yl,
                margin=dict(l=50, r=20, t=20, b=50),
                height=280,
                font=dict(size=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            # 内联 plotly.js 确保离线可用
            img_html = pio.to_html(fig_h, full_html=False, include_plotlyjs=True)
            html_parts.append(f"<div class='plot-card'><div class='plot-header'>{titles_6[idx]}</div>{img_html}</div>")

        html_parts.append("</div></div>")  # 闭合 grid-3col 和 card

        # 卡片 3：视窗二扩展传感器（如果存在）
        if results_tab2:
            html_parts.append("""
<div class='card'>
<div class='card-title'>📂 模块三：视窗二扩展传感器对比曲线</div>
<ul>
<li><strong>扩展时域裁剪区间:</strong> """ + f"{ext_crop_start}s ~ {ext_crop_end}s" + """ &nbsp;&nbsp;|&nbsp;&nbsp; <strong>通道专属滤波状态:</strong> """ + extra_filter_option + """</li>
</ul>
<div class='grid-1col'>
<div class='plot-card'>
<div class='plot-header'>📂 新导入传感器多通道叠加比对波形</div>
""")
            fig_ext_h = go.Figure()
            for exp_id, ch_dict in results_tab2.items():
                for ch_label, trace_data in ch_dict.items():
                    fig_ext_h.add_trace(go.Scatter(
                        x=trace_data['x'], y=trace_data['y'],
                        mode='lines', name=trace_data['name'],
                        line=dict(color=trace_data['color'], width=2)
                    ))
            fig_ext_h.update_layout(
                xaxis_title=extra_x_label_input,
                yaxis_title=extra_y_label_input,
                margin=dict(l=50, r=20, t=20, b=50),
                height=350,
                font=dict(size=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            img_ext_html = pio.to_html(fig_ext_h, full_html=False, include_plotlyjs=True)
            html_parts.append(img_ext_html)
            html_parts.append("</div></div></div>")  # 闭合 plot-card, grid-1col, card

        html_parts.append("</div></body></html>")
        full_html_data = "\n".join(html_parts)
        html_report_ready = True

    # 渲染最终下载按钮
    if html_report_ready:
        st.sidebar.download_button(
            label="📥 点击下载高级定制 HTML 报告",
            data=full_html_data,
            file_name=f"MME_Advanced_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )
        st.sidebar.success("🎉 报告组装完毕，请点击上方按钮下载！")
