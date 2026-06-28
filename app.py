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
    if type == 60 or type == 180 or type == 600 or type == 1000:
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
        scan_df = [col for col in df.columns if new_channel_name in col]
        last_col = df.shape[1]
        if len(scan_df) == 0:
            df.insert(last_col, new_channel_name, filtered_list, allow_duplicates=False)
        elif len(scan_df) == 1:
            prefix = "(1)"
            new_channel_name = new_channel_name + prefix
            df.insert(last_col, new_channel_name, filtered_list, allow_duplicates=False)
        else:
            new_channel_name_temp = scan_df[len(scan_df) - 1]
            prefix_temp = int(new_channel_name_temp[-2:-1]) + 1
            new_channel_name = f"{new_channel_name}({prefix_temp})"
            df.insert(last_col, new_channel_name, filtered_list, allow_duplicates=False)
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
        if step == 0:
            y_t = round(input_list[0], precision)
            filtered_list[step] = y_t
        elif step == 1:
            t = step
            Xt = input_list[t]
            y_t = round(Xt, precision)
            filtered_list[step] = y_t
        else:
            t = step
            Xt = input_list[t]
            Xtminus1 = input_list[t - 1]
            Xtminus2 = input_list[t - 2]
            Ytminus1 = filtered_list[t - 1]
            Ytminus2 = filtered_list[t - 2]
            y_t = a0 * Xt + a1 * Xtminus1 + a2 * Xtminus2 + b1 * Ytminus1 + b2 * Ytminus2
            y_t = round(y_t, precision)
            filtered_list[step] = y_t
    return filtered_list

# ===================== 页面配置 =====================
st.set_page_config(page_title="线性冲击实验基本分析看板", layout="wide")
st.title("🛡️线性冲击实验标准分析看板")
st.markdown("以ZF_LF内LIP实验文件夹格式的输入参考对象")

# ===================== 数据解析函数 =====================
@st.cache_data
def process_multi_experiment_files(uploaded_files):
    if not uploaded_files:
        return {}
    raw_groups = {}
    for f in uploaded_files:
        if '.' not in f.name:
            continue
        base_name, ext = f.name.rsplit('.', 1)
        ext_lower = ext.lower()
        if base_name not in raw_groups:
            raw_groups[base_name] = {'chn': None, 'data_files': []}
        if ext_lower == 'chn':
            raw_groups[base_name]['chn'] = f
        elif ext_lower == 'mme':
            continue
        else:
            raw_groups[base_name]['data_files'].append(f)
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
                if not line_str:
                    continue
                if ':' in line_str:
                    parts = line_str.split(':', 1)
                    metadata[parts[0].strip().lower()] = parts[1].strip()
                else:
                    try:
                        data_values.append(float(line_str))
                    except ValueError:
                        pass
            if len(data_values) == 0:
                continue
            sampling_interval = float(metadata.get('sampling interval', 0.0001))
            time_of_first_sample = float(metadata.get('time of first sample', 0.0))
            data_array = np.array(data_values, dtype=float)
            num_samples = len(data_array)
            time_array = np.linspace(
                time_of_first_sample,
                time_of_first_sample + (num_samples * sampling_interval),
                num_samples,
                endpoint=False
            )
            file_suffix = f.name.split('.')[-1]
            raw_channel_meaning = metadata.get('name of the channel', f"Channel_{file_suffix}")
            raw_unit = metadata.get('unit', '未知')
            channels_dict[file_suffix] = {
                'data': data_array,
                'time': time_array,
                'sampling_interval': sampling_interval,
                'raw_meaning': raw_channel_meaning,
                'unit': raw_unit,
                'default_name': f"通道 {file_suffix}"
            }
        if files['chn'] and channels_dict:
            chn_lines = files['chn'].getvalue().decode('utf-8', errors='ignore').splitlines()
            sorted_suffixes = sorted(channels_dict.keys())
            chn_names = []
            for i, line in enumerate(chn_lines):
                if i >= 2 and line.strip():
                    if len(line) >= 45:
                        chn_names.append(line[29:45].strip())
                    elif len(line) > 29:
                        chn_names.append(line[29:].strip())
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

# ===================== 侧边栏：文件导入 =====================
st.sidebar.header("📁 数据导入中心")
uploaded_files = st.sidebar.file_uploader(
    "同时拖入mme、chn和对应通道里所有通道所有文件，支持拖入多次实验的全部组件文件：",
    accept_multiple_files=True
)
all_experiments = process_multi_experiment_files(uploaded_files)

if not all_experiments:
    st.info("💡 提示：请在左侧边栏上传试验数据集（包含 MME、chn、数据通道后缀等文件）以激活数据视图。")
else:
    # ===================== 控制区 =====================
    st.sidebar.header("📋控制区间")
    st.sidebar.markdown("### 🎯 核心分析实验与控制参数设定")
    experiment_names = list(all_experiments.keys())
    selected_exps = st.sidebar.multiselect("1. 选择要对比的实验（可多选）：", options=experiment_names, default=experiment_names[:1])
    if not selected_exps:
        st.sidebar.warning("请至少选择一个实验。")
        st.sidebar.stop()

    ref_exp = selected_exps[0]
    exp_chns = all_experiments[ref_exp]
    channel_options = {info['channel_name']: suffix for suffix, info in exp_chns.items()}
    ac_defaults = [name for name in channel_options.keys() if "AC" in name.upper()]
    default_ac = ac_defaults[0] if ac_defaults else list(channel_options.keys())[0]

    # 加速度通道选择
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.sidebar.columns(4)
    with ctrl_col1:
        selected_ac_name = st.sidebar.selectbox("2. 选择基本加速度通道：", options=list(channel_options.keys()), index=list(channel_options.keys()).index(default_ac))
        suffix = channel_options[selected_ac_name]
        ch_info_ref = exp_chns[suffix]
    t_raw = ch_info_ref['time']
    t_min, t_max = float(t_raw[0]), float(t_raw[-1])

    with ctrl_col2:
        mass_kg = st.sidebar.number_input("3. 冲击块质量 (Mass, kg)：", value=6.80, step=0.1)
        g_factor = st.sidebar.number_input("重力加速度系数 (g ➔ m/s²)", value=9.81, step=0.01)
    with ctrl_col3:
        reverse_integration = st.sidebar.checkbox("碰撞减速脉冲积分 (v0 - ∫a dt)", value=True)
    with ctrl_col4:
        filter_option = st.sidebar.selectbox("5. 加速度滤波 (CFC)：", ["无滤波", "CFC60", "CFC180"], index=0)
        tick_step = st.sidebar.selectbox("6. X轴刻度间隔：", ["自动", "10 ms", "20 ms", "50 ms", "100 ms"], index=0)
        tick_step_map = {"自动": None, "10 ms": 0.01, "20 ms": 0.02, "50 ms": 0.05, "100 ms": 0.1}
        dtick = tick_step_map[tick_step]

    # 每个实验的初速度输入
    st.sidebar.markdown("#### 🚀 各实验初速度设置")
    v0_inputs = {}
    cols_v0 = st.sidebar.columns(min(len(selected_exps), 4))
    for i, exp_name in enumerate(selected_exps):
        col = cols_v0[i % 4]
        with col:
            v0_inputs[exp_name] = st.sidebar.number_input(
                f"{exp_name} 初速度 v0 (m/s)",
                value=3.04,
                step=0.01,
                key=f"v0_{exp_name}"
            )

    # 裁剪区间
    st.sidebar.markdown("#### ⏱️ 曲线裁剪区间控制 (Crop Range)")
    crop_col1, crop_col2 = st.sidebar.columns(2)
    with crop_col1:
        crop_start = st.sidebar.number_input("裁剪起点时间 (Start Time, 秒 s)：", value=max(0.0, t_min), min_value=t_min, max_value=t_max, step=0.001, format="%.4f")
    with crop_col2:
        crop_end = st.sidebar.number_input("裁剪终点时间 (End Time, 秒 s)：", value=min(0.150, t_max), min_value=t_min, max_value=t_max, step=0.001, format="%.4f")
    if crop_start >= crop_end:
        st.sidebar.error("❌ 错误：起点时间必须小于终点时间！")
        st.sidebar.stop()

    # ===================== 物理计算 =====================
    color_cycle = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    exp_color_map = {exp_name: color_cycle[i % len(color_cycle)] for i, exp_name in enumerate(selected_exps)}

    results = {}
    all_extra_data = {}

    for exp_name in selected_exps:
        exp_data = all_experiments[exp_name]
        if suffix not in exp_data:
            st.warning(f"实验 {exp_name} 中未找到加速度通道 {suffix}，跳过。")
            continue
        ch_info = exp_data[suffix]
        t_raw_exp = ch_info['time']
        acc_raw_exp = ch_info['data']
        dt_exp = ch_info['sampling_interval']

        start_idx = np.abs(t_raw_exp - crop_start).argmin()
        end_idx = np.abs(t_raw_exp - crop_end).argmin() + 1
        time_cropped = t_raw_exp[start_idx:end_idx]
        acc_cropped = acc_raw_exp[start_idx:end_idx]

        # 滤波
        if filter_option != "无滤波":
            df_temp = pd.DataFrame({"Time": time_cropped, "acc": acc_cropped})
            if filter_option == "CFC60":
                filt_type = 60
            else:
                filt_type = 180
            filtered = filter_cfc(df_temp, "acc", filt_type, append_df=False)
            if filtered is not None:
                acc_cropped = np.array(filtered)

        # 物理量计算
        v0 = v0_inputs[exp_name]
        acc_m_s2 = acc_cropped * g_factor
        vel_delta = cumulative_trapezoid(acc_m_s2, dx=dt_exp, initial=0)
        if reverse_integration:
            vel_cropped = v0 - vel_delta
        else:
            vel_cropped = v0 + vel_delta
        disp_cropped = cumulative_trapezoid(vel_cropped, dx=dt_exp, initial=0)
        force_cropped = acc_m_s2 * mass_kg
        energy_cropped = cumulative_trapezoid(force_cropped, disp_cropped, initial=0)

        results[exp_name] = {
            'time': time_cropped,
            'acc': acc_cropped,
            'vel': vel_cropped,
            'disp': disp_cropped,
            'force': force_cropped,
            'energy': energy_cropped
        }

        # 提取所有通道的裁剪数据（用于额外通道对比）
        all_extra_data[exp_name] = {}
        for ch_suffix, ch_info in exp_data.items():
            t_raw_extra = ch_info['time']
            val_raw_extra = ch_info['data']
            start_idx_extra = np.abs(t_raw_extra - crop_start).argmin()
            end_idx_extra = np.abs(t_raw_extra - crop_end).argmin() + 1
            all_extra_data[exp_name][ch_info['channel_name']] = {
                'time': t_raw_extra[start_idx_extra:end_idx_extra],
                'value': val_raw_extra[start_idx_extra:end_idx_extra]
            }

    if not results:
        st.error("所有选定实验均无法计算，请检查通道选择。")
        st.stop()

    # ===================== 绘图 =====================
    st.markdown("---")
    st.subheader("📊 八面展示曲线对比看板（多实验叠加）")

    all_channel_names = sorted(list(next(iter(all_extra_data.values())).keys())) if all_extra_data else []


    def create_figure(x_key, y_key, x_label, y_label, title, use_results=True, extra_ch_name=None):
        fig = go.Figure()
        if use_results:
            for exp_name, res in results.items():
                color = exp_color_map[exp_name]
                fig.add_trace(go.Scatter(
                    x=res[x_key], y=res[y_key],
                    mode='lines',
                    name=exp_name,
                    line=dict(color=color, width=2)
                ))
        else:
            for exp_name, extra_dict in all_extra_data.items():
                if extra_ch_name in extra_dict:
                    data = extra_dict[extra_ch_name]
                    color = exp_color_map[exp_name]
                    fig.add_trace(go.Scatter(
                        x=data['time'], y=data['value'],
                        mode='lines',
                        name=exp_name,
                        line=dict(color=color, width=2)
                    ))

        # 设置 x 轴刻度间隔（如果有）
        if dtick is not None:
            fig.update_xaxes(dtick=dtick)

        # 横轴样式：标题字体、刻度字体、标题与轴线间距
        fig.update_xaxes(
            title_font=dict(size=12),  # 横轴标题字体大小
            tickfont=dict(size=10),  # 横轴刻度字体大小
            title_standoff=15  # 标题与轴线之间的像素距离（防止重叠）
        )
        # 纵轴样式（与横轴保持一致）
        fig.update_yaxes(
            title_font=dict(size=12),
            tickfont=dict(size=10)
        )

        # 整体布局：增大底部边距，为横轴留足空间
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            margin=dict(l=60, r=60, t=30, b=80),  # 底部边距增加到80px
            height=320,
            font=dict(size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    # 前六页
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.markdown("**1. 加速度曲线 vs 时间**")
        fig1 = create_figure('time', 'acc', "时间 Time (s)", "加速度 Acceleration (g)", "")
        st.plotly_chart(fig1, use_container_width=True)
    with row1_col2:
        st.markdown("**2. 速度曲线 vs 时间 (一阶积分)**")
        fig2 = create_figure('time', 'vel', "时间 Time (s)", "速度 Velocity (m/s)", "")
        st.plotly_chart(fig2, use_container_width=True)
    with row1_col3:
        st.markdown("**3. 位移曲线 vs 时间 (二阶积分)**")
        fig3 = create_figure('time', 'disp', "时间 Time (s)", "位移 Displacement (m)", "")
        st.plotly_chart(fig3, use_container_width=True)

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        st.markdown("**4. 力曲线 vs 时间 ($F = m \\cdot a$)**")
        fig4 = create_figure('time', 'force', "时间 Time (s)", "冲击力 Force (N)", "")
        st.plotly_chart(fig4, use_container_width=True)
    with row2_col2:
        st.markdown("**5. 弹性表现曲线 (力 vs 位移)**")
        fig5 = create_figure('disp', 'force', "位移 Displacement (m)", "冲击力 Force (N)", "")
        st.plotly_chart(fig5, use_container_width=True)
    with row2_col3:
        st.markdown("**6. 吸能曲线 (能量 vs 位移)**")
        fig6 = create_figure('disp', 'energy', "位移 Displacement (m)", "吸能量 Energy (J)", "")
        st.plotly_chart(fig6, use_container_width=True)

    # 新增第7、第8页（每个页面独立选择通道）
    st.markdown("---")
    st.subheader("📊 额外通道对比（可自由选择通道）")

    col7, col8 = st.columns(2)
    with col7:
        st.markdown("**7. 额外通道对比**")
        selected_extra_1 = st.selectbox("选择第7页显示的通道：", options=all_channel_names, key="extra1")
        fig7 = create_figure(None, None, "时间 Time (s)", selected_extra_1, "", use_results=False, extra_ch_name=selected_extra_1)
        st.plotly_chart(fig7, use_container_width=True)

    with col8:
        st.markdown("**8. 额外通道对比**")
        selected_extra_2 = st.selectbox("选择第8页显示的通道：", options=all_channel_names, key="extra2")
        fig8 = create_figure(None, None, "时间 Time (s)", selected_extra_2, "", use_results=False, extra_ch_name=selected_extra_2)
        st.plotly_chart(fig8, use_container_width=True)

    # ===================== 保存为 HTML 报告 =====================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 保存报告")
    if st.sidebar.button("💾 保存当前看板为 HTML 报告"):
        # 收集所有图表和标题
        figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]
        titles = [
            "1. 加速度曲线 vs 时间",
            "2. 速度曲线 vs 时间",
            "3. 位移曲线 vs 时间",
            "4. 力曲线 vs 时间",
            "5. 弹性表现曲线 (力 vs 位移)",
            "6. 吸能曲线 (能量 vs 位移)",
            f"7. {selected_extra_1} 对比",
            f"8. {selected_extra_2} 对比"
        ]

        # 构建 HTML 内容
        html_parts = []
        html_parts.append("""<!DOCTYPE html>
    <html>
    <head>
    <meta charset='utf-8'>
    <title>线性冲击实验分析报告</title>
    <style>
    body { font-family: Arial, sans-serif; margin: 20px; background-color: #f9f9f9; }
    h1 { color: #333; text-align: center; }
    h2 { color: #555; font-size: 14px; margin: 0 0 5px 0; }
    .grid-container {
        display: grid;
        gap: 15px;
        margin-bottom: 20px;
    }
    .grid-3col {
        grid-template-columns: repeat(3, 1fr);
    }
    .grid-2col {
        grid-template-columns: repeat(2, 1fr);
    }
    .plot-card {
        background: white;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .plot-wrapper {
        width: 100%;
        height: 340px; /* 略大于图表高度，避免滚动条 */
    }
    .report-info { text-align: center; color: #777; margin-bottom: 20px; }
    hr { margin: 20px 0; }
    /* 响应式：小屏幕时自动折叠 */
    @media (max-width: 1200px) {
        .grid-3col { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 800px) {
        .grid-3col, .grid-2col { grid-template-columns: 1fr; }
    }
    </style>
    </head>
    <body>
    <h1>线性冲击实验标准分析报告</h1>
    <div class="report-info">
    <p>生成时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    <p>实验列表: """ + ", ".join(selected_exps) + """</p>
    <p>滤波选项: """ + filter_option + """</p>
    <p>裁剪区间: """ + f"{crop_start:.4f}s ~ {crop_end:.4f}s" + """</p>
    </div>
    <hr>
    """)

        # 前6个图表：2行3列
        html_parts.append('<div class="grid-container grid-3col">')
        for i in range(6):
            html_parts.append(f'<div class="plot-card"><h2>{titles[i]}</h2><div class="plot-wrapper">')
            html_parts.append(pio.to_html(figs[i], include_plotlyjs=True, full_html=False))
            html_parts.append('</div></div>')
        html_parts.append('</div>')

        # 后2个图表：1行2列
        html_parts.append('<div class="grid-container grid-2col">')
        for i in range(6, 8):
            html_parts.append(f'<div class="plot-card"><h2>{titles[i]}</h2><div class="plot-wrapper">')
            html_parts.append(pio.to_html(figs[i], include_plotlyjs=True, full_html=False))
            html_parts.append('</div></div>')
        html_parts.append('</div>')

        html_parts.append("</body></html>")
        full_html = "\n".join(html_parts)

        # 提供下载按钮
        st.sidebar.download_button(
            label="📥 下载 HTML 报告",
            data=full_html,
            file_name=f"crash_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )
    # ===================== 数据导出 =====================
    st.markdown("---")
    st.subheader("📥 经裁剪及积分运算后的完整试验报告数据预览")
    combined_dfs = []
    for exp_name, res in results.items():
        df_exp = pd.DataFrame({
            "实验": exp_name,
            "时间 Time (s)": res['time'],
            "加速度 Acc (g)": res['acc'],
            "速度 Velocity (m/s)": res['vel'],
            "位移 Displacement (m)": res['disp'],
            "冲击力 Force (N)": res['force'],
            "吸能累积 Energy (J)": res['energy']
        })
        if selected_extra_1 in all_extra_data.get(exp_name, {}):
            df_exp[selected_extra_1] = all_extra_data[exp_name][selected_extra_1]['value'][:len(res['time'])]
        if selected_extra_2 in all_extra_data.get(exp_name, {}):
            df_exp[selected_extra_2] = all_extra_data[exp_name][selected_extra_2]['value'][:len(res['time'])]
        combined_dfs.append(df_exp)
    if combined_dfs:
        report_df = pd.concat(combined_dfs, ignore_index=True)
        st.dataframe(report_df.head(100), use_container_width=True)
        st.download_button("📥 导出当前视窗 CSV 报告数据", report_df.to_csv(index=False), "MME_MultiExp_Report.csv", "text/csv")
