import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.spatial.distance import cdist  
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning) 
import time
import subprocess
import os
import sys
import json
import tkinter as tk
from tkinter import filedialog
import math
import shutil
import ctypes

st.set_page_config(page_title="LFS Aero-Mapper", layout="wide")
st.title("LFS Aerodynamic Optimization Framework")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'doe_points' not in st.session_state:
    st.session_state.doe_points = None
if 'is_initialized' not in st.session_state:
    st.session_state.is_initialized = False
if 'sim_params' not in st.session_state:
    st.session_state.sim_params = ["(Scan .sim file to populate)"]
if 'sim_reports' not in st.session_state:
    st.session_state.sim_reports = ["(Scan .sim file to populate)"]
if 'sim_ffs' not in st.session_state:
    st.session_state.sim_ffs = ["(Scan .sim file to populate)"]
if 'sim_file_path' not in st.session_state:
    st.session_state.sim_file_path = "Choose .sim file from folder"
if 'cad_files' not in st.session_state:
    st.session_state.cad_files = []
if 'cad_mappings' not in st.session_state:
    st.session_state.cad_mappings = {}

# ==========================================
# WINDOWS NATIVE FILE BROWSERS
# ==========================================
def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1) 
    file_path = filedialog.askopenfilename(filetypes=[("STAR-CCM+ Simulation", "*.sim")])
    root.destroy()
    return file_path

def open_cad_dialog():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1) 
    file_paths = filedialog.askopenfilenames(filetypes=[("CAD Files", "*.prt *.step *.stp *.iges *.igs *.x_t *.stl")])
    root.destroy()
    return list(file_paths)

# ==========================================
# SIDEBAR WIZARD ROUTING
# ==========================================
step = st.session_state.current_step

# ------------------------------------------
# WIZARD STEP 1: SIMULATION & CAD GEOMETRY
# ------------------------------------------
if step == 0:
    st.sidebar.header("Step 1: Simulation & Geometry")

    st.sidebar.markdown("**STAR-CCM+ Executable Path:**")
    starccm_exe = st.sidebar.text_input(
        "Executable Path:", 
        value=st.session_state.get('perm_starccm_exe', r"D:\Program Files\Siemens\20.04.007\STAR-CCM+20.04.007\star\bin\starccm+.bat"), 
        label_visibility="collapsed"
    )

    st.sidebar.markdown("**Simulation File (.sim):**")
    col_path, col_browse = st.sidebar.columns([4, 1])
    sim_file_input = col_path.text_input("Path:", value=st.session_state.sim_file_path, label_visibility="collapsed")

    if col_browse.button("📁", key="sim_browse"):
        selected_file = open_file_dialog()
        if selected_file:
            st.session_state.sim_file_path = selected_file
            st.rerun() 
    else:
        st.session_state.sim_file_path = sim_file_input

    if st.sidebar.button("🔍 Scan .sim File", width="stretch"):
        if not os.path.exists(st.session_state.sim_file_path):
            st.sidebar.error("Simulation file not found!")
        elif not os.path.exists(starccm_exe):
            st.sidebar.error("STAR-CCM+ Executable not found!")
        else:
            with st.spinner('Probing simulation tree...'):
                try:
                    result = subprocess.run([starccm_exe, "-batch", "probe_parameters.java", st.session_state.sim_file_path], check=True, capture_output=True, text=True)
                    if os.path.exists("sim_metadata.json"):
                        with open("sim_metadata.json", "r") as f:
                            data = json.load(f)
                            st.session_state.sim_params = [p for p in data.get("parameters", []) if p] or ["(No Parameters Found)"]
                            st.session_state.sim_reports = [r for r in data.get("reports", []) if r] or ["(No Reports Found)"]
                            st.session_state.sim_ffs = [f for f in data.get("field_functions", []) if f] or ["(No Field Functions Found)"]
                        st.sidebar.success("Successfully loaded simulation metadata!")
                    else:
                        st.sidebar.error("Macro ran, but failed to generate the JSON file.")
                except subprocess.CalledProcessError as e:
                    st.sidebar.error("STAR-CCM+ Execution Failed!")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### CAD Geometry Swap (Optional)")
    st.sidebar.caption("Select new .x_t or .step files to test in this sweep.")
    
    if st.sidebar.button("➕ Select CAD Parts", width="stretch"):
        new_cad_files = open_cad_dialog()
        if new_cad_files:
            st.session_state.cad_files.extend(new_cad_files)
            st.session_state.cad_files = list(set(st.session_state.cad_files)) 
            st.rerun()

    if st.session_state.cad_files:
        st.sidebar.markdown("**Assign Geometry Tags:**")
        categories = ["Ignore", "Chassis", "Front Wing", "Rear Wing", "Floor", "Front Left Wheel", "Front Right Wheel", "Rear Left Wheel", "Rear Right Wheel"]
        
        for i, fpath in enumerate(st.session_state.cad_files):
            fname = os.path.basename(fpath)
            st.session_state.cad_mappings[fpath] = st.sidebar.selectbox(
                f"⚙️ {fname}", categories, key=f"cad_sel_{i}"
            )
            
        if st.sidebar.button("🗑️ Clear Selected CAD"):
            st.session_state.cad_files = []
            st.session_state.cad_mappings = {}
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("Next: Sweep Setup ➡️", type="primary", width="stretch"):
        st.session_state.perm_starccm_exe = starccm_exe
        st.session_state.current_step = 1
        st.rerun()

# ------------------------------------------
# WIZARD STEP 2: PARAMETERS & CONSTRAINTS
# ------------------------------------------
elif step == 1:
    st.sidebar.header("Step 2: Sweep Setup")
    
    uploaded_csv = st.sidebar.file_uploader("📂 Upload Baseline CSV (Optional):", type="csv")
    prev_df = pd.read_csv(uploaded_csv) if uploaded_csv is not None else None
    
    sweep_type = st.sidebar.radio("Sweep Dimensionality:", ["1-Parameter Sweep (2D Curve)", "2-Parameter Sweep (3D Surface)"])
    is_2d = "2-Parameter" in sweep_type
    
    current_dim = 2 if is_2d else 1
    if 'sweep_dim' not in st.session_state:
        st.session_state.sweep_dim = current_dim
    if st.session_state.sweep_dim != current_dim:
        st.session_state.is_initialized = False
        st.session_state.sweep_dim = current_dim

    use_adaptive = False
    al_targets = []
    if prev_df is not None:
        use_adaptive = st.sidebar.checkbox("🧬 Enable Active Learning (Target Max Uncertainty)", value=False)
        if use_adaptive:
            available_targets = [col for col in prev_df.columns if col not in ['Run_ID', prev_df.columns[1], prev_df.columns[2] if is_2d else None] and not col.startswith('95%')]
            default_t = [t for t in available_targets if "Downforce" in t or "Drag" in t]
            al_targets = st.sidebar.multiselect("Select Uncertainty Targets to Minimize:", available_targets, default=default_t)

    num_runs = st.sidebar.number_input("Number of CFD Runs in Sweep", min_value=1, max_value=200, value=5 if use_adaptive else 15, step=1)

    if 'prev_num_runs' not in st.session_state:
        st.session_state.prev_num_runs = num_runs
    if st.session_state.prev_num_runs != num_runs:
        st.session_state.is_initialized = False
        st.session_state.prev_num_runs = num_runs

    st.sidebar.markdown("### Parameter 1 (X-Axis)")
    p1_default_idx = 0
    if prev_df is not None and len(prev_df.columns) > 1:
        if prev_df.columns[1] in st.session_state.sim_params:
            p1_default_idx = st.session_state.sim_params.index(prev_df.columns[1])
            
    param_1 = st.sidebar.selectbox("Select Target", st.session_state.sim_params, index=p1_default_idx, key="p1")
    col1, col2 = st.sidebar.columns(2)
    p1_min = col1.number_input("Min", value=0.000, step=0.005, format="%.4f", key="p1_min")
    p1_max = col2.number_input("Max", value=0.000, step=0.005, format="%.4f", key="p1_max")

    if is_2d:
        st.sidebar.markdown("### Parameter 2 (Y-Axis)")
        p2_default_idx = 0
        if prev_df is not None and len(prev_df.columns) > 2:
            if prev_df.columns[2] in st.session_state.sim_params:
                p2_default_idx = st.session_state.sim_params.index(prev_df.columns[2])
                
        param_2 = st.sidebar.selectbox("Select Target", st.session_state.sim_params, index=p2_default_idx, key="p2")
        col3, col4 = st.sidebar.columns(2)
        p2_min = col3.number_input("Min", value=0.000, step=0.005, format="%.4f", key="p2_min")
        p2_max = col4.number_input("Max", value=0.000, step=0.005, format="%.4f", key="p2_max")
    else:
        param_2, p2_min, p2_max = None, None, None

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Kinematic Constraints")
    if is_2d:
        apply_constraints = st.sidebar.checkbox("Apply Constraints", value=False)
        if apply_constraints:
            col_k1, col_k2 = st.sidebar.columns(2)
            limit_frh = col_k1.number_input("Min P1", value=1.0, step=1.0, key="lim_f")
            limit_rrh = col_k2.number_input("Max P2", value=1.0, step=1.0, key="lim_r")
            max_rrh_ratio = limit_rrh / limit_frh if limit_frh != 0 else 1.0
        else:
            max_rrh_ratio = None
    else:
        apply_constraints = False
        st.sidebar.info("Constraints disabled for 1D sweeps.")
        
    st.sidebar.markdown("---")
    col_back1, col_next1 = st.sidebar.columns(2)
    if col_back1.button("⬅️ Back", width="stretch"):
        st.session_state.current_step = 0
        st.rerun()
    if col_next1.button("Next ➡️", type="primary", width="stretch"):
        st.session_state.perm_p1_min = p1_min
        st.session_state.perm_p1_max = p1_max
        st.session_state.perm_p1 = param_1
        if is_2d:
            st.session_state.perm_p2_min = p2_min
            st.session_state.perm_p2_max = p2_max
            st.session_state.perm_p2 = param_2
            st.session_state.perm_apply_constraints = apply_constraints
            st.session_state.perm_max_rrh_ratio = max_rrh_ratio
        st.session_state.current_step = 2
        st.rerun()

# ------------------------------------------
# WIZARD STEP 3: OUTPUTS & EXECUTION
# ------------------------------------------
elif step == 2:
    st.sidebar.header("Step 3: Outputs & Execution")
    
    uploaded_csv = st.sidebar.file_uploader("📂 Confirm Baseline CSV (Optional):", type="csv", key="confirm_csv")
    prev_df = pd.read_csv(uploaded_csv) if uploaded_csv is not None else None
    
    is_2d = st.session_state.sweep_dim == 2
    if is_2d:
        apply_constraints = st.session_state.get('perm_apply_constraints', False)
        max_rrh_ratio = st.session_state.get('perm_max_rrh_ratio', None)
    else:
        apply_constraints = False

    available_reports = st.session_state.sim_reports
    if prev_df is not None:
        default_reports = [col for col in prev_df.columns if col in available_reports]
    else:
        default_reports = [r for r in ["Downforce (Cl)", "Drag (Cd)", "CLA", "CDA"] if r in available_reports]

    targets = st.sidebar.multiselect("Select Global Variables to Track:", available_reports, default=default_reports if default_reports else None)
    selected_ff = st.sidebar.multiselect("Export Volumetric Field Functions:", st.session_state.sim_ffs, default=None)
    export_csv_table = st.sidebar.checkbox("Generate Consolidated Results .CSV", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Execution Control")
    
    is_appending = uploaded_csv is not None
    
    if is_appending:
        st.sidebar.success("🔗 Appending to Baseline SDoE")
        run_name = uploaded_csv.name.replace(".csv", "")
        st.sidebar.text_input("Target Folder & CSV Name:", value=run_name, disabled=True)
    else:
        st.sidebar.info("✨ Starting a Fresh Sweep")
        run_name = st.sidebar.text_input("New Run Name (Folder & CSV):", value="Sweep_001", key="fresh_run_name")
    
    starccm_exe = st.session_state.get('perm_starccm_exe', r"D:\Program Files\Siemens\20.04.007\STAR-CCM+20.04.007\star\bin\starccm+.bat")
    cores = st.sidebar.number_input("STAR-CCM+ Cores (-np)", min_value=1, max_value=128, value=8, step=1)

    col_btn1, col_btn2 = st.sidebar.columns(2)
    init_button = col_btn1.button("⚙️ Initialise DoE", width="stretch")
    run_button = col_btn2.button("🚀 Launch Batch", width="stretch", type="primary")

    if st.sidebar.button("🚨 Abort Active Sweep", type="primary", width="stretch"):
        with open("stop_sweep.txt", "w") as f: 
            f.write("ABORT")
        st.sidebar.error("Abort signal sent!")
        
    st.sidebar.markdown("---")
    if st.sidebar.button("⬅️ Back to Sweep Setup", width="stretch"):
        st.session_state.current_step = 1
        st.rerun()

# ==========================================
# MATH: LATIN HYPERCUBE & ACTIVE LEARNING
# ==========================================
if step == 2:
    p1_min = st.session_state.perm_p1_min
    p1_max = st.session_state.perm_p1_max
    if is_2d:
        p2_min = st.session_state.perm_p2_min
        p2_max = st.session_state.perm_p2_max
    param_1 = st.session_state.perm_p1
    param_2 = st.session_state.perm_p2 if is_2d else None
    num_runs = st.session_state.prev_num_runs
    use_adaptive = False 

def generate_doe():
    def get_smart_decimals(val):
        if val == 0: return 3
        mag = math.floor(math.log10(abs(val)))
        return max(0, -(mag - 1))
    
    l_p1_min = st.session_state.perm_p1_min
    l_p1_max = st.session_state.perm_p1_max
    l_p2_min = st.session_state.get('perm_p2_min', 0)
    l_p2_max = st.session_state.get('perm_p2_max', 0)
    l_num_runs = st.session_state.prev_num_runs
    l_is_2d = st.session_state.sweep_dim == 2
    
    l_apply_constraints = st.session_state.get('perm_apply_constraints', False)
    l_max_rrh_ratio = st.session_state.get('perm_max_rrh_ratio', None)

    p1_decimals = get_smart_decimals(l_p1_min)
    p2_decimals = get_smart_decimals(l_p2_min) if l_is_2d else 0

    if not l_is_2d:
        sampler = qmc.LatinHypercube(d=1)
        raw_samples = sampler.random(n=l_num_runs)
        scaled = qmc.scale(raw_samples, [l_p1_min], [l_p1_max])
        scaled[:, 0] = np.round(scaled[:, 0], p1_decimals)
        return scaled
    else:
        sampler = qmc.LatinHypercube(d=2)
        if l_apply_constraints and l_max_rrh_ratio is not None:
            raw_samples = sampler.random(n=l_num_runs * 10) 
            scaled = qmc.scale(raw_samples, [l_p1_min, l_p2_min], [l_p1_max, l_p2_max])
            valid_mask = scaled[:, 1] <= (scaled[:, 0] * l_max_rrh_ratio)
            valid_points = scaled[valid_mask][:l_num_runs]
            valid_points[:, 0] = np.round(valid_points[:, 0], p1_decimals)
            valid_points[:, 1] = np.round(valid_points[:, 1], p2_decimals)
            return valid_points
        else:
            raw_samples = sampler.random(n=l_num_runs)
            scaled = qmc.scale(raw_samples, [l_p1_min, l_p2_min], [l_p1_max, l_p2_max])
            scaled[:, 0] = np.round(scaled[:, 0], p1_decimals)
            scaled[:, 1] = np.round(scaled[:, 1], p2_decimals)
            return scaled

if step == 2 and init_button:
    st.session_state.doe_points = generate_doe()
    st.session_state.is_initialized = True

# ==========================================
# MAIN DASHBOARD: TABS
# ==========================================
tab1, tab2 = st.tabs(["📊 1. Pre-Run Evaluation", "🏁 2. Results Dashboard"])

with tab1:
    if not st.session_state.is_initialized:
        st.info("👈 **Awaiting Initialization:** Proceed to Step 3 in the sidebar and click **Initialise DoE**.")
    else:
        st.markdown("### Design of Experiments (DoE) Validation")
        fig = go.Figure()
        doe_pts = st.session_state.doe_points
        l_p1_min = st.session_state.perm_p1_min
        l_p1_max = st.session_state.perm_p1_max
        param_1 = st.session_state.perm_p1
        l_is_2d = st.session_state.sweep_dim == 2
        l_num_runs = st.session_state.prev_num_runs
        
        if not l_is_2d:
            fig.add_trace(go.Scatter(x=doe_pts[:, 0], y=np.zeros(l_num_runs), mode='markers+text', marker=dict(size=12, color='black'), text=[str(i+1) for i in range(l_num_runs)], textposition="top center", name='New SDoE Points'))
            fig.update_layout(xaxis_title=param_1, yaxis=dict(visible=False), xaxis=dict(range=[l_p1_min, l_p1_max]), height=300, template='plotly_white')
            st.plotly_chart(fig, width="stretch")
        else:
            l_p2_min = st.session_state.perm_p2_min
            l_p2_max = st.session_state.perm_p2_max
            param_2 = st.session_state.perm_p2
            
            l_apply_constraints = st.session_state.get('perm_apply_constraints', False)
            l_max_rrh_ratio = st.session_state.get('perm_max_rrh_ratio', None)
            
            has_baseline = prev_df is not None
            pt_label = 'New SDoE Points' if has_baseline else 'CFD Test Points'
            pt_color = 'blue' if has_baseline else 'black'
            
            fig.add_trace(go.Scatter(x=[l_p1_min, l_p1_max, l_p1_max, l_p1_min, l_p1_min], y=[l_p2_min, l_p2_min, l_p2_max, l_p2_max, l_p2_min], mode='lines', line=dict(color='black', width=3), name='Domain Boundary', hoverinfo='skip'))
            
            if l_apply_constraints and l_max_rrh_ratio is not None:
                x_shade = np.linspace(l_p1_min, l_p1_max, 100)
                y_limit = x_shade * l_max_rrh_ratio
                y_upper = np.full_like(x_shade, l_p2_max)
                y_lower = np.clip(y_limit, l_p2_min, l_p2_max)
                fig.add_trace(go.Scatter(x=x_shade, y=y_upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(
                    x=x_shade, y=y_lower, fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', 
                    mode='lines', line=dict(color='darkred', dash='dash', width=2), name='Ground Clash Limit'
                ))
            
            if has_baseline:
                fig.add_trace(go.Scatter(x=prev_df[param_1], y=prev_df[param_2], mode='markers', marker=dict(size=8, color='lightgray', symbol='x'), name='Baseline CFD Points'))
            
            fig.add_trace(go.Scatter(x=doe_pts[:, 0], y=doe_pts[:, 1], mode='markers+text', marker=dict(size=10, color=pt_color), text=[str(i+1) for i in range(l_num_runs)], textposition="top right", name=pt_label))
            
            pad_x, pad_y = (l_p1_max - l_p1_min) * 0.1, (l_p2_max - l_p2_min) * 0.1
            fig.update_layout(
                xaxis_title=param_1, yaxis_title=param_2,
                xaxis=dict(range=[l_p1_min - pad_x, l_p1_max + pad_x]), 
                yaxis=dict(range=[l_p2_min - pad_y, l_p2_max + pad_y], scaleanchor="x", scaleratio=1), 
                height=700, template='plotly_white'
            )
            st.plotly_chart(fig, width="stretch")
            with st.expander("View Raw Run Coordinates"):
                st.dataframe(pd.DataFrame(doe_pts, columns=[param_1, param_2], index=range(1, l_num_runs+1)), width="stretch")

with tab2:
    if step == 2:
        real_results = None 
        
        if uploaded_csv is not None and not run_button:
            st.success(f"📂 Rendering baseline data from: {uploaded_csv.name}")
            real_results = prev_df
                
        elif run_button:
            if not st.session_state.is_initialized:
                st.error("⚠️ You must Initialise the DoE before launching the solver!")
            elif not targets:
                st.error("⚠️ Please select at least one Target Output in the sidebar.")
            else:
                actual_runs = len(st.session_state.doe_points)
                status_box = st.empty()
                
                run_dir = os.path.join(os.getcwd(), run_name)
                os.makedirs(run_dir, exist_ok=True)
                
                matrix_df = pd.DataFrame(st.session_state.doe_points.copy(), columns=[param_1, param_2] if is_2d else [param_1])
                def prepare_values(col_name, val): 
                    return math.radians(val) if "Angle" in col_name else val 
                    
                matrix_df[param_1] = matrix_df[param_1].apply(lambda x: prepare_values(param_1, x))
                if is_2d: 
                    matrix_df[param_2] = matrix_df[param_2].apply(lambda x: prepare_values(param_2, x))
                matrix_df.to_csv("sweep_matrix.csv", index=False)
                
                with open("sweep_config.txt", "w") as f:
                    f.write(",".join(targets) + "\n" + ",".join(selected_ff) + "\n" + run_dir.replace("\\", "/") + "\n") 
                
                # --- MAP EXACT TAGS FOR STAR-CCM+ ---
                with open("geometry_swap.csv", "w") as f:
                    for cad_path, target_name in st.session_state.cad_mappings.items():
                        if target_name != "Ignore":
                            if "Front" in target_name and "Wheel" in target_name:
                                target_tag = "Front Wheel"
                            elif "Rear" in target_name and "Wheel" in target_name:
                                target_tag = "Rear Wheel"
                            else:
                                target_tag = target_name 
                                
                            f.write(f"{cad_path},{target_name},{target_tag}\n")
                    
                status_box.info(f"🚀 Launching STAR-CCM+ on {cores} Cores... Telemetry will stream below and save to '{run_name}'.")
                
                # --- KEEP WINDOWS AWAKE ---
                ES_CONTINUOUS = 0x80000000
                ES_SYSTEM_REQUIRED = 0x00000001
                try:
                    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
                except Exception:
                    pass
                
                try:
                    sweep_start_time = time.time() - 5
                    log_file_path = os.path.join(run_dir, "starccm_batch.log")
                    
                    process = subprocess.Popen([
                        starccm_exe, 
                        "-np", str(cores), 
                        "-batch", "Geometry_Janitor.java,master_sweep.java", 
                        st.session_state.sim_file_path
                    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    
                    csv_path = os.path.join(run_dir, "Aero_Map_Results.csv")
                    live_monitor = st.empty()
                    
                    with open(log_file_path, "w") as log_file:
                        last_telemetry_check = time.time()
                        for line in iter(process.stdout.readline, ""):
                            # 1. Print directly to standard console
                            sys.stdout.write(line)
                            sys.stdout.flush()
                            
                            # 2. Write to log file
                            log_file.write(line)
                            log_file.flush()
                            
                            # 3. Periodically update Streamlit Telemetry Table
                            if time.time() - last_telemetry_check > 3:
                                if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
                                    try:
                                        live_results = pd.read_csv(csv_path)
                                        if prev_df is not None:
                                            max_id = prev_df['Run_ID'].max() if 'Run_ID' in prev_df.columns else len(prev_df)
                                            live_results['Run_ID'] += max_id
                                            live_display = pd.concat([prev_df, live_results], ignore_index=True)
                                        else: 
                                            live_display = live_results
                                            
                                        with live_monitor.container():
                                            st.markdown(f"### 📡 Live Telemetry: Run {len(live_results)} / {actual_runs} Completed")
                                            st.dataframe(live_display, width='stretch')
                                    except Exception:
                                        pass
                                last_telemetry_check = time.time()
                    
                    process.wait()
                    live_monitor.empty()
                    
                    if process.returncode == 0 or os.path.exists("stop_sweep.txt"): 
                        status_box.success("✅ Sweep Complete!")
                    else: 
                        status_box.error(f"🚨 STAR-CCM+ Execution Failed. Exit code: {process.returncode}. Check starccm_batch.log for details.")
                    
                    if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
                        try:
                            new_results = pd.read_csv(csv_path)
                            if prev_df is not None:
                                max_id = prev_df['Run_ID'].max() if 'Run_ID' in prev_df.columns else len(prev_df)
                                
                                for folder_name in os.listdir(run_dir):
                                    folder_path = os.path.join(run_dir, folder_name)
                                    if os.path.isdir(folder_path) and os.path.getmtime(folder_path) > sweep_start_time:
                                        if folder_name.startswith("Run_"):
                                            parts = folder_name.split("_", 2)
                                            if len(parts) >= 2 and parts[1].isdigit():
                                                new_id = int(parts[1]) + max_id
                                                new_folder_name = f"Run_{new_id}_{parts[2]}" if len(parts) > 2 else f"Run_{new_id}"
                                                try: os.rename(folder_path, os.path.join(run_dir, new_folder_name))
                                                except Exception: pass
                                
                                new_results['Run_ID'] += max_id
                                real_results = pd.concat([prev_df, new_results], ignore_index=True)
                                real_results.to_csv(os.path.join(run_dir, f"{run_name}.csv"), index=False)
                                try: os.remove(csv_path)
                                except Exception: pass
                            else:
                                real_results = new_results
                                real_results.to_csv(os.path.join(run_dir, f"{run_name}.csv"), index=False)
                                try: os.remove(csv_path)
                                except Exception: pass
                        except Exception as csv_err:
                            st.error(f"🚨 Failed to process the results CSV: {csv_err}")
                    else:
                        st.error("🚨 Output CSV was empty or missing. Check starccm_batch.log to see the Java error log!")
                        
                except Exception as e:
                    status_box.error(f"🚨 Process Launch Failed: {e}")
                    
                finally:
                    try:
                        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                    except Exception:
                        pass
                
        else:
            st.info("Review your Pre-Run Evaluation. Once satisfied, hit **Launch Batch**.")

        # ==========================================
        # --- UNIFIED POST-RUN RESULTS RENDERING ---
        # ==========================================
        if real_results is not None:
            if run_button:
                if os.path.exists("sweep_matrix.csv"): shutil.move("sweep_matrix.csv", os.path.join(run_dir, "sweep_matrix.csv"))
                if os.path.exists("sweep_config.txt"): shutil.move("sweep_config.txt", os.path.join(run_dir, "sweep_config.txt"))
                if os.path.exists("geometry_swap.csv"): shutil.move("geometry_swap.csv", os.path.join(run_dir, "geometry_swap.csv"))

            param_1 = real_results.columns[1]
            col2_name = real_results.columns[2].lower()
            
            is_2d_plot = any(x in col2_name for x in ['height', 'angle', 'yaw', 'pitch', 'roll', 'sweep', 'radius'])
            param_2_plot = real_results.columns[2] if is_2d_plot else None
            plot_targets = list(real_results.columns[3:]) if is_2d_plot else list(real_results.columns[2:])
            plot_targets = [t for t in plot_targets if not t.startswith('95%')]
            
            if not plot_targets:
                st.warning("⚠️ No valid target columns found to plot in the data.")
            else:
                result_tabs = st.tabs(plot_targets)
                p1_min_true, p1_max_true = real_results[param_1].min(), real_results[param_1].max()
                
                for i, target in enumerate(plot_targets):
                    with result_tabs[i]:
                        st.markdown(f"### Surrogate Response Model: {target}")
                        x_true = real_results[param_1].values 
                        z_true = real_results[target].values
                        
                        if not is_2d_plot:
                            X_train = x_true.reshape(-1, 1)
                            scaler = MinMaxScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(0.2, 10.0), nu=1.5)
                            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=15, normalize_y=True)
                            gp.fit(X_train_scaled, z_true)
                            x_grid = np.linspace(p1_min_true, p1_max_true, 100).reshape(-1, 1)
                            y_pred, sigma = gp.predict(scaler.transform(x_grid), return_std=True)
                            margin_95 = sigma * 1.96
                            
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(x=np.concatenate([x_grid.flatten(), x_grid.flatten()[::-1]]), y=np.concatenate([y_pred - margin_95, (y_pred + margin_95)[::-1]]), fill='toself', fillcolor='rgba(0, 0, 255, 0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", name='95% Confidence Interval'))
                            htemp_1d = f"{param_1}: %{{x:.4f}}<br>{target}: %{{y:.4f}}<br>95% Confidence Bound (±): %{{customdata:.5f}}<extra></extra>"
                            fig2.add_trace(go.Scatter(x=x_grid.flatten(), y=y_pred, customdata=margin_95, mode='lines', line=dict(color='blue', width=3), name='GP Surrogate Mean', hovertemplate=htemp_1d))
                            htemp_truth = f"Run ID: %{{text}}<br>{param_1}: %{{x:.4f}}<br>{target}: %{{y:.4f}}<extra></extra>"
                            fig2.add_trace(go.Scatter(x=x_true, y=z_true, mode='markers+text', marker=dict(size=10, color='black'), text=[str(j) for j in real_results['Run_ID']], textposition="top center", name='CFD Truth Data', hovertemplate=htemp_truth))
                            fig2.update_layout(xaxis_title=param_1, yaxis_title=target, height=500, template='plotly_white')
                            st.plotly_chart(fig2, width="stretch")
                        else:
                            y_true = real_results[param_2_plot].values
                            p2_min_true, p2_max_true = real_results[param_2_plot].min(), real_results[param_2_plot].max()
                            X_train = np.c_[x_true, y_true]
                            scaler = MinMaxScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(0.2, 10.0), nu=1.5)
                            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=15, normalize_y=True)
                            gp.fit(X_train_scaled, z_true)
                            x = np.linspace(p1_min_true, p1_max_true, 50)
                            y = np.linspace(p2_min_true, p2_max_true, 50)
                            X, Y = np.meshgrid(x, y)
                            X_grid_scaled = scaler.transform(np.c_[X.ravel(), Y.ravel()])
                            y_pred, sigma = gp.predict(X_grid_scaled, return_std=True)
                            Z_pred = y_pred.reshape(X.shape) 
                            Margin_95_pred = (sigma * 1.96).reshape(X.shape)
                            
                            htemp_3d = f"{param_1}: %{{x:.4f}}<br>{param_2_plot}: %{{y:.4f}}<br>{target}: %{{z:.4f}}<br>95% Confidence Bound (±): %{{customdata:.5f}}<extra></extra>"
                            fig2 = go.Figure()
                            fig2.add_trace(go.Surface(z=Z_pred, x=X, y=Y, customdata=Margin_95_pred, colorscale='Viridis', name='GP Surface', opacity=0.9, hovertemplate=htemp_3d))
                            htemp_truth_3d = f"Run ID: %{{text}}<br>{param_1}: %{{x:.4f}}<br>{param_2_plot}: %{{y:.4f}}<br>{target}: %{{z:.4f}}<extra></extra>"
                            fig2.add_trace(go.Scatter3d(x=x_true, y=y_true, z=z_true, mode='markers+text', marker=dict(size=5, color='black'), text=[str(j) for j in real_results['Run_ID']], textposition="top right", name='CFD Truth Data', hovertemplate=htemp_truth_3d))
                            fig2.update_layout(scene=dict(xaxis_title=param_1, yaxis_title=param_2_plot, zaxis_title=target), height=700)
                            st.plotly_chart(fig2, width="stretch")

            if export_csv_table:
                st.markdown("---")
                col_raw, col_dense = st.columns(2)
                with col_raw:
                    st.markdown("### Coarse Truth Data")
                    display_df = real_results.copy()
                    st.dataframe(display_df, width="stretch")
                    csv_raw = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download Coarse Data", data=csv_raw, file_name=f"{run_name}.csv", mime="text/csv", type="primary")

                with col_dense:
                    st.markdown("### Dense VD Map (2,500 points)")
                    vd_df = pd.DataFrame()
                    if is_2d_plot:
                        vd_df[param_1], vd_df[param_2_plot] = X.flatten(), Y.flatten()
                        X_train = np.c_[x_true, y_true]
                        scaler = MinMaxScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_grid_scaled = scaler.transform(np.c_[X.ravel(), Y.ravel()])
                        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(0.2, 10.0), nu=1.5)
                        for tgt in plot_targets:
                            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=15, normalize_y=True)
                            gp.fit(X_train_scaled, real_results[tgt].values)
                            preds, sigmas = gp.predict(X_grid_scaled, return_std=True)
                            vd_df[tgt] = preds
                            vd_df[f"95%_Bound_±_{tgt}"] = sigmas * 1.96
                    else:
                        vd_df[param_1] = x_grid.flatten()
                        X_train = x_true.reshape(-1, 1)
                        scaler = MinMaxScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_grid_scaled = scaler.transform(x_grid)
                        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(0.2, 10.0), nu=1.5)
                        for tgt in plot_targets:
                            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=15, normalize_y=True)
                            gp.fit(X_train_scaled, real_results[tgt].values)
                            preds, sigmas = gp.predict(X_grid_scaled, return_std=True)
                            vd_df[tgt] = preds
                            vd_df[f"95%_Bound_±_{tgt}"] = sigmas * 1.96
                    
                    st.dataframe(vd_df, width="stretch")
                    csv_dense = vd_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Download Dense VD Map", data=csv_dense, file_name=f"{run_name}_VD_Dense.csv", mime="text/csv", type="secondary")