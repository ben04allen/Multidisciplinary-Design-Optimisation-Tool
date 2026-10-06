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
if 'nx_assembly_path' not in st.session_state:
    st.session_state.nx_assembly_path = "Choose NX Assembly (.prt) file from folder"

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

def open_prt_dialog():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1) 
    file_path = filedialog.askopenfilename(filetypes=[("NX Part Files", "*.prt")])
    root.destroy()
    return file_path

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
    st.sidebar.markdown("### NX CAD Integration")
    st.sidebar.caption("Link the Master Assembly for geometric sweeps.")

    col_nx_path, col_nx_browse = st.sidebar.columns([4, 1])
    nx_file_input = col_nx_path.text_input("Master Assembly (.prt) Path:", value=st.session_state.nx_assembly_path, label_visibility="collapsed")

    if col_nx_browse.button("📁", key="nx_browse"):
        selected_nx_file = open_prt_dialog()
        if selected_nx_file:
            # tkinter sometimes returns forward slashes; standardize them
            st.session_state.nx_assembly_path = selected_nx_file.replace("/", "\\")
            st.rerun() 
    else:
        st.session_state.nx_assembly_path = nx_file_input

    if st.sidebar.button("🔍 Scan NX Parameters", width="stretch"):
        if not os.path.exists(st.session_state.nx_assembly_path):
            st.sidebar.error("NX Assembly file not found!")
        else:
            with st.spinner("Booting NX in headless mode..."):
                nx_executable = r"C:\Program Files\Siemens\NX1926\NXBIN\run_journal.exe"
                command = [nx_executable, "nx_probe.py", "-args", st.session_state.nx_assembly_path]
                result = subprocess.run(command, capture_output=True, text=True)

                if os.path.exists("nx_metadata.json"):
                    with open("nx_metadata.json", "r") as f:
                        data = json.load(f)
                    
                    st.session_state.nx_params = data.get("nx_parameters", [])
                    st.sidebar.success(f"Loaded {len(st.session_state.nx_params)} NX parameters!")
                else:
                    st.sidebar.error("Failed to read NX parameters.")
                    st.sidebar.code(result.stderr)

    if st.session_state.get('nx_params'):
        with st.sidebar.expander("Detected NX Parameters"):
            for p in st.session_state.nx_params:
                st.write(f"- {p}")

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
    all_sweep_params = st.session_state.sim_params + st.session_state.get('nx_params', [])
    p1_default_idx = 0
    if prev_df is not None and len(prev_df.columns) > 1:
        if prev_df.columns[1] in st.session_state.sim_params:
            p1_default_idx = st.session_state.sim_params.index(prev_df.columns[1])
            
    param_1 = st.sidebar.selectbox("Select Target", all_sweep_params, index=p1_default_idx, key="p1")
    col_b1, col1, col2 = st.sidebar.columns(3)
    p1_base = col_b1.number_input("Base", value=0.033, step=0.005, format="%.4f", key="p1_base")
    p1_min = col1.number_input("Min", value=0.000, step=0.005, format="%.4f", key="p1_min")
    p1_max = col2.number_input("Max", value=0.050, step=0.005, format="%.4f", key="p1_max")

    if is_2d:
        st.sidebar.markdown("### Parameter 2 (Y-Axis)")
        p2_default_idx = 0
        if prev_df is not None and len(prev_df.columns) > 2:
            if prev_df.columns[2] in st.session_state.sim_params:
                p2_default_idx = st.session_state.sim_params.index(prev_df.columns[2])
                
        param_2 = st.sidebar.selectbox("Select Target", all_sweep_params, index=p2_default_idx, key="p2")
        col_b2, col3, col4 = st.sidebar.columns(3)
        p2_base = col_b2.number_input("Base", value=0.033, step=0.005, format="%.4f", key="p2_base")
        p2_min = col3.number_input("Min", value=0.000, step=0.005, format="%.4f", key="p2_min")
        p2_max = col4.number_input("Max", value=0.050, step=0.005, format="%.4f", key="p2_max")
    else:
        param_2, p2_base, p2_min, p2_max = None, None, None, None

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
        st.session_state.perm_use_adaptive = use_adaptive
        st.session_state.perm_al_targets = al_targets    
        
        st.session_state.perm_p1_min = p1_min
        st.session_state.perm_p1_max = p1_max
        st.session_state.perm_p1 = param_1
        if is_2d:
            st.session_state.perm_p2_base = p2_base
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
    param_1 = st.session_state.perm_p1
    param_2 = st.session_state.get('perm_p2', None)

def generate_doe():
    def get_smart_decimals(val):
        if val == 0: return 3
        mag = math.floor(math.log10(abs(val)))
        return max(0, -(mag - 1))
    
    l_p1_base = st.session_state.get('perm_p1_base', 0.0)
    l_p1_min = st.session_state.perm_p1_min
    l_p1_max = st.session_state.perm_p1_max
    
    l_p2_base = st.session_state.get('perm_p2_base', 0.0)
    l_p2_min = st.session_state.get('perm_p2_min', 0)
    l_p2_max = st.session_state.get('perm_p2_max', 0)
    
    l_num_runs = st.session_state.prev_num_runs
    l_is_2d = st.session_state.sweep_dim == 2
    
    l_apply_constraints = st.session_state.get('perm_apply_constraints', False)
    l_max_rrh_ratio = st.session_state.get('perm_max_rrh_ratio', None)
    
    # FETCH ADAPTIVE SETTINGS
    l_use_adaptive = st.session_state.get('perm_use_adaptive', False)
    l_al_targets = st.session_state.get('perm_al_targets', [])
    is_appending = prev_df is not None

    p1_decimals = get_smart_decimals(l_p1_min)
    p2_decimals = get_smart_decimals(l_p2_min) if l_is_2d else 0
    
    # =========================================================
    # 1. ANCHOR GENERATION (Boundary Augmentation)
    # =========================================================
    anchors = []
    if not is_appending:
        # A. Always inject the user's Baseline as Run 1
        anchors.append([l_p1_base, l_p2_base] if l_is_2d else [l_p1_base])
        
        # B. Inject Domain Extremes
        if not l_is_2d:
            anchors.extend([[l_p1_min], [l_p1_max]])
        else:
            # Generate the 4 absolute corners + 3 potential constraint intersections
            raw_corners = [
                [l_p1_min, l_p2_min],
                [l_p1_max, l_p2_min],
                [l_p1_min, l_p2_max],
                [l_p1_max, l_p2_max]
            ]
            
            if l_apply_constraints and l_max_rrh_ratio is not None:
                if l_max_rrh_ratio > 0:
                    raw_corners.append([l_p2_max / l_max_rrh_ratio, l_p2_max])  # Top bound intersection
                raw_corners.append([l_p1_min, l_p1_min * l_max_rrh_ratio])      # Left bound intersection
                raw_corners.append([l_p1_max, l_p1_max * l_max_rrh_ratio])      # Right bound intersection

            # Filter vertices mathematically to ensure they sit inside the feasible domain
            for pt in raw_corners:
                p1_c, p2_c = pt[0], pt[1]
                if not (l_p1_min - 1e-5 <= p1_c <= l_p1_max + 1e-5): continue
                if not (l_p2_min - 1e-5 <= p2_c <= l_p2_max + 1e-5): continue
                if l_apply_constraints and l_max_rrh_ratio is not None:
                    if p2_c > (p1_c * l_max_rrh_ratio) + 1e-5: continue
                anchors.append([p1_c, p2_c])

    # C. Format, Round, and Deduplicate
    if len(anchors) > 0:
        anchors_arr = np.array(anchors)
        anchors_arr[:, 0] = np.round(anchors_arr[:, 0], p1_decimals)
        if l_is_2d:
            anchors_arr[:, 1] = np.round(anchors_arr[:, 1], p2_decimals)
            
        # Deduplicate while strictly preserving the run sequence (Baseline must stay Run 1)
        _, unique_indices = np.unique(anchors_arr, axis=0, return_index=True)
        anchors_arr = anchors_arr[np.sort(unique_indices)]
    else:
        anchors_arr = np.array([]).reshape(0, 2 if l_is_2d else 1)

    # Calculate remaining run budget
    runs_to_generate = l_num_runs if is_appending else l_num_runs - len(anchors_arr)

    # =========================================================
    # 2. ACTIVE LEARNING / LHS (For remaining budget)
    # =========================================================
    new_points = np.array([]).reshape(0, 2 if l_is_2d else 1)
    
    if runs_to_generate > 0:
        if is_appending and l_use_adaptive and len(l_al_targets) > 0:
            target = l_al_targets[0]
            x_true = prev_df[param_1].values
            z_true = prev_df[target].values
            
            scaler = MinMaxScaler()
            kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0] if l_is_2d else 1.0, length_scale_bounds=(0.2, 10.0), nu=1.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=5, normalize_y=True)
            
            if l_is_2d:
                y_true = prev_df[param_2].values
                X_train_scaled = scaler.fit_transform(np.c_[x_true, y_true])
            else:
                X_train_scaled = scaler.fit_transform(x_true.reshape(-1, 1))
                
            gp.fit(X_train_scaled, z_true)
            
            sampler = qmc.LatinHypercube(d=2 if l_is_2d else 1)
            candidates_raw = sampler.random(n=5000)
            
            if l_is_2d:
                candidates = qmc.scale(candidates_raw, [l_p1_min, l_p2_min], [l_p1_max, l_p2_max])
                if l_apply_constraints and l_max_rrh_ratio is not None:
                    candidates = candidates[candidates[:, 1] <= (candidates[:, 0] * l_max_rrh_ratio)]
            else:
                candidates = qmc.scale(candidates_raw, [l_p1_min], [l_p1_max])
                
            _, sigma = gp.predict(scaler.transform(candidates), return_std=True)
            highest_uncert_indices = np.argsort(sigma)[::-1]
            selected_indices = highest_uncert_indices[::25][:runs_to_generate]
            if len(selected_indices) < runs_to_generate:
                 selected_indices = highest_uncert_indices[:runs_to_generate]
                 
            new_points = candidates[selected_indices]
            
        else:
            sampler = qmc.LatinHypercube(d=2 if l_is_2d else 1)
            if not l_is_2d:
                new_points = qmc.scale(sampler.random(n=runs_to_generate), [l_p1_min], [l_p1_max])
            else:
                if l_apply_constraints and l_max_rrh_ratio is not None:
                    raw_samples = sampler.random(n=runs_to_generate * 10) 
                    scaled = qmc.scale(raw_samples, [l_p1_min, l_p2_min], [l_p1_max, l_p2_max])
                    new_points = scaled[scaled[:, 1] <= (scaled[:, 0] * l_max_rrh_ratio)][:runs_to_generate]
                else:
                    new_points = qmc.scale(sampler.random(n=runs_to_generate), [l_p1_min, l_p2_min], [l_p1_max, l_p2_max])

        # Format decimal places for newly generated points
        new_points[:, 0] = np.round(new_points[:, 0], p1_decimals)
        if l_is_2d:
            new_points[:, 1] = np.round(new_points[:, 1], p2_decimals)

    # =========================================================
    # 3. FINAL ASSEMBLY
    # =========================================================
    if not is_appending:
        final_points = np.vstack((anchors_arr, new_points)) if new_points.size else anchors_arr
        return final_points[:l_num_runs] # Truncates if budget is smaller than anchor count
    else:
        return new_points

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
                
                if os.path.exists("stop_sweep.txt"):
                    os.remove("stop_sweep.txt")
                
                matrix_df = pd.DataFrame(st.session_state.doe_points.copy(), columns=[param_1, param_2] if is_2d else [param_1])
                
                nx_params = st.session_state.get('nx_params', [])
                uses_nx = param_1 in nx_params or (is_2d and param_2 in nx_params)
                
                ES_CONTINUOUS = 0x80000000
                ES_SYSTEM_REQUIRED = 0x00000001
                try: ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
                except Exception: pass
                
                try:
                    log_file_path = os.path.join(run_dir, "starccm_batch.log")
                    live_monitor = st.empty()
                    
                    nx_exe = r"C:\Program Files\Siemens\NX1926\NXBIN\run_journal.exe"
                    cumulative_results = prev_df if prev_df is not None else pd.DataFrame()
                    
                    with open(log_file_path, "w", encoding="utf-8", errors="replace") as log_file:
                        for idx, row in matrix_df.iterrows():
                            if os.path.exists("stop_sweep.txt"):
                                status_box.warning("🚨 Sweep aborted by user.")
                                break
                                
                            run_id = int(idx) + 1
                            if prev_df is not None:
                                run_id += prev_df['Run_ID'].max() if 'Run_ID' in prev_df.columns else len(prev_df)
                                
                            status_box.info(f"🚀 Processing Run {idx+1}/{actual_runs} (ID: {run_id})...")
                            
                            # --- 1. CREATE DYNAMIC SUBFOLDER ---
                            subfolder_name = f"Run_{run_id}"
                            for col in matrix_df.columns:
                                subfolder_name += f"_{col}_{row[col]}"
                            specific_run_dir = os.path.join(run_dir, subfolder_name)
                            os.makedirs(specific_run_dir, exist_ok=True)
                            
                            # Route STAR-CCM+ outputs to the specific subfolder
                            with open("sweep_config.txt", "w") as f:
                                f.write(",".join(targets) + "\n" + ",".join(selected_ff) + "\n" + specific_run_dir.replace("\\", "/") + "\n") 
                            
                            # --- 2. NX CAD MORPHING ---
                            parasolid_name = f"Run_{run_id}_Geometry.x_t" if uses_nx else "Baseline_Geometry.x_t"
                            parasolid_path = os.path.join(run_dir, parasolid_name)
                            
                            if uses_nx or (not uses_nx and not os.path.exists(parasolid_path)):
                                status_box.info(f"⚙️ Morphing CAD in NX for Run {run_id}...")
                                nx_config = {
                                    "assembly_path": st.session_state.nx_assembly_path,
                                    "export_path": parasolid_path,
                                    "parameters": {}
                                }
                                if param_1 in nx_params: nx_config["parameters"][param_1] = row[param_1]
                                if is_2d and param_2 in nx_params: nx_config["parameters"][param_2] = row[param_2]
                                
                                json_config_path = os.path.abspath("nx_run_config.json")
                                with open(json_config_path, "w") as f: json.dump(nx_config, f)
                                
                                nx_process = subprocess.run([nx_exe, "nx_morph.py", "-args", json_config_path], capture_output=True, text=True)
                                
                                if not os.path.exists(parasolid_path):
                                    st.error(f"🚨 NX failed to export the geometry for Run {run_id}!\n\nNX Error Log:\n{nx_process.stderr}")
                                    with open("stop_sweep.txt", "w") as f: f.write("ABORT")
                                    break
                            
                            # --- 3. WRITE CSVs FOR STAR-CCM+ ---
                            single_row_df = pd.DataFrame([row])
                            single_row_df.to_csv("sweep_matrix.csv", index=False)
                            
                            with open("geometry_swap.csv", "w") as f:
                                f.write(f"{parasolid_path},Parasolid_Import,Master_Assembly\n")
                            
                            # --- 4. LAUNCH CFD SOLVER ---
                            status_box.info(f"🚀 Launching STAR-CCM+ for Run {run_id}...")
                            process = subprocess.Popen([
                                starccm_exe, 
                                "-np", str(cores), 
                                "-batch", "Geometry_Janitor.java,master_sweep.java", 
                                st.session_state.sim_file_path
                            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding="utf-8", errors="replace")
                            
                            for line in iter(process.stdout.readline, ""):
                                sys.stdout.write(line)
                                sys.stdout.flush()
                                log_file.write(line)
                                log_file.flush()
                            
                            process.wait()
                            
                            # --- 5. PROCESS RESULTS & CLEANUP ---
                            csv_path = os.path.join(specific_run_dir, "Aero_Map_Results.csv")
                            if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
                                try:
                                    run_result = pd.read_csv(csv_path)
                                    run_result['Run_ID'] = run_id
                                    cumulative_results = pd.concat([cumulative_results, run_result], ignore_index=True)
                                    
                                    # Save the master CSV to the root folder
                                    cumulative_results.to_csv(os.path.join(run_dir, f"{run_name}.csv"), index=False)
                                    
                                    with live_monitor.container():
                                        st.markdown(f"### 📡 Live Telemetry: Run {idx+1} / {actual_runs} Completed")
                                        st.dataframe(cumulative_results, width='stretch')
                                        
                                except Exception as csv_err:
                                    st.error(f"🚨 Failed to process results CSV for Run {run_id}: {csv_err}")
                                    
                            if uses_nx and os.path.exists(parasolid_path):
                                try: os.remove(parasolid_path)
                                except: pass

                    if not os.path.exists("stop_sweep.txt"):
                        status_box.success("✅ Batch Complete!")
                        
                except Exception as e:
                    status_box.error(f"🚨 Process Launch Failed: {e}")
                    
                finally:
                    try: ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                    except Exception: pass
                    
                    if not cumulative_results.empty:
                        real_results = cumulative_results

        # ==========================================
        # --- UNIFIED POST-RUN RESULTS RENDERING ---
        # ==========================================
        if real_results is not None:
            if run_button:
                if os.path.exists("sweep_matrix.csv"): shutil.move("sweep_matrix.csv", os.path.join(run_dir, "sweep_matrix.csv"))
                if os.path.exists("sweep_config.txt"): shutil.move("sweep_config.txt", os.path.join(run_dir, "sweep_config.txt"))
                if os.path.exists("geometry_swap.csv"): shutil.move("geometry_swap.csv", os.path.join(run_dir, "geometry_swap.csv"))

           # --- DYNAMIC DIMENSION DETECTION ---
            known_outputs = ["cla", "cda", "cma", "cra", "cya", "balance", "cells", "solver", "cpu", "cop", "force", "moment", "area", "run_id"]
            
            # Identify columns that do NOT contain target report keywords
            parameter_cols = [col for col in real_results.columns if not any(kw in col.lower() for kw in known_outputs)]
            
            if len(parameter_cols) >= 2:
                is_2d_plot = True
                param_1 = parameter_cols[0]
                param_2_plot = parameter_cols[1]
                # Targets are everything else except Run_ID
                plot_targets = [col for col in real_results.columns if col not in parameter_cols and col != "Run_ID"]
            elif len(parameter_cols) == 1:
                is_2d_plot = False
                param_1 = parameter_cols[0]
                param_2_plot = None
                plot_targets = [col for col in real_results.columns if col not in parameter_cols and col != "Run_ID"]
            else:
                st.warning("⚠️ Could not detect parameters in the CSV.")
                plot_targets = []
                
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
                            
                            # --- NEW TOGGLE UI ---
                            view_style = st.radio("Visualization Style:", ["3D Surface (Interactive)", "2D Aero Map (Static)"], horizontal=True, key=f"view_{target}")
                            
                            if "3D" in view_style:
                                htemp_3d = f"{param_1}: %{{x:.4f}}<br>{param_2_plot}: %{{y:.4f}}<br>{target}: %{{z:.4f}}<br>95% Confidence Bound (±): %{{customdata:.5f}}<extra></extra>"
                                fig2 = go.Figure()
                                fig2.add_trace(go.Surface(z=Z_pred, x=X, y=Y, customdata=Margin_95_pred, colorscale='Viridis', name='GP Surface', opacity=0.9, hovertemplate=htemp_3d))
                                htemp_truth_3d = f"Run ID: %{{text}}<br>{param_1}: %{{x:.4f}}<br>{param_2_plot}: %{{y:.4f}}<br>{target}: %{{z:.4f}}<extra></extra>"
                                fig2.add_trace(go.Scatter3d(x=x_true, y=y_true, z=z_true, mode='markers+text', marker=dict(size=5, color='black'), text=[str(j) for j in real_results['Run_ID']], textposition="top right", name='CFD Truth Data', hovertemplate=htemp_truth_3d))
                                fig2.update_layout(scene=dict(xaxis_title=param_1, yaxis_title=param_2_plot, zaxis_title=target), height=700)
                                st.plotly_chart(fig2, width="stretch")
                                
                            else:
                                # --- NEW PLOTLY 2D AERO MAP ---
                                from plotly.subplots import make_subplots
                                
                                fig_2d = make_subplots(rows=1, cols=2, subplot_titles=(f"{target} Response", "Uncertainty"))
                                htemp_2d = f"{param_1}: %{{x:.4f}}<br>{param_2_plot}: %{{y:.4f}}<br>{target}: %{{z:.4f}}<br>95% Bound (±): %{{customdata:.4f}}<extra></extra>"
                                
                                # -----------------------------------
                                # PLOT 1: Clean Aero Map (20 Levels)
                                # -----------------------------------
                                fig_2d.add_trace(go.Contour(
                                    z=Z_pred, x=x, y=y, customdata=Margin_95_pred,
                                    colorscale='Viridis',
                                    hovertemplate=htemp_2d,
                                    ncontours=20,  # Forces higher contour density
                                    colorbar=dict(title=target, x=0.45, len=1.0), 
                                ), row=1, col=1)
                                
                                # -----------------------------------
                                # PLOT 2: Fog Overlay + Scatter
                                # -----------------------------------
                                # A. Base vivid layer (20 Levels)
                                fig_2d.add_trace(go.Contour(
                                    z=Z_pred, x=x, y=y, customdata=Margin_95_pred,
                                    colorscale='Viridis',
                                    showscale=False,
                                    ncontours=100,  # Matches Plot 1
                                    contours=dict(showlines=False),
                                    hovertemplate=htemp_2d
                                ), row=1, col=2)
                                
                                # B. The White Fog Layer (100 Levels, Smooth Fade)
                                fog_colorscale = [[0.0, 'rgba(255, 255, 255, 0.0)'], [1.0, 'rgba(255, 255, 255, 0.1)']]
                                fig_2d.add_trace(go.Contour(
                                    z=Margin_95_pred, x=x, y=y,
                                    colorscale=fog_colorscale,
                                    hoverinfo='skip',
                                    showscale=False,  # Removes the ugly black colorbar
                                    ncontours=100,    # High resolution for the fog
                                    contours=dict(showlines=False)  # Hides fog lines for a smooth gradient fade
                                ), row=1, col=2)
                                
                                # C. Truth Data Scatter (Black X markers)
                                htemp_scatter = f"Run ID: %{{text}}<br>{param_1}: %{{x:.4f}}<br>{param_2_plot}: %{{y:.4f}}<br>{target}: %{{customdata:.4f}}<extra></extra>"
                                fig_2d.add_trace(go.Scatter(
                                    x=x_true, y=y_true, customdata=z_true,
                                    mode='markers+text', 
                                    marker=dict(size=8, color='black', symbol='x'),
                                    text=[str(j) for j in real_results['Run_ID']], 
                                    textposition="top right", 
                                    hovertemplate=htemp_scatter,
                                    showlegend=False
                                ), row=1, col=2)
                                
                                # -----------------------------------
                                # FORMATTING & RENDERING
                                # -----------------------------------
                                fig_2d.update_layout(height=600, template='plotly_white')
                                fig_2d.update_xaxes(title_text=param_1, row=1, col=1)
                                fig_2d.update_yaxes(title_text=param_2_plot, row=1, col=1)
                                fig_2d.update_xaxes(title_text=param_1, row=1, col=2)
                                fig_2d.update_yaxes(title_text=param_2_plot, row=1, col=2)
                                
                                st.plotly_chart(fig_2d, width="stretch")

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
