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
import math
import random

# Lock the stochastic optimizer for DoE repeatability on the cloud
np.random.seed(42)
random.seed(42)

st.set_page_config(page_title="LFS Aero-Mapper", layout="wide")
st.title("LFS Aerodynamic Optimization Framework (Cloud Viewer)")

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
    st.session_state.sim_params = ["(Upload CSV to populate)"]
if 'sim_reports' not in st.session_state:
    st.session_state.sim_reports = ["(Upload CSV to populate)"]
if 'cad_files' not in st.session_state:
    st.session_state.cad_files = []
if 'cad_mappings' not in st.session_state:
    st.session_state.cad_mappings = {}

# ==========================================
# SIDEBAR WIZARD ROUTING
# ==========================================
step = st.session_state.current_step

# ------------------------------------------
# WIZARD STEP 1: SIMULATION & CAD GEOMETRY
# ------------------------------------------
if step == 0:
    st.sidebar.header("Step 1: Simulation & Geometry")
    
    st.sidebar.info("☁️ **Cloud Mode:** Local STAR-CCM+ execution and .sim file probing are disabled. Proceed to Step 2 to upload a baseline CSV.")

    st.sidebar.markdown("**Simulation File (.sim):**")
    st.sidebar.file_uploader("Upload .sim file (Disabled in Web Viewer)", type=["sim"], disabled=True)

    if st.sidebar.button("🔍 Scan .sim File", width="stretch"):
        st.sidebar.success("Cloud Demo: Metadata scanning disabled. Proceed to upload baseline CSV.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### CAD Geometry Swap (Optional)")
    st.sidebar.caption("Select new .x_t or .step files to test in this sweep.")
    
    new_cad_files = st.sidebar.file_uploader("➕ Select CAD Parts", accept_multiple_files=True)
    if new_cad_files:
        for f in new_cad_files:
            if f.name not in st.session_state.cad_files:
                st.session_state.cad_files.append(f.name)
        
    if st.session_state.cad_files:
        st.sidebar.markdown("**Assign Geometry Tags:**")
        categories = ["Ignore", "Chassis", "Front Wing", "Rear Wing", "Floor", "Front Left Wheel", "Front Right Wheel", "Rear Left Wheel", "Rear Right Wheel"]
        
        for i, fname in enumerate(st.session_state.cad_files):
            st.session_state.cad_mappings[fname] = st.sidebar.selectbox(
                f"⚙️ {fname}", categories, key=f"cad_sel_{i}"
            )
            
        if st.sidebar.button("🗑️ Clear Selected CAD"):
            st.session_state.cad_files = []
            st.session_state.cad_mappings = {}
            st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("Next: Sweep Setup ➡️", type="primary", width="stretch"):
        st.session_state.current_step = 1
        st.rerun()

# ------------------------------------------
# WIZARD STEP 2: PARAMETERS & CONSTRAINTS
# ------------------------------------------
elif step == 1:
    st.sidebar.header("Step 2: Sweep Setup")
    
    uploaded_csv = st.sidebar.file_uploader("📂 Upload Baseline CSV:", type="csv")
    prev_df = pd.read_csv(uploaded_csv) if uploaded_csv is not None else None
    
    # Auto-populate parameters based on uploaded CSV for the cloud demo
    if prev_df is not None:
        st.session_state.sim_params = list(prev_df.columns)
        st.session_state.sim_reports = list(prev_df.columns)
    
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
    
    uploaded_csv = st.sidebar.file_uploader("📂 Confirm Baseline CSV:", type="csv", key="confirm_csv")
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

    col_btn1, col_btn2 = st.sidebar.columns(2)
    init_button = col_btn1.button("⚙️ Initialise DoE", width="stretch")
    run_button = col_btn2.button("🚀 Launch Batch", width="stretch", type="primary")
        
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
            st.info("☁️ **Cloud Demo Mode:** Local STAR-CCM+ execution is disabled in the web viewer. To view results, please upload a baseline CSV in Step 2.")
        else:
            st.info("Review your Pre-Run Evaluation. Upload a CSV to view the aerodynamic surrogate surfaces.")

        # ==========================================
        # --- UNIFIED POST-RUN RESULTS RENDERING ---
        # ==========================================
        if real_results is not None:
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
                    st.download_button(label="📥 Download Coarse Data", data=csv_raw, file_name=f"Viewer_Export.csv", mime="text/csv", type="primary")

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
                    st.download_button(label="📥 Download Dense VD Map", data=csv_dense, file_name=f"Viewer_Dense_Map.csv", mime="text/csv", type="secondary")
