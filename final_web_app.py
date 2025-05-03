import streamlit as st
import os
import matplotlib.pyplot as plt
from ase.data import covalent_radii
from io import BytesIO
from ase.io import read
from ase.visualize.plot import plot_atoms
from ase.neighborlist import neighbor_list
import ase
from ase.atoms import Atoms
from llmatdesign.core.agent import Agent
from llmatdesign.modules.llms import AskLLM
from llmatdesign.core.discover import discover_bandgap
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
def plot_structure_3d(atoms: Atoms, bond_cutoff=1.1):
    xyz = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    n_atoms = len(atoms)

    # map element to a color
    symbols = list(dict.fromkeys(syms))
    cmap = plt.get_cmap('tab20', len(symbols))
    palette = {
        sym: '#%02x%02x%02x' % tuple(int(255 * x) for x in cmap(i)[:3])
        for i, sym in enumerate(symbols)
    }
    colors = [palette.get(s, 'grey') for s in syms]

    # Compute pairwise distances
    distances = squareform(pdist(xyz))
    
    # Compute covalent radii sum matrix
    radii = np.array([covalent_radii[atom.number] for atom in atoms])
    cutoff_matrix = bond_cutoff * (radii[:, None] + radii[None, :])

    # Build bonded pairs (including O–O etc.)
    pairs = set()
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if distances[i, j] <= cutoff_matrix[i, j]:
                pairs.add((i, j))

    # Plot atoms
    atom_trace = go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode='markers',
        marker=dict(size=12, color=colors, line=dict(width=1, color='black')),
        text=syms,
        hoverinfo='text'
    )

    # Plot bonds
    bond_x, bond_y, bond_z = [], [], []
    for ai, aj in pairs:
        x0, y0, z0 = xyz[ai]
        x1, y1, z1 = xyz[aj]
        bond_x += [x0, x1, None]
        bond_y += [y0, y1, None]
        bond_z += [z0, z1, None]

    bond_trace = go.Scatter3d(
        x=bond_x, y=bond_y, z=bond_z,
        mode='lines',
        line=dict(color='white', width=4),
        hoverinfo='none'
    )

    fig = go.Figure([bond_trace, atom_trace])
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

# 0) PAGE CONFIG
st.set_page_config(
    page_title="Autonomous Materials Discovery",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1) DARK THEME CSS
st.markdown(
    """
    <style>
      /* overall background */
      body, .stApp {
        background-color: #000000 !important;
        color: #CCCCCC !important;
      }
      /* sidebar */
      .sidebar .sidebar-content {
        background-color: #111111 !important;
        color: #CCCCCC !important;
      }
      /* headers & titles */
      h1, h2, h3, h4, h5, h6 {
        color: #00FFFF !important;  /* cyan */
      }
      /* subheaders and markdown bold */
      .stMarkdown b, .stMarkdown strong {
        color: #FF00FF !important;  /* magenta */
      }
      /* metric labels */
      .stMetricLabel {
        color: #00FF00 !important;  /* lime */
      }
      /* buttons */
      .stButton>button {
        background-color: #1A1A1A !important;
        color: #00FFFF !important;
        border: 1px solid #00FFFF !important;
      }
      .stButton>button:hover {
        background-color: #333333 !important;
      }
      /* progress bar */
      .stProgress>div>div>div>div {
        background-color: #00FFFF !important;
      }
      /* expander caret */
      button[aria-expanded="false"] > .css-1d391kg {
        color: #FF00FF !important;
      }
      button[aria-expanded="true"] > .css-1d391kg {
        color: #00FF00 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🧪 Autonomous Materials Discovery")
# 2) Sidebar: LLM & Inputs
st.sidebar.header("🔧 Configuration")
llm_option = st.sidebar.selectbox("LLM Model", ["gemini-2.0-flash", "Llama-3.3-70B-Instruct"])
api_key     = st.sidebar.text_input("API Key", type="password")
chemical_formula = st.sidebar.text_input("Formula", "CdCu2GeS4")
target_bandgap   = st.sidebar.number_input("Target band gap", 1.4, step=0.1)
max_iters        = st.sidebar.slider("Max iters", 10, 100, 55)

if not api_key:
    st.sidebar.error("Please enter your API key.")
    st.stop()

# Init
llm = AskLLM(llm_option, api_key=api_key)
agent = Agent(
    llm,
    save_path="outputs/cifs/",
    forcefield_config_path=os.path.abspath("checkpoints/matdeeplearn/force_field/config.yml"),
    bandgap_config_path=os.path.abspath("checkpoints/matdeeplearn/band_gap/config.yml"),
    formation_energy_config_path=os.path.abspath("checkpoints/matdeeplearn/formation_energy/config.yml"),
    mp_api_key="HykOG4IhaN8Xi2kH3dq0lr42nLpcMBZE"
)

if st.button("🚀 Run Discovery"):
    progress = st.progress(0)
    status   = st.empty()

    # three tabs
    tab_iters, tab_plot, tab_struct = st.tabs(
        ["Iterations","Convergence","Final Structure"]
    )

    results = []
    with st.spinner("Running..."):
        for step in discover_bandgap(
            agent, chemical_formula, None, None, target_bandgap, max_iters
        ):
            it   = step["iteration"]
            bg   = step["band_gap"]
            unc  = step["uncertainty"]
            mod  = step["modification"]
            refl = step["reflection"]
            alert = step["threshold_exceeded"]

            progress.progress(min(it/max_iters,1.0))
            status.markdown(f"### Iteration {it}")

            # iteration details
            with tab_iters:
                cols = st.columns([2,2,1,3])
                cols[0].markdown(f"**Mod:** `{mod}`")
                cols[1].metric("Band gap", f"{bg:.3f} eV", delta=f"{bg-target_bandgap:+.3f}")
                cols[2].metric("Uncertainty", f"{unc:.3f} eV", "" )
                if alert:
                    cols[2].warning("⚠️ High uncertainty, need DFT re-evaluation.")
                cols[3].info(refl)

            results.append(step)
            if step["converged"]:
                st.success(f"Converged at step {it}!")
                break

        if not results or not results[-1]["converged"]:
            st.warning("❌ Did not converge in allotted iterations.")

    # convergence plot
    with tab_plot:
        band_gaps = [r["band_gap"] for r in results]
        fig, ax = plt.subplots(1,1,figsize=(6,3))
        ax.plot(range(1,len(band_gaps)+1), band_gaps, "-o")
        ax.axhline(target_bandgap, color="r", ls="--", label="Target")
        ax.set(xlabel="Iteration", ylabel="Predicted band gap (eV)")
        ax.legend()
        st.pyplot(fig)

    # final structure
    with tab_struct:
        st.markdown("## 🏗️ Final 3D Structure")
        if results:
            final_atoms = results[-1]["new_structure"]

            # use your plotly 3D viewer
            fig3d = plot_structure_3d(final_atoms, bond_cutoff=1.6)
            st.plotly_chart(fig3d, use_container_width=True)

            # optionally let user download CIF
            # cif_buf = final_atoms.write(format="cif")
            # st.download_button(
            #     "📥 Download CIF",
            #     data=cif_buf,
            #     file_name=f"{chemical_formula}_final.cif",
            #     mime="chemical/x-cif"
            # )
        else:
            st.error("No final structure available.")

