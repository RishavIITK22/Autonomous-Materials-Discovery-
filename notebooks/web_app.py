#A WEB APPLICATION FOR LLMatDesign Framework
import streamlit as st
import json
from llmatdesign.core.agent import Agent
from llmatdesign.modules.llms import AskLLM
from llmatdesign.core.discover import discover_bandgap
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Autonomous Materials Discovery")
st.sidebar.header("LLM Configuration")
llm_option = st.sidebar.selectbox("Select LLM Model", ["gemini-2.0-flash", "Llama-3.3-70B-Instruct"])
api_key = st.sidebar.text_input("API Key", value="", type="password")

llm=None

# Initialize the LLM based on your chosen model
if llm_option == "gemini-2.0-flash":
    # Set up for Gemini (adjust as necessary)
    llm = AskLLM(llm_option, api_key=api_key)
elif llm_option == "Llama-3.3-70B-Instruct":
    llama_api_url = st.sidebar.text_input("Llama API URL", value="https://cloud.olakrutrim.com/v1")
    # Set up for Llama (adjust as necessary)
    llm = AskLLM(llm_option, api_key=api_key)

if llm is None:
    st.error("Please select a supported LLM model in the sidebar.")
    st.stop()

# Initialize the agent
agent = Agent(
    llm,
    save_path="outputs/cifs/",
    forcefield_config_path="../checkpoints/matdeeplearn/force_field/config.yml",
    bandgap_config_path="../checkpoints/matdeeplearn/band_gap/config.yml",
    formation_energy_config_path="../checkpoints/matdeeplearn/formation_energy/config.yml",
    mp_api_key="HykOG4IhaN8Xi2kH3dq0lr42nLpcMBZE"
)
#Input for user
st.header("Material Input")
chemical_formula = st.text_input("Chemical Formula", value="CdCu2GeS4")
target_bandgap = st.number_input("Target Band Gap (eV)", value=1.4, min_value=0.0, step=0.1)

if st.button("Run Materials Discovery"):
    max_iters = 55  # or whatever your default is
    band_gaps = []
    uncertainty_thresh = 0.1  # 10% by default; you could make this configurable

    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []  # keep for the final plot

    with st.spinner("Running discovery process..."):
        try:
            # Iterate the generator live
            for step in discover_bandgap(
                agent,
                chemical_formula=chemical_formula,
                target_value=target_bandgap,
                max_iterations=max_iters
            ):
                it = step["iteration"]
                mod = step["modification"]
                bg  = step["band_gap"]
                unc = step["uncertainty"]
                alert = step["threshold_exceeded"]
                refl = step["reflection"]

                # Update progress
                progress_bar.progress(min(it / max_iters, 1.0))
                status_text.markdown(f"### Iteration {it}")

                # Show this step in a container
                with st.container():
                    st.markdown(f"**Modification:** `{mod}`")
                    st.markdown(f"**Predicted Band Gap:** {bg:.4f} eV")
                    st.markdown(f"**Uncertainty:** {unc:.4f} eV")
                    if alert:
                        st.warning("⚠️ Uncertainty above threshold – consider DFT re‑evaluation.")
                    st.markdown("**Reflection:**")
                    st.info(refl)

                results.append(step)

                # If we’ve already converged, break early
                if step["converged"]:
                    st.success(f"Converged in {it} steps!")
                    break

            # Final convergence check if never broke
            if not results[-1]["converged"]:
                st.warning(f"Did not converge within {max_iters} iterations.")

            # Plot theBand Gap convergence
            band_gaps = [r["band_gap"] for r in results]
            fig, ax = plt.subplots()
            ax.plot(range(1, len(band_gaps)+1), band_gaps, marker='o', label="Predicted Band Gap")
            ax.axhline(y=target_bandgap, color='r', linestyle='--', label="Target")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Band Gap (eV)")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during discovery: {e}")

