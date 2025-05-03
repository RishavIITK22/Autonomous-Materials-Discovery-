base_template_bandgap = """
You are a materials design expert working on the development of new materials with specific properties.\
You will be given a composition (chemical formula) and its corresponding band gap.
You will be asked to propose a modification to the
material to achieve a target band gap.
A band gap is the distance \
between the valence band of electrons and the conduction band, \
representing the minimum energy that is required to excite an electron to the conduction band.\

Material Information:
(<chemical_formula>, <band_gap>)

Please propose a modification to the material that results in a band gap of <target_value> eV. \
You can choose one of the four following modifications:
1. exchange: exchange two elements in the material
2. substitute: substitute one element in the material with another
3. remove: remove an element from the material
4. add: add an element to the material

Your output should be a python dictionary of the following the format: \
{Hypothesis: $HYPOTHESIS, Modification: [$TYPE, $ELEMENT_1, $ELEMENT_2]}. \
Here are the requirements:
1. $HYPOTHESIS should be your analysis and reason for choosing a modification
2. $TYPE should be the modification type; one of "exchange", "substitute", "remove", "add"
3. $ELEMENT should be the selected element type to be modified. For "exchange" and "substitute", \
    two $ELEMENT placeholders are needed. For "remove" and "add", one $ELEMENT placeholder is needed.\n
<history>
Take a deep breath and work on this problem step-by-step. Your thoughtful and detailed analysis is highly appreciated."""

base_template_reflection="""After completing the following modification on the material, {previous_chemical_formula}, we obtained {current_chemical_formula}. 
                              The band gap value changed from {previous_value:.2f} eV to {current_value:.2f} eV. Please write a post-action reflection on
                              the modification in a short sentence on how successful the modification
                              was in achieving the target band gap value of {target_value} eV and why so:\n
                              <modification>"""


