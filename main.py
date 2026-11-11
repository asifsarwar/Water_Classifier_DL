import pickle
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model():
    with open("water-classifier-model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

features = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']
labels = ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity']

st.title("Water Classifier App using Deep Learning ANN")

st.text("""1. ph: pH of 1. water (0 to 14).
2. Hardness: Capacity of water to precipitate soap in mg/L.
3. Solids: Total dissolved solids in ppm.
4. Chloramines: Amount of Chloramines in ppm.
5. Sulfate: Amount of Sulfates dissolved in mg/L.
6. Conductivity: Electrical conductivity of water in μS/cm.
7. Organic_carbon: Amount of organic carbon in ppm.
8. Trihalomethanes: Amount of Trihalomethanes in μg/L.
9. Turbidity: Measure of light emiting property of water in NTU.
10. Potability: Indicates if water is safe for human consumption. Potable -1 and Not potable -0""")

df = pd.read_csv('water_potability.csv')
df_water = df.fillna(df.mean())
x = df_water[df_water.columns[0:-1]]
y = df_water['Potability']

event = st.dataframe(
    df_water,
    on_select = 'rerun',
    selection_mode = 'single-row'
)

rows = event.selection.rows

if rows and st.session_state.get('last_row') != rows[0]:
    row = df_water.iloc[rows[0]]
    for key, col in zip(features, x.columns):
        st.session_state[key] = float(row[col])
    st.session_state['last_row'] = rows[0]
elif not rows:
    st.session_state['last_row'] = None

for (label, key) in zip(labels, features):
    st.number_input(f'{label}', key=key, format='%.14f')


if st.button('Predict'):
    values = [st.session_state[key] for key in features]
    x_input = pd.DataFrame([values], columns=x.columns)
    st.write(x_input)
    model = load_model()
    proba = float(model.predict(x_input, verbose=0)[0][0])   # probability of potable
    pred = int(proba > 0.5)                                  # 1 = potable, 0 = not

    st.write(f'Prediction: {pred}')
    st.write(f'Probability of potable: {proba:.1%}')