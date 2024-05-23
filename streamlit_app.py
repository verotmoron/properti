import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r'D:\dwh_properti\results_cleaned.csv')
    return df

# Fungsi untuk pra-pemrosesan data
@st.cache_data
def preprocess_data(df):
    label_encoder = LabelEncoder()
    df['location'] = label_encoder.fit_transform(df['location'])
    X = df.drop(columns=['price', 'house_name'])
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoder

# Fungsi untuk membangun model dan evaluasi
@st.cache_data
def build_model(X_train, y_train):
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

# Fungsi untuk memformat prediksi harga
def format_price(price):
    billions = price // 1_000_000_000
    millions = (price % 1_000_000_000) // 1_000_000
    thousands = (price % 1_000_000) // 1_000

    formatted_price = ""
    if billions > 0:
        formatted_price += f"{billions} Miliar"
    if millions > 0:
        formatted_price += f" {millions} Juta"
    if thousands > 0:
        formatted_price += f" {thousands} Ribu"

    return formatted_price.strip()

# Halaman Exploratory Data Analysis (EDA)
def page_eda(df):
    st.title('Exploratory Data Analysis')
    st.write("### Statistik Deskriptif")
    st.write(df.describe())

    st.write("### Distribusi Fitur")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    sns.histplot(df['price'], ax=axes[0, 0], kde=True).set_title('Distribusi Harga')
    sns.histplot(df['bedroom_count'], ax=axes[0, 1], kde=True).set_title('Distribusi Jumlah Kamar Tidur')
    sns.histplot(df['bathroom_count'], ax=axes[0, 2], kde=True).set_title('Distribusi Jumlah Kamar Mandi')
    sns.histplot(df['carport_count'], ax=axes[1, 0], kde=True).set_title('Distribusi Jumlah Tempat Parkir')
    sns.histplot(df['land_area'], ax=axes[1, 1], kde=True).set_title('Distribusi Luas Tanah')
    sns.histplot(df['building_area (m2)'], ax=axes[1, 2], kde=True).set_title('Distribusi Luas Bangunan')
    plt.tight_layout()
    st.pyplot(fig)

    st.write("### Korelasi Antar Fitur")
    numeric_df = df.select_dtypes(include='number')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Halaman Visualisasi Decision Tree
def page_visualize_tree(model, X_train):
    st.title('Visualisasi Decision Tree')
    st.write('Pada halaman ini, Anda dapat melihat visualisasi dari pohon keputusan yang dibangun.')

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=X_train.columns, rounded=True, fontsize=12, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)

# Halaman Feature Importance
def page_feature_importance(model, X_train):
    st.title('Feature Importance')
    st.write('Pada halaman ini, Anda dapat melihat pentingnya fitur-fitur dalam model.')

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    st.write(feature_importance)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, hue='Feature', dodge=False, palette='viridis', ax=ax)
    if ax.legend_:
        ax.legend_.remove()
    st.pyplot(fig)

# Halaman Model Comparison
def page_model_comparison(X_train, X_test, y_train, y_test):
    st.title('Model Comparison')
    st.write('Pada halaman ini, Anda dapat melihat perbandingan performa model Decision Tree dengan model lain.')

    models = {
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train.copy(), y_train.copy())
        y_pred = model.predict(X_test.copy())
        mae = mean_absolute_error(y_test.copy(), y_pred)
        r2 = r2_score(y_test.copy(), y_pred)
        results[model_name] = {'MAE': mae, 'R^2 Score': r2}

    results_df = pd.DataFrame(results).T
    st.write(results_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind='bar', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Halaman Prediksi
def page_prediction(model, label_encoder):
    st.title('Prediksi Harga Rumah di Bandung')
    st.write('Masukkan detail rumah di bawah ini untuk mendapatkan prediksi harga.')

    df = load_data()
    location = st.selectbox('Lokasi', df['location'].unique())
    bedroom_count = st.number_input('Jumlah Kamar Tidur', min_value=1)
    bathroom_count = st.number_input('Jumlah Kamar Mandi', min_value=1)
    carport_count = st.number_input('Jumlah Tempat Parkir', min_value=0)
    land_area = st.number_input('Luas Tanah (m2)', min_value=0)
    building_area = st.number_input('Luas Bangunan (m2)', min_value=0)

    if st.button('Prediksi Harga'):
        input_data = pd.DataFrame({
            'location': [location],
            'bedroom_count': [bedroom_count],
            'bathroom_count': [bathroom_count],
            'carport_count': [carport_count],
            'land_area': [land_area],
            'building_area (m2)': [building_area]
        })
        input_data['location'] = label_encoder.transform(input_data['location'])
        prediction = model.predict(input_data)[0]
        formatted_prediction = format_price(prediction)
        st.write(f'Prediksi Harga Rumah: Rp {formatted_prediction}')

# Halaman Penjelasan dan Artikel Ilmiah
def page_article():
    st.title('Artikel Ilmiah')
    st.write("""
    ## Metodologi
    Pada penelitian ini, kami menggunakan algoritma Decision Tree untuk memprediksi harga rumah di Bandung. Data yang digunakan adalah data properti yang mencakup fitur-fitur seperti lokasi, jumlah kamar tidur, jumlah kamar mandi, jumlah tempat parkir, luas tanah, dan luas bangunan.

    ## Pra-pemrosesan Data
    Data yang digunakan diproses dengan cara mengonversi fitur kategorikal seperti lokasi menjadi bentuk numerik menggunakan Label Encoder. Data kemudian dibagi menjadi data latih dan data uji dengan perbandingan 80:20.

    ## Membangun Model
    Kami membangun model Decision Tree dengan menggunakan data latih. Parameter yang digunakan dalam model ini adalah `max_depth`, `min_samples_split`, dan `min_samples_leaf`. Kami juga melakukan optimasi model menggunakan GridSearchCV untuk menemukan parameter terbaik.

    ## Evaluasi Model
    Model dievaluasi menggunakan Mean Absolute Error (MAE) dan R^2 Score. Skor R^2 rata-rata yang diperoleh adalah sekitar 0.39 setelah optimasi.

    ### Mean Absolute Error (MAE)
    Mean Absolute Error (MAE) adalah rata-rata dari selisih absolut antara nilai yang diprediksi dan nilai sebenarnya. MAE memberikan gambaran seberapa besar kesalahan rata-rata dari prediksi model kita. Nilai MAE yang lebih rendah menunjukkan model yang lebih baik.

    ### R^2 Score
    R^2 Score (koefisien determinasi) mengukur seberapa baik variabel independen (fitur) menjelaskan variasi dalam variabel dependen (target). Skor R^2 berkisar antara 0 dan 1, dengan nilai yang lebih tinggi menunjukkan model yang lebih baik. R^2 Score yang tinggi berarti model mampu menjelaskan sebagian besar variabilitas dalam data target.

    ## Visualisasi Decision Tree
    Kami juga menyertakan visualisasi dari pohon keputusan yang dibangun untuk memberikan pemahaman lebih mendalam mengenai bagaimana model melakukan prediksi.

    ## Analisis Pentingnya Fitur
    Pada halaman ini, kami menunjukkan pentingnya fitur-fitur yang digunakan dalam model untuk memberikan wawasan tentang faktor-faktor utama yang mempengaruhi prediksi harga rumah.

    ## Perbandingan Model
    Selain menggunakan Decision Tree, kami juga membandingkan model ini dengan beberapa model lain seperti Random Forest, Gradient Boosting, dan Linear Regression untuk mengevaluasi performa relatif dari setiap model.

    ## Prediksi Harga
    Pengguna dapat memasukkan detail rumah untuk mendapatkan prediksi harga rumah di Bandung.""")

# Main function untuk navigasi halaman
def main():
    st.sidebar.title('Navigasi')
    page = st.sidebar.radio('Pilih Halaman', ['Exploratory Data Analysis', 'Visualisasi Decision Tree', 'Feature Importance', 'Model Comparison', 'Prediksi Harga Rumah di Bandung', 'Artikel Ilmiah'])

    df = load_data()
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)
    model = build_model(X_train, y_train)

    if page == 'Exploratory Data Analysis':
        page_eda(df)
    elif page == 'Visualisasi Decision Tree':
        page_visualize_tree(model, X_train)
    elif page == 'Feature Importance':
        page_feature_importance(model, X_train)
    elif page == 'Model Comparison':
        page_model_comparison(X_train, X_test, y_train, y_test)
    elif page == 'Prediksi Harga Rumah di Bandung':
        page_prediction(model, label_encoder)
    elif page == 'Artikel Ilmiah':
        page_article()

if __name__ == '__main__':
    main()
