import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Membaca data dari file CSV
df = pd.read_csv('populasi_kabupaten_bekasi.csv')

# Luas wilayah Kabupaten Bekasi (km²)
area_kabupaten_bekasi = 2400  # km²

# Menghitung kepadatan penduduk untuk setiap tahun
df['Kepadatan Penduduk'] = df['Populasi'] / area_kabupaten_bekasi

# Mengatur format tampilan angka agar tidak menggunakan notasi ilmiah dan tanpa koma
pd.options.display.float_format = '{:.0f}'.format  # Tanpa koma

# Data tahun dan populasi untuk model prediksi
years = df['Tahun'].values.reshape(-1, 1)
population = df['Populasi'].values

# Membuat model regresi linier untuk memprediksi populasi
model = LinearRegression()
model.fit(years, population)

# Menentukan rentang tahun prediksi, misalnya dari tahun data terakhir sampai 2039
pred_years = np.arange(df['Tahun'].iloc[-1], 2040).reshape(-1, 1)

# Prediksi populasi untuk tahun-tahun prediksi
pred_population = model.predict(pred_years)

# Menghitung kepadatan prediksi untuk tahun-tahun tersebut
pred_density = pred_population / area_kabupaten_bekasi

# Menambahkan data prediksi ke dalam dataframe
pred_df = pd.DataFrame({
    'Tahun': pred_years.flatten(),
    'Populasi': pred_population,
    'Kepadatan Penduduk': pred_density
})

# Gabungkan data historis dan prediksi
df_combined = pd.concat([df, pred_df], ignore_index=True)

# Menampilkan output historis dan prediksi dengan format yang lebih jelas
print("Tahun  Populasi  Kepadatan Penduduk(jiwa/km²)")
for index, row in df_combined.iterrows():
    # Menampilkan data tanpa koma atau notasi ilmiah
    print(f"{int(row['Tahun']):<6} {int(row['Populasi']):<10} {int(row['Kepadatan Penduduk']):<17}")

# Visualisasi populasi dan kepadatan penduduk
plt.figure(figsize=(12, 6))

# Visualisasi populasi
plt.subplot(1, 2, 1)
plt.scatter(df_combined['Tahun'], df_combined['Populasi'], color='blue', label='Data Populasi', zorder=5)
plt.plot(df_combined['Tahun'], model.predict(df_combined['Tahun'].values.reshape(-1, 1)), color='red', label='Prediksi Populasi', linestyle='--')
plt.title('Prediksi Populasi Kabupaten Bekasi')
plt.xlabel('Tahun')
plt.ylabel('Populasi')
plt.legend()

# Visualisasi kepadatan penduduk
plt.subplot(1, 2, 2)
plt.scatter(df_combined['Tahun'], df_combined['Kepadatan Penduduk'], color='green', label='Kepadatan Penduduk', zorder=5)
plt.plot(df_combined['Tahun'], df_combined['Kepadatan Penduduk'], color='orange', label='Tren Kepadatan', linestyle='--')
plt.title('Kepadatan Penduduk Kabupaten Bekasi')
plt.xlabel('Tahun')
plt.ylabel('Kepadatan (jiwa/km²)')
plt.legend()

plt.tight_layout()
plt.show()
