from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model saat aplikasi dimulai
with open('tree_classifier.pkl', 'rb') as r:
    model = joblib.load(r)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.json

            usia_ibu = float(data['usia_ibu'])
            usia_kandungan = float(data['usia_kandungan'])
            golongan_darah = float(data['golongan_darah'])      
            rhesus = float(data['rhesus'])
            hamil_ke_brp = float(data['hamil_ke_brp'])
            jumlah_persalinan = float(data['jumlah_persalinan'])
            jumlah_keguguran = float(data['jumlah_keguguran'])
            kehamilan_diinginkan = float(data['kehamilan_diinginkan'])
            penggunaan_alkohol = float(data['penggunaan_alkohol'])
            perokok = float(data['perokok'])
            narkoba = float(data['narkoba'])
            polusi = float(data['polusi'])
            pendarahaan_pasca_lahir = float(data['pendarahaan_pasca_lahir'])
            pendarahan_ketika_hamil = float(data['pendarahan_ketika_hamil'])
            gadget = float(data['gadget'])
            riwayat_kelainan = float(data['riwayat_kelainan'])
            alergi = float(data['alergi'])
            pernah_caesar = float(data['pernah_caesar'])
            riwayat_caesar = float(data['riwayat_caesar'])
            riwayat_penyakit = float(data['riwayat_penyakit'])
            penyakit_turunan = float(data['penyakit_turunan'])

            input_data = np.array([usia_ibu, usia_kandungan, golongan_darah, rhesus, hamil_ke_brp, jumlah_keguguran, kehamilan_diinginkan, penggunaan_alkohol, perokok, narkoba, polusi, pendarahaan_pasca_lahir, pendarahan_ketika_hamil, gadget, riwayat_kelainan, alergi, pernah_caesar, riwayat_caesar, riwayat_penyakit, penyakit_turunan])
            input_data = input_data.reshape(1, -1)

            isJanin = model.predict(input_data)

            result = {'result': int(isJanin[0])}  # Konversi hasil prediksi ke integer

            return jsonify(result), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
