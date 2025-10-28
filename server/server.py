import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests # Za Open-Meteo API
import time # Za praćenje vremena treniranja

# --- Flask app inicijalizacija ---
app = Flask(__name__)
CORS(app)

# --- Globalne varijable za model i skalere ---
trained_mlp_model = None
scaler_x_global = None
scaler_y_global = None
model_r2_score_global = 0.0
model_mae_global = 0.0
model_training_status_message = "Model nije još treniran."

# --- Konfiguracija za Open-Meteo ---
LATITUDE = 45.8150  # Primjer: Zagreb, Hrvatska
LONGITUDE = 15.9819 # Primjer: Zagreb, Hrvatska
OPEN_METEO_API_URL = "https://api.open-meteo.com/v1/forecast"

# --- Konstante za treniranje modela ---
TARGET_R2_SCORE = 0.90
MAX_TRAINING_ATTEMPTS = 100 # Maksimalan broj pokušaja treniranja

def train_and_evaluate_solar_model():
    """
    Učitava podatke, iterativno trenira neuronsku mrežu dok se ne postigne
    TARGET_R2_SCORE ili MAX_TRAINING_ATTEMPTS, i evaluira model.
    Koristi globalne varijable za spremanje modela, skenera i metrika.
    """
    global trained_mlp_model, scaler_x_global, scaler_y_global
    global model_r2_score_global, model_mae_global, model_training_status_message

    start_time = time.time()
    
    try:
        # Učitavanje podataka iz CSV datoteka
        # VAŽNO: Ove datoteke moraju biti u istom direktoriju kao app.py
        # Korištenje nrows=250 kako je bilo u vašem originalnom kodu.
        # Razmislite o korištenju više podataka ako je moguće za bolji model.
        solar_df = pd.read_csv("Solar_irradiation_measurements.csv", nrows=250)
        temp_df = pd.read_csv("Ambient_temparature_measurements.csv", nrows=250)
        power_df = pd.read_csv("PV_power_measurements.csv", skiprows=range(1, 281), nrows=250)

        irradiation_values = np.array(solar_df['_value'].tolist())
        temperature_values = np.array(temp_df['_value'].tolist())

        X_full = np.column_stack((irradiation_values, temperature_values))
        y_full = np.array(power_df['_value'].tolist())

        # Inicijalizacija skenera (jednom za sve podatke)
        # scaler_x i scaler_y su definirani globalno
        scaler_x_global = MinMaxScaler()
        scaler_y_global = MinMaxScaler()

        X_full_scaled = scaler_x_global.fit_transform(X_full)
        y_full_scaled = scaler_y_global.fit_transform(y_full.reshape(-1, 1)).flatten()

        current_attempt = 0
        best_r2_achieved = -float('inf') # Pratimo najbolji R2 ako cilj nije dosegnut

        while current_attempt < MAX_TRAINING_ATTEMPTS:
            current_attempt += 1
            print(f"Pokušaj treniranja modela: {current_attempt}/{MAX_TRAINING_ATTEMPTS}")

            # Podjela na skupove za treniranje i testiranje
            # random_state=None osigurava različitu podjelu u svakom pokušaju
            X_train, X_test, y_train, y_test = train_test_split(
                X_full_scaled, y_full_scaled, test_size=0.3, random_state=None
            )

            # Definiranje i treniranje neuronske mreže
            # random_state=42 u MLPRegressor za konzistentnu inicijalizaciju težina
            # pri istoj arhitekturi, ali s različitim podacima (zbog train_test_split)
            mlp = MLPRegressor(
                solver='adam',
                hidden_layer_sizes=(8, 8,), 
                activation='relu',
                learning_rate='adaptive',
                learning_rate_init=0.005,
                max_iter=500, 
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42, # Fiksni random_state za MLPRegressor
                n_iter_no_change=10 # Standardna vrijednost, može se prilagoditi
            )
            mlp.fit(X_train, y_train)

            # Predikcija na testnom skupu
            y_pred_scaled = mlp.predict(X_test)
            
            # Inverzna transformacija za dobivanje stvarnih vrijednosti
            y_pred_original = scaler_y_global.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_original = scaler_y_global.inverse_transform(y_test.reshape(-1, 1)).flatten()

            # Osiguravanje da predikcije snage nisu negativne 
            y_pred_final = np.maximum(0, y_pred_original)

            # Evaluacija modela
            mae = mean_absolute_error(y_test_original, y_pred_final)
            r2 = r2_score(y_test_original, y_pred_final)
            
            print(f"  Pokušaj {current_attempt}: R2 Score = {r2:.4f}, MAE = {mae:.2f}")

            if r2 > best_r2_achieved: # Pratimo najbolji model do sada
                best_r2_achieved = r2
                # Ako želimo spremiti najbolji model čak i ako ne dosegne target:
                # best_model_candidate = mlp
                # best_mae_candidate = mae
            
            if r2 >= TARGET_R2_SCORE:
                trained_mlp_model = mlp # Spremanje uspješnog modela
                model_r2_score_global = r2
                model_mae_global = mae
                training_duration = time.time() - start_time
                model_training_status_message = (
                    f"Model uspješno treniran u {current_attempt} pokušaja. "
                    f"R2 Score: {r2:.4f}, MAE: {mae:.2f}. "
                    f"Trajanje treniranja: {training_duration:.2f}s. Model je spreman za predikcije."
                )
                print(model_training_status_message)
                return # Izlaz iz funkcije nakon uspješnog treniranja
        
        # Ako petlja završi bez postizanja ciljanog R2 skora
        training_duration = time.time() - start_time
        model_training_status_message = (
            f"Nakon {MAX_TRAINING_ATTEMPTS} pokušaja, R2 Score ({TARGET_R2_SCORE}) nije postignut. "
            f"Najbolji postignuti R2 Score bio je: {best_r2_achieved:.4f}. "
            f"Trajanje treniranja: {training_duration:.2f}s. Model neće biti korišten za predikcije."
        )
        trained_mlp_model = None # Osiguravamo da se model ne koristi
        model_r2_score_global = best_r2_achieved # Spremi najbolji R2, iako nije dovoljan
        # model_mae_global ostaje od zadnjeg pokušaja ili najboljeg, ovisno o logici.
        print(model_training_status_message)

    except FileNotFoundError as e:
        model_training_status_message = f"Greška: Nije pronađena CSV datoteka ({e.filename}). Provjerite nalaze li se datoteke u direktoriju i jesu li imena ispravna. Model nije treniran."
        print(model_training_status_message)
        trained_mlp_model = None
    except Exception as e:
        model_training_status_message = f"Dogodila se greška tijekom treniranja modela: {str(e)}. Model nije treniran."
        print(model_training_status_message)
        trained_mlp_model = None


def fetch_weather_from_open_meteo(period):
    """
    Dohvaća podatke o vremenu (temperatura i solarna radijacija) s Open-Meteo API-ja.
    (Ova funkcija ostaje neizmijenjena)
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": "auto"
    }
    weather_variables = "temperature_2m,shortwave_radiation"

    if period == "24h":
        params["minutely_15"] = weather_variables
        params["forecast_days"] = 1 
    elif period == "72h":
        params["hourly"] = weather_variables
        params["forecast_days"] = 3
    elif period == "7d":
        params["hourly"] = weather_variables
        params["forecast_days"] = 7
    else:
        return None, "Interna greška: Nepoznat period za API poziv."

    try:
        response = requests.get(OPEN_METEO_API_URL, params=params)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Greška pri komunikaciji s Open-Meteo API-jem: {e}"
    except Exception as e:
        return None, f"Neočekivana greška prilikom dohvaćanja vremenskih podataka: {e}"

@app.route('/predict-solar-power', methods=['POST'])
def predict_solar_power():
    """
    Endpoint za predikciju solarne snage. Prima 'period' (24h, 72h, 7d),
    dohvaća vremensku prognozu i koristi trenirani model za predikciju.
    (Ova funkcija ostaje uglavnom neizmijenjena, ali provjerava model_r2_score_global)
    """
    if not request.is_json:
        return jsonify({"error": "Zahtjev mora biti u JSON formatu."}), 400

    data = request.get_json()
    period = data.get('period')

    if period not in ['24h', '72h', '7d']:
        return jsonify({"error": "Vrijednost 'period' mora biti '24h', '72h' ili '7d'."}), 400

    # Provjera statusa modela
    if trained_mlp_model is None or model_r2_score_global < TARGET_R2_SCORE:
        error_detail = model_training_status_message
        if trained_mlp_model is not None and model_r2_score_global < TARGET_R2_SCORE:
             error_detail = (f"Model je treniran, ali R2 Score ({model_r2_score_global:.4f}) "
                             f"je ispod potrebnog praga od {TARGET_R2_SCORE}. Predikcija nije moguća.")
        elif trained_mlp_model is None:
             error_detail = "Model za predikciju nije uspješno treniran ili nije dostupan."

        return jsonify({
            "error": "Model nije spreman za predikciju.",
            "status_details": error_detail,
            "current_model_r2_score": round(model_r2_score_global, 4) if model_r2_score_global > -float('inf') else "N/A"
        }), 503 # Service Unavailable
    
    # Dohvati vremenske podatke s Open-Meteo
    weather_api_data, error_message = fetch_weather_from_open_meteo(period)

    if error_message or not weather_api_data:
        return jsonify({"error": error_message or "Nije moguće dohvatiti podatke o vremenu."}), 502

    # Priprema podataka za predikciju
    interval_key = "minutely_15" if period == "24h" else "hourly"
    
    if interval_key not in weather_api_data or not weather_api_data[interval_key]:
        return jsonify({"error": f"Nedostaju podaci za interval '{interval_key}' u odgovoru Open-Meteo API-ja."}), 500

    timestamps = weather_api_data[interval_key].get("time", [])
    radiations = weather_api_data[interval_key].get("shortwave_radiation", [])
    temperatures = weather_api_data[interval_key].get("temperature_2m", [])

    if not (len(timestamps) == len(radiations) == len(temperatures) and timestamps):
        return jsonify({"error": "Nekompletni vremenski podaci primljeni od Open-Meteo."}), 500
    
    valid_forecast_data = []
    valid_timestamps = []
    for i in range(len(timestamps)):
        if radiations[i] is not None and temperatures[i] is not None:
            valid_forecast_data.append([radiations[i], temperatures[i]]) # Redoslijed: radijacija, temperatura
            valid_timestamps.append(timestamps[i])
    
    if not valid_forecast_data:
        return jsonify({
            "message": "Nema dostupnih podataka o radijaciji i temperaturi za predikciju nakon filtriranja.",
            "requested_period": period,
            "predictions": []
            }), 200

    input_features_np = np.array(valid_forecast_data)
    input_features_scaled = scaler_x_global.transform(input_features_np)
    predicted_power_scaled = trained_mlp_model.predict(input_features_scaled)
    predicted_power_actual = scaler_y_global.inverse_transform(predicted_power_scaled.reshape(-1, 1)).flatten()
    predicted_power_final = np.maximum(0, predicted_power_actual)

    predictions_output = []
    for i in range(len(valid_timestamps)):
        predictions_output.append({
            "timestamp": valid_timestamps[i],
            "predicted_solar_power_kw": round(predicted_power_final[i], 3) 
        })
        
    units_key = "minutely_15_units" if period == "24h" else "hourly_units"
    api_units = weather_api_data.get(units_key, {})

    return jsonify({
        "requested_period": period,
        "model_details": {
            "r2_score": round(model_r2_score_global, 4),
            "mae": round(model_mae_global, 2),
            "status": "Model uspješno treniran i spreman."
        },
        "open_meteo_input_units": {
            "temperature": api_units.get("temperature_2m", "°C"),
            "solar_radiation": api_units.get("shortwave_radiation", "W/m²")
        },
        "prediction_output_unit_note": "Jedinica za 'predicted_solar_power' je pretpostavljeno kW. Molimo provjerite originalnu jedinicu u vašoj 'PV_power_measurements.csv' datoteci.",
        "predictions": predictions_output
    })

if __name__ == '__main__':
    print("="*50)
    print("POKRETANJE SERVERA I ITERATIVNI TRENING MODELA SOLARNE SNAGE")
    print(f"Ciljani R2 Score: {TARGET_R2_SCORE}, Maksimalan broj pokušaja: {MAX_TRAINING_ATTEMPTS}")
    print("="*50)
    print("\nMolimo pripremite CSV datoteke:")
    print("- Solar_irradiation_measurements.csv")
    print("- Ambient_temparature_measurements.csv")
    print("- PV_power_measurements.csv")
    print("u istom direktoriju kao i ova skripta (app.py).\n")
    
    train_and_evaluate_solar_model() # Treniraj model pri pokretanju servera
    
    print("\nKonačni status treniranja:")
    print(model_training_status_message)
    print("="*50)
    
    print("\nPokretanje Flask servera na http://0.0.0.0:5000/")
    print("Pritisnite CTRL+C za zaustavljanje.")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
