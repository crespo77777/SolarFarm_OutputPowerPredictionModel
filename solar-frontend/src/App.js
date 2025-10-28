import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

import Header from './components/Header';
import Controls from './components/Controls';
import PredictionChart from './components/PredictionChart';
import LoadingSpinner from './components/LoadingSpinner';

// URL vašeg backend servera - PROMIJENITE AKO JE POTREBNO
const API_URL = 'http://172.20.10.4:5000/predict-solar-power'; 

function App() {
  const [predictionData, setPredictionData] = useState(null); // null za inicijalno stanje
  const [selectedPeriodForChart, setSelectedPeriodForChart] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchPredictionData = async (period) => {
    setIsLoading(true);
    setError('');
    setPredictionData(null); // Očisti prethodne podatke prije novog dohvaćanja
    setSelectedPeriodForChart(period);

    try {
      console.log(`Šaljem zahtjev za period: ${period} na ${API_URL}`);
      const response = await axios.post(API_URL, { period });
      console.log('Odgovor primljen:', response.data);

      if (response.data && Array.isArray(response.data.predictions)) {
        if (response.data.predictions.length === 0) {
            // Backend je vratio prazan niz predikcija
            setPredictionData([]); // Postavi na prazan niz da PredictionChart može prikazati "Nema podataka"
            setError(''); // Očisti eventualnu prethodnu grešku
        } else {
            setPredictionData(response.data.predictions);
        }
      } else {
        // Odgovor nije u očekivanom formatu
        setError('Odgovor servera nije u očekivanom formatu.');
        setPredictionData([]); // Postavi na prazan niz za konzistentnost
      }
    } catch (err) {
      console.error('Greška prilikom API poziva:', err);
      let errorMessage = 'Dogodila se greška prilikom dohvaćanja podataka.';
      if (err.response) {
        // Server je odgovorio sa statusom izvan 2xx raspona
        errorMessage = `Greška od servera: ${err.response.status} - `;
        if (err.response.data && err.response.data.error) {
            errorMessage += `${err.response.data.error}`;
            if(err.response.data.status_details){
                errorMessage += ` (${err.response.data.status_details})`;
            }
        } else {
            errorMessage += err.response.statusText;
        }
      } else if (err.request) {
        // Zahtjev je poslan, ali nije primljen odgovor
        errorMessage = 'Server nije odgovorio. Provjerite je li backend pokrenut i dostupan na ' + API_URL;
      } else {
        // Nešto se dogodilo pri postavljanju zahtjeva što je izazvalo grešku
        errorMessage = `Greška pri slanju zahtjeva: ${err.message}`;
      }
      setError(errorMessage);
      setPredictionData([]); // Osiguraj da se ne prikazuje stari graf
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <Header />
      <main>
        <Controls onFetchPrediction={fetchPredictionData} isLoading={isLoading} />
        
        {isLoading && <LoadingSpinner />}
        
        {!isLoading && error && <p className="error-message">{error}</p>}
      
        {!isLoading && predictionData !== null && (
          <PredictionChart predictionData={predictionData} period={selectedPeriodForChart} />
        )}
        
        {!isLoading && !error && predictionData === null && (
          <div className="initial-message">
            Odaberite period i kliknite "Dohvati Prognozu i Predikciju" za prikaz podataka.
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
